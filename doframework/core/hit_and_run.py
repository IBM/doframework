#
# Copyright IBM Corporation 2022
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from itertools import islice, combinations
from typing import Optional
from functools import partial
# from multiprocess import Pool
from ray.util.multiprocessing import Pool

import numpy as np
import numpy.linalg as la
from scipy.stats import norm, uniform
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def box_sample(box_lows, box_highs, d, N: int=1) -> np.array:
    '''
    Sample N points of dimension d in a box whose lower bounds are given by box_lows and whose upper bounds are given by box_highs.
    '''
    
    return np.random.uniform(box_lows,box_highs,(N,d))

def scale(x: np.array, points: np.array, normals: np.array):
    
    shifts = x.flatten() - points
    dists = [abs(normal @ shift.T) for normal, shift in zip(normals, shifts)]
    r = min(dists)
    
    return min([r,1])

def sphere_sample(d: int, N: int=1, r: float=1.):
    
    us = np.random.normal(size=(N,d))
    norms = la.norm(us,axis=1)[:,None]
    
    return r * (us/norms) 

def ball_sample(d: int, N: int=1, R: float=1.):
    
    us = np.random.normal(size=(N,d))
    norms = la.norm(us,axis=1)[:,None]
    rs = R*np.power(np.random.uniform(size=N),1/d)[:,None]
    
    return rs * (us/norms) 

def projection(x, x1, x2) -> np.array:
    '''
    Produce the projection of point x onto the line between x1 and x2.
    
            Parameters:
                    x (np.array): point.
                    x1 (np.array): line point.
                    x2 (np.array): line point.
                    
            Returns:
                    projection point.
                    
    '''    
    
    nhat = (x2-x1)/np.sqrt((x2-x1)@(x2-x1).T)
    
    return x1 + ((x-x1)@nhat.T)*nhat

def chord_intersection(x: np.array, ray: np.array, A: np.array):
    '''
    Produce the two intersection points of a convex polytope with a line.
    The line is given by a point x and a unit vector ray in the line direction.
    The convex polytope is given by the matrix A such that Ax <= 0 defines the convex polytope. 
    Each row of A[:,:-1] is a normal to a face of the convex polytope.
    We assume x is inside the given polytope!
    
            Parameters:
                    x (np.array): point.
                    ray (np.array): direction.
                    A (np.array): matrix defining a convex polytope Ax <= 0.
                    
            Returns:
                    x_minus, x_plus the two intersection points.
                    
    '''    
    
    pos_mask = A[:,:-1] @ ray.T >= 0
    pos_hull = A[pos_mask]
    ts_plus = ((-pos_hull[:,-1:] - (pos_hull[:,:-1]@x.T)[:,None]) / (pos_hull[:,:-1]@ray.T)[:,None]).min()

    neg_mask = A[:,:-1] @ ray.T <= 0
    neg_hull = A[neg_mask]
    ts_minus = ((-neg_hull[:,-1:] - (neg_hull[:,:-1]@x.T)[:,None]) / (neg_hull[:,:-1]@ray.T)[:,None]).max()

    x_minus = ts_minus*ray+x
    x_plus = ts_plus*ray+x
    
    return x_minus, x_plus

def get_hull(points: np.array):

    hull = ConvexHull(points,qhull_options='QJ')
    vertices = points[hull.vertices,:]
    simplices = points[hull.simplices,:]
    
    return hull, vertices, simplices

def _singular(points: np.array):
    
    M = points.T
    U, S, _ = la.svd(M,full_matrices=False)

    svalmax = S[0]
    svecmax = U[:,0][:,None]
    svalmin = S[-1]    
    svecmin = U[:,-1][:,None]

    return svalmax, svecmax, svalmin, svecmin

def _tranformation(svalmax: float, svecmax: np.array, d: int):
    
    return np.eye(d) + (1/svalmax-1) * svecmax@svecmax.T

def rounding(points: np.array, threshold: float=0.1, M: int=500, tol: float=1e-8):
    '''
    Round a convex polytope. The polytope is given as the convex hull of points.
    We assume the polytope contains a the unit ball around the origin.
    Rounding will terminate once the diff between the high and low singular values of points is below 
    the threshold times the high singular value. The lower the ratio (sval_high-sval_low)/sval_high,
    the rounder we consider the polytope to be.
    
            Parameters:
                    points (np.array): points.
                    threshold (float): stopping condition for rounding (default: 10% or 0.1).
                    M (int): terminate after M rounding attempts (default: 500).
                    
            Returns:
                    round_points: the new points after rounding.
                    T: the rounding linear transformation. Reverse-apply with round_points@la.inv(T).
                    
    '''    

    svalmax, svecmax, svalmin, _ = _singular(points)

    d = points.shape[-1]
    T = np.eye(d)
    round_points = points
    attempts = 1

    while (svalmax-svalmin)/svalmax > threshold:
        A = _tranformation(svalmax, svecmax, d)
        round_points, T = (A@round_points.T).T, A@T
        svalmax, svecmax, svalmin, _ = _singular(round_points)
        if attempts>M:
            print(f'Reached max {M} rounding attempts. Rounding failed to reach the rounding threshold {threshold:.3f}.')
            break
        else:
            attempts += 1
        
    assert np.all(np.isclose((T@(points).T).T,round_points,atol=tol)), \
    'Something went wrong ... transformation T should produce rounded points from initial points.'
    
    return round_points, T

def in_domain(xs, A: np.array, R: Optional[float]=None, **kwargs) -> np.array:
    '''
    Check whether xs are inside the intersection of a convex polytope and a ball of radius R.
    The convex polytope is given by the matrix A such that Ax<=0 defines it.
    When the radius is not specified, this restriction is dropped.
    
            Parameters:
                    xs (np.array): points.
                    A (np.array): matrix defining a convex polytope Ax <= 0.
                    R (float): radius (default: None).
                    
            Returns:
                    a boolean numpy array that indicates whether the points are inside the polytope.
                    
    '''    
    
    tol = kwargs['tol'] if 'tol' in kwargs else 1e-8
    N = xs.shape[0]
    multi = A @ np.vstack([xs.T,np.ones(N)[None,:]])
    multi = np.where(np.isclose(multi,0,atol=tol),0,multi)
    
    inside = np.all(multi <= 0, axis=0) * (la.norm(xs,axis=1) <= R) if R else np.all(multi <= 0, axis=0)
    
    return inside 

def _warm_yield(origin: np.array, sigma_sq: float):
    
    d = origin.shape[-1]
    
    while True:
        
        yield multivariate_normal(mean=origin.flatten(), cov=sigma_sq*np.eye(d)).rvs()

def warm_start(mu: np.array, A: np.array, sigma_sq: float, N: int=1, **kwargs) -> np.array:
    '''
    Sample a spherical Gaussian inside a convex polytope by exclusion.
    The convex polytope is given by the matrix A such that Ax<=0 defines it.
    The spherical gaussian is centered at mu with variance sigma_sq.
    
            Parameters:
                    mu (np.array): center of spherical Gaussian.
                    A (np.array): matrix defining a convex polytope Ax <= 0.
                    sigma_sq (float): variance of spherical Gaussian.
                    
            Returns:
                    numpy array of samples.
                    
    '''    
            
    return np.array(list(islice(filter(lambda x: in_domain(x[None,:],A,**kwargs)[0],_warm_yield(mu,sigma_sq)),N)))

def _chord_interval(mu: np.array, x1: np.array, x2: np.array) -> tuple:
    '''
    Project line passing through x1 and x2 to the reals with mu at the origin.
    mu is assumed to be a point on the line.
    
            Parameters:
                    mu (np.array): point.
                    x1 (np.array): point on chord end.
                    x2 (np.array): point on chord end.

                    
            Returns:
                    a 2-tuple (a,b) indicating the real coordinates of x1 and x2 projections relative to mu.
                    
    '''
    
    if (x2-mu)@(mu-x1).T>0: # mu between x1 and x2
        a, b = -la.norm(mu-x1), la.norm(x2-mu)
        
    elif (x2-x1)@(x1-mu).T>0: # x1 between mu and x2
        a, b = la.norm(x1-mu), la.norm(x2-mu)
        
    elif (x1-x2)@(x2-mu).T>0: # x2 between mu and x1
        a, b = -la.norm(x1-mu), -la.norm(x2-mu)
        
    else:
        a, b = None, -la.norm(x2-mu)
                    
    return a, b
    
def chord_sample(mu: np.array, x1: np.array, x2: np.array, **kwargs):    
    '''
    Sample a single point along a chord with end points x1 and x2. 
    The point mu is located on a line defined by x1 and x2 (not necessarily in between, i.e., on the chord).
    When variance (sigma_sq) is provded in kwargs, point is sampled from a 1D Gaussian along the chord centered at mu.
    Otherwise, point is sampled uniformaly along the chord.
    
            Parameters:
                    mu (np.array): point.
                    x1 (np.array): point on chord end.
                    x2 (np.array): point on chord end.

                    
            Returns:
                    a single sample (np.array) from the chord.
                    
    '''
                
    (a,b) = _chord_interval(mu,x1,x2)        
    chord_length = b - a # must be positive!
    chord_normal = (x2-x1)/ chord_length
        
    if 'sigma_sq' in kwargs: # sample from gaussian projected on chord
                
        sigma_sq = kwargs['sigma_sq']
        chord_gaussian = norm(loc=0.0, scale=np.sqrt(sigma_sq))
        
        if chord_length > 2*np.sqrt(sigma_sq):
            c = chord_gaussian.rvs()
            accept = c<=b and c>=a
            y = mu + c*chord_normal
        else:
            c = box_sample([a,0],[b,1],2).flatten()
            accept = chord_gaussian.pdf(c[0]) >= c[1]
            y = mu + c[0]*chord_normal
            
    else: # sample uniformly from chord
        
        chord_uniform = uniform(loc=a, scale=chord_length)    
        
        c = chord_uniform.rvs()
        accept = True
        y = mu + c*chord_normal
                            
    return y, accept

def _hit_and_run_single(origin: np.array, A: np.array, delta: float, T: int=1, tol: float=1e-8, dist_dict: dict={}) -> np.array:
    '''
    Sample a single point from a spherical Gaussian or the uniform distribution restricted to a convex polytope given by Ax<=0.
    The spherical gaussian is centered at origin with variance sigma_sq given in dist_dict.
    When no variance (sigma_sq) is provded in dist_dict, point is sampled uniformaly.
    Sampling is done using the Hit & Run algorithm following Ben Cousins in Efficient High-dimensional Sampling and Integration (PhD Thesis, 2017).
    
            Parameters:
                    origin (np.array): center of spherical Gaussian.
                    A (np.array): matrix defining a convex polytope Ax <= 0.
                    delta (float): variance of spherical Gaussian for warm start.
                    T (int): random walk mixing time (default: 1).
                    tol (float): tolerance level to near zero results (default: 1e-8).
                    dist_dict (dict): dictionary containing distribution info (default: {}, i.e., sample uniformaly).

                    
            Returns:
                    a single sample (np.array).
                    
    '''
    
    np.random.seed() # prevent sampling from same random state on different pool threads
    
    x = warm_start(origin,A,delta,tol=tol).flatten()
    d = origin.shape[-1]
    
    count = 1

    while count<=T:

        ray = sphere_sample(d).flatten()
        x1, x2 = chord_intersection(x,ray,A)
        mu = projection(origin,x1,x2)

        try:
            if 'sigma_sq' in dist_dict:                    
                y, accept = chord_sample(mu,x1,x2,sigma_sq=dist_dict['sigma_sq'])
            else:
                y, accept = chord_sample(mu,x1,x2)
        except:
            continue

        if accept:
            x = y   
            count += 1
    
    return x

def hit_and_run(N: int,
                origin: np.array,
                A: np.array, 
                T: int=1,
                delta: float=0.1,
                tol: float=1e-8,
                **kwargs):

    num_cpus = kwargs['num_cpus'] if 'num_cpus' in kwargs else 1

    dist_tuples = tuple({'sigma_sq': kwargs['sigma_sq']} for _ in range(N)) if 'sigma_sq' in kwargs else tuple({} for _ in range(N))

    with Pool(processes=num_cpus) as pool:

        res = pool.map_async(partial(_hit_and_run_single,origin,A,delta,T,tol), dist_tuples)
        xs = res.get()

    return np.array(xs)

def hit_and_run_square_test(N: int=5000, T: int=20, sigma_sq: float=1., L: float=10.0, delta: float=0.1, 
                            linspace_num: int=500, coord_projection: int=0, figsize: tuple=(7,8), num_cpus: int=4, tol: float=1e-8):
    '''
    Test the Hit & Run algorithm by sampling in a large box a relatively small spherical Gaussian.
    Multivariate normality is tested with the Henze-Zikler test.
    When no variance (sigma_sq) is provded in dist_dict, point is sampled uniformaly.
    Sampling is done using the Hit & Run algorithm following Ben Cousins in Efficient High-dimensional Sampling and Integration (PhD Thesis, 2017).
    
            Parameters:
                    origin (np.array): center of spherical Gaussian.
                    A (np.array): matrix defining a convex polytope Ax <= 0.
                    delta (float): variance of spherical Gaussian for warm start.
                    T (int): random walk mixing time (default: 1).
                    tol (float): tolerance level to near zero results (default: 1e-8).
                    dist_dict (dict): dictionary containing distribution info (default: {}, i.e., sample uniformaly).

                    
            Returns:
                    a single sample (np.array).
                    
    '''

    import pingouin as pg # Henze-Zikler test of multivariate normality
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    
    points = np.array([[-L,-L],[-L,L],[L,-L],[L,L]])
    d = points.shape[-1]

    hull = ConvexHull(points,qhull_options='QJ')
    vertices = points[hull.vertices,:]
    origin = np.average(vertices,axis=0)[None,:]

    X = hit_and_run(N,origin,hull.equations,T,delta,sigma_sq=sigma_sq,num_cpus=num_cpus,tol=tol)
    henze_zirkler_test = pg.multivariate_normality(X, alpha=.05)
    print('Henze-Zirkler multivariate normality sample test:',henze_zirkler_test.normal)

    x = X[-1,:]
    ray = sphere_sample(d).flatten()
    x1, x2 = chord_intersection(x,ray,hull.equations)
    mu = projection(origin,x1,x2)
    interval = np.vstack([x1,x2])
    y, accept = chord_sample(mu, x1, x2, sigma_sq=sigma_sq)

    xproj = X[:,coord_projection]
    xmin = xproj.min()
    xmax = xproj.max()
    kde = gaussian_kde(xproj)
    xs = np.linspace(xmin,xmax,linspace_num)
    density = kde.pdf(xs)

    fig = plt.figure(figsize=figsize)
    
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(xs,density,label='H&R Samples')
    ax1.plot(xs,norm(loc=0.,scale=sigma_sq**(0.5)).pdf(xs),label='Gaussian Samples')
    ax1.set_title(f'Hit & Run vs. Spherical Gaussian Samples with STD {sigma_sq**(0.5)} ({N} samples projected onto the {coord_projection}-th coordinate)')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(2,1,2)
    convex_hull_plot_2d(hull, ax=ax2)
    ax2.plot(np.atleast_2d(interval)[:,0],np.atleast_2d(interval)[:,1],c='pink',label='chord')
    ax2.scatter(np.atleast_2d(interval)[:,0],np.atleast_2d(interval)[:,1],s=60,c='pink')
    ax2.scatter(np.atleast_2d(X)[:,0],np.atleast_2d(X)[:,1],s=20,c='y',alpha=0.3,label='Xs')
    ax2.scatter(np.atleast_2d(origin)[:,0],np.atleast_2d(origin)[:,1],s=80,c='black',label='origin')
    ax2.scatter(np.atleast_2d(y)[:,0],np.atleast_2d(y)[:,1],s=40,c='r',label='y')
    ax2.scatter(np.atleast_2d(mu)[:,0],np.atleast_2d(mu)[:,1],s=40,c='b',label='mu')
    ax2.set_title(f'Hit & Run sample along a chord (sample accepted={accept})')
    ax2.grid()
    ax2.legend(loc='lower right')  
    
    plt.show()

def plot_polytope(hull: ConvexHull, origin: np.array=np.array([]), X: np.array=np.array([]), figsize: tuple=(7,7), elevation: int=25, azimute: int=150, **kwargs):

    import matplotlib.pyplot as plt
    
    points = hull.points
    d = points.shape[-1]
    plot_sample = False
    
    if X.size>0:
        assert X.shape[-1]==d, f'Dimension mismatch between samples of dimension {X.shape[-1]} and origin of dimension {d}.'
        plot_sample = True
        
    plot_origin = origin.size>0
        
    if d == 2:        

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        convex_hull_plot_2d(hull, ax=ax)
        
        if plot_sample:
            ax.scatter(np.atleast_2d(X)[:,0],np.atleast_2d(X)[:,1],s=20,c='y',alpha=0.3,label='Xs')
            
        if plot_origin:
            ax.scatter(np.atleast_2d(origin)[:,0],np.atleast_2d(origin)[:,1],s=80,c='black',label='origin')

        ax.scatter(np.atleast_2d(points)[:,0],np.atleast_2d(points)[:,1],color='b',s=40,label='points')        
            
        plt.grid()
        plt.legend(loc='lower right')
        
        if 'xlims' in kwargs:
            plt.xlim(kwargs['xlims'])
        if 'ylims' in kwargs:
            plt.ylim(kwargs['ylims'])
            

    if d==3:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
                
        simplices = points[hull.simplices,:]

        for i,s in enumerate(simplices):

            edges = [np.vstack(e) for e in combinations(s, 2)]

            for j,edge in enumerate(edges):
                if i+j==0:
                    ax.plot(edge[:,0],edge[:,1],edge[:,2],color='grey',label='polytope')
                else:
                    ax.plot(edge[:,0],edge[:,1],edge[:,2],color='grey')

        if plot_sample:
            ax.scatter(np.atleast_2d(X)[:,0],np.atleast_2d(X)[:,1],np.atleast_2d(X)[:,2],c='y',s=20,alpha=0.3,label='Xs')
        if plot_origin:
            ax.scatter(np.atleast_2d(origin)[:,0],np.atleast_2d(origin)[:,1],np.atleast_2d(origin)[:,2],color='black',s=40,label='origin')  
        ax.scatter(np.atleast_2d(points)[:,0],np.atleast_2d(points)[:,1],np.atleast_2d(points)[:,2],color='b',s=40,label='points')        

        ax.view_init(elev=elevation, azim=azimute)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.draw()
        plt.legend()