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

import itertools as it
import logging
from typing import Optional, List

import numpy as np
import numpy.linalg as la
from scipy.stats import rv_continuous
from scipy.stats import norm, uniform
from scipy.stats import gmean
from scipy.spatial import ConvexHull

from doframework.core.pwl import PWL
from doframework.core.hit_and_run import scale, get_hull, in_domain, rounding, hit_and_run

def X_hypothesis_sampler_legacy(hypothesis, I: int, weights: list, **kwargs):
    
    while True:
        
        i = np.random.choice(np.arange(I),replace=True,p=weights)
    
        yield np.atleast_2d(hypothesis(**{k: v[i] for k,v in kwargs.items()}).rvs())

def X_sampler_legacy(f: PWL, hypothesis, N: int, weights: list, **kwargs):
    '''
    Sample from a mixed distribution given a hypothesis. Samples will be in Supp(f).
    
            Parameters:
                    f (PWL): piecewise linear function.
                    hypothesis: multivariate distribution [scipy.stats].
                    N (int): sample size.
                    weights (list): weight of each distribution in the mix [adds up to 1].
                    
            Returns:
                    Sample of size N from mixed multivariate distribution defined by hypothesis and kwargs.
                    
    '''

    Is = [len(v) for v in kwargs.values()] + [len(weights)]
    
    assert min(Is) == max(Is), 'Unequal dimensions in kwargs and weights.'

    epsilon = 1e-10
    assert abs(sum(weights)-1)<epsilon, 'Weights must add up to 1.'
            
    return np.concatenate(list(it.islice(filter(lambda x: f.isin(x)[0],
                                                X_hypothesis_sampler_legacy(hypothesis,
                                                                    Is[0],
                                                                    weights,
                                                                    **kwargs)),N)))

def D_sampler_legacy(f: PWL, hypothesis, N: int, weights: list, noise: float, **kwargs):
    
    X = X_sampler_legacy(f,hypothesis,N,weights,**kwargs)
    y = f.evaluate(X) + norm(loc=0,scale=noise).rvs(size=X.shape[0])
    D = np.concatenate((X, y.reshape(-1,1)), 1)
    
    return D       

def sample_f_values(f_range: List, N: int=1) -> np.array:
    '''
    Sample function values from a given range.
    
            Parameters:
                    f_range (List): range of function values.
                    N (int): sample size.
                    
            Returns:
                    Sample of size N from uniform distribution defined by f_range.
                    
    '''
    
    return uniform(f_range[0],f_range[1]-f_range[0]).rvs(N)

def omega_hypothesis_sampler(hypothesis: rv_continuous, J: int, i: int, **kwargs):
    
    while True:
        
        yield [hypothesis(**{k: v[i][j] for k,v in kwargs.items()}).rvs() for j in range(J)]

def omega_sampler(f: Optional[PWL], hypothesis: rv_continuous, num_tries: int=10, tolerance: float=1e-8, **kwargs) -> np.array:
    '''
    Sample a single omega given a hypothesis. Sampled omega vertices will be in dom(f).
    
            Parameters:
                    f (PWL): PWL constrains samples to dom(f), otherwise no constraint when None.
                    hypothesis: continuous distribution [scipy.stats].
                    num_tries (int): number of tries to sample vertex in dom(f). Default is 10.
                    tolerance (float): tolerance to near zero. Default is 1e-8.
                    
            Returns:
                    Sample of shape JxI where J (number of vertices) and I (dimension) depend on loc and scale in kwargs.                    
    '''
    
    assert ('loc' in kwargs) and ('scale' in kwargs), 'kwargs must include loc and scale for hypothesis sampling.'
    
    Is = [v.shape[0] for v in kwargs.values()]
    Js = [v.shape[1] for v in kwargs.values()]
    
    assert list(map(min,[Is,Js])) == list(map(max,[Is,Js])), 'Unequal dimensions in kwargs.'
    
    I, J = Is[0], Js[0]
    
    locs = kwargs['loc']
    scales = kwargs['scale']

    # sample around origin and reflect to pos
    kwargs_at_zero = {'loc': np.zeros(locs.shape), 'scale': scales}

    omega_samples = []

    for i,loc in enumerate(locs):
        
        sample_in = False

        if f is not None:

            loc_in = False

            for poly in f.polylins:  # find loc poly
                if poly.ison(np.atleast_2d(loc),tolerance=tolerance): # reduce tol since convex hull shifted locs
                    loc_in = True
                    break

            if loc_in:
                
                for _ in range(num_tries): # try sampling                                     
                    
                    origin_sample = np.abs(next(omega_hypothesis_sampler(hypothesis,J,i,**kwargs_at_zero)))

                    X = np.atleast_2d(poly.points)
                    Xshift = X - np.atleast_2d(loc)
                    A = Xshift[~np.all(np.isclose(Xshift,0.0,atol=tolerance),axis=1)]
                    
                    assert A.shape[0] == A.shape[-1], 'More than one polyhedron vertex identified as loc, or no polyhedron vertex identified as loc.'

                    sample = origin_sample @ A + np.array(loc)  # map sample to poly

                    if f.isin(np.atleast_2d(sample)):
                        sample_in = True
                        break

        else:
            
            sample_in = True

            sample = np.array(loc) + np.abs(next(omega_hypothesis_sampler(hypothesis,J,i,**kwargs_at_zero)))

        if not sample_in: # did not locate loc in f or did not get sample in dom(f) after num_tries

            omega_samples = None

            break

        else:

            omega_samples.append(sample)
            
    return np.atleast_2d(omega_samples) if omega_samples is not None else omega_samples

def X_sampler(Ps: np.array, N: int, weights: list, **kwargs):
    '''
    Sample from a mixed Gaussian distribution restricted to a union of polytopes Ps. 
    Samples will be in the convex hull of the union of Ps.
    Sampling is done using the Hit & Run algorithm following the work of
    Ben Cousins in Efficient High-dimensional Sampling and Integration.
    
            Parameters:
                    Ps (np.array): Polytopes.
                    N (int): Overall sample size.
                    weights (list): Weight of each Gaussian in the mix [must add up to 1].
                    num_cpus (int): Number of CPUs to parallelize sampling (default: 1).

                    
            Returns:
                    N samples (np.array).
                    
    '''

    means = kwargs['mean']
    covariances = kwargs['cov']
    
    is_round = kwargs['is_round'] if 'is_round' in kwargs else True 
    round_threshold = kwargs['round_threshold'] if 'round_threshold' in kwargs else 0.1
    upper_bound = kwargs['upper_bound'] if 'upper_bound' in kwargs else np.inf 
    lower_bound = kwargs['lower_bound'] if 'lower_bound' in kwargs else 1.0
    T = kwargs['T'] if 'T' in kwargs else 1 
    tol = kwargs['tol'] if 'tol' in kwargs else 1e-8
    num_cpus = kwargs['num_cpus'] if 'num_cpus' in kwargs else 1 
    
    objective_id = kwargs['objective_id'] if 'objective_id' in kwargs else ''
    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    
    assert abs(sum(weights)-1)<tol, 'Weights must add up to 1.'

    lens = [len(means),len(covariances),len(weights)]    
    assert min(lens) == max(lens), 'Unequal number of policies, covariances, and weights.'    

    Ns = [int(w*N) for w in weights]
    Ns[-1] += N-sum(Ns)

    points = np.unique(np.vstack(Ps),axis=0)
    hull, _, _ = get_hull(points)
    d = points.shape[-1]
    
    Xs = []
    
    sample_slice = kwargs['sample_slice'] if 'sample_slice' in kwargs else slice(0,len(Ns))

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info(f'Sampling {N} data points for objective {objective_id} (mix time T={T}) with {num_cpus} cpus.')
        log.info(f'rounding={is_round}, rounding threshold={round_threshold:.3f}, sigma upper_bound={upper_bound:.3f}, sigma lower_bound={lower_bound:.3f}.')

    for n, mean, covariance in zip(Ns[sample_slice], means[sample_slice], covariances[sample_slice]):

        u, s, uT = la.svd(covariance,full_matrices=False)
        B = np.diag(s**(0.5)) @ uT # square root of covariance
        Binv = u @ np.diag(s**(-0.5))
        
        assert np.all(np.isclose(B@Binv, np.eye(d), atol=tol)), 'Root of covariance and its inverse multiply to I.'
        assert np.all(np.isclose(covariance, (u @ np.diag(s**(0.5))) @ (np.diag(s**(0.5)) @ uT), atol=tol)), 'Retrieve original covariance from SVD decomposition.'

        points_init = points @ B
        mean_init = mean @ B
        hull_init, _, simplices_init = get_hull(points_init)
        shift_init = mean_init
        scale_init = scale(mean_init, simplices_init[:,0,:], hull_init.equations[:,:-1])

        # shift & scale for unit ball around mean inside polytope
        points_tmp = (points_init-shift_init)/scale_init
        mean_tmp = (mean_init-shift_init)/scale_init

        # round polytope. no rounding iff A = I.
        if is_round:
            points_tmp, A = rounding(points_tmp,round_threshold,tol=tol)
        else:
            A = np.eye(d)
        Ainv = la.inv(A)
        Avals = la.eigvals(A)
        
        mean_tmp = (A@mean_tmp.T).T
        hull_tmp, _, simplices_tmp = get_hull(points_tmp)

        # shift & scale for unit ball around mean inside rounded polytope
        shift_tmp = mean_tmp
        scale_tmp = scale(mean_tmp, simplices_tmp[:,0,:], hull_tmp.equations[:,:-1])
        points_tmp = (points_tmp-shift_tmp)/scale_tmp
        mean_tmp = (mean_tmp-shift_tmp)/scale_tmp

        hull_tmp, _, _ = get_hull(points_tmp)

        # when A = I, sigma_sq is scaled to produce samples from cov.
        # when A != I, produced samples are not from cov, but from A @ cov @ A^T.
        # then, sigma_sq is additionally scaled by A evals.
        sigma = max(min(upper_bound,1/scale_init * gmean(Avals) * 1/scale_tmp),lower_bound)
        sigma_sq = sigma**2

        # sigma for warm start
        warm_sigma = kwargs['warm_sigma'] if 'warm_sigma' in kwargs else 0.1*sigma
        warm_sigma_sq = warm_sigma**2

        if logger_name:
            log = logging.getLogger(logger_name)
            log.info(f'Producing {n} samples at policy {mean} and sigma {sigma:.3f}.')

        # add epsilon to ensure sampled points are well inside 
        hull_tmp_equations_eps = hull_tmp.equations+np.hstack([np.zeros((hull_tmp.equations.shape[0],hull_tmp.equations.shape[-1]-1)),
                                                                        tol * np.ones(hull_tmp.equations.shape[0])[:,None]])
            
        X = hit_and_run(n,mean_tmp,hull_tmp_equations_eps,T,warm_sigma_sq,sigma_sq=sigma_sq,num_cpus=num_cpus,tol=tol)

        # reverse apply transformations
        X = scale_tmp*X+shift_tmp
        X = X@Ainv
        X = scale_init*X+shift_init
        X = X@Binv

        Xs.append(X)
    
    X = np.vstack(Xs)
    np.random.shuffle(X)
    
    assert np.all(in_domain(X, hull.equations, tol=tol)), f'SANITY: samples not inside polytope for objective {objective_id}!'
    
    if len(Ns[sample_slice])==len(Ns):
        assert np.unique(X,axis=1).shape[0]==N, f'SANITY: {np.unique(X,axis=1).shape[0]} unique samples generated, but required {N} samples.'
    
    return X

def D_sampler(f: PWL, N: int, weights: list, noise: float, **kwargs): 
        
    domain_scale = np.power(f.volume(),1/f.dim)
    
    X = X_sampler(f.Ps,N,weights,upper_bound=1.2*domain_scale,lower_bound=0.1*domain_scale,**kwargs)
    y = f.evaluate(X) + norm(loc=0,scale=noise).rvs(size=X.shape[0])
    D = np.concatenate((X, y.reshape(-1,1)), 1)
    
    return D       