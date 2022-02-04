import itertools
import numpy as np
from numpy import linalg
from scipy.spatial import ConvexHull
from typing import Callable, Any, List
from dataclasses import dataclass, field

from doframework.core.utils import sample_standard_simplex

def constraint(x: float, y: float, z: float, coeffs: np.array) -> np.array:
    
    return (coeffs @ np.array([x,y,z],dtype='object').T)

def box_sampler(mins,maxs,d):
    while True:
        yield np.random.uniform(mins,maxs).reshape(1,d)

def hyperplanePartition(H: np.array,
                        X: np.array, 
                        d: int, 
                        O: np.array=np.array([]),
                        scale_factor: float=1.0,
                        tolerance: float=1e-10) -> np.array:
    '''
    Identify the spatial relation of a vector X to a hyperplane H. Spatial relation is determined 
    by the generalized cross-product of hyperplane basis vectors and the vector, relative to a 
    given orientation O. The hyperplane basis vectors are {v_1-v_d,...,v_{d-1}-v_d}, 
    where {v_1,...,v_d} are the given vectors that determine the H. The orientation O determines which 
    side of the hyperplane is considered positive (i.e. "up") relative to this basis.
    
            Parameters:
                    H (np.array): array of d-dimensional hyperplanes determined by d+1 vectors.
                    X (np.array): 2D array of vectors to test in relation to the hyperplanes.
                    d (int): dimension.
                    O (np.array): array of orientation (+-1) per hyperplane. Set to +1 by default.
                    scale_factor (float): scale factor to assess how close we are numerically to 0. Defaults to 1.
                    tolerance (float): a matrix is singular when abs(determinant) < tolerance.
                    
            Returns:
                    2D numpy array with [i][j] == 1 / -1 / 0 iff X[i] lies "above" / "below" / on 
                    hyperplane H[j] relative to the orientation determined by O[j].
                    
    '''
    
    n = H.shape[0]
    m = X.shape[0]
    
    O = O if O.shape[0] else np.ones(n)

    SFirst = np.tile(H[:,:-1,:],(m,1,1,1))
    SLast = np.tile(np.tile(H[:,-1,:],(1,d)).reshape(n,d,d),(m,1,1,1)) 
    XRepeat = np.swapaxes(np.tile(X,(n,1,1)),0,1).reshape(m,n,1,d) 
    dets = linalg.det(np.append(SFirst,XRepeat,axis=2)-SLast) # shape m x n
    
    dets = dets/scale_factor

    return np.sign(np.where(np.isclose(dets,0.0,atol=tolerance),0.0,dets)*O)

def argopt(arrs: List[np.array], key: Callable[..., Any]) -> tuple:
    '''
    Find the index of argmax / argmin of list of numpy arrays of different shapes. 
    
            Parameters:
                    arrs (List[np.array]): list of numpy arrays of varying shapes. 
                    key (Callable): accepts either np.nanargmax or np.nanargmin.
                    
            Returns:
                    Index of argmax/argmin as a tuple. The 0-th entry indicates the list member where max/min was identified.

    >>> argopt([np.array([[1,2,3],[1,2,3]]),np.array([[5],[5],[5]])],np.nanargmin)
    (0,0,0)
    >>> argopt([np.array([[1,2,3],[1,2,3]]),np.array([[5],[5],[5]])],np.nanargmax)
    (1,0,0)
    '''

    arrs_float = [a.astype(float) for a in arrs] # np.pad converts np.nan for dtype int
    lmax = np.array([a.shape for a in arrs_float]).max(axis=0)    
    arrs_nanpad = np.stack([np.pad(a,
                                   np.vstack([np.zeros(lmax.shape),lmax-a.shape]).T.astype(int),
                                   'constant',
                                   constant_values=(0,np.nan)) for a in arrs_float])

    return np.unravel_index(key(arrs_nanpad, axis=None), arrs_nanpad.shape)

@dataclass
class Polyhedron:
    '''
    Class for polyhedra.
    '''
        
    points: np.array
    dim: int = field(init=False)
    hull: ConvexHull = field(init=False)
    vertices: np.array = field(init=False)
    simplices: np.array = field(init=False)
    orientation: np.array = field(init=False)
        
    def __post_init__(self):
        self.dim = self.points.shape[-1]
        self.hull = ConvexHull(self.points,qhull_options='QJ') # rescale option to avoid QHull errors    
        self.simplices = self.points[self.hull.simplices,:]
        self.vertices = self.points[self.hull.vertices,:]
        self.orientation = np.sign(hyperplanePartition(self.simplices,self.points,self.dim,scale_factor=self.volume()).sum(axis=0))

    def ison(self, X: np.array, tolerance: float=1e-12) -> np.array:
        '''
        Identify when vectors are vertices of the Polyhedron.
        
                Parameters:
                        X (np.array): array of vectors to test.
                        tolerance (float): A matrix will be considered singular when abs(determinant) < tolerance.
                        
                Returns:
                        1D array with [i] == True iff X[i] is vertex of Polyhedron.                        
        '''

        Y = np.atleast_2d(X)

        Y3D = np.swapaxes(np.vstack([[Y] for _ in range(self.points.shape[0])]),1,2)
        poly3D = np.swapaxes(np.swapaxes(np.tile(self.points,(Y.shape[0],1,1)),0,1),1,2)
        
        return np.any(np.isclose((poly3D-Y3D).sum(axis=1),0,atol=tolerance),axis=0)
        
    def isin(self, X: np.array, tolerance: float=1e-12) -> np.array:
        '''
        Identify when vectors are inside the Polyhedron (including the boundary).
        
                Parameters:
                        X (np.array): array of vectors to test.
                        tolerance (float): A matrix will be considered singular when abs(determinant) < tolerance.
                        
                Returns:
                        1D array with [i] == True iff X[i] is inside Polyhedron or on its boundary.                        
        '''

        # TODO: improve performance - if True at i for ison, remove from Y for hyperplanePartition calc
        Y = np.atleast_2d(X)
        YonPoly = self.ison(Y,tolerance=tolerance)
        YreltoPoly = hyperplanePartition(self.simplices,Y,self.dim,self.orientation,scale_factor=self.volume(),tolerance=tolerance)
        YposreltoPoly = np.all(YreltoPoly >= np.zeros(self.simplices.shape[0]),axis=1)

        return YposreltoPoly+YonPoly
    
    def sample(self, N: int=1) -> np.array:

        d = self.dim
        sample_standard = sample_standard_simplex(d+1,N)
        sample_proj = np.atleast_2d(sample_standard)[:,:-1]

        X = self.points

        A = (X-np.tile(X[-1:,:],(X.shape[0],1)))[:-1,:]
        sample_in = np.atleast_2d(sample_proj) @ A + X[-1,:]

        return sample_in

    def volume(self):
        
        return self.hull.volume

@dataclass
class PolyLinear(Polyhedron):
    '''
    Class for affine functions defined over polyhedra. Value arrays are assumed flat! [TBD: flatten]
    '''
        
    points: np.array
    values: np.array
    coeffs: np.array = field(init=False)
        
    def __post_init__(self):
        super().__post_init__()
        self.coeffs = linalg.lstsq(np.hstack([self.points,np.ones(self.points.shape[0]).reshape(-1,1)]),self.values,rcond=None)[0]

    def set_values(self, values: np.array):

        self.values = values
        
    def evaluate(self, X: np.array) -> np.array:
        
        return np.where(self.isin(X), np.hstack([X,np.ones(X.shape[0]).reshape(-1,1)])@self.coeffs, np.nan)

@dataclass
class PWL:
    '''
    Class for piecewise linear functions.
    '''
    
    Ps: List[np.array] #### TODO: enable init with a list of PolyLinear objects / enable init with 3D array
    Vs: List[np.array] # arrays must be flat !!
    polylins: List[PolyLinear] = field(init=False)
    dim: int = field(init=False)

    def __post_init__(self):
        self.polylins = [PolyLinear(points,values) for points,values in zip(self.Ps,self.Vs)]
        dims = [poly.dim for poly in self.polylins]
        assert min(dims) == max(dims),'Unequal dimensions in polyhedra points.'
        self.dim = dims[0]
        
    def isin(self, X: np.array) -> np.array:
        return np.any(np.array([polylin.isin(X) for polylin in self.polylins]),axis=0)

    def evaluate(self, X: np.array) -> np.array:

        # [i][j] == 1 iff i = min_j {j|p_j in V(P_i)}
        XinPoly = np.array([polylin.isin(X) for polylin in self.polylins])
        XIndex = np.zeros(XinPoly.shape)
        XIndex[XinPoly.argmax(axis=0),np.arange(XinPoly.shape[-1])] = 1 # single 1 on first Poly with X
        vals = np.einsum('ij,ji->i', 
                         XIndex.T, 
                         np.nan_to_num(np.array([polylin.evaluate(X) for polylin in self.polylins])))

        poly_vals = [self.Vs[i] for i in XIndex.argmax(axis=0)]
        poly_val_mins = np.array([vals.min() for vals in poly_vals])
        poly_val_maxs = np.array([vals.max() for vals in poly_vals])
        legit_vals = (vals >= poly_val_mins)*(vals <= poly_val_maxs)        
        
        return np.where(self.isin(X)*legit_vals, vals, np.nan)
    
    def argmax(self) -> np.array:
        
        argind = argopt(self.Vs,np.nanargmax)
        
        return self.Ps[argind[0]][argind[1]]
    
    def argmin(self) -> np.array:
        
        argind = argopt(self.Vs,np.nanargmin)
        
        return self.Ps[argind[0]][argind[1]]
    
    def sample(self, N: int=1) -> np.array:

        d = self.dim

        vols = np.array([poly.volume() for poly in self.polylins])
        f_weights = vols/vols.sum()

        I = [*range(len(self.polylins))] # list of polyhedra indices

        indices = np.random.choice(I,size=N,replace=True,p=f_weights) # chosen polyhedra indices by weight

        sample_standard = sample_standard_simplex(d+1,N)
        sample_proj = np.atleast_2d(sample_standard)[:,:-1] # project standard (d-1)-simplex on (d-1)-dimensional Euclidean space

        X = np.array([self.polylins[i].points for i in indices]) # 3D array, [1..N] on the 0th axis
        A = (X-np.tile(X[:,-1:,:],(X.shape[1],1)))[:,:-1,:] # linear transformation from projected standard simplex to shifted polyhedron
        sample_in = (np.atleast_2d(sample_proj)[:,None,:] @ A + X[:,-1,:][:,None,:]).reshape(N,d) # affine transformation applied to projected samples

        return sample_in # 2D array
        
    def volume(self):
        
        return sum([poly.volume() for poly in self.polylins])
    