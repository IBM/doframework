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

import json
import numpy as np
from numpy import linalg
from scipy.stats import beta
from collections.abc import Iterable
from numbers import Number
from typing import Callable, Any, List
import logging
from botocore.exceptions import ClientError
from scipy.stats import dirichlet

def is_jsonable(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, OverflowError):
        return False

def logger(func):
    
    def inner(*args,**kwargs):
        
        try:
            
            return func(*args,**kwargs)

        except ClientError as be:
            if 'logger_name' in kwargs:
                logger_name = kwargs['logger_name']
                if logger_name:
                    logger = logging.getLogger(logger_name) 
                    logger.error(f'Connection error in {func.__name__}.')
                    logger.error(be)
            if 'is_raised' in kwargs:
                is_raised = kwargs['is_raised']
                if is_raised:
                    raise be
    
        except AssertionError as ae:
            if 'logger_name' in kwargs:
                logger_name = kwargs['logger_name']
                if logger_name:
                    logger = logging.getLogger(logger_name) 
                    logger.error(f'Assertion failed in {func.__name__}.')
                    logger.error(ae)
            if 'is_raised' in kwargs:
                is_raised = kwargs['is_raised']
                if is_raised:
                    raise ae    

        except Exception as e:
            if 'logger_name' in kwargs:
                logger_name = kwargs['logger_name']
                if logger_name:
                    logger = logging.getLogger(logger_name) 
                    logger.error(f'Something went wrong in {func.__name__}....')
                    logger.error(e)
            if 'is_raised' in kwargs:
                is_raised = kwargs['is_raised']
                if is_raised:
                    raise e    
       
    return inner  

def flatten(arr: List) -> List:
    '''
    Flatten a list one level down.
    '''
    
    flat = []
    
    for a in arr:
        
        if isinstance(a,list):
            flat += a
        else:
            flat.append(a)
            
    return flat

def try_fail(try_max: int, fn: Callable[..., Any], **kwargs) -> Any:

    tries = 0
    val = None
    
    while tries<try_max: 
        try:
            val = fn(**kwargs)
            tries = try_max
        except:
            pass
        tries += 1

    return val

def scale(arr: Iterable) -> float:
    
    assert all([isinstance(a, Number) for a in arr]), 'Scaling only allowed for iterables of numbers.'
    
    log_arr = [np.log10(a) for a in arr if a]
    scale = 1 if len(log_arr)==0 else min([np.power(10,np.round(log_a)) for log_a in log_arr])
    
    return scale

def remove(vectors: np.array, vector: np.array) -> np.array:
    '''
    Remove vector from array of vectors.
    
            Parameters:
                    vectors (np.array): array of vectors.
                    vector (np.array): vector to be removed.
                    
            Returns:
                    New array of vectors.
                    
    '''    
    #### TODO:  assert same dimension
    return vectors[~np.all((vectors-vector)==0,axis=1)]

#### TODO: replace == with close enough to 0 !!!
def row_index_2d(arr: np.array, row: np.array) -> np.array:
    '''
    Return all indices of a given row in an array.
    
            Parameters:
                    arr (np.array): source array.
                    row (np.array): row to locate.
                    
            Returns:
                    Array of indices.                    
    '''        
    return np.where(np.all(arr == row,axis=1))[0]

def subarray(a: np.array, b: np.array):
    '''
    Produce array of indices i such that b[i,:] is in a.
    
            Parameters:
                    a (np.array): source array.
                    b (np.array): target array.
                    
            Returns:
                    Array of 0-axis indices of b.
                    
    '''        
    
    indices = [row_index_2d(b,row) for row in a]
    
    return np.array([i for i in indices if i.shape[0]>0],np.int32)

def degenerate(P: np.array, tolerance: float=1e-12) -> bool:
    '''
    Find if a set of vectors overdetermine their Span.
    
            Parameters:
                    P (np.array): array of vectors.
                    tolerance (float): threshold for singularity.
                    
            Returns:
                    True if vectors overdetermine their span, False otherwise.
                    
    '''        
    
    return np.isclose(linalg.det(P[:-1,:]-np.tile(P[-1:,:],(P.shape[-1],1))),0,atol=tolerance)

def order_stats(t: np.array, is_minimum: bool=True) -> float:
    '''
    Compare two equally-sized samples X_0 and X_1.
    
            Parameters:
                    t (np.array): array of dimension Nx2 with t=(X_0,X_1)
                    is_minimum (bool): whether to produce stats for X_0 > X_1 or X_0 < X_1.
                    
            Returns:
                    P[X_0 > X_1] if is_minimum, otherwise, P[X_0 < X_1].
                    
    '''        
    
    t = np.atleast_2d(t)
    N = t.shape[0]
    
    assert t.shape[-1]==2, 'Array must be of dimension Nx2.'
    
    return (t[:,0]>t[:,1]).sum()/N if is_minimum else (t[:,0]<t[:,1]).sum()/N

def minimum_stats(xs: List[float]):
    
    if not isinstance(xs, list):
        
        return None
    
    else:
        
        if not all([isinstance(x,Number) for x in xs]):
            
            return None
        
        else:
    
            total = sum(xs)
            num = len(xs)

            if num < 2:

                return 1

            else:

                return beta(1,num-1).cdf(min(xs)/total)

def sample_standard_simplex(d: int, N: int=1) -> np.array:
    '''
    Sample coefficients for convex span of canonical basis of d-dimensional space,
    i.e., sample a point in the (d-1)-dimensional standard simplex. Coefficients
    will be >= 0 and add up to  1.
    
            Parameters:
                    d (int): The resultant simplex has d-1 geometric dimension. It is embeded in d-dimensional space.
                    
            Returns:
                    Array of coefficients for convex span of canoncal d-dimensional basis.

    >>> sample_standard_simplex(1)
    array([1])
    >>> sample_standard_simplex(5)
    array([0.02709838, 0.3306366 , 0.02140897, 0.03307449, 0.58778157])
    >>> sample_standard_simplex(3,2)
    array([[0.25108299, 0.45567031, 0.2932467 ], [0.52033301, 0.45166188, 0.02800511]])
    '''
    
    alpha = np.ones(d)
    
    points = dirichlet(alpha).rvs().flatten() if N==1 else dirichlet(alpha).rvs(N)

    return points                                                                                 
