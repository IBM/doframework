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

from typing import Optional, List
import numpy as np
from scipy.stats import rv_continuous
from scipy.stats import norm, uniform
import itertools as it

from doframework.core.pwl import PWL, Polyhedron

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

def X_hypothesis_sampler(hypothesis, I: int, weights: list, **kwargs):
    
    while True:
        
        i = np.random.choice(np.arange(I),replace=True,p=weights)
    
        yield np.atleast_2d(hypothesis(**{k: v[i] for k,v in kwargs.items()}).rvs())

def X_sampler(f: PWL, hypothesis, N: int, weights: list, **kwargs):
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
                                                X_hypothesis_sampler(hypothesis,
                                                                    Is[0],
                                                                    weights,
                                                                    **kwargs)),N)))

def D_sampler(f: PWL, hypothesis, N: int, weights: list, noise: float, **kwargs):
    
    X = X_sampler(f,hypothesis,N,weights,**kwargs)
    y = f.evaluate(X) + norm(loc=0,scale=noise).rvs(size=X.shape[0])
    D = np.concatenate((X, y.reshape(-1,1)), 1)
    
    return D       

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
