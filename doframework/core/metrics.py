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

from typing import Optional
import numpy as np
from scipy.stats import rv_continuous
from scipy.spatial import ConvexHull
from scipy.stats import norm
from scipy.stats import multivariate_normal

from doframework.core.pwl import PWL
from doframework.core.sampler import omega_sampler

#### !!! BAD !!! ADDRESS ANALYTICALLY !!!
def prob_in_omega(x: np.array, f: Optional[PWL], hypothesis: rv_continuous, N: int=1000, **kwargs) -> np.array:
    '''
    Estimate the probability of x in Omega given the hypothesis on the Omega distribution.
    
            Parameters:
                    x (np.array): a single vector to be tested.
                    f (PWL): piecewise linear function used to constrain omega samples to Supp(f), otherwise f=None.
                    hypothesis: multivariate distribution [scipy.stats].
                    N (int): number of samples drawn from Omega distribution.
                    
            Returns:
                    Estimate of the probability that x is in Omega.
                    
    '''    
    
    omegas = [omega_sampler(f, hypothesis, **kwargs) for _ in range(N)]
    d = omegas.shape[1] # dimension    
    
    equations = [ConvexHull(omega.T).equations for omega in omegas.T]    
    n = max([eqns.shape[0] for eqns in equations]) # max num of eqns of omega
    
    A = np.concatenate([np.pad(eqns,((0,n-eqns.shape[0]),(0,0))).reshape(1,n,d+1) for eqns in equations],axis=0)    
    B = np.atleast_3d(np.tile(np.hstack([x.flatten(),np.array([1])]),(N,1)))    
    C = A@B # evaluate x on each set of omega eqns
    
    return np.all(C<=0,axis=1).sum()/N
