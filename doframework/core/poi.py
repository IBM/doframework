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

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from GPy.models import GPRegression

from doframework.core.utils import order_stats

def plot_joint_distribution(samples: np.array, **kwargs):

    import seaborn as sns
    
    assert samples.shape[-1]==2, 'Array must be of dimension Nx2. Array is a draw from a bivariate distribution.'
    
    cols = kwargs['cols'] if 'cols' in kwargs else [f'x{i}' for i in range(2)]

    if 'x_min' not in kwargs or 'x_max' not in kwargs:
        x_min = samples.min(axis=0).min()*0.5
        x_max = samples.max(axis=0).max()*1.5
    else:
        x_min = kwargs['x_min']
        x_max = kwargs['x_max']

    lims = np.array([x_min,x_max])[:,None]

    df = pd.DataFrame(samples,columns=cols)
    dl = pd.DataFrame(np.hstack([lims,lims]),columns=cols)

    sns.set(style="white", color_codes=True)
    sns.jointplot(data=df, x=cols[0], y=cols[1], kind="hex", xlim=lims, ylim=lims)
    sns.lineplot(data=dl, x=cols[0], y=cols[1])

class POI(object):
    '''
    Class for probability of improvement outcomes.
    '''
    
    def __init__(self, point: np.array, probability: float, **kwargs):
        
        self.point = point
        assert all([probability>=0.0,probability<=1.0]), f'Probability value should be in [0,1]. Received {probability:.2f}.'
        self.probability = probability
                                    
        self.upper_bound = kwargs['upper_bound'] if 'upper_bound' in kwargs else True
        self.reference = kwargs['reference'] if 'reference' in kwargs else np.array([])
        self.threshold = kwargs['threshold'] if 'threshold' in kwargs else None
            
    def __repr__(self):
        return 'POI('+''.join([f'point={self.point},',
                                f' probability={self.probability},',
                                f' upper_bound={self.upper_bound}',
                                ','*any([self.reference.size > 0]),
                                f' reference={self.reference}'*(self.reference.size > 0),
                               ','*any([self.threshold is not None]),
                                f' threshold={self.threshold}'*(self.threshold is not None)])+')'

def probability_of_improvement(solutions: np.array, references: np.array, model: GPRegression,
                               sample_num: int=100000, is_constraint: bool=False, upper_bound: bool=True, plot_joint_gaussians: bool=False,
                               **kwargs) -> List[POI]:
    
    sols = np.atleast_2d(solutions)
    d = sols.shape[-1]
    is_minimum = not upper_bound

    if is_constraint:

        refs = np.atleast_2d(references.flatten()).T

    else:

        refs = np.atleast_2d(references)

    ref_num = refs.shape[0]
    ref_dim = refs.shape[-1]

    assert ref_dim == d or ref_dim == 1, \
    'Input reference row dimension must either be:\n(1) identical to solution row dimension (POI for objective target, is_constraint=False)\n(2) or equal to 1 (POI for constraint satisfaction, is_constraint=True).\nYour input has inferred dimension {} for solution vectors and inferred dimension {} for reference vectors (is_constraint={}).'.format(d,ref_dim,is_constraint)

    sols_rep = np.tile(sols, (1,ref_num)).reshape(ref_num*sols.shape[0],sols.shape[-1])
    refs_rep = np.tile(refs, (sols.shape[0],1))

    N = sols_rep.shape[0]*(sols_rep.shape[0] == refs_rep.shape[0])

    pois = []

    for i in range(N):

        if is_constraint:
            mu, cov = model.predict(np.vstack([sols_rep[i]]),full_cov=True)
            samples = multivariate_normal(mean=mu.flatten(),cov=cov).rvs(size=sample_num)
            samples = np.hstack([samples[:,None],np.tile(refs_rep[i:i+1],(samples.size,1))])
        else:
            mu, cov = model.predict(np.vstack([sols_rep[i],refs_rep[i]]),full_cov=True)
            samples = multivariate_normal(mean=mu.flatten(),cov=cov).rvs(size=sample_num)

        if is_constraint:            
            pois.append(POI(sols_rep[i],order_stats(samples,is_minimum),upper_bound=upper_bound,threshold=refs_rep[i]))
        else:
            pois.append(POI(sols_rep[i],order_stats(samples,is_minimum),upper_bound=upper_bound,reference=refs_rep[i]))

        if plot_joint_gaussians and not is_constraint:

            kwargs = {'cols': ['f({})'.format(np.around(sols_rep[i],2)),
                               'f({})'.format(np.around(refs_rep[i]),2)]}
            
            plot_joint_distribution(samples=samples, **kwargs)

    return pois