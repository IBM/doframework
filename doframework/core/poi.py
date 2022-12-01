import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal
from GPy.models import GPRegression

from dataclasses import dataclass
from typing import List

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

@dataclass
class POI:
    '''
    Class for probability of improvement outcomes.
    '''
        
    solution: np.array
    reference: np.array
    probability: float
    is_minimum: bool

def probability_of_improvement(solutions: np.array, references: np.array, model: GPRegression,
                               sample_num: int=100000, is_constraint: bool=False, is_minimum: bool=True, plot_joint_gaussians: bool=False,
                               **kwargs) -> List[POI]:
    
    sols = np.atleast_2d(solutions)
    d = sols.shape[-1]

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
            
        pois.append(POI(sols_rep[i],refs_rep[i],order_stats(samples,is_minimum),is_minimum))

        if plot_joint_gaussians and not is_constraint:

            kwargs = {'cols': ['f({})'.format(np.around(sols_rep[i],2)),
                               'f({})'.format(np.around(refs_rep[i]),2)]}
            
            plot_joint_distribution(samples=samples, **kwargs)

    return pois