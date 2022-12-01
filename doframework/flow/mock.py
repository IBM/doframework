import time
import logging
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

from doframework.core.inputs import generate_id
from doframework.core.utils import sample_standard_simplex
from doframework.flow.solutions import _ids_from_data
from doframework.flow.metrics import _ids_from_solution
from doframework.core.optimizer import optimalSolution

sleep_time = 5

def generate_objective_mock(meta_input: dict, meta_name: str, **kwargs) -> Tuple[dict, str]:
    '''
    generate_objective test for end-to-end integration.
    
            Parameters:
                    meta_input (dict): Meta data for objective target generation. 
                    meta_name (str): Name of meta data file.
                    
            Returns:
                    Returns a mock objective target dictionary and the name of the objective target file.
    
    >>> meta = {'data': {'N': 500, 'noise': 0.01}, 'input_file_name': 'input_test.json'}
    >>> objective, generated_objective_file = generate_objective_mock(meta, meta['input_file_name'])
    >>> objective
    {'data': {'N': 500, 'noise': 0.01},
    'input_file_name': 'input_test.json',
    'objective_id': 'negzm03t',
    'generated_file_name': 'objective_negzm03t.json'}
    >>> generated_objective_file
    'objective_negzm03t.json'
    '''
    
    output_prefix = 'objective'
    output_suffix = 'json'

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    is_raised = kwargs['is_raised'] if 'is_raised' in kwargs else False

    time.sleep(sleep_time)
    
    objective_id = generate_id()
    
    output = {}    
    output['data'] = {}
    
    data = meta_input['data']
    
    output['data']['N'] = data['N']
    output['data']['noise'] = data['noise']
    
    assert meta_name == meta_input['input_file_name'], \
    'Mismatch between file name recorded in json input and given file name.'
    
    output['input_file_name'] = meta_name
    output['objective_id'] = objective_id
    generated_file = ''.join(['_'.join([output_prefix,objective_id]),'.',output_suffix])
    output['generated_file_name'] = generated_file

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info('Finished output for objective ID {}.'.format(objective_id))

    return output, generated_file

def generate_dataset_mock(obj_input: dict, obj_name: str, **kwargs) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    '''
    generate_dataset test for end-to-end integration. The mock dataset will be sampled for the function (x**2 + y**2)*np.sin(np.pi*x*y/16).
    
            Parameters:
                    obj_input (dict): Objective target dictionary. 
                    obj_name (str): Name of objective target file.
                    
            Returns:
                    Returns a mock dataset as a pd.DataFrame and the name of the dataset file.

    >>> objective = {'data': {'N': 500, 'noise': 0.01},'input_file_name': 'input_test.json','objective_id': 'negzm03t','generated_file_name': 'objective_negzm03t.json'}
    >>> df, generated_df_file = generate_dataset_mock(objective,objective['generated_file_name'],is_raised=True)
    >>> df
     	x0 	        x1 	        y
    0 	0.737314 	0.058099 	0.010967
    1 	-1.052423 	0.217132 	-0.045152
    2 	0.452205 	0.058047 	-0.005346
    3 	1.834529 	-0.095925 	-0.116584
    4 	0.978658 	-0.086900 	-0.013328
    ...
    >>> generated_df_file
    'data_negzm03t_dsgwujaz.csv'
    '''
    
    input_prefix = 'objective'
    input_suffix = 'json'
    output_prefix = 'data'
    output_suffix = 'csv'

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    is_raised = kwargs['is_raised'] if 'is_raised' in kwargs else False
    
    time.sleep(sleep_time)

    try:
        
        objective_id = re.match(input_prefix+'_'+'(\w+)'+'.'+input_suffix,obj_name).group(1)
        assert objective_id == obj_input['objective_id'], 'Mismatch between file name recorded in json and file name.'

        data_id = generate_id()

        N = obj_input['data']['N']
        noise = obj_input['data']['noise']
        
        d = 2
        f = lambda x, y: (x**2 + y**2)*np.sin(np.pi*x*y/16)
        fvect = np.vectorize(f)

        mu = np.zeros(d)
        sigma = np.diag(d*sample_standard_simplex(d))
        X = multivariate_normal(mean=mu,cov=sigma).rvs(size=N)        
        y = fvect(X[:,0],X[:,1]) + norm.rvs(loc=0,scale=noise,size=X.shape[0])
        D = np.hstack([X,y[:,None]])
        df = pd.DataFrame(D,columns=[f'x{i}' for i in range(D.shape[1]-1)]+['y'])

        generated_file = ''.join(['_'.join([output_prefix,objective_id,data_id]),'.',output_suffix])

        return df, generated_file    

    except AssertionError as e:
        if logger_name:
            log = logging.getLogger(logger_name)
            log.error(e)
        if is_raised: raise e
    except Exception as e:
        if logger_name:
            log = logging.getLogger(logger_name)
            log.error(e)
        if is_raised: raise e
            
    return None, None

def generate_solution_mock(predict_optimize, data_input: pd.DataFrame, data_name: str, **kwargs) -> Tuple[Optional[dict], Optional[str]]:
    '''
    generate_solution test for end-to-end integration. The solutions will be generated for the test function (x**2 + y**2)*np.sin(np.pi*x*y/16).
    The optimization will be constrained to -2 =< x,y <= 2. The solutions produced will be on the vertices [+-2, +-2].
    
            Parameters:
                    predict_optimize: Predit-then-optimize algorithm.
                    data_input (pd.DataFrame): Dataset. 
                    data_name (str): Name of dataser file.
                    
            Returns:
                    Returns a mock solution output dictionary and the name of the solution file.

    >>> from doframework.core.optimizer import predict_optimize
    >>> meta = {'data': {'N': 500, 'noise': 0.01}, 'input_file_name': 'input_test.json'}
    >>> objective, _ = generate_objective_mock(meta, meta['input_file_name'])
    >>> extra = {'objective': objective}
    >>> df, _ = generate_dataset_mock(objective,objective['generated_file_name'],is_raised=True)
    >>> solution, generated_solution_file = generate_solution_mock(predict_optimize, df, generated_df_file, **extra)
    >>> solution
    {'omega': {'constraints': [[1.0, 0.0, -2.0],
    [0.0, 1.0, -2.0],
    [-1.0, -0.0, -2.0],
    [-0.0, -1.0, -2.0]]},
    'solution': {'min': {'arg': [-2.0, -2.0], 'pred': -0.10076280464617189}},
    'objective_id': 'negzm03t',
    'data_id': 'dsgwujaz',
    'solution_id': 'p9t0nm07',
    'generated_file_name': 'solution_negzm03t_dsgwujaz_p9t0nm07.json'}
    >>> generated_solution_file
    'solution_negzm03t_dsgwujaz_p9t0nm07.json'                    
    '''
        
    output_prefix = 'solution'
    output_suffix = 'json'
    extra_input = ['objective']

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    is_raised = kwargs['is_raised'] if 'is_raised' in kwargs else False  
    
    time.sleep(sleep_time)
    
    if 'is_minimum' in kwargs:    
        is_minimum = kwargs['is_minimum']
        extra = kwargs        
    else:        
        is_minimum = True
        extra = {**kwargs,**{'is_minimum': is_minimum}}
    
    if all([k in kwargs for k in extra_input]) and all([v is not None for k,v in kwargs.items() if k in extra_input]):        

        objective_id, data_id = _ids_from_data(data_name)

        solution_id = generate_id()
        objective = kwargs['objective']
        
        D = data_input.to_numpy()
        d = D.shape[-1]-1
        
        output = {}

        output['omega'] = {}
        constraints = np.hstack([np.vstack([np.eye(d),-np.eye(d)]), -d*np.ones(2*d)[:,None]])
        output['omega']['constraints'] = [list(c) for c in constraints]

        arg, val, model = predict_optimize(D, constraints, **extra)
        solution = optimalSolution(arg, val)

        output['solution'] = {}
        opt = 'min' if is_minimum else 'max'

        if all([solution.arg is not None,solution.val is not None]):
            output['solution'][opt] = {}
            output['solution'][opt]['arg'] = list(solution.arg)            
            output['solution'][opt]['pred'] = solution.val
        else:
            output['solution'][opt] = 'FAILED'

        output['objective_id'] = objective_id
        output['data_id'] = data_id
        output['solution_id'] = solution_id
        generated_file = ''.join(['_'.join([output_prefix,objective_id,data_id,solution_id]),'.',output_suffix])
        output['generated_file_name'] = generated_file

        return output, generated_file
    
    else:

        return None, None

def generate_metric_mock(solution_input: dict, solution_name: str, **kwargs) -> Tuple[Optional[dict], Optional[str]]:    
    '''
    generate_metric test for end-to-end integration.
    
            Parameters:
                    predict_optimize: Predit-then-optimize algorithm.
                    data_input (pd.DataFrame): Dataset. 
                    data_name (str): Name of dataser file.
                    
            Returns:
                    Returns a mock metric output dictionary and the name of the metric file.

    >>> from doframework.core.optimizer import predict_optimize
    >>> meta = {'data': {'N': 500, 'noise': 0.01}, 'input_file_name': 'input_test.json'}
    >>> objective, _ = generate_objective_mock(meta, meta['input_file_name'])
    >>> df, _ = generate_dataset_mock(objective,objective['generated_file_name'],is_raised=True)
    >>> extra = {'objective': objective, 'data': df}
    >>> metric, generated_metric_file = generate_metric_mock(solution, generated_solution_file, **extra)
    >>> metric
    {'objective_id': 'negzm03t',
    'data_id': 'dsgwujaz',
    'solution_id': 'p9t0nm07',
    'generated_file_name': 'metrics_negzm03t_dsgwujaz_p9t0nm07.json'}
    >>> generated_metric_file
    'metrics_negzm03t_dsgwujaz_p9t0nm07.json'
    '''

    output_prefix = 'metrics'
    output_suffix = 'json'
    extra_input = ['objective','data']
    
    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    is_raised = kwargs['is_raised'] if 'is_raised' in kwargs else False  
    is_mcmc = kwargs['is_mcmc'] if 'is_mcmc' in kwargs else False
    
    time.sleep(sleep_time)    

    if all([k in kwargs for k in extra_input]) and all([v is not None for k,v in kwargs.items() if k in extra_input]):

        objective_id, data_id, solution_id = _ids_from_solution(solution_name)

        objective = kwargs['objective']
        data = kwargs['data']
        D = data.to_numpy()
        
        constraints = np.array(solution_input['omega']['constraints'])

        output = {}

        output['objective_id'] = objective_id
        output['data_id'] = data_id
        output['solution_id'] = solution_id

        generated_file = ''.join(['_'.join([output_prefix,objective_id,data_id,solution_id]),'.',output_suffix])
        output['generated_file_name'] = generated_file

        return output, generated_file
            
    else:

        return None, None