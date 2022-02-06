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

import os
import getpass
import logging
import argparse
from typing import Optional, Tuple
from pathlib import Path

import re
import yaml
import json
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial import ConvexHull

from doframework.core.inputs import generate_id, setup_logger
from doframework.core.sampler import omega_sampler
from doframework.core.pwl import PWL
from doframework.core.optimizer import optimalSolution
from doframework.core.utils import is_jsonable

def _ids_from_data(data_name: str) -> Tuple[str,str]:

    input_prefix = 'data'
    input_suffix = 'csv'

    data_id = re.match(input_prefix+'_'+'\w+_(\w+)'+'.'+input_suffix,data_name).group(1)
    objective_id = re.match(input_prefix+'_'+'(\w+)_\w+'+'.'+input_suffix,data_name).group(1)

    return objective_id, data_id

def files_from_data(data_name: str) -> dict:

    objective_id, _ = _ids_from_data(data_name)

    return {'objective': f'objective_{objective_id}.json'}

def generate_solution(predict_optimize, data_input: pd.DataFrame, data_name: str, **kwargs) -> Tuple[Optional[dict], Optional[str]]:

    output_prefix = 'solution'
    output_suffix = 'json'
    extra_input = ['objective']

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None

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
        Ps = np.array(objective['f']['polyhedrons'])
        Vs = np.array(objective['f']['values'])
        f = PWL(Ps,Vs)

        omega_hypothesis = objective['omega']['hypothesis']
        omega_hypothesis_obj = getattr(scipy.stats,omega_hypothesis)
        omega_locs = np.array(objective['omega']['locs'])
        omega_scales = np.array(objective['omega']['scales'])
        lower_bound = omega_locs.min(axis=0)
        upper_bound = omega_locs.max(axis=0)
        init_value = omega_locs[0,:]

        extra = {**extra,**{'lower_bound': lower_bound, 'upper_bound': upper_bound, 'init_value': init_value}}

        output = {}

        if logger_name:
            log = logging.getLogger(logger_name)
            log.info('Started solution output for objective {} and dataset {}.'.format(objective_id,data_id))

        omega_approx = omega_sampler(f,omega_hypothesis_obj,2*D.shape[-1],loc=omega_locs,scale=omega_scales)
        if omega_approx is None:
            omega_approx = np.vstack([np.array(P) for P in objective['omega']['polyhedrons']])
            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Omega sampling failed. Possibly due to omega_locs not in dom(f), or scale is too big and hit max number of tries. Reverted back to original Omega.')
        omega_approx_hull = ConvexHull(omega_approx)
        omega_approx_hull_vertices = omega_approx_hull.points[omega_approx_hull.vertices,:]

        output['omega'] = {}
        output['omega']['vertices'] = [list(v) for v in omega_approx_hull_vertices]
        constraints = np.unique(omega_approx_hull.equations,axis=0)
        output['omega']['constraints'] = [list(c) for c in constraints]

        arg, val, model = predict_optimize(D, constraints, **extra)
        solution = optimalSolution(arg, val)

        if logger_name:
            log = logging.getLogger(logger_name)                
            if all([solution.arg is not None, solution.val is not None]):
                log.info('Predicition and optimization succeeded for objective {} and dataset {}.'.format(objective_id,data_id))
            else:
                log.info('Predicition and optimization **failed** for objective {} and dataset {}.'.format(objective_id,data_id))

        output['solution'] = {}
        opt = 'min' if is_minimum else 'max'

        if all([solution.arg is not None,solution.val is not None]):
            output['solution'][opt] = {}
            output['solution'][opt]['arg'] = list(solution.arg)            
            output['solution'][opt]['value'] = f.evaluate(np.atleast_2d(solution.arg))[0]
            output['solution'][opt]['pred'] = solution.val
        else:
            output['solution'][opt] = 'FAILED'

        output['policies'] = {}
        policies = np.array(objective['data']['policies'])
        policy_values = f.evaluate(policies)
        policy_preds = model.predict(policies) if model is not None else np.full(policies.shape[0], None)
        for i, (policy,value,pred) in enumerate(zip(policies,policy_values,policy_preds)):
            output['policies'][f'policy{i}'] = {'pos': list(policy), 'value': value, 'pred': pred}

        if model is not None:
            output['model'] = {}
            output['model']['name'] = type(model).__name__
            for k,v in model.__dict__.items():
                if not any([k.endswith('_'),k.startswith('_')]):
                    if is_jsonable(v):
                        output['model'][k] = v
                    else:
                        output['model'][k] = str(v)

        output['objective_id'] = objective_id
        output['data_id'] = data_id
        output['solution_id'] = solution_id
        generated_file = ''.join(['_'.join([output_prefix,objective_id,data_id,solution_id]),'.',output_suffix])
        output['generated_file_name'] = generated_file

        return output, generated_file
    
    else:

        return None, None

def main(data_root: str, args, logger_name: str=None, is_raised=False):

    from doframework.core.optimizer import predict_optimize # test with built-in model

    for p in Path(os.path.join(data_root,'data')).rglob('*.csv'):
        
        try:
            
            with open(os.path.join(data_root,'data',p.name)) as file:  
                data_input = pd.read_csv(file)
                data_name = p.name
                
            with open(os.path.join(data_root,'objectives',files_from_data(data_name)['objective'])) as file:  
                objective = json.load(file)
            
            if logger_name:

                log = logging.getLogger(logger_name)
                log.info('Reading data from {}.'.format(data_name))
                if objective is not None:
                    log.info('Using information from {} to generate constraints.'.format(objective['generated_file_name']))
                    log.info('Generating {} constraints and solving for each'.format(args.regions))
                else:
                    log.info('No matching objective found for given dataset. No solutions generated.')

            for _ in range(args.regions):

                opt_output, opt_file = generate_solution(predict_optimize, data_input, data_name, logger_name=logger_name, is_raised=is_raised, objective=objective)
                
                if (opt_output is not None) and (opt_file is not None):
                    opt_path = os.path.join(data_root,'solutions',opt_file)
                    with open(opt_path,'w') as file:
                        json.dump(opt_output, file)                        

        except FileNotFoundError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error(e)
            if is_raised: raise e
        except IOError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Unable to read data csv or load objective json.\n')
                log.error(e)
            if is_raised: raise e
        except TypeError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Failed to dump generated solution into json.\n')
                log.error(e)
            if is_raised: raise e
        except AssertionError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error(e)
            if is_raised: raise e
        except Exception as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Something went wrong while processing optimal solution...\n')
                log.error(e)
            if is_raised: raise e
                
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--regions", type=int, default=1, help="Number of feasibility regions [i.e., omegas] to generate per dataset [default: 1].")
    parser.add_argument("-l", "--logger", action="store_true", help="Enable logging.")
    args = parser.parse_args()

    configs_path = os.environ['HOME']
    configs_file = 'configs.yaml'

    with open(os.path.join(configs_path,configs_file),'r') as file:
        try:
            configs = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print('CRITICAL ... Unable to load configs yaml.\n.')
            print(e)
            raise e    

    user = getpass.getuser()
    data_root = configs[user]['data']

    now = datetime.now().strftime('%Y-%m-%d_%H%M')
    log_file = 'generanted_solution_{}.log'.format(now)
    log_path = os.path.join(data_root,'logs',log_file)
    logger_name = 'generanted_solutions_log' if args.logger else None
    setup_logger(logger_name, log_path)            

    if logger_name:                
        log = logging.getLogger(logger_name)                
        log.info('Running on user %s', user)
        log.info('Data root %s', data_root)

    main(data_root, args, logger_name)