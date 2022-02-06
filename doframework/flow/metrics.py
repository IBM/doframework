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
from pathlib import Path

import re
import yaml
import json
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from doframework.core.pwl import PWL
from doframework.core.inputs import setup_logger
from doframework.core.metrics import prob_in_omega
from doframework.core.gp import gp_model

def _ids_from_solution(solution_name: str) -> Tuple[str,str,str]:

    input_prefix = 'solution'
    input_suffix = 'json'

    objective_id = re.match(input_prefix+'_'+'(\w+)_\w+_\w+'+'.'+input_suffix,solution_name).group(1)
    data_id = re.match(input_prefix+'_'+'\w+_(\w+)_\w+'+'.'+input_suffix,solution_name).group(1)
    solution_id = re.match(input_prefix+'_'+'\w+_\w+_(\w+)'+'.'+input_suffix,solution_name).group(1)

    return objective_id, data_id, solution_id

def files_from_solution(solution_name: str) -> dict:

    objective_id, data_id, _ = _ids_from_solution(solution_name)

    return {'objective': f'objective_{objective_id}.json', 'data': f'data_{objective_id}_{data_id}.csv'}

def generate_metric(solution_input: dict, solution_name: str, is_mcmc: bool=False, logger_name: Optional[str]=None, is_raised: Optional[bool]=False, **kwargs) -> Tuple[Optional[dict], Optional[str]]:    

    output_prefix = 'metrics'
    output_suffix = 'json'
    extra_input = ['objective','data']

    if all([k in kwargs for k in extra_input]) and all([v is not None for k,v in kwargs.items() if k in extra_input]):

        objective_id, data_id, solution_id = _ids_from_solution(solution_name)

        objective = kwargs['objective']
        data = kwargs['data']
        D = data.to_numpy()
        
        constraints = np.array(solution_input['omega']['constraints'])
        D_feasible = D[np.all(np.pad(D[:,:-1],((0,0),(0,1)),constant_values=1) @ constraints.T < 0, axis=1)]
        X = D_feasible[:,:-1]
        y = D_feasible[:,-1:]
        y_mean = y.mean()

        output = {}

        gp = gp_model(X,y-y_mean,is_mcmc=is_mcmc) # centered at 0

        if gp is not None:
        
            output['gp'] = {}

            policies = np.array(objective['data']['policies'])
            policy_predicts = gp.predict(policies)
            
            # output['probability'] = {}
            # output['probability']['in_omega'] = {} 

            for opt in solution_input['solution']:

                if solution_input['solution'][opt] != 'FAILED':

                    xs = np.array(solution_input['solution'][opt]['arg'])[None,:]

                    # TODO: fix performance of prob_in_omega
                    # omega_hypothesis = objective['omega']['hypothesis']
                    # omega_hypothesis_obj = getattr(scipy.stats,omega_hypothesis)
                    # omega_locs = np.array(objective['omega']['locs'])
                    # omega_scales = np.array(objective['omega']['scales'])
                    # output['probability']['in_omega'][opt] = prob_in_omega(x=xs,f=None,hypothesis=omega_hypothesis_obj,loc=omega_locs,scale=omega_scales)

                    output['gp'][opt] = {}
                    output['gp'][opt]['mu'] = gp.predict(xs)[0][0][0]+y_mean # shift back 
                    output['gp'][opt]['sigma'] = gp.predict(xs)[1][0][0]

            for i, (policy, mu, sigma) in enumerate(zip(policies,policy_predicts[0]+y_mean,policy_predicts[1])):    
                output['gp'][f'policy{i}'] = {'pos': list(policy), 'mu': mu[0], 'sigma': sigma[0]}

        else:

            output['gp'] = 'FAILED'

        output['objective_id'] = objective_id
        output['data_id'] = data_id
        output['solution_id'] = solution_id

        generated_file = ''.join(['_'.join([output_prefix,objective_id,data_id,solution_id]),'.',output_suffix])
        output['generated_file_name'] = generated_file

        return output, generated_file
            
    else:

        return None, None

def main(data_root: str, is_mcmc: bool=False, logger_name: str=None, is_raised=True):

    for p in Path(os.path.join(data_root,'solutions')).rglob('*.json'):
        
        try:
            
            with open(os.path.join(data_root,'solutions',p.name)) as file:  
                solution_input = json.load(file)
                solution_name = p.name
            
            files = files_from_solution(solution_name)

            with open(os.path.join(data_root,'objectives',files['objective'])) as file:  
                objective = json.load(file)
            
            with open(os.path.join(data_root,'data',files['data'])) as file:  
                data = pd.read_csv(file)
        
            if logger_name:

                log = logging.getLogger(logger_name)
                log.info('Reading solution from {}.'.format(solution_name))
                if (objective is not None) and (data is not None):
                    log.info('Using information from {} to generate metrics.'.format(objective['generated_file_name']))
                else:
                    log.info('No matching objective and/or dataset found for given solution. No metrics generated.')

            mertic_output, mertic_file = generate_metric(solution_input,solution_name,is_mcmc=is_mcmc,logger_name=logger_name,is_raised=is_raised,objective=objective,data=data)
            
            if (mertic_output is not None) and (mertic_file is not None):
                metric_path = os.path.join(data_root,'metrics',mertic_file)
                with open(metric_path,'w') as file:
                    json.dump(mertic_output, file)
                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.info('Finished output of metrics for {}.'.format(solution_name))

        except FileNotFoundError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error(e)
            if is_raised: raise e
        except IOError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Unable to read data csv or load objective or solution json.\n')
                log.error(e)
            if is_raised: raise e
        except TypeError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Failed to dump generated metrics into json.\n')
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
                log.error('Something went wrong while processing metrics...\n')
                log.error(e)
            if is_raised: raise e

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-mc", "--mcmc", action="store_true", help="Optimize GP kernel hyperparameters with MCMC [otherwise, maximum likelihood].")
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
    log_file = 'generanted_metrics_{}.log'.format(now)
    log_path = os.path.join(data_root,'logs',log_file)
    logger_name = 'generanted_metrics_log' if args.logger else None
    setup_logger(logger_name, log_path)            

    if logger_name:                
        log = logging.getLogger(logger_name)                
        log.info('Running on user %s', user)
        log.info('Data root %s', data_root)

    main(data_root, args.mcmc, logger_name)