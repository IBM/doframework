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
import re
import getpass
import logging
import argparse
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from doframework.core.inputs import generate_id, setup_logger
from doframework.core.pwl import PWL
from doframework.core.sampler import D_sampler, D_sampler_legacy

def generate_dataset(obj_input: dict, obj_name: str, **kwargs) -> Tuple[pd.DataFrame, str]:
    
    input_prefix = 'objective'
    input_suffix = 'json'
    output_prefix = 'data'
    output_suffix = 'csv'

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    num_cpus = kwargs['num_cpus'] if 'num_cpus' in kwargs else 1

    objective_id = re.match(input_prefix+'_'+'(\w+)'+'.'+input_suffix,obj_name).group(1)
    assert objective_id == obj_input['objective_id'], 'Mismatch between file name recorded in json and file name.'

    data_id = generate_id()

    Ps = np.array(obj_input['f']['polyhedrons'])
    Vs = np.array(obj_input['f']['values'])
    f = PWL(Ps,Vs)

    N = obj_input['data']['N']
    noise = obj_input['data']['noise']
    weights = obj_input['data']['weights']
    policies = [np.array(policy) for policy in obj_input['data']['policies']]
    covariances = [np.array(cov) for cov in obj_input['data']['covariances']]
    data_hypothesis = obj_input['data']['hypothesis']
    data_hypothesis_obj = getattr(scipy.stats,data_hypothesis)

    # D = D_sampler_legacy(f, data_hypothesis_obj, N, weights, noise, mean=policies, cov=covariances)
    D = D_sampler(f, N, weights, noise, mean=policies, cov=covariances, objective_id=objective_id, logger_name=logger_name, num_cpus=num_cpus)

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info('Sampling dataset {} for objective {} resulted in {} NaN values.'.format(data_id,objective_id,D[np.isnan(D).any(axis=1)].shape[0]))

    df = pd.DataFrame(D,columns=[f'x{i}' for i in range(D.shape[1]-1)]+['y'])
    generated_file = ''.join(['_'.join([output_prefix,objective_id,data_id]),'.',output_suffix])

    return df, generated_file

def main(data_root: str, args: argparse.Namespace, **kwargs):

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    is_raised = args.is_raised

    for p in Path(os.path.join(data_root,'objectives')).rglob('*.json'):
        
        try:
            
            with open(os.path.join(data_root,'objectives',p.name)) as file:  
                obj_input = json.load(file)
                obj_name = p.name
                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.info('Loaded objective {}.'.format(obj_name))
                
            obj_id = obj_name # obj_input['objective_id']
            
            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Sampling {} datasets for {} with {} CPUs.'.format(args.datasets,obj_id,args.cpus))        
                
            for i in range(args.datasets):
                
                df, gen_data_file = generate_dataset(obj_input,obj_name,logger_name=logger_name,is_raised=is_raised,num_cpus=args.cpus)              
                gen_data_path = os.path.join(data_root,'data-dest',gen_data_file)
                df.to_csv(gen_data_path,index=False)
                
        except IOError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Unable to load json from objective file.\n')
                log.error(e)
            if is_raised: raise e
        except json.JSONDecodeError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Error occured while decoding obective json.\n')
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
                log.error('Something went wrong while processing objective...\n')
                log.error(e)
            if is_raised: raise e

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, help="Specify the absolute path of the configs file.")
    parser.add_argument("-s", "--datasets", type=int, default=1, help="Number of datasets to generate (default: 1).")
    parser.add_argument("-l", "--logger", action="store_true", help="Enable logging.")
    parser.add_argument("-r", "--is_raised", action="store_true", help="Raise assertions and terminate run.")
    parser.add_argument("--cpus", type=int, default=1, help="Number of CPUs to sample data (default: 1).")
    args = parser.parse_args()

    configs_path = args.configs

    with open(configs_path,'r') as file:
        try:
            configs = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print('CRITICAL ... Unable to load configs yaml.\n.')
            print(e)
            raise e    

    user = getpass.getuser()
    data_root = configs[user]['data']

    now = datetime.now().strftime('%Y-%m-%d_%H%M')
    log_file = 'generanted_data_{}.log'.format(now)
    log_path = os.path.join(data_root,'logs',log_file)
    logger_name = 'generanted_data_log' if args.logger else None
    setup_logger(logger_name, log_path)            

    if logger_name:                
        log = logging.getLogger(logger_name)                
        log.info('Running on user %s', user)
        log.info('Data root %s', data_root)

    main(data_root, args, logger_name=logger_name)