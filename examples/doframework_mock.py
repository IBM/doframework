#
# Copyright IBM Corporation 2021
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

# Run mock script:
# $ python doframework_mock.py -c configs.yaml

import argparse
import numpy as np
import doframework as dof
from doframework.core.optimizer import predict_optimize
from doframework.core.storage import Storage
from doframework.core.inputs import get_configs

@dof.resolve
def predict_optimize_resolved(data: np.array, constraints: np.array, **kwargs):
    return predict_optimize(data, constraints, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, help="Path to user configs.yaml relative to current working directory.")
    parser.add_argument("-o", "--objectives", type=int, default=3, help="Number of objectives to generate per meta input (default: 3).")
    parser.add_argument("-d", "--datasets", type=int, default=2, help="Number of datasets to generate per objective (default: 2).")
    parser.add_argument("-r", "--feasibility_regions", type=str, default=1, help="Number of feasibility regions to generate per dataset (default: 1).")
    args = parser.parse_args()
    
    configs = get_configs(args.configs)
    storage = Storage(configs)
    buckets = storage.buckets()

    num_inputs_at_start = storage.count(buckets['inputs'],'json')+storage.count(buckets['inputs_dest'],'json')
    num_objectives_at_start = storage.count(buckets['objectives'],'json')+storage.count(buckets['objectives_dest'],'json')
    num_datasets_at_start = storage.count(buckets['data'],'csv')+storage.count(buckets['data_dest'],'csv')
    num_solutions_at_start = storage.count(buckets['solutions'],'json')+storage.count(buckets['solutions_dest'],'json')

    expected_objectives = args.objectives
    expected_datasets = expected_objectives*args.datasets
    expected_solutions = expected_datasets*args.feasibility_regions

    bucket = buckets['inputs']
    name = 'input_test.json'
    content = {"data": {"N": 500, "noise": 0.01}, "input_file_name": name}
    response = storage.put(bucket,content,name,'json')
    assert response, 'Failed to upload test input file to inputs bucket.'

    dof.run(
        predict_optimize_resolved, 
        args.configs, 
        objectives=args.objectives, 
        datasets=args.datasets, 
        feasibility_regions=args.feasibility_regions, 
        after_idle_for=60, 
        logger=True, 
        mock=True # mock mode
    )

    num_inputs = storage.count(buckets['inputs'],'json')+storage.count(buckets['inputs_dest'],'json')-num_inputs_at_start
    num_objectives = storage.count(buckets['objectives'],'json')+storage.count(buckets['objectives_dest'],'json')-num_objectives_at_start
    num_datasets = storage.count(buckets['data'],'csv')+storage.count(buckets['data_dest'],'csv')-num_datasets_at_start
    num_solutions = storage.count(buckets['solutions'],'json')+storage.count(buckets['solutions_dest'],'json')-num_solutions_at_start

    print(f'SANITY: compare the number of generated files to the expected:')
    print(f'Generated {num_objectives} objectives out of expected {expected_objectives}.')
    print(f'Generated {num_datasets} datasets out of expected {expected_datasets}.')
    print(f'Generated {num_solutions} solutions out of expected {expected_solutions}.')