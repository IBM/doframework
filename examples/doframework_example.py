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

import argparse
import numpy as np
import doframework as dof
from doframework.core.optimizer import predict_optimize

@dof.resolve
def predict_optimize_resolved(data: np.array, constraints: np.array, **kwargs):
    return predict_optimize(data, constraints, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, help="User configs.yaml.")
    args = parser.parse_args()
    
    dof.run(predict_optimize_resolved, args.configs, objectives=2, datasets=2)
