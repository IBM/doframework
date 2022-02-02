#!/bin/sh

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

doframework_version=0.1.0

config="doframework.yaml"
namespace="ray"
cpu="1"
mem="2G"
project_dir=""
project_requirements_file=""
project_dependencies="gcc" # GPy

while [ -n "$1" ]; do
    case "$1" in
        -h|--help) help="1"; break;;
        -c|--config) shift; config="$1";;
        -n|--namespace) shift; namespace="$1";;
        --skip) skip="1";;
        --cpu) shift; cpu=$1;;
        --mem) shift; mem=$1;;
        --example) example="1";;
        --version) version="1";;
        --install-project) install_project="1";;
        --project-dir) shift; project_dir=$1;;
        --project-dep) shift; project_dependencies="$project_dependencies $1";;
        --project-requirements) shift; project_requirements_file=$1;;
    esac
    shift
done

if [ -n "$help" ]; then
    cat << EOF
Configure and launch Rayvens-enabled Ray cluster on Kubernetes cluster.

Usage: doframework-setup.sh [options]
    -c --config <doframework.yaml>  cluster configuration file to use/generate (defaults to "doframework.yaml" in current working directory)
    -n --namespace <namespace>      kubernetes namespace to target (defaults to "ray")
    --cpu <cpus>                    cpu quota for each Ray node (defaults to 1)
    --mem <mem>                     memory quota for each Ray node (defaults to 2G)
    --skip                          reuse existing cluster configuration file (skip generation)
    --example                       generate example file "doframework_example.py" in current working directory
    --version                       shows the version of this script

    --install-project               install project on cluster nodes, use --project-dir <absolute_dir_path>, --project-requirements <absolute_file_path> or --requirements-in-project-dir
      --project-dir <relative_dir_path>             directory of the user project to be pip installed on the cluster nodes
      --project-requirements <relative_dir_path>    file containing python dependencies to be pip installed on the cluster nodes via requirements file
      --project-dep <dep>                           system project dependency that will use "apt-get install -y <dep>"
EOF
    exit 0
fi

if [ -n "$version" ]; then
  echo $doframework_version
  exit 0
fi

params=()
params+=("--config $config")
params+=("--namespace $namespace")

if [ -z "$skip" ]; then
    params+=("--cpu $cpu")
    params+=("--mem $mem")
    if [ -n "$install_project" ]; then
        params+=("--install-project")
        params+=("--project-dir $PWD/$project_dir")
    fi
    if [ -n "$project_dependencies" ]; then
        params+=("--project-dep $project_dependencies")
    fi
    if [ -n "$project_requirements_file" ]; then
        params+=("--project-requirements $PWD/$project_requirements_file")
    fi
else
    params+=("--skip")
fi

echo "--- running rayvens-setup.sh on params: ${params[@]}"
rayvens-setup.sh ${params[@]}

if [ -n "$example" ]; then
    echo "--- generating example file doframework_example.py"
    echo "--- try running 'python doframework_example.py $config' "
    cat > doframework_example.py << EOF
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
import e2e
from doframework.core.optimizer import predict_optimize

@doframework.resolve
def predict_optimize_resolved(data: np.array, constraints: np.array, **kwargs):
    return predict_optimize(data, constraints, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, help="User configs.yaml.")
    args = parser.parse_args()
    
    doframework.run(predict_optimize_resolved, args.configs, objectives=2, datasets=2)
EOF
fi
