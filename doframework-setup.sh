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

doframework_version=0.2.0

yaml="doframework.yaml"
configs=""
namespace="ray"
kind=""
cpu="1"
mem="8G"
max_workers="2"
project_dir=""
project_requirements_file=""
project_pip_dep="doframework"
project_dep="gcc" # GPy

while [ -n "$1" ]; do
    case "$1" in
        -h|--help) help="1"; break;;
        -y|--yaml) shift; yaml="$1";;
        -c|--configs) shift; configs="$1";;
        -n|--namespace) shift; namespace="$1";;
        --kind) kind="1";;
        --skip) skip="1";;
        --cpu) shift; cpu=$1;;
        --mem) shift; mem=$1;;
        --max-workers) shift; max_workers=$1;;
        --example) example="1";;
        --version) version="1";;
        --project-dir) shift; project_dir="$1";;
        --project-requirements) shift; project_requirements_file="$1";;
    esac
    shift
done

if [ -n "$help" ]; then
    cat << EOF
yamlure and launch Rayvens-enabled Ray cluster on Kubernetes cluster.

Usage: doframework-setup.sh [options]
    -y --yaml <doframework.yaml>            cluster yaml to generate in working directory (defaults to "doframework.yaml")
    -y --configs <absolute_configs_path>    configs yaml containing COS configurations (defaults to "configs.yaml" in working directory)
    -n --namespace <namespace>              kubernetes namespace to target (defaults to "ray")
    --cpu <cpus>                            cpu quota for each Ray node (defaults to 1)
    --mem <mem>                             memory quota for each Ray node (defaults to 2G)
    --max-workers <max_workers>             the maximum number of workers the Ray cluster will have at any given time (defaults to 2)
    --skip                                  reuse existing cluster configuration file (skip generation)
    --example                               generate example file "doframework_example.py" in current working directory
    --version                               shows the version of this script

    --project-dir <absolute_dir_path>             directory of the user project to be pip installed on the cluster nodes
    --project-requirements <absolute_file_path>   requirements file containing python dependencies to be pip installed on the cluster nodes

    --kind                                  setup a development Kind cluster on localhost instead of deploying to current Kubernetes context
                                            (destroy existing Kind cluster if any, set Kubernetes context to Kind)
EOF
    exit 0
fi

if [ -n "$version" ]; then
  echo $doframework_version
  exit 0
fi

params=()
params+=("--config $yaml")
params+=("--namespace $namespace")

if [ -n "$configs" ]; then
    project_mount=$configs
else
    project_mount="$PWD/configs.yaml"
fi

if [ -z "$skip" ]; then

    params+=("--kind $kind")
    params+=("--cpu $cpu")
    params+=("--mem $mem")
    params+=("--max-workers $max_workers")
    params+=("--project-mount $project_mount")
    params+=("--project-pip-dep $project_pip_dep")
    params+=("--project-dep $project_dep")

    if [ -n "$project_dir" ]; then
        params+=("--project-dir $project_dir")
    fi

    if [ -n "$project_requirements_file" ]; then
        params+=("--project-requirements $project_requirements_file")
    fi

else

    params+=("--skip")

fi

echo "--- running rayvens-setup.sh on params: ${params[@]}"
rayvens-setup.sh ${params[@]}

if [ -n "$example" ]; then
    echo "--- generating example file doframework_example.py"
    echo "--- try running 'python doframework_example.py --configs configs.yaml' "
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
import doframework as dof
from doframework.core.optimizer import predict_optimize

@dof.resolve
def predict_optimize_resolved(data: np.array, constraints: np.array, **kwargs):
    return predict_optimize(data, constraints, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, help="User configs.yaml.")
    args = parser.parse_args()
    
    dof.run(predict_optimize_resolved, args.configs, objectives=2, datasets=2, logger=True)
EOF
fi
