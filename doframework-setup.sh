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

yaml="doframework.yaml"
configs="configs.yaml"
namespace="ray"
cpu="1"
mem="8G"
project_dir=""
project_requirements_file=""
project_dependencies="gcc" # GPy
project_pip="doframework"

while [ -n "$1" ]; do
    case "$1" in
        -h|--help) help="1"; break;;
        -y|--yaml) shift; yaml="$1";;
        -c|--configs) shift; configs="$1";;
        -n|--namespace) shift; namespace="$1";;
        --skip) skip="1";;
        --cpu) shift; cpu=$1;;
        --mem) shift; mem=$1;;
        --example) example="1";;
        --version) version="1";;
        --install-project) install_project="1";;
        --project-dir) shift; project_dir=$1;;
        --project-requirements) shift; project_requirements_file=$1;;
    esac
    shift
done

if [ -n "$help" ]; then
    cat << EOF
yamlure and launch Rayvens-enabled Ray cluster on Kubernetes cluster.

Usage: doframework-setup.sh [options]
    -y --yaml <doframework.yaml>            cluster yaml to use/generate (defaults to "doframework.yaml" in current working directory)
    -y --configs <absolute_configs_path>    yaml file containing COS configurations (defaults to "configs.yaml" in current working directory)
    -n --namespace <namespace>              kubernetes namespace to target (defaults to "ray")
    --cpu <cpus>                            cpu quota for each Ray node (defaults to 1)
    --mem <mem>                             memory quota for each Ray node (defaults to 2G)
    --skip                                  reuse existing cluster configuration file (skip generation)
    --example                               generate example file "doframework_example.py" in current working directory
    --version                               shows the version of this script

    --project-requirements <absolute_file_path>     file containing python dependencies to be pip installed on the cluster nodes via requirements file
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

if [ -z "$skip" ]; then
    params+=("--cpu $cpu")
    params+=("--mem $mem")
else
    params+=("--skip")
fi

echo "--- running rayvens-setup.sh on params: ${params[@]}"
rayvens-setup.sh ${params[@]}

if [ -z "$skip" ]; then

    cat >> "$yaml" << EOF
file_mounts:
    {
        "/home/ray/$configs": $PWD/$configs,
EOF

    if [ ! -z "$project_requirements_file" ]; then

        if [ -z "$(dirname "${project_requirements_file}")" ]; then
            echo "ERROR: project requirements file specified: ${project_requirements_file} but it is not an absolute path"
            exit 1
        fi

        requirements_file_name="$(basename "${project_requirements_file}")"

        if [ -z "${requirements_file_name}" ]; then
            echo "ERROR: project requirements file missing from path: ${project_requirements_file}"
            exit 1
        fi

        cat >> "$yaml" << EOF
        "/home/ray/$requirements_file_name": "$project_requirements_file"
EOF

    fi

    cat >> "$yaml" << EOF
    }
file_mounts_sync_continuously: false
head_setup_commands:
    - sudo apt-get update
    - sudo apt-get -y install $project_dependencies
    - pip install $project_pip
EOF

    if [ ! -z "$requirements_file_name" ]; then
        cat >> "$yaml" << EOF
    - pip install -r /home/ray/$requirements_file_name
EOF
    fi

    cat >> "$yaml" << EOF
worker_setup_commands:
    - sudo apt-get update
    - sudo apt-get -y install $project_dependencies
    - pip install $project_pip
EOF

    if [ ! -z "$requirements_file_name" ]; then
        cat >> "$yaml" << EOF
    - pip install -r /home/ray/$requirements_file_name
EOF
    fi

    ray up "$yaml" --no-config-cache --yes

fi

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
