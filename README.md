<!--
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
-->

# DOFramework

`doframework` is a testing framework for data-driven decision-optimization algorithms. 

`doframework` integrates with the user's data-driven decision-optimization application written in Python.

# Requirements

`doframework` is written for Python version >= 3.9.0. 

The testing framework can run locally or remotely. For optimal performance, run it on an OpenShift cluster.

The framework relies on cloud object storage (COS) to interact with simulation products.

# Configs

The user provides COS specifications in the form of a `configs.yaml`. 

The `configs.yaml` includes the list of source and target bucket names (under `s3:buckets`). Credentials are added under designated fields.

```
s3:
  buckets: 
    inputs: '<inputs-bucket>'
    inputs_dest: '<inputs-dest-bucket>'
    objectives: '<objectives-bucket>'
    objectives_dest: '<objectives-dest-bucket>'
    data: '<data-bucket>'
    data_dest: '<data-dest-bucket>'
    solutions: '<solutions-bucket>'
  aws_secret_access_key: 'xxxx'
  aws_access_key_id: 'xxxx'
  endpoint_url: 'https://xxx.xxx.xxx'
  region: 'xx-xxxx'
```

# Input

The application will run end to end and produce results assuing a `input.json` files are uploaded to `<inputs_bucket>`. 

Here is an input file (see input samples `input_basic.json` and `input_all.json` under `inputs`).


```
{     
    "f": {
        "vertices": {
            "num": 7,
            "range": [[5.0,20.0],[0.0,10.0]],
        },
        "values": {
            "range": [0.0,5.0]
        },
    },
    "omega" : {
        "ratio": 0.8,
        "scale": 0.01
    },
    "data" : {
        "N": 750,
        "noise": 0.01,
        "policy_num": 2,
        "scale": 0.4
    },
    "input_file_name": "user_input.json"
}
```

`f:vertices:num`: The number of vertices in the objective target piece-wise linear graph.<br>
`f:vertices:range`: Objective target domain will be contained in this box range.<br>
`f:values:range`: Objective target values will range within this interval.<br>
`omega:ratio`: feasibility regions volume / objective target domain volume > ratio.<br>
`omega:scale`: Upper bound on standard deviation of sampled feasibility region jitter (as a ratio of domain diameter).<br>
`data:N`: Number of data points to sample.<br>
`data:noise`: Response variable noise.<br>
`data:policy_num`: Number Gaussians in mixed Gaussian distribution of data points.<br>
`data:scale`: Upper bound on standard deviation of Gaussians in mixed Gaussian distribution of data points as a ratio of objective target domain diameter.

The jupyter notebook `inputs.ipynb` under `notebooks` allows you to automatically generate input files and upload them to `<inputs_bucket>`

# Install

To run `doframework` scripts locally install with

```
$ pip install doframework
```

To run `doframework` on an OpenShift cluster, `cd` into your project's folder and run the setup bash script `doframework-setup.sh`. Make sure to log in first (see OpenShift Login below). The setup script will generate a `doframework.yaml` file in your project's folder.
```
$ cd <user_project_folder>
$ doframework-setup.sh
```

To run `doframework` on a KiND cluster, run the setup bash script with the `--kind` option. 
```
$ cd <user_project_folder>
$ doframework-setup.sh --kind
```

# Test

Run the setup bash script `doframework-setup.sh` with the `--example` flag to generate the test script  `doframework_example.py` in your project folder.
```
$ cd <user_project_folder>
$ doframework-setup.sh --example
```
Then run the test script locally
```
$ python doframework_example.py --configs configs.yaml
```
Make sure to upload input json files to `<inputs_bucket>`.

# Adaptations

You have the option to adapt `doframework.yaml` to fit your application. 

Using the option `--project-dir` will allow you to mount your application and `pip install .` it on cluster nodes with:
```
$ doframework-setup.sh --project-dir <relative_dir_path>
```
Using the option `--project-requirements` will allow you to specify additional application requirements and `pip install -r` then on cluster nodes with
```
$ doframework-setup.sh --project-requirements <relative_dir_path>
```
Using the option `--project-dep` will allow you to specify additional application requirements and `apt-get install -y` them on cluster nodes with
```
$ doframework-setup.sh --project-dep <dep>
```

# Run

The testing framework is invoked within a user's application, for example, `module.py` below. 

Your model will be integrated into the testing framework, once it is decorated with `doframework.resolve`. 

The framework supports the following inputs to your model: 

`data`: 2D np.array with features X = data[ : , :-1] and response variable y = data[ : ,-1].<br>
`constraints`: Linear constraints as a 2D numpy array A. x satisfies constraints if A[ : , :-1]@x + A[ : ,-1] $\leq 0$.<br>
`lower_bound`: Lower bound per feature variable.<br>
`upper_bound`: Upper bound per feature variable.<br>
`init_value`: Optimization initial value.<br>

An experiment runs with `doframework.run()`, which accepts your decorated model and a relative path to your `configs.yaml`. It also accepts the keyword arguments:

`objectives`: The number of objective targets to generate per input file.<br>
`datasets`: The number of datasets to generate per objective target.<br>
`feasibility_regions`: The number of feasibility regions to generate per objective and dataset.<br>
`distribute`: True to run distributively, False to run sequentially.<br>
`logger`: True to see logs, False otherwise.<br>
`after_idle_for`: Stop running when event stream is idle after this many seconds.<br>

Here is an example `module.py` application. 

```
import doframework as dof

@dof.resolve
def model(data: np.array, constraints: np.array, **kwargs):
    ...    
    return optimal_arg, optimal_val, regression_model

if __name__ == '__main__':
    
    dof.run(model, 'configs.yaml', objectives=5, datasets=3)
```

# KiND

A KiND cluster simulates a Kubernetes cluster without the need to dockerize. However, it still requires [Docker](https://docs.docker.com/get-docker/ "Get Docker") installed and configured.

Here's how to set up KiND on a Mac. First install `Go`. Here's how that's done with `homebrew`.
```
$ export GOPATH="${HOME}/.go"
$ export GOROOT="$(brew --prefix golang)/libexec"
$ export PATH="$PATH:${GOPATH}/bin:${GOROOT}/bin"
$ test -d "${GOPATH}" || mkdir "${GOPATH}"
$ test -d "${GOPATH}/src/github.com" || mkdir -p "${GOPATH}/src/github.com"
$ brew install go
```

Now install [KiND](https://kind.sigs.k8s.io/docs/user/quick-start "KiND quick start")
```
$ brew install kind
```

Set up the cluster by running the bash script `doframework-setup.sh`. The ` --skip` flag will tell the script not to generate `doframework.yaml` again. 
```
$ cd <user_project_folder>
$ doframework-setup.sh --kind  --skip
```

Run an application `module.py` on the [KiND](https://kind.sigs.k8s.io/docs/user/quick-start "KiND quick start") cluster with.
```
$ ray submit doframework.yaml module.py
```
At this point we may encounter the following.
```
RuntimeError: Head node of cluster (rayvens-cluster) not found!
```
This is typically a resource allocation issue. To investigate this issue further, we would want to have [kubectl](https://kubernetes.io/docs/tasks/tools/ "Install kubectl") installed, so we can identify our head node and see what gives
```
$ kubectl get pods -n ray
$ kubectl describe pod rayvens-cluster-head-xxsfh -n ray
```
The `decsribe` command will spit out something like
```
Events:
  Type     Reason            Age                 From               Message
  ----     ------            ----                ----               -------
  Warning  FailedScheduling  24s (x13 over 11m)  default-scheduler  0/1 nodes are available: 1 Insufficient memory.
```
One quick way to resolve the issue is to go to the Docker Desktop and look under Resources -> Advanced. Play with the number of CPUs / Memory GBs, but don't go crazy, otherwise your machine will start running really ... really ... s-l-o-w.

Any changes to `doframework.yaml` can be updated with
```
$ ray up doframework.yaml --no-config-cache --yes
```

Once you're done, clean up with
```
$ kind delete cluster
$ docker stop registry
$ docker rm registry
```

# OpenShift

This assumes you've already set up your cluster. It assumes you have `kubectl`, `ibmcloud` and `oc` CLI installed.

Log into IBM cloud services and follow the instructions.
```
$ ibmcloud login --sso
```
Generate a token for your openshift cluster (good for 24HR). Go to https://cloud.ibm.com (make sure your connection has access rights). Click on OpenShift Web Console (top right). Click on `IAM#user` and look for `Copy Login Command`. Copy the login command (`oc login ...`).
```
$ oc login --token=shaxxx~xxxx --server=https://xxx.xx-xx.xx.cloud.ibm.com:xxxxx
```

If you haven't already, define a new ray project [done once]. 
```
$ oc new-project ray
```
If you have already defined the "ray" project, you'll find it under `oc projects`.

Upload `input.json` file(s) to your S3 bucket `<inputs_bucket>`.

Run the bash script `doframework-setup.sh` to establish the `ray` cluster. Use the `--skip` flag to skip generating a new `doframework.yaml` file.
```
$ cd <user_project_folder>
$ doframework-setup.sh --skip
```
If you are familiar with `ray` you can run instead 
```
$ ray up doframework.yaml --no-config-cache --yes
```

Submit the application to the `ray` cluster
```
$ ray submit doframework.yaml module.py
```

Changing cluster resource allocation is done through `doframework.yaml`. Change `max_workers` to the max CPUs you have available [but keep max_workers of head node at 0]. You can play with `resources: requests: cpu, memory` and `resources: limits: cpu, memory` for the head and worker nodes. 

After introducing changes to `doframework.yaml`, update with
```
$ ray up doframework.yaml --no-config-cache --yes
```

To see the ray dashboard, open a separate terminal and run
```
$ oc -n ray port-forward service/rayvens-cluster-head 8265:8265
```
In your browser, connect to `http://localhost:8265`.

Some useful health-check commands. Check status of ray pods
```
$ kubectl get pods -n ray
```
Check status of ray head node
```
$ kubectl describe pod rayvens-cluster-head-mcsfh -n ray
```
Monitor autoscaling with
```
$ ray exec doframework.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```
Connect to a terminal on the head node:
```
$ ray attach doframework.yaml
$ ...
$ exit
```
Get a remote shell to the cluster manually (find head node ID with `kubectl describe`):
```
$ kubectl -n ray exec -it rayvens-cluster-head-z97wc -- bash
```
Shutdown the `ray` cluster with:
```
$ ray down -y doframework.yaml
```

# CPLEX

In case your application relies on a commercial solver, such as `CPLEX`, you will need to mount it onto cluster nodes, if you wish to run your application on an OpenShift cluster.

To allow for a silent installation of `CPLEX`, create a `installer.properties` file under your project folder. Add the following lines to your `installer.properties` file:
```
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
USER_INSTALL_DIR=/home/ray
INSTALLER_LOCALE=en
```


Add the following to your `doframework.yaml` file
```
file_mounts:
    {
        ...,
        "/home/ray/cplex.bin": "/path/to/ILOG_COS_20.10_LINUX_X86_64.bin",
        ...,
        "/home/ray/installer.properties": "./installer.properties"
    }
file_mounts_sync_continuously: false   
setup_commands:
    - chmod u+x /home/ray/cplex.bin
    - sudo bash /home/ray/cplex.bin -f "/home/ray/installer.properties"
    - echo 'export PATH="$PATH:/home/ray/cplex/bin/x86-64_linux"' >> ~/.bashrc
head_setup_commands:
    ...
```
Make sure you are mounting the Linux OS `ILOG_COS_XX.XX_LINUX_X86_64.bin` binary. 

Now update your `ray` cluster
```
$ ray up doframework.yaml --no-config-cache --yes 
```

# Issues

## Idle

Issues relating to an experiment apparently hanging or not going full cycle may have to do with `after_idle_for`. There is enough time window for simulation products to make it to the next stage. Set a large value (in seconds) to be on the safe side.

## Autoscaling
 
 If you're having problems with scaling, e.g.., the app is only running on the head node, you can start by checking the `ray` logs with
```
$ ray exec doframework.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```
or just the error logs
```
$ ray exec doframework.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor.err'
```
You may see an error as follows
```
mkdir: cannot create directory '/home/ray/e2e': Permission denied
```
The problem is that openshift generates worker nodes on random user ids which do not have [permissions](https://docs.openshift.com/container-platform/3.11/admin_guide/manage_scc.html#enable-dockerhub-images-that-require-root) for file mounts. To fix the permissions issue, run
```
$ oc adm policy add-scc-to-group anyuid system:authenticated
```

---------------

## Consumed Uploaded Files

When the application fails to shutdown properly, some `kamel` processes may keep running, consuming any file uploaded to COS S3 buckets. You’ll be able to identify them as source type processes with
```
$ kubectl get all
```
To delete, use
```
$ kamel delete source-data source-inputs 
```

---------------

## SSH Unavailable
Running the bash script `doframework-setup.sh`, you may encounter the following 
```
Error from server (BadRequest): pod rayvens-cluster-head-mcsfh does not have a host assigned
    SSH still not available (Exit Status 1): kubectl -n ray exec -it rayvens-cluster-head-mcsfh -- bash --login -c -i 'true && source ~/.bashrc && export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (uptime)', retrying in 5 seconds.
```
Just wait. Eventually it'll go through. 

---------------

## Login
When running `doframework-setup.sh`, you may see
```
--- creating namespace
error: You must be logged in to the server (Unauthorized)
--- installing Kamel operator
error: You must be logged in to the server (Unauthorized)
error: failed to create clusterrolebinding: Unauthorized
error: You must be logged in to the server (Unauthorized)
```
This is an indication that you haven't logged into your cluster. See login instructions below.

---------------

## rayvens Image Update
Any version updates of rayvens [image](https://quay.io/repository/ibm/rayvens?tab=tags) can be editted in `doframework.yaml` under `containers: ... image: quay.io/ibm/rayvens:0.X.X`.

---------------

## Nothing Generated
If you only see `kamel` subprocesses after hitting `ray submit`, it's likely you forgot to upload `input.json` files to S3 bucket `<inputs_bucket>`. You can upload then now -- no need to stop the experiment.

---------------

## RayOutOfMemoryError

You may run into insufficient memory errors such as `RayOutOfMemoryError: More than 95% of the memory on node rayvens-cluster-head-xxxxx is used`. Make sure you have enough memory on your cluster and increase memory allowance in `doframework.yaml` under `resources:requests:memory:` and `resources:limits:memory:`.

# OpenShift Login

This assumes you have `ibmcloud` and `oc` CLI set up.

Log into IBM cloud services and follow the instructions.
```
$ ibmcloud login --sso
```
Generate a token for your openshift cluster (good for 24HR). Go to https://cloud.ibm.com (make sure you web connection has access rights). Click on OpenShift Web Console [top right]. Click on IAM\#user and look for Copy Login Command. Copy the login command [`oc login ...`].
```
$ oc login --token=shaxxx~xxxx --server=https://xxx.xx-xx.xx.cloud.ibm.com:xxxxx
```
