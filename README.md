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

`doframework` is a testing framework for data-driven decision-optimization algorithms. It integrates with the user's data-driven decision-optimization application (written in Python).

`doframework` randomly generates optimization problems (f,O,D,x*): 
* f is continuous piece-wise linear functions defined over a domain in d-dimensional space (d>1),
* O is a region in dom(f) defined by linear constraints,
* D = (X,y) is a dataset derived for f,
* x* is the true optimum of f in O.

The testing framework feeds the constraints and the data (O,D) into the user's algorithm, and collects its predicted optimum. The algorithm's predicted optimal value can then be conpared to the true optimal value f(x*). By comparing the two over many optimization problems, `doframework` produces a performance profile for data-driven decision-optimization algorithms.

# Design

`doframework` was designed for optimal cloud distribution following an event-driven approach. 

`doframework` was built on top of [ray](https://www.ray.io/ "Ray -- fast and simple distributed computing") for cloud distribution and [rayvens](https://github.com/project-codeflare/rayvens "Rayvens augments Ray with events") for event driven management.

# Requirements

`doframework` was written for Python version >= 3.9.0. 

`doframework` can run either locally or remotely. For optimal performance, run it on a Kubernetes cluster. Cloud configuration is currently available for OpenShift clusters.

The framework relies on [Cloud Object Storage](https://www.ibm.com/cloud/object-storage) (COS) to interact with simulation products.

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
The buckets must be **distinct**.

# Install

To run `doframework` locally, install with

```
$ pip install doframework
```

To run `doframework` on an OpenShift cluster, `cd` into your project's folder and run the setup bash script `doframework-setup.sh`. Make sure to log into your cluster first (see OpenShift Login below). 

The setup script `doframework-setup.sh` will generate the cluster configuration file `doframework.yaml` in your project's folder.
```
$ cd <user_project_folder>
$ doframework-setup.sh --configs configs.yaml
```

To run `doframework` on a KiND cluster, run the setup bash script with the `--kind` option. 
```
$ cd <user_project_folder>
$ doframework-setup.sh --kind --configs configs.yaml
```

# Inputs

The application will run end to end and produce results once `input.json` files are uploaded to `<inputs_bucket>`. The `input.json` files provide meta data for the random genration of optimization problems.

Here is an example of an input file (see input samples `input_basic.json` and `input_all.json` under `./inputs`).


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
    "input_file_name": "input.json"
}
```

`f:vertices:num`: number of vertices in the piece-wise linear graph of f.<br>
`f:vertices:range`: f domain will be inside this box range.<br>
`f:values:range`: range of f values.<br>
`omega:ratio`: vol(O) / vol(dom(f)) >= ratio.<br>
`omega:scale`: max jitter in sampling feasibility regions (as a ratio of f domain diameter).<br>
`data:N`: number of data points to sample.<br>
`data:noise`: response variable $y$ noise.<br>
`data:policy_num`: number of Gaussians in Gaussian mix distribution of data.<br>
`data:scale`: max STD of Gaussians in Gaussian mix distribution of data (as a ratio of f domain diameter).

The jupyter notebook `./notebooks/inputs.ipynb` allows you to automatically generate input files and upload them to `<inputs_bucket>`.

It's a good idea to start experimenting on low dimensional problems. 

# Outputs

`doframework` produces three types of files as experiment byproducts:

* `objective.json`: containing information on (f,O,x*) 
* `data.csv`: containing the dataset the algorithm accepts as input
* `solution.json`: containing the algorithm's predicted optimum

Find sample files under `./outputs`/

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

Using the option `--project-dir` will allow you to mount your application and `pip install .` it on cluster nodes.
```
$ doframework-setup.sh --project-dir <relative_dir_path>
```
Using the option `--project-requirements` will allow you to specify application requirements and `pip install -r` then on cluster nodes.
```
$ doframework-setup.sh --project-requirements <relative_dir_path>
```
Using the option `--project-dep` will allow you to specify application dependencies and `apt-get install -y` them on cluster nodes.
```
$ doframework-setup.sh --project-dep <dep>
```

# Run

The testing framework is invoked within the user's application. 

The user's model will be integrated into the testing framework, when it is decorated with `doframework.resolve`. 

`doframework` supports the following inputs to the model: 

`data`: 2D np.array with features X = data[ : , :-1] and response variable y = data[ : ,-1].<br>
`constraints`: linear constraints as a 2D numpy array A. A data point x satisfies the constraints when A[ : , :-1]*x + A[ : ,-1] <= 0.<br>
`lower_bound`: lower bound per feature variable.<br>
`upper_bound`: upper bound per feature variable.<br>
`init_value`: optional initial value.<br>

A `doframework` experiment runs with `doframework.run()`. The `run()` utility accepts the decorated model and a relative path to the `configs.yaml`. It also accepts the keyword arguments:

`objectives`: number of objective targets to generate per input file.<br>
`datasets`: number of datasets to generate per objective target.<br>
`feasibility_regions`: number of feasibility regions to generate per objective and dataset.<br>
`distribute`: True to run distributively, False to run sequentially.<br>
`logger`: True to see logs, False otherwise.<br>
`after_idle_for`: stop running when event stream is idle after this many seconds.<br>
`alg_num_cpus`: number of CPUs provisioned for user model per worker.<br>

Here is an example user application. 

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

A [KiND](https://kind.sigs.k8s.io/docs/user/quick-start "KiND quick start") cluster simulates a Kubernetes cluster without the need to dockerize. It still requires [Docker](https://docs.docker.com/get-docker/ "Get Docker") installed and configured.

Here's how to set up KiND on a Mac. First install `Go`.
```
$ export GOPATH="${HOME}/.go"
$ export GOROOT="$(brew --prefix golang)/libexec"
$ export PATH="$PATH:${GOPATH}/bin:${GOROOT}/bin"
$ test -d "${GOPATH}" || mkdir "${GOPATH}"
$ test -d "${GOPATH}/src/github.com" || mkdir -p "${GOPATH}/src/github.com"
$ brew install go
```

Now install [KiND]
```
$ brew install kind
```

Set up the cluster by running the bash script `doframework-setup.sh`. The ` --skip` flag will tell the script not to generate `doframework.yaml` again. 
```
$ cd <user_project_folder>
$ doframework-setup.sh --kind  --skip
```

Run your application `module.py` on the [KiND] cluster with.
```
$ ray submit doframework.yaml module.py
```
At this point you may encounter
```
RuntimeError: Head node of cluster (rayvens-cluster) not found!
```
This is typically a resource allocation issue. To investigate this issue, make sure you have the [kubectl](https://kubernetes.io/docs/tasks/tools/ "Install kubectl") CLI installed. Look for the `rayvens-cluster-head` node
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

Any resource allocation changes to `doframework.yaml` can be updated with
```
$ ray up doframework.yaml --no-config-cache --yes
```

Once you're done with KiND, clean up
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

Upload `input.json` file(s) to your `<inputs_bucket>`.

Run the bash script `doframework-setup.sh` to establish the `ray` cluster. Use the `--skip` flag to skip generating a new `doframework.yaml` file.
```
$ cd <user_project_folder>
$ doframework-setup.sh --skip
```
Otherwise, in case you are familiar with `ray`, run instead 
```
$ ray up doframework.yaml --no-config-cache --yes
```

Submit your application to the `ray` cluster
```
$ ray submit doframework.yaml module.py
```

Changing cluster resource allocation is done through `doframework.yaml`. Change `max_workers` to the max CPUs you have available [but keep max_workers of head node at 0]. You can play with `resources: requests: cpu, memory` and `resources: limits: cpu, memory` for the head and worker nodes. 

After introducing changes to `doframework.yaml`, update with
```
$ ray up doframework.yaml --no-config-cache --yes
```
To see the `ray` dashboard, open a separate terminal and run
```
$ oc -n ray port-forward service/rayvens-cluster-head 8265:8265
```
In your browser, connect to `http://localhost:8265`.

Some useful health-check commands: check the status of `ray` pods
```
$ kubectl get pods -n ray
```
Check status of the `ray` head node
```
$ kubectl describe pod rayvens-cluster-head-xxxxx -n ray
```
Monitor autoscaling with
```
$ ray exec doframework.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```
Connect to a terminal on the head node
```
$ ray attach doframework.yaml
$ ...
$ exit
```
Get a remote shell to the cluster manually (find head node ID with `kubectl describe`)
```
$ kubectl -n ray exec -it rayvens-cluster-head-z97wc -- bash
```
Shutdown the `ray` cluster with
```
$ ray down -y doframework.yaml
```

# CPLEX

In case your application relies on a solver, such as `CPLEX`, you will need to mount it onto cluster nodes, if you wish to run your application on a cluster.

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

# OpenShift Login

This assumes you have `ibmcloud` and `oc` CLI set up. Log into IBM cloud services and follow the instructions.
```
$ ibmcloud login --sso
```
Generate a token for your openshift cluster (good for 24hrs). Go to https://cloud.ibm.com (make sure your web connection has access rights). Click on `OpenShift Web Console` (top right). Click on your IAM\#user and look for `Copy Login Command`. Copy the login command [`oc login ...`]. Now run it
```
$ oc login --token=shaxxx~xxxx --server=https://xxx.xx-xx.xx.cloud.ibm.com:xxxxx
```

# Issues

## Timing

Timing can be a delicate issue when running a `doframework` experiment. Ray workers may get throttled by too many tasks, which reduces the compute resources per task, effectively choking that worker.

One way to tackle this is to `ray submit` the application when the `<inputs_bucket>` is empty and then upload new `input.json` files at controlled time intervals. Finding the optimal rate may involve some trial and error.

## Idle

When an experiment goes idle, or it does not go through full cycle, this may have to do with `after_idle_for`. 

The `after_idle_for` time window should be sufficiently large for simulation products to make it through to the next stage. This is especially true when optimization problem dimensions are higher, or when your algorithm takes longer.

## Autoscaling on OpenShift
 
 If you're having problems with scaling, for instance, the application is only running on the head node, you can start by checking the `ray` logs with
```
$ ray exec doframework.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```
or just the error logs
```
$ ray exec doframework.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor.err'
```
You may see an error
```
mkdir: cannot create directory '/home/ray/..': Permission denied
```
The problem is that OpenShift generates worker nodes on random user ids which do not have [permissions](https://docs.openshift.com/container-platform/3.11/admin_guide/manage_scc.html#enable-dockerhub-images-that-require-root) for file mounts. To fix the permissions issue, run
```
$ oc adm policy add-scc-to-group anyuid system:authenticated
```

---------------

## Consumed Uploads

When files magically disappear when you upload them to the COS buckets, it may be that some `kamel` processes are still running, consuming any uploaded file. 

You may be able to identify these `kamel` processes as source-type processes with
```
$ kubectl get all
```
To delete, use
```
$ kamel delete source-data source-inputs 
```
If that doesn't work, try shutting down `ray`.

---------------

## SSH Unavailable
Running the bash script `doframework-setup.sh`, or the `ray up` command, you may encounter the following 
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
This is an indication that you haven't logged into your cluster (see login instructions above). 

The `doframework.yaml` was generated, though!

---------------

## rayvens Image Update

Any updates to the rayvens [image](https://quay.io/repository/ibm/rayvens?tab=tags) you wish to make can be editted in `doframework.yaml` under `containers: ... image: quay.io/ibm/rayvens:0.X.X`.

---------------

## Nothing Generated

If you only see `kamel` subprocesses after hitting `ray submit`, it's likely you haven't uploaded `input.json` files to `<inputs_bucket>`. You can upload then now -- no need to stop the experiment.

---------------

## RayOutOfMemoryError

You may run into insufficient memory errors such as `RayOutOfMemoryError: More than 95% of the memory on node rayvens-cluster-head-xxxxx is used`. 

Make sure you have enough memory on your cluster and increase memory allowance in `doframework.yaml` under `resources:requests:memory:` and `resources:limits:memory:`.

