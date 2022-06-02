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

`doframework` is a testing framework for decision-optimization model learning algorithms. Such algorithms learn part or all of a decision-optimization model from data and solve the model to produce a predicted optimal solution. 

`doframework` randomly generates multiple optimization problems (f,O,D,x*) for your algorithm to learn and solve: 
* f is a continuous piece-wise linear function defined over a domain in d-dimensional space (d>1),
* O is a feasibility region in dom(f) defined by linear constraints,
* D = (X,y) is a dataset derived from f,
* x* is the true optimum of f in O (minimum or maximum).

The testing framework feeds your algorithm constraints and data (O,D) and collects its predicted optimum. The algorithm's predicted optimal value can then be compared to the true optimal value f(x*). By comparing the two over multiple randomly generated optimization problems, `doframework` produces a **prediction profile** for your algorithm.

`doframework` integrates with your algorithm (written in Python).

# Design

`doframework` was designed for optimal cloud distribution following an event-driven approach. 

`doframework` was built on top of [ray](https://www.ray.io/ "Ray -- fast and simple distributed computing") for cloud distribution and [rayvens](https://github.com/project-codeflare/rayvens "Rayvens augments Ray with events") for event driven management.

# Requirements

`doframework` was written for Python version >= 3.8.0. 

`doframework` can run either locally or remotely. For optimal performance, run it on a Kubernetes cluster. Cloud configuration is currently available for AWS and IBM Cloud [OpenShift](https://docs.openshift.com/ "RedHat OpenShift Documentation") clusters.

The framework relies on Cloud Object Storage (COS) to interact with simulation products. Configuration is currently available for [AWS](https://aws.amazon.com/s3/ "AWS S3") or [IBM COS](https://www.ibm.com/cloud/object-storage "IBM Cloud Object Storage").

# Install

To run `doframework` locally, install with

```
$ pip install doframework
```

# Configs

COS specifications are provided in a `configs.yaml`. 

The `configs.yaml` includes the list of source and target bucket names (under `s3:buckets`). Credentials are added under designated fields.

Currently, two cloud service providers are available under `s3:cloud_service_provider`: `aws` and `ibm`.

`s3:endpoint_url` is optional for AWS.

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
    cloud_service_provider: 'aws'

```
**Bucket names above must be distinct**.

# Inputs

`input.json` files provide the necessary metadata for the random genration of optimization problems.

`doframework` will run end to end, once `input.json` files are uploaded to `<inputs_bucket>`. 

The jupyter notebook `./notebooks/inputs.ipynb` allows you to automatically generate input files and upload them to `<inputs_bucket>`.

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
`omega:scale`: scale of jitter when sampling feasibility regions (as a ratio of domain diameter).<br>
`data:N`: number of data points to sample.<br>
`data:noise`: response variable noise.<br>
`data:policy_num`: number of centers in Gaussian mix distribution of data.<br>
`data:scale`: max STD of Gaussian mix distribution of data (as a ratio of domain diameter).

It's a good idea to start experimenting on low-dimensional problems. 

# User App Integration

Your algorithm will be integrated together with `doframework` once it is decorated with `doframework.resolve`. 

A `doframework` experiment runs with `doframework.run()`. The `run()` utility accepts the decorated model and a path to the `configs.yaml`.

Here is an example user application `module.py`.

```
import doframework as dof

@dof.resolve
def alg(data: np.array, constraints: np.array, **kwargs):
    ...    
    return optimal_arg, optimal_val, regression_model

if __name__ == '__main__':
    
    dof.run(alg, 'configs.yaml', objectives=5, datasets=3, **kwargs)
```

The testing framework supports the following inputs to your algorithm: 

`data`: 2D np.array with features X = data[ : , :-1] and response variable y = data[ : ,-1].<br>
`constraints`: linear constraints as a 2D numpy array A. A data point x satisfies the constraints when A[ : , :-1]*x + A[ : ,-1] <= 0.<br>
`lower_bound`: lower bound per feature variable.<br>
`upper_bound`: upper bound per feature variable.<br>
`init_value`: optional initial value.<br>

The `run()` utility accepts the arguments:

`objectives`: number of objective targets to generate per input file.<br>
`datasets`: number of datasets to generate per objective target.<br>
`feasibility_regions`: number of feasibility regions to generate per objective and dataset.<br>
`distribute`: True to run distributively, False to run sequentially.<br>
`logger`: True to see logs, False otherwise.<br>
`after_idle_for`: stop running when event stream is idle after this many seconds.<br>

# Algorithm Prediction Profile

Once you are done running a `doframework` experiment, run the notebook `notebooks/profile.ipynb`. It will fetch the relevant experiment products from the target COS buckets and produce the algorithm's prediction profile and prediction probabilities.

`doframework` produces three types of experiment products files:

* `objective.json`: containing information on (f,O,x*) 
* `data.csv`: containing the dataset the algorithm accepts as input
* `solution.json`: containing the algorithm's predicted optimum

See sample files under `./outputs`/

# Kubernetes Cluster

To run `doframework` on a K8S cluster, make sure you are on the cluster's local `kubectl` context. Log into your cluster, if necessary (applicable to OpenShift, see doc).

You can check your local `kubectl` context and change it if necessary with
```
$ kubectl config current-context
$ kubectl config get-contexts
$ kubectl config use-context cluster_name
>> Switched to context "cluster_name".
```

Now `cd` into your project's folder and run the setup bash script `doframework-setup.sh`. The setup script will generate the cluster configuration file `doframework.yaml` in your project's folder. The setup script requires the absolute path to your `configs.yaml`. Otherwise, it assumes a file `configs.yaml` is located under your project's folder. Running the setup script will establish the `ray` cluster. 

```
$ cd <user_project_folder>
$ doframework-setup.sh --configs ~/path/to/configs.yaml
```

You have the option to adapt `doframework.yaml` to fit your application. 

Use the flag `--project-requirements` to specify the absolute path to your `requirements.txt` file. It will be `pip install -r requirements.txt` on your cluster nodes. 

Use the flag `--project-dir` to specify the absolute path to your project. It will be pip installed on your cluster nodes. 
```
$ doframework-setup.sh --configs ~/path/to/configs.yaml --project-requirements <absolute_requirements_path> --project-dir <absolute_project_path>
```

Use the `--skip` flag to skip re-generating the `doframework.yaml`.
```
$ doframework-setup.sh --skip
```
Or, in case you are familiar with `ray`, run instead 
```
$ ray up doframework.yaml --no-config-cache --yes
```
Upload `input.json` file(s) to your `<inputs_bucket>`. Now you can submit your application `module.py` to the cluster
```
$ ray submit doframework.yaml module.py
```

# Ray Cluster

To observe the `ray` dashboard, connect to `http://localhost:8265` in your browser. See the OpenShift doc for OpenShift-specific instructions.

Some useful health-check commands: 

* Check the status of `ray` pods
```
$ kubectl get pods -n ray
```
* Check the status of the `ray` head node
```
$ kubectl describe pod rayvens-cluster-head-xxxxx -n ray
```
* Monitor autoscaling with
```
$ ray exec doframework.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```
* Connect to a terminal on the head node
```
$ ray attach doframework.yaml
$ ...
$ exit
```
* Get a remote shell to the cluster manually (find the head node ID with `kubectl describe`)
```
$ kubectl -n ray exec -it rayvens-cluster-head-z97wc -- bash
```

After introducing manual changes to `doframework.yaml`, update with
```
$ ray up doframework.yaml --no-config-cache --yes
```

Shutdown the `ray` cluster with
```
$ ray down -y doframework.yaml
```

# Test

Run the setup bash script `doframework-setup.sh` with the `--example` flag to generate the test script  `doframework_example.py` in your project folder.
```
$ cd <user_project_folder>
$ doframework-setup.sh  --configs ~/path/to/configs.yaml --example
```

To run the test script locally, use
```
$ python doframework_example.py --configs ~/path/to/configs.yaml
```

To run the test script on your K8S cluster, use
```
$ ray submit doframework.yaml doframework_example.py --configs configs.yaml
```
[NOTE: we are using the path to the `configs.yaml` that was mounted on cluster nodes under `$HOME`.]

Make sure to upload input json files to `<inputs_bucket>` once you run `doframework_example.py`.

