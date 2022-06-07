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

# Issues

---------------

## Idle

When an experiment goes idle, or it does not go through full cycle, this may have to do with `after_idle_for`. 

The `after_idle_for` time window should be sufficiently large for simulation products to make it through to the next stage. This is especially true when optimization problem dimensions are higher, or when your algorithm takes longer.

---------------

## Autoscaling on OpenShift
 
If you're having problems with autoscaling, for instance, the application is only running on the head node, you can start by checking the `ray` logs with
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
Just wait. Eventually it'll go through. If it ultimately fails, this may be a resources issue -- your cluster may be too small for the resources requested. Set the values of the variables `--mem` and / or `--cpu` to reflect your cluster resources when you run `doframework-setup.sh`.

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
This is an indication that you haven't logged into your cluster (see login instructions for OpenShift). 

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

