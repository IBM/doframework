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

# OpenShift

This assumes you've already set up your OpenShift cluster. It assumes you have `kubectl`, `ibmcloud` and `oc` CLI installed.

Log into IBM cloud services and follow the instructions.
```
$ ibmcloud login --sso
```
Generate a token for your OpenShift cluster (good for 24HR). Go to https://cloud.ibm.com (make sure your connection has access rights). Click on OpenShift Web Console (top right). Click on `IAM#user` and look for `Copy Login Command`. Copy the login command (`oc login ...`).
```
$ oc login --token=shaxxx~xxxx --server=https://xxx.xx-xx.xx.cloud.ibm.com:xxxxx
```

If you haven't already, define a new ray project [done once]. 
```
$ oc new-project ray
```
If you have already defined the "ray" project, you'll find it under
```
$ oc projects
```

Run the bash script `doframework-setup.sh` to establish the `ray` cluster. The setup script will generate the cluster configuration file `doframework.yaml` in your project's folder. The setup script requires the absolute path to your `configs.yaml`. Otherwise, the setup script assumes a file `configs.yaml` is located under your project's folder.

```
$ cd <user_project_folder>
$ doframework-setup.sh --configs ~/path/to/configs.yaml [--skip]
```
Use the `--skip` flag to skip generating a new `doframework.yaml` file. Otherwise, in case you are familiar with `ray`, run 
```
$ ray up doframework.yaml --no-config-cache --yes
```

Upload `input.json` file(s) to your `<inputs_bucket>`.

Submit your application to the `ray` cluster
```
$ ray submit doframework.yaml module.py
```

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