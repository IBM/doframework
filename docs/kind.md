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

# KiND [on a Mac]

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

Now install
```
$ brew install kind
```

Set up the KiND cluster by running the bash script `doframework-setup.sh`. This will generate the cluster `doframework.yaml` in your working directory. The setup script accepts the absolute path to your `configs.yaml`.

The ` --skip` flag will tell the `doframework-setup.sh` script not to re-generate `doframework.yaml`.
```
$ cd <user_project_folder>
$ doframework-setup.sh --configs ~/path/to/configs.yaml --kind  [--skip]
```

Run your application `module.py` on the KiND cluster with.
```
$ ray submit doframework.yaml module.py
```
At this point you may encounter
```
RuntimeError: Head node of cluster (rayvens-cluster) not found!
```
This is typically a resource allocation issue. To investigate this issue, make sure you have the [kubectl](https://kubernetes.io/docs/tasks/tools/ "Install kubectl") installed. Look for the `rayvens-cluster-head` node
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
One quick way to resolve the issue is to go to `Docker Desktop` and look under `Resources -> Advanced`. Play with the number of CPUs / Memory GBs, but don't go crazy, otherwise your machine will start running really ... really ... s-l-o-w.

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
