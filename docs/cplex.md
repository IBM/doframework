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

# CPLEX

In case your application relies on a solver, such as `CPLEX`, you will need to mount it onto the cluster nodes.

To allow for a silent installation of `CPLEX`, create a `installer.properties` file under your project folder. Add the following lines to your `installer.properties` file:
```
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
USER_INSTALL_DIR=/home/ray
INSTALLER_LOCALE=en
```


Add the following to your `doframework.yaml` file
```
file_mounts: {
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

