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

# Create K8S Cluster on AWS EC2

AWS does not have native support for Kubernetes. [Here](https://zero-to-jupyterhub.readthedocs.io/en/latest/kubernetes/amazon/step-zero-aws.html "K8S cluster on AWS") is a guide for setting up a K8S cluster on an AWS EC2 instance.

## Launch EC2 instance

This node will be used to provision, manage, and ultimately tear down the K8S cluster we will create.

* AWS Dashboard ==> Services ==> EC2 ==> Launch instances
* Pick one of the small (free tier) instance types (e.g., `t2.micro`)
* Launch the instance with a security keypair. This will produce a `name.pem` file.
* Download the `name.pem` file and store it somewhere safe
* Restrict permissions to your `name.pem` file with
```
$ sudo chmod 400 ~/path/to/name.pem
```

## SSH EC2 Instance

[SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html "SSH EC2 Instance") into your small EC2 instance.

* Find your instance’s public IPv4 address under Details [on the EC2 Dashboard]
* If you created an Amazon Linux instance, you’ll need to prefix the IP address with `ec2-user`, if your cluster is ubuntu, you’ll need to prefix it with `ubuntu`. We will assume an Amazon Linux instance throughout this demo.
```
$ ssh -i ~/path/to/name.pem ec2-user@55.1111.222.33
>> ‘authenticity of host … can’t be established … ECDSA key fingerprint is … Are you sure you want to continue …?’
```
* Type ‘yes’ and ENTER. You will see ‘Warning: Permanently added …’
* If you’re getting WARNING: UNPROTECTED PRIVATE KEY FILE! you have a permissions issue on your `name.pem` file.

## Install KOPS

Install [kops](https://github.com/kubernetes/kops/blob/HEAD/docs/install.md "Install kops") on your EC2 instance to create and manage K8S clusters.
```
$ curl -Lo kops https://github.com/kubernetes/kops/releases/download/$(curl -s https://api.github.com/repos/kubernetes/kops/releases/latest | grep tag_name | cut -d '"' -f 4)/kops-linux-amd64
$ chmod +x ./kops
$ sudo mv ./kops /usr/local/bin/
```
Test your installation with
```
$ kops version
```

## Install kubectl

Install the [K8S CLI](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/ "K8S cli") `kubectl` on your EC2 instance.
```
$ curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
$ sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```
Test your installation with
```
$ kubectl version --client
```

## DNS Name

To create a K8S cluster with kops, you'll need to use a valid [DNS name](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/CreatingHostedZone.html "DNS Name"). Since we are not using pre-configured DNS, we will use the suffix “.k8s.local”. When the DNS name ends in .k8s.local the cluster will use internal hosted DNS.
```
$ export NAME=cluster_name.k8s.local
```

## SSH Key Pair

Generate SSH key pair
```
$ ssh-keygen
```
And follow the steps ...

## KOPS State Store

Create a dedicated S3 bucket to store your K8S cluster state: AWS Dashboard ==> Services ==> Storage ==> S3 ==> Buckets: Create bucket. 

Once the S3 bucket is created, set up the variable
```
$ export KOPS_STATE_STORE=s3://cluster-log
```

## AWS CLI

Install AWS CLI and set up its credentials. Use `apt-get` if you chose a ubuntu instance, or `yum` if an Amazon instance.
```
$ sudo yum update
$ sudo yum install awscli 
$ aws configure
```
And follow the steps ...

## Region and Zone

Set up the region and zones variables for your K8S cluster.
```
$ export REGION=`curl -s http://169.254.169.254/latest/dynamic/instance-identity/document|grep region|awk -F\" '{print $4}'`
$ export ZONES=$(aws ec2 describe-availability-zones --region $REGION | grep ZoneName | awk '{print $2}' | tr -d '"')
$ echo $ZONES [check values]
```
A trailing comma in zone list may cause an issue ... You may have to fix that manually.

## Create K8S Cluster
```
$ kops create cluster $NAME \
--zones "$ZONES" \
--state $KOPS_STATE_STORE \
--master-size c5.2xlarge \
--node-count 2 \
--node-size c5.2xlarge \
--yes
>> kOps has set your kubectl context to cluster_name.k8s.local
>> Cluster is starting.  It should be ready in a few minutes
```
If you absolutely must, add the option `--authorization RBAC` in the `kops create` command for role-based access control.
```
$ kops validate cluster --wait 5m
>> Validation Failed … [set up may take a few minutes]
```
The default Security group for your new cluster's master / nodes is `cluster_name.k8s.local` [see under Dashboard Security groups].

Check that all nodes are on status Ready.
```
$ kubectl get nodes
```
Or check
```
$ kubectl cluster-info
>> Kubernetes control plane is running at https://api-dof-k8s-local-f862dq-108762079.us-east-1.elb.amazonaws.com
>> CoreDNS is running at https://api-dof-k8s-local-f862dq-108762079.us-east-1.elb.amazonaws.com/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
```

Optional: SSH to the master node
```
$ ssh -i ~/.ssh/id_rsa ec2-user@api.cluster_name.k8s.local
```

## Local kubectl Context 

Configure your local `kubectl` contexts to include your newly created K8S cluster.

Generate your `cluster_name.k8s.local` `/.kube/config` file at default location `$HOME`
```
$ kops export kubecfg
```
If you’re getting a denied access, try
```
$ kops export kubecfg $NAME --admin
```

Now you can finally exit the instance
```
$ exit
```

Copy the kube config file to your local machine
```
$ scp -i ~/path/to/name.pem ec2-user@55.1111.222.33:/home/ec2-user/.kube/config ~/tmp_kubeconfig
```

Copy the contents of `tmp_kubeconfig` and paste to your local `~/.kube/config`. Switch `kubectl` context to your new K8S cluster.
```
$ kubectl config current-context
$ kubectl config get-contexts
$ kubectl config use-context cluster_name.k8s.local
>> Switched to context "cluster_name.k8s.local".
```

Check access and status of your K8S cluster nodes
```
$ kubectl get nodes
```

## Delete cluster

Do NOT use the EC2 dashboard to delete your K8S cluster!!! `kops` will handle a clean delete.

SSH into your small EC2 instance as above.
```
$ ssh -i ~/path/to/name.pem ec2-user@55.1111.222.33
$ export NAME=cluster_name.k8s.local
$ kops delete cluster --name=$NAME --yes
```