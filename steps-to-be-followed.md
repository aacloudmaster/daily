## 1. Installing Required Libraries
You’ll need to install a few libraries for machine learning tasks and dependencies like NumPy, Pandas, and scikit-learn. These can be installed with `pip`.

### NumPy Installation
To install NumPy, run:
```bash
$ pip install numpy
```

### Pandas Installation
To install Pandas, run:
```bash
$ pip install pandas
```

### scikit-learn Installation
To install or upgrade scikit-learn, run:
```bash
$ pip install -U scikit-learn
```

## 2. Running Docker Container Locally
You can run a Docker container for the machine learning app and make predictions locally.

### Pull the Docker Image
First, pull the Docker image for the application:
```bash
$ docker pull ssadcloud/mlapp:latest
```

### Run the Docker Container
Now, run the Docker container:
```bash
$ docker run -p 5000:5000 ssadcloud/mlapp:latest
```

This starts the application on port `5000`.

### Test the Application
In another terminal tab, use `curl` to send a prediction request:
```bash
$ curl http://localhost:5000/predict
```

## 3. Kubernetes Setup
If you want to run the app on Kubernetes (e.g., for scaling), follow these steps:

### Installing Kubernetes (kind)
To install `kind` (Kubernetes in Docker), run the following command on Windows:
```bash
curl.exe -Lo kind-windows-amd64.exe https://kind.sigs.k8s.io/dl/v0.24.0/kind-windows-amd64
Move-Item .\kind-windows-amd64.exe c:\kind\kind.exe
```

### Create a Kubernetes Cluster
Create a local Kubernetes cluster using `kind`:
```bash
$ kind create cluster --name main-k8s-cluster
```

Verify that the cluster is up and running:
```bash
$ kubectl get nodes
$ kubectl get pods
$ kubectl get deployments
```

### Deploying the Application on Kubernetes
Use the following commands to deploy the machine learning app.

#### Apply Deployment Configuration:
```bash
$ kubectl apply -f manifests/mlapp-deployment.yaml
```

#### Check Pods and Port-Forward:
After applying the deployment, verify the pods:
```bash
$ kubectl get pods
```

Then, forward the container's port to your local machine:
```bash
$ kubectl port-forward <<pod-name>> 5000:5000
```

#### Delete Deployment:
When you're done, you can delete the deployment with:
```bash
$ kubectl delete -f manifests/mlapp-deployment.yaml
```

### Deploying the Service
Apply the service configuration:
```bash
$ kubectl apply -f manifests/mlapp-service.yaml
```

Check the services:
```bash
$ kubectl get svc
```

Port-forward the service to access it locally:
```bash
$ kubectl port-forward svc/mlapp-service 5000:5000
```

#### Delete Service:
When you no longer need the service, you can delete it with:
```bash
$ kubectl delete -f manifests/mlapp-service.yaml
```

## 4. Kubeflow Setup
Kubeflow Pipelines help manage and deploy machine learning workflows.

### Install Kubectl
If you haven't installed `kubectl` yet, you can follow these steps:

#### For Windows:
Download and install `kubectl` from the official Kubernetes website:
[Install kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/)

#### For Ubuntu:
Use the following commands to install `kubectl` on Ubuntu:
```bash
sudo apt update
sudo apt install kubectl
```

### Kubeflow Deployment
You can deploy Kubeflow Pipelines locally using [this guide](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/localcluster-deployment/). 

Install Kubeflow Pipelines:
```bash
pip install kfp==2.0.0b13
```

Once Kubeflow is set up, you can access the Pipelines UI by port-forwarding:
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Open your browser and navigate to `http://localhost:8080` to access the Kubeflow Pipelines UI.



## 5. Creating and Setting Up a Virtual Environment (for Ubuntu Linux)
A virtual environment is a self-contained directory that contains a Python installation for a specific version of Python, along with several additional packages.

### Step 1: Install Python (if not already installed)
Check if Python is installed:
```bash
$ python3 --version
```

If it’s not installed, install it:
```bash
$ sudo apt update
$ sudo apt install python3 python3-pip
```

### Step 2: Install `python3-venv`
If the `venv` module is not installed by default, run:
```bash
$ sudo apt install python3-venv
```

### Step 3: Create the Virtual Environment
Navigate to the directory where you want the virtual environment:
```bash
$ cd ~/my_project
```

Create the virtual environment:
```bash
$ python3 -m venv myenv
```

This will create a folder called `myenv` that contains the isolated Python environment.

### Step 4: Activate the Virtual Environment
To activate the virtual environment, run:
```bash
$ source myenv/bin/activate
```

After activation, your command prompt should change to show the virtual environment name, e.g., `(myenv)`.

### Step 5: Install Packages into the Virtual Environment
With the virtual environment activated, install necessary packages:
```bash
$ pip install requests
```

You can replace `requests` with any Python package you need, e.g., NumPy, pandas, scikit-learn, etc.

### Step 6: Deactivate the Virtual Environment
Once you're done with the virtual environment, deactivate it by running:
```bash
$ deactivate
```

### Step 7: (Optional) Remove the Virtual Environment
If you no longer need the virtual environment, delete the `myenv` folder:
```bash
$ rm -rf myenv
```



## 6. Port-Forwarding for AWS EC2
When using Kubernetes and Kubeflow on AWS EC2, you might want to port-forward services to make them accessible locally.

For example, you can use the following command to forward the Kubeflow UI:
```bash
$ kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow --address='0.0.0.0'
```

This will allow you to access the Kubeflow Pipelines UI at `http://<your-ec2-public-ip>:8080` from your browser.



### Summary
This document covers the installation and configuration steps for:
1. Installing necessary Python libraries for machine learning.
2. Running the application in a Docker container.
3. Setting up Kubernetes using `kind` for local cluster management.
4. Deploying the app and accessing it through Kubernetes services.
5. Setting up Kubeflow Pipelines to manage ML workflows.
6. Creating and managing a Python virtual environment on Ubuntu.
7. Port-forwarding Kubernetes services on AWS EC2.

# AWS CLI Commands:
aws --version
aws configiure
  AWS_ACCESS_KEY_ID:
  AWS_SECRET_ACCESS_KEY_ID:
  REGION: us-east-2
  output: json
aws s3 ls
aws s3 mb s3://rajbt123

# HELM FOR PROMETHEUS AND GRAFANA

$ helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
"prometheus-community" has been added to your repositories

$ helm install main-prometheus-release prometheus-community/prometheus --version 26.0.0
NAME: main-prometheus-release
LAST DEPLOYED: Fri Nov 29 08:21:23 2024
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
The Prometheus server can be accessed via port 80 on the following DNS name from within your cluster:
main-prometheus-release-server.default.svc.cluster.local


Get the Prometheus server URL by running these commands in the same shell:
  export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=prometheus,app.kubernetes.io/instance=main-prometheus-release" -o jsonpath="{.items[0].metadata.name}")
  kubectl --namespace default port-forward $POD_NAME 9090


The Prometheus alertmanager can be accessed via port 9093 on the following DNS name from within your cluster:        
main-prometheus-release-alertmanager.default.svc.cluster.local


Get the Alertmanager URL by running these commands in the same shell:
  export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=alertmanager,app.kubernetes.io/instance=main-prometheus-release" -o jsonpath="{.items[0].metadata.name}")
  kubectl --namespace default port-forward $POD_NAME 9093
#################################################################################
######   WARNING: Pod Security Policy has been disabled by default since    #####
######            it deprecated after k8s 1.25+. use                        #####
######            (index .Values "prometheus-node-exporter" "rbac"          #####
###### .          "pspEnabled") with (index .Values                         #####
######            "prometheus-node-exporter" "rbac" "pspAnnotations")       #####
######            in case you still need it.                                #####
#################################################################################


The Prometheus PushGateway can be accessed via port 9091 on the following DNS name from within your cluster:
main-prometheus-release-prometheus-pushgateway.default.svc.cluster.local


Get the PushGateway URL by running these commands in the same shell:
  export POD_NAME=$(kubectl get pods --namespace default -l "app=prometheus-pushgateway,component=pushgateway" -o jsonpath="{.items[0].metadata.name}")
  kubectl --namespace default port-forward $POD_NAME 9091

For more information on running Prometheus, visit:
https://prometheus.io/

# GRAFANA
helm repo add grafana https://grafana.github.io/helm-charts
helm install main-grafana-release grafana/grafana --version 8.6.3

NAME: main-grafana-release
LAST DEPLOYED: Fri Nov 29 08:24:32 2024
NAMESPACE: default
STATUS: deployed
REVISION: 1
NOTES:
1. Get your 'admin' user password by running:

   kubectl get secret --namespace default main-grafana-release -o jsonpath="{.data.admin-password}" | base64 --decode ; echo

2. The Grafana server can be accessed via port 80 on the following DNS name from within your cluster:

   main-grafana-release.default.svc.cluster.local

   Get the Grafana URL to visit by running these commands in the same shell:
     export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=main-grafana-release" -o jsonpath="{.items[0].metadata.name}")
     kubectl --namespace default port-forward $POD_NAME 3000

3. Login with the password from step 1 and the username: admin
#################################################################################
######   WARNING: Persistence is disabled!!! You will lose your data when   #####
######            the Grafana pod is terminated.                            #####
#################################################################################