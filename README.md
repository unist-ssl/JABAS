# JABAS

## Introduction
*JABAS* (*J*oint *A*daptive *B*atching and *A*utomatic *S*caling) is a novel DNN training system for a heterogeneous GPU cluster.
Major components of JABAS are a DNN training framework called IIDP, which provides the same theoretical convergence rate of distributed SGD in a heterogeneous GPU cluster,
a fine-grained adaptive batching technique with dynamic configuration,
and a coarse-grained automatic resource scaling technique that leverages the prediction of global batch size changes for an epoch to auto-scale GPU resources optimally.

For more details, please refer to EuroSys '25 paper entitled **JABAS: Joint Adaptive Batching and Automatic Scaling for DNN Training on Heterogeneous GPUs** (link will be uploaded).

## Table of Contents

<!-- TOC GFM -->

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Software Packages Installation](#software-packages-installation)
  * [Remote storage - NFS setup](#remote-storage---nfs-setup)
* [Code structure](#code-structure)
* [QuickStart](#quickstart-for-resnet-50)
* [Run JABAS](#run-jabas)

<!-- /TOC -->

## Getting Started
### Prerequisites
* Ubuntu >= 16.04
* Anaconda3 4.13.0
* Python 3.8
* NVIDIA driver >= 450.80.02
* CUDA 11.1
* cuDNN 8.2.1
* Remote storage (e.g, NFS, AWS S3)

### Software Packages Installation
Install CUDA and CuDNN
- CUDA download toolkit [[link]](https://developer.nvidia.com/cuda-toolkit-archive). Make sure that `/usr/local/cuda` is linked to `/usr/local/cuda-11.1`.
- CuDNN download toolikt [[link]](https://developer.nvidia.com/rdp/cudnn-archive).

Install Anaconda (Optional) - If Anaconda has already been installed, skip this step.
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

Prepare conda environment
```bash
CONDA_ENV=jabas
conda create -n $CONDA_ENV python=3.8 -y
conda activate $CONDA_ENV
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install -c pytorch magma-cuda111 -y # For CUDA 11.1
```
Install IIDP by building source code (NOTE: IIDP will be released before the final version is completed.)
```bash
BASE=$HOME # Set the custom base path
JABAS_HOME=$BASE/JABAS
IIDP_HOME=$JABAS_HOME/IIDP
PYTORCH_HOME=$BASE/pytorch
VISION_HOME=$BASE/vision

cd $BASE

git clone https://github.com/unist-ssl/JABAS

git clone --recursive -b v1.8.1 https://github.com/pytorch/pytorch.git
cd $PYTORCH_HOME
patch -p1 < $IIDP_HOME/pytorch.patch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install

cd $BASE
git clone -b v0.9.1 https://github.com/pytorch/vision.git
cd $VISION_HOME
pip install pillow==10.4.0
python setup.py install

cd $IIDP_HOME
pip install -r requirements.txt
python setup.py install
```

Install JABAS
```bash
cd $JABAS_HOME
pip install -r requirements.txt
python setup.py install
```

### Remote storage - NFS setup
We provide a guideline to setup NFS (Network File System), assuming mount point is `/mnt`.

Install NFS packages
```bash
sudo apt-get install nfs-kernel-server nfs-common -y
```

Make NFS mount directory
```bash
USER=`whoami`
sudo mkdir /mnt/nfs
sudo chown -R $USER:$USER /mnt/nfs
sudo chmod -R 777 /mnt/nfs
```

/etc/exports file setup

In /etc/exports file, write the below configuration.
```
$ sudo vi /etc/exports
----------------------------------------------------
/mnt/nfs *(rw,no_root_squash,no_all_squash,async)
----------------------------------------------------
```

On a `main node (NFS server)`:

Start NFS server
```bash
sudo systemctl restart nfs-server
```
Check if NFS server status is active
```bash
sudo systemctl status nfs-server
```

On `NFS client nodes`:

Set user's NFS server ip
```bash
$ NFS_SERVER_IP=<User NFS server ip>
```
```bash
sudo mount -t nfs $NFS_SERVER_IP:/mnt/nfs /mnt/nfs
```
Check if NFS directory is mounted
```bash
df -h
```

## Code structure
- `IIDP/iidp/`: Implementation of IIDP.
- `jabas/train/trainer.py`: Core runtime of JABAS to cooperate with adaptive batching and automatic scaling.
- `jabas/profiler/` and `IIDP/iidp/profiler/`: Profiler.
- `jabas/config/` and `IIDP/iidp/config/`: Configuration Solver.
- `jabas/elastic/`: Main code for elastic training. gRPC communication stack (jabas/elastic/runtime/) is mainly borrowed from [stanford-futuredata/gavel](https://github.com/stanford-futuredata/gavel/tree/master/scheduler/runtime).
- `examples/`: Example (benchmark) codes for JABAS.

## QuickStart for ResNet-50
Refer to [README.md](examples/resnet50/quickstart/) in ```examples/resnet50/quickstart/``` directory.

## Run JABAS
Refer to [README.md](examples/) in `examples/` directory.
