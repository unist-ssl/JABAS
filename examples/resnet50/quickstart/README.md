# QuickStart for ResNet-50

We assume that current path is ```$JABAS_HOME/examples/resnet50```.

We provide 3 types of profile data on the below hardware setup:
1) Computation: ```quickstart/cluster_comp_profile_data```
2) Memory: ```quickstart/cluster_mem_profile_data```
3) Communication (All-reduce): ```quickstart/comm_profile_data``` and ```quickstart/bucket_profile_data```

## Hardware setup
**Requirement**: Nodes with NVIDIA V100 and P100 GPU must be equipped for quickstart example.

Assume the below hardware setup of two GPU nodes.
```
Hostname: node0                               # example hostname
GPU: 4 x Tesla V100-PCIE-16GB (TFPLOS: 14.13)
PCIe bandwidth: 15.75 GB/s
InfiniBand (IB) bandwidth: 100Gbps
```
```
Hostname: node1                               # example hostname
GPU: 4 x Tesla P100-PCIE-16GB (TFPLOS: 9.52)
PCIe bandwidth: 15.75 GB/s
InfiniBand (IB) bandwidth: 100Gbps
```

## Before getting stared
[**Important**] For all instructions listed below, if not mentioning a specific node, follow the procedure on **every node**.

**Current path** is ```$JABAS_HOME/examples/resnet50```.

## Prerequisites
1. Activate conda environment
    ```bash
    conda activate $CONDA_ENV
    ```
2. Set remote storage (e.g, NFS) and dataset storage path on every node
    ```bash
    $ export JABAS_REMOTE_STORAGE=<remote storage path>
    $ export JABAS_DATA_STORAGE=<dataset storage path>
    ```
    Example configuration
    ```
    $ export JABAS_REMOTE_STORAGE=/mnt/nfs
    $ export JABAS_DATA_STORAGE=/mnt/dataset
    ```
3. ImageNet dataset on every node

    3-1. Download the ImageNet dataset from http://www.image-net.org/ to ```JABAS_DATA_STORAGE```.

    Two .tar files are downloaded: ```ILSVRC2012_img_train.tar``` and ```ILSVRC2012_img_val.tar```

    3-2. Run the script to extract and make labeled folders.
        Dataset is stored to ```JABAS_DATA_STORAGE/imagenet```.
    ```bash
    ./scripts/utils/prepare_dataset.sh
    ```

## Replace profile data and JABAS configuration files with user node setup

As the provided profile data and JABAS configuration files (`config.json` and `cluster_info.json`) assume the hostname of two GPU nodes as `node0` and `node1`,
we provide instructions on how to replace it with the user environment.


Set User's V100 node and P100 node hostname.
```bash
$ export NODE0=<User V100 node hostname>
$ export NODE1=<User P100 node hostname>
```
Run the below command to replace configuration with the user node setup.
```bash
./quickstart/scripts/prepare_config.sh
```
In case you want to revert to an initial given setup, execute the below command.
```bash
./quickstart/scripts/initialize_config.sh
```

## Run configuration solver for global batch size of 128
Execute the below command to reproduce the result.
```bash
./quickstart/scripts/run_config_solver.sh
```
```bash
...
========================================================================================================
[INFO] Solution - GBS: 128 | LBS: 32 | weight sync method: overlap | config: ['node0:4GPU,VSW:1,GA:0']
========================================================================================================
```
Output of initial JABAS configuration indicates:
1) Local Batch Size (LBS) is ```32```.
2) One-way weight synchronization method on IIDP is ```overlapping```.
3) For each GPU in  ```node0 (V100)```, the number of Virtual Stream Workers (VSWs) is ```1```, and the Gradient Accumulation (GA) step is ```0```.
4) ```node1 (P100)``` is not used for the initial runtime phase.

## Run ResNet-50 on JABAS

[Preparation] Before executing the command on every terminal, please check if the conda environment is activated. If not, execute the below command.
  ```bash
  conda activate $CONDA_ENV
  ```

[Runtime behavior] In an initial phase for training, JABAS only runs on the V100 node (Node 0). The P100 node (Node 1) will eventually be used as global batch size increases during training. This is the merit of JABAS, which jointly takes advantage of auto-scaling with adaptive batching.

[Final result] Total train time and cost are shown on the screen. The log file is saved to ```JABAS_REMOTE_STORAGE/resnet50_jabas_convergence_log```.

On V100 node (terminal 0):
```bash
./quickstart/scripts/run_elastic_agent.sh
```
```bash
jabas.elastic.runtime.rpc.scheduler_server:INFO [YYYY-MM-DD H:M:S] Starting server at <V100 node ip>:<port>
```

After the agent server execution starts,

On V100 node (terminal 1):
```bash
$ export JABAS_REMOTE_STORAGE=<remote storage path>
$ export JABAS_DATA_STORAGE=<dataset storage path>
$ NODE0=<User V100 node hostname>
```
```bash
./quickstart/scripts/run.sh 0
```

On P100 node (terminal 0):
```bash
$ export JABAS_REMOTE_STORAGE=<remote storage path>
$ export JABAS_DATA_STORAGE=<dataset storage path>
$ NODE0=<User V100 node hostname>
```
```bash
./quickstart/scripts/run.sh 1
```
