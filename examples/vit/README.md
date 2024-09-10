# ViT

The original code of this example comes from [yitu-opensource/T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)

## Table of Contents

<!-- TOC GFM -->

* [Prerequisites](#prerequisites)
* [Configuration of JABAS](#configuration-of-jabas)
  * [1. Memory profiler](#1-memory-profiler)
  * [2. Computation profiler](#2-computation-profiler)
  * [3. DDP bucket profiler](#3-ddp-bucket-profiler)
  * [4. Synchronize profile data](#4-synchronize-computation-and-memory-profile-data-for-heterogeneous-gpus)
  * [5. Prepare config file (JSON)](#5-prepare-config-file-json-for-jabas)
* [How to run on JABAS](#how-to-run-on-jabas)

<!-- /TOC -->

## Prerequisites
- Activate conda environment
  ```bash
  conda activate $CONDA_ENV
  ```
- Software packages
  ```bash
  pip install -r requirements.txt --no-deps
  ```
- ImageNet dataset
  1) Download the ImageNet dataset from http://www.image-net.org/ to ```JABAS_DATA_STORAGE```.
      - Two .tar files are downloaded: ```ILSVRC2012_img_train.tar``` and ```ILSVRC2012_img_val.tar```
  2) Run the script to extract and make labeled folders.

      Dataset is stored to ```JABAS_DATA_STORAGE/imagenet```.
      ```bash
      ./scripts/utils/prepare_dataset.sh
      ```

## Configuration of JABAS
We provide the configuration solver that finds the best *dynamic* configuration of IIDP *running upon JABAS*
with repect to 1) local batch size 2) the number of VSWs 3) gradient accumulation steps.

To get the configuration, the following steps are required.

### 1. Memory Profiler
Memory profiler finds 1) the ranges of local batch size 2) the maximum number of VSWs with each local batch size.
If LBS search mode is 'static', the user-specified local batch size is only searched.

Generate memory profile data on each node.
```
./profiler/memory/scripts/run_mem_profiler.sh [Local Batch Size] [profile dir (default: mem_profile_data)] [LBS search mode (default: dynamic)]
```
Example
```
# On node0, node1
./profiler/memory/scripts/run_mem_profiler.sh 32
```

### 2. Computation Profiler
Generate computation profile data on each node with memory profile data.

For each server, execute computation profiler to get profile data json file.

GPU pause time is set to 300 seconds by default because the temperature of the GPUs can affect the accuracy of time measurement, potentially impacting the performance model.
  ```
  ./profiler/comp/scripts/run_comp_profiler_with_max_mem.sh [local mem profile dir] [local comp profile dir] [GPU reuse pause time (default: 300 sec)]
  ```
Example
```
# On node0, node1
./profiler/comp/scripts/run_comp_profiler_with_max_mem.sh mem_profile_data comp_profile_data
```

### 3. DDP Bucket Profiler
Generate profile data on each node.
```
./profiler/ddp_bucket/scripts/run_bucket_profiler.sh [profile dir (default: bucket_profile_data)] [plot dir (default: bucket_profile_plot)] [visible GPU ID (optional)]
```
Example
```
# On node0, node1
./profiler/ddp_bucket/scripts/run_bucket_profiler.sh
```

### 4. Synchronize computation and memory profile data for heterogeneous GPUs
All-gather and broadcast profile data results on every node.

After finishing the profile step on every node, execute the below script.
  ```
  ./profiler/utils/sync/sync_data_across_servers.sh [local profile dir] [all-gathered profile dir] [master node] [slave nodes]
  ```
Example
```
# On node0, node1
./profiler/utils/sync/sync_data_across_servers.sh mem_profile_data cluster_mem_profile_data node0 node1
./profiler/utils/sync/sync_data_across_servers.sh comp_profile_data cluster_comp_profile_data node0 node1
```

### 5. Prepare config file (JSON) for JABAS
- Required format of configuration file
  ```json
  {
    "memory_profile_dir": "{mem profile dir}",
    "comp_profile_dir": "{comp profile dir}",
    "comm_profile_dir": "{comm profile dir}",
    "bucket_profile_dir": "{bucket profile dir}",
    "gpu_cluster_info": "{gpu cluster info file (JSON)}",
    "batch_size_lower_bound": "{minimum global batch size}",
    "batch_size_upper_bound": "{maximum global batch size}",
    "similarity_target": "{similarity target value}",
    "available_servers": ["{first node}", "{second node}"]
  }
  ```
  Example config file: ```jabas_config/example.json```
  ```json
  {
    "memory_profile_dir": "cluster_mem_profile_data",
    "comp_profile_dir": "cluster_comp_profile_data",
    "comm_profile_dir": "../common/comm_profile_data",
    "bucket_profile_dir": "bucket_profile_data",
    "gpu_cluster_info": "../common/cluster_config/example_cluster_info.json",
    "batch_size_lower_bound": 512,
    "batch_size_upper_bound": 32768,
    "similarity_target": 0.3,
    "available_servers": ["node0", "node1"]
  }
  ```
  We provide more optional configuration parameters. Please refer to [jabas/docs/CONFIG.md](../../jabas/docs/CONFIG.md).

## How to run on JABAS
- Before executing the command on every terminal, please check if the conda environment is activated. If not, execute the below command.
  ```bash
  conda activate $CONDA_ENV
  ```
- Initial configuration

  This step is unnecessary on every node, just one node is enough.
  ```bash
  ./scripts/config/run_config_solver.sh [config file (JSON)] [global batch size] [weight sync method (default: recommend)]
  ```
  Example
  - node0 (main node)
    ```bash
    ./scripts/config/run_config_solver.sh jabas_config/example.json 512
    ```
- JABAS scheduler

  Scheduler helps elastic training to execute on assigned GPUs on main node.
  ```bash
  python -m jabas.elastic.run_scheduler -c ../common/initial_elastic_config/{initial GPU config JSON file}
  ```
  Example
  - node0 (main node)
    ```
    python -m jabas.elastic.run_scheduler -c ../common/initial_elastic_config/2_node_config.json
    ```
- JABAS worker
  ```
  ./scripts/convergence/distributed/distributed_run.sh [node rank] [local batch size] [number of VSWs] [accum step] [weight sync method] [jabas config file] [master] [scheduler ip (default: master)]
  ```
  Example
  - node0
    ```
    ./scripts/convergence/distributed/distributed_run.sh 0 32 1 2 recommend jabas_config/example.json node0
    ```
  - node1
    ```
    ./scripts/convergence/distributed/distributed_run.sh 1 32 1 0 recommend jabas_config/example.json node0
    ```