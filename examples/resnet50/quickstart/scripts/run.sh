#!/bin/bash

if [[ -z $NODE0 ]] ; then
    echo "NODE0 must be exported, but NODE0: "$NODE0
    exit 1
fi

node_rank=$1
if [ -z $node_rank ] ; then
    echo "[USAGE] [node rank]"
    exit 1
fi

local_batch_size=32
num_vsws=2
ga_steps=0
weight_sync_method=overlap
user_config_file=quickstart/config.json

./scripts/convergence/distributed/distributed_run.sh $node_rank $local_batch_size $num_vsws $ga_steps $weight_sync_method $user_config_file $NODE0
