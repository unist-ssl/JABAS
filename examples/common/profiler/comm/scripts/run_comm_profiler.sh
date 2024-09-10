#!/bin/bash

rank=$1
world_size=$2  # Number of nodes
master=$3
bucket_cap_mb=$4
intra=$5
inter=$6
profile_dir=$7

if [ $# -lt 7 ]; then
  echo "[USAGE] [rank] [world size] [master] [bucket cap (MB)] [intra (bytes/sec)] [inter (bytes/sec)] [profile dir]"
  exit 1
fi

if [ $world_size == "1" ]; then
  master=`hostname`
  echo "Single-node training - master: "$master
else
  if [ -z $master ]; then
    echo "[ERROR] If world_size > 1, master must be configured"
    exit 1
  fi
fi

python -m iidp.profiler.comm.profiler \
    --dist-url 'tcp://'${master}':20000' \
    --multiprocessing-distributed \
    --rank $rank \
    --world-size $world_size \
    --bucket-cap-mb $bucket_cap_mb \
    --network-bandwidths $intra $inter \
    --profile-dir $profile_dir
