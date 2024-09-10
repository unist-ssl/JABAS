#!/bin/bash

mem_profile_dir=$1
comp_profile_dir=$2
gpu_reuse_pause_time=${3:-"300"}

if [ $# -lt 2 ]; then
  echo "[USAGE] [memory profile dir] [comp profile dir] [GPU reuse pause time (default: 300 sec)]"
  exit 1
fi

python profiler/comp/comp_profiler_driver_with_max_mem.py \
    --mem-profile-dir $mem_profile_dir \
    --profile-dir $comp_profile_dir \
    -p $gpu_reuse_pause_time