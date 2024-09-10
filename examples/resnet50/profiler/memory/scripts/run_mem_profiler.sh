#!/bin/bash

local_batch_size=$1
profile_dir=${2:-mem_profile_data}
lbs_search_mode=${3:-dynamic}

if [ $# -lt 1 ]; then
  echo "[USAGE] [Local Batch Size] [profile dir (default: mem_profile_data)] [LBS search mode (dynamic, static)]"
  exit 1
fi

timestamp=`date +%Y%m%d_%H%M%S`
profile_log_dir=${timestamp}_${profile_dir}_log
echo "Make memory profile log dir:" $profile_log_dir
mkdir -p $profile_log_dir

if [ $lbs_search_mode == "dynamic" ]; then

    python profiler/memory/memory_profiler_driver.py \
        --profile-dir $profile_dir \
        --min-batch-size $local_batch_size 2>&1 | tee -i $profile_log_dir/log.txt

elif [ $lbs_search_mode == "static" ]; then

    python profiler/memory/memory_profiler_driver.py \
        --profile-dir $profile_dir \
        -lbs $local_batch_size 2>&1 | tee -i $profile_log_dir/log.txt

else
    echo "Not support such local batch size search mode (Choose among dynamic, static):" $lbs_search_mode
    exit 1
fi