#!/bin/bash

profile_dir=${1:-"bucket_profile_data"}
visible_gpu=${2:-"0"}

CUDA_VISIBLE_DEVICES=$visible_gpu python profiler/ddp_bucket/main.py \
  --profile-dir $profile_dir