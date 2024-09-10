#!/bin/bash

config_file=$1
gbs=$2
weight_sync_method=${3:-"recommend"}

if [ $# -lt 2 ]; then
  echo "[USAGE] [config file (JSON)] [global batch size] [weight sync method (default: recommend)]"
  exit 1
fi

timestamp=`date +%Y%m%d_%H%M%S`
dir_name=${timestamp}_config_solver_log
mkdir $dir_name
log_file=$dir_name/"config_log.txt"
echo "log file: "$log_file

python -m jabas.config.api.configuration_solver \
    -c $config_file \
    -gbs $gbs \
    -wsm $weight_sync_method 2>&1 | tee -i $log_file
