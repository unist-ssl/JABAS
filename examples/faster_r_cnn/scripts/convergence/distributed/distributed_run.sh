#!/bin/bash

rank=$1
local_batch_size=$2
num_models=$3
accum_step=$4
weight_sync_method=$5
jabas_config_file=$6
master=$7
scheduler_ip=$8

if [ $# -lt 7 ]; then
  echo "[USAGE] [rank] [local batch size] [number of VSWs] [accum step] [weight sync method] [jabas config file] [master] [scheduler ip (default: master)]"
  exit 1
fi

python -m jabas.config.api.user_config_checker -c $jabas_config_file
if [ $? -eq 0 ]; then
  echo "[script] Success to check user-defined config JSON file:" $jabas_config_file
else
  echo "[script] Error with user-defined config JSON file:" $jabas_config_file
  exit 1
fi

if [ -z ${JABAS_REMOTE_STORAGE} ] || [ -z ${JABAS_DATA_STORAGE} ]; then
    echo "JABAS_REMOTE_STORAGE and JABAS_DATA_STORAGE must be required," \
         "but JABAS_REMOTE_STORAGE: "${JABAS_REMOTE_STORAGE}" and " \
         "JABAS_DATA_STORAGE: "${JABAS_DATA_STORAGE}
    exit 1
fi

dataset_dir=$JABAS_DATA_STORAGE'/coco2017'
if [ ! -d $dataset_dir ]; then
  echo "No such data dir:"$dataset_dir
  exit 1
fi

model='fasterrcnn_resnet50_fpn'

# To avoid the error: "Too many open files"
ulimit -n 65535

CMD="python train.py \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --model $model \
  --weight-sync-method $weight_sync_method \
  --jabas-config-file $jabas_config_file \
  --data-dir $dataset_dir"
echo "[script] static execution command: "$CMD

if [ -z $scheduler_ip ]; then
  scheduler_ip=$master
fi
echo "[script] scheduler_ip: "$scheduler_ip

check_dir_and_copy_backup() {
  dir_name=$1

  timestamp=`date +%Y%m%d_%H%M%S`
  mv $JABAS_REMOTE_STORAGE/$dir_name $JABAS_REMOTE_STORAGE/${timestamp}_${dir_name}
  echo "[WARNING] Directory: "$JABAS_REMOTE_STORAGE/$dir_name "exists, move it to "$JABAS_REMOTE_STORAGE/${timestamp}_${dir_name}
}
log_dir=$JABAS_REMOTE_STORAGE/faster_r_rcnn_jabas_convergence_log
if [ -d $log_dir ] && [ $rank == "0" ] ; then
  check_dir_and_copy_backup faster_r_rcnn_jabas_convergence_log
fi
elastic_ckpt_dir=$JABAS_REMOTE_STORAGE/faster_r_rcnn_elastic_convergence_ckpt
if [ -d $elastic_ckpt_dir ] && [ $rank == "0" ] ; then
  check_dir_and_copy_backup faster_r_rcnn_elastic_convergence_ckpt
fi

python -m jabas.elastic.launch \
  -r $rank -i $scheduler_ip \
  -s 40000 \
  --jabas-config-file $jabas_config_file \
  --cmd "$CMD" \
  --dist-url 'tcp://'${master}':22201' \
  --num-models $num_models \
  --accum-step $accum_step \
  --local-batch-size $local_batch_size \
  --log_dir $log_dir \
  --elastic_checkpoint_dir $elastic_ckpt_dir