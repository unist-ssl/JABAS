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

data_dir=$JABAS_DATA_STORAGE/pytorch_wmt16_en_de
if [ ! -d $data_dir ]; then
  echo "No such data dir:"$data_dir
  exit 1
fi

num_layers=4
echo "Number of layers: "$num_layers
epochs=20

eval_dir=$JABAS_REMOTE_STORAGE/gnmt_elastic_eval_dir
CMD="python train.py \
  --dist-backend 'nccl' \
  --dataset-dir $data_dir \
  --math fp32 \
  --seed 2 \
  --num-layers $num_layers \
  --weight-sync-method $weight_sync_method \
  --target-bleu 24.61 \
  --epochs $epochs \
  --eval-dir $eval_dir \
  --jabas-config-file $jabas_config_file"
echo "[script] static execution command: "$CMD

# To avoid the error: "Too many open files"
ulimit -n 65535

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
log_dir=$JABAS_REMOTE_STORAGE/gnmt_jabas_convergence_log
if [ -d $log_dir ] && [ $rank == "0" ] ; then
  check_dir_and_copy_backup gnmt_jabas_convergence_log
fi
elastic_ckpt_dir=$JABAS_REMOTE_STORAGE/gnmt_elastic_convergence_ckpt
if [ -d $elastic_ckpt_dir ] && [ $rank == "0" ] ; then
  check_dir_and_copy_backup gnmt_elastic_convergence_ckpt
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
  --elastic_checkpoint_dir $elastic_ckpt_dir \
  --log_dir $log_dir
