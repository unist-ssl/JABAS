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

data_dir=$JABAS_DATA_STORAGE/imagenet
if [ ! -d $data_dir ]; then
  echo "No such data dir:"$data_dir
  exit 1
fi

lr=0.0005
epochs=90

CMD="python main.py \
    --model t2t_vit_14 \
    --weight-sync-method $weight_sync_method \
    --jabas-config-file $jabas_config_file \
    --lr $lr \
    --weight-decay .05 \
    --img-size 224 \
    --epochs $epochs \
    --data-dir $data_dir"

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
log_dir=$JABAS_REMOTE_STORAGE/vit_jabas_convergence_log
if [ -d $log_dir ] && [ $rank == "0" ] ; then
  check_dir_and_copy_backup vit_jabas_convergence_log
fi
elastic_ckpt_dir=$JABAS_REMOTE_STORAGE/vit_elastic_convergence_ckpt
if [ -d $elastic_ckpt_dir ] && [ $rank == "0" ] ; then
  check_dir_and_copy_backup vit_elastic_convergence_ckpt
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
