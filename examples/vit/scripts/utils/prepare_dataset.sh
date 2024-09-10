#!/bin/bash

# Reference: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

TAR_FILE_DIR=${JABAS_DATA_STORAGE}
if [ ! -d $TAR_FILE_DIR ] || [ -z ${JABAS_DATA_STORAGE} ]; then
    echo "export JABAS_DATA_STORAGE to the existing directory"
    exit 1
fi

data_dir=${JABAS_DATA_STORAGE}/imagenet
if [ -d $data_dir ]; then
    echo "Data dir: "$data_dir" already exists"
    exit 1
fi

mkdir -p ${data_dir}/train
TRAIN_DIR=${data_dir}/train

mkdir -p ${data_dir}/val
VAL_DIR=${data_dir}/val

# Train dataset
mv ILSVRC2012_img_train.tar ${TRAIN_DIR} && cd ${TRAIN_DIR}
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

# Validation dataset
cd $TAR_FILE_DIR
mv ILSVRC2012_img_val.tar ${VAL_DIR} && cd ${VAL_DIR} && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# Check total files after extract
#
#  $ find ${TRAIN_DIR}/ -name "*.JPEG" | wc -l
#  1281167
#  $ find ${VAL_DIR}/ -name "*.JPEG" | wc -l
#  50000
#