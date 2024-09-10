#!/bin/bash

# Reference: https://github.com/linzhenyuyuchen/Dataset-Download/blob/master/coco/coco2017.sh

if [ ! -d $JABAS_DATA_STORAGE ] || [ -z ${JABAS_DATA_STORAGE} ]; then
    echo "export JABAS_DATA_STORAGE to the existing directory"
    exit 1
fi

start=`date +%s`

data_dir=${JABAS_DATA_STORAGE}/coco2017
if [ -d $data_dir ]; then
    echo "Data dir: "$data_dir" already exists"
    exit 1
fi
mkdir -p $data_dir
cd $data_dir
mkdir -p ./images
mkdir -p ./annotations

# Download the image data.
cd ./images
echo "Downloading MSCOCO train images ..."
curl -LO http://images.cocodataset.org/zips/train2017.zip
echo "Downloading MSCOCO val images ..."
curl -LO http://images.cocodataset.org/zips/val2017.zip

cd ../

# Download the annotation data.
cd ./annotations
echo "Downloading MSCOCO train/val annotations ..."
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Finished downloading. Now extracting ..."

# Unzip data
echo "Extracting train images ..."
unzip -q ../images/train2017.zip -d ../images
echo "Extracting val images ..."
unzip -q ../images/val2017.zip -d ../images
echo "Extracting annotations ..."
unzip -q ./annotations_trainval2017.zip

#echo "Removing zip files ..."
#rm ../images/train2014.zip
#rm ../images/val2014.zip
#rm ./annotations_trainval2014.zip

echo "Creating trainval35k dataset..."

# Download annotations json
echo "Downloading trainval35k annotations from S3"
curl -LO https://s3.amazonaws.com/amdegroot-datasets/instances_trainval35k.json.zip

# combine train and val
echo "Combining train and val images"
mkdir ../images/trainval35k
cd ../images/train2017
find -maxdepth 1 -name '*.jpg' -exec cp -t ../trainval35k {} + # dir too large for cp
cd ../val2017
find -maxdepth 1 -name '*.jpg' -exec cp -t ../trainval35k {} +


end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"