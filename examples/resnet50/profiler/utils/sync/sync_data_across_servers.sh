#!/bin/bash

data_dir=$1
result_dir=$2
master=$3
slaves=$4

if [ $# -lt 4 ] ; then
    echo "[USAGE] [local profile data dir - {profile dir}/{model}] [synchornized result dir] [main server] [slave servers]"
    exit 1
fi

if [ ! -d $data_dir ] ; then
    echo "data_dir: "$data_dir "must exist"
    exit 1
fi

local_host=`hostname`

if [ "$local_host" == "$master" ] ; then
    if [ -d $result_dir ] ; then
        echo "result_dir: "$result_dir "already exist"
        exit 1
    else
        mkdir -p $result_dir
        echo "Make result dir:" $result_dir
    fi
    cp -r $data_dir/* $result_dir
else
    scp -P 51234 -r $data_dir/* $master:$PWD/$result_dir
fi

echo "Sleep 10 sec to wait for all local data gathered .."
sleep 10
if [ "$local_host" == "$master" ] ; then
    IFS=',' read -r -a slave_array <<< "$slaves"
    type_nargs=""
    for slave in "${slave_array[@]}"
    do
        scp -P 51234 -r $result_dir $slave:$PWD
    done
fi

