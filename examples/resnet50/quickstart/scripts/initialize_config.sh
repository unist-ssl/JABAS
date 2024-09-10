#!/bin/bash

if [[ -z $NODE0 || -z $NODE1 ]] ; then
    echo "NODE0 and NODE1 must be exported, but NODE0: "$NODE0" and NODE1: "$NODE1
    exit 1
fi

python quickstart/utils/replace_config_to_user_node_hostname.py -n0 $NODE0 -n1 $NODE1 --initialize