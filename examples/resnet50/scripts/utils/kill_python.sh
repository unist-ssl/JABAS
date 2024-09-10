#!/bin/bash

echo "pid: "`ps | grep python | grep -v defunct | awk -F" " '{print $1}'`
kill -9 `ps | grep python | awk -F" " '{print $1}'`
