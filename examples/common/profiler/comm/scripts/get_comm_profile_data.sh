
rank=$1
world_size=$2
master=$3
intra=$4
inter=$5
profile_dir=$6
gpu_pause_time=${7:-300}

if [ $# -lt 6 ]; then
  echo "[USAGE] [rank] [world size] [master] [intra (bytes/sec)] [inter (bytes/sec)] [profile dir] [GPU pause time (default: 300 sec)]"
  exit 1
fi

log_file=comm_profile_log.txt
if [ $world_size == "1" ]; then
  master=`hostname`
  echo "Single-node training - master: "$master
  log_file=intra_comm_profile_log.txt
else
  if [ -z $master ]; then
    echo "[ERROR] If world_size > 1, master must be configured"
    exit 1
  fi
  log_file=inter_comm_profile_log.txt
fi

for cap in 1 5 10 15 20 25 50 100 150 ; do
    if [ $rank == "0" ]; then
        ./profiler/comm/scripts/run_comm_profiler.sh $rank $world_size $master $cap $intra $inter $profile_dir 2>&1 | tee -a -i $log_file
    else
        ./profiler/comm/scripts/run_comm_profiler.sh $rank $world_size $master $cap $intra $inter $profile_dir
    fi
    echo "Sleep "$gpu_pause_time" sec .."
    sleep $gpu_pause_time
done
