import argparse
import os
import subprocess
import socket

from jabas.elastic.worker import Worker
from jabas.utils.json_utils import read_json


def check_nfs(dir_path):
    proc = subprocess.Popen(f'stat --file-system --format=%T {dir_path}', stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    filesystem_type = out.decode('utf-8').replace('\n','')
    if filesystem_type != 'nfs':
        raise ValueError(
            f'[jabas/elastic/launch.py] checkpoint path: {dir_path} must be stored NFS, but : {filesystem_type}')


def check_cluster_topology(config_params):
    available_server_list = config_params['available_servers']
    for rank, server_name in enumerate(available_server_list):
        if socket.gethostname() == server_name:
            if opt_dict['rank'] != rank:
                raise AssertionError(
                    "--rank must be equal to the order of available servers, but "
                    f"--rank: {opt_dict['rank']} of server: {socket.gethostname()}"
                    f" != rank: {rank} in available_servers: {available_server_list}")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Elastic IIDP Launcher')
    parser.add_argument('-r', '--rank', type=int, required=True,
                        help='Unique Rank among Hetero Clusters')
    parser.add_argument('-i', '--ip_addr', type=str, required=True,
                        help='IP address for scheduler server')
    parser.add_argument('--elastic_checkpoint_dir', type=str, required=True,
                        help='Directory where checkpoints is stored')
    parser.add_argument('--jabas-config-file', type=str, required=True,
                        help='Adaptive training configuration file path (json)')
    parser.add_argument('--cmd', type=str, required=True,
                        help='Job cmd to run')
    parser.add_argument('--accum-step', type=int, required=True,
                        help='Gradient accumulation step')
    parser.add_argument('--num-models', type=int, required=True,
                        help='Number of VSWs')
    parser.add_argument('--local-batch-size', type=int, required=True,
                        help='Local batch size')
    parser.add_argument('--dist-url', type=str, required=True,
                        help='URL used to set up distributed training')

    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory where log is stored')
    parser.add_argument('-s', '--sched_port', type=int, default=50060,
                        help='Port number for scheduler server')
    parser.add_argument('-w', '--worker_port', type=int, default=50061,
                        help='Port number for worker server')

    args = parser.parse_args()
    opt_dict = vars(args)

    # [Assumption] Checkpoint and log path must be stored on NFS
    ckpt_dir = opt_dict['elastic_checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    #check_nfs(ckpt_dir)

    log_dir = opt_dict['log_dir']
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        #check_nfs(log_dir)

    config_params = read_json(opt_dict['jabas_config_file'])
    gpu_cluster_info_file_path = config_params['gpu_cluster_info']

    check_cluster_topology(config_params)

    worker = Worker(opt_dict['rank'],
                    opt_dict['ip_addr'],
                    opt_dict['sched_port'],
                    opt_dict['worker_port'],
                    opt_dict['cmd'],
                    opt_dict['dist_url'],
                    (opt_dict['num_models'], opt_dict['accum_step']),
                    opt_dict['local_batch_size'],
                    ckpt_dir,
                    gpu_cluster_info_file_path,
                    log_dir)
    worker.join()
