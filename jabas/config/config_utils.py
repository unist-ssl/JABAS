import torch

from iidp.utils.distributed import print_one_rank
from iidp.utils.json_utils import read_json

import os
import socket

from jabas.utils.global_vars import REQUIRED_CONFIG_JSON_KEYS, REQUIRED_CONFIG_FILES
from iidp.config.config_utils import get_possible_batch_size_across_cluster, \
                                     recommend_weight_sync_method_by_bucket_profile_data, \
                                     print_table, sorted_listdir, check_server_profile_data_exists


def sorted_config_map_by_rank(config_map):
    return {key:value for key, value in \
                sorted(config_map.items(), key=lambda item: int(item[0]))}


def check_user_config_is_valid(config_file):
    config_params = read_json(config_file)
    print_one_rank(f'[INFO] configuration parameters: {config_params}')
    # == JABAS == #
    if config_params.get('enable_adjust', None) is None:
        config_params["enable_adjust"] = True
    if isinstance(config_params["enable_adjust"], str): # handle to parse from json file
        config_params["enable_adjust"] = bool(config_params["enable_adjust"] == "True")
    if config_params["enable_adjust"] is False:
        print_one_rank('[PASS] If enable_adjust = False, no need to check configuration')
        return
    # =========== #

    # [CHECK 1] Required JSON keys
    is_config_json_has_required_keys = set(REQUIRED_CONFIG_JSON_KEYS).issubset(set(config_params.keys()))
    if is_config_json_has_required_keys is False:
        missing_keys = ','.join(
            list(filter(lambda elem: elem not in list(config_params.keys()), REQUIRED_CONFIG_JSON_KEYS))
        )
        raise ValueError(
            f'[FAIL] Configuration JSON \"{config_file}\" misses the required keys: ```{missing_keys}``` '
            f'among required keys: {REQUIRED_CONFIG_JSON_KEYS}')
    else:
        print_one_rank(f'[PASS] Configuration JSON \"{config_file}\" has all requied JSON keys: {REQUIRED_CONFIG_JSON_KEYS}')

    # [CHECK 2] Required profile dir exists
    for key in REQUIRED_CONFIG_FILES:
        if os.path.exists(config_params[key]) is False:
            raise ValueError(
                f'[FAIL] File \"{config_params[key]}\" must exist')
        else:
            print_one_rank(f'[PASS] \"{config_params[key]}\" exists')

    # [CHECK 3] Structure of Comp and mem profile dir
    comp_profile_dir = config_params['comp_profile_dir']
    memory_profile_dir = config_params['memory_profile_dir']
    if comp_profile_dir == memory_profile_dir:
        raise ValueError(
            f'[FAIL] Path of Computation and memory profile dir must be different, but same path: {comp_profile_dir}')
    # Check the range of local batch sizes
    if len(os.listdir(comp_profile_dir)) != len(os.listdir(memory_profile_dir)):
        raise ValueError(
            f'[FAIL] Computation and memory profile data for range of local batch size '
            f'must be equal, but comp: {sorted_listdir(comp_profile_dir)} | mem: {sorted_listdir(memory_profile_dir)}')

    # Check the same server profile data in computation and memory profile dir
    for lbs in sorted_listdir(memory_profile_dir):
        static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, lbs)
        static_lbs_mem_profile_dir = os.path.join(memory_profile_dir, lbs)
        if os.listdir(static_lbs_comp_profile_dir) != os.listdir(static_lbs_mem_profile_dir):
            raise ValueError(
                f'[FAIL] For static LBS of {lbs}, server profile data is not consistent among comp and memory profile dir!\n'
                f'comp profile dir - {static_lbs_comp_profile_dir} : {os.listdir(static_lbs_comp_profile_dir)}\n'
                f'memory profile dir - {static_lbs_mem_profile_dir} : {os.listdir(static_lbs_mem_profile_dir)}')

    # Check at least one profile data exists on current server. If not, the current server cannot be registered in available_servers
    if socket.gethostname() in config_params['available_servers']:
        if check_server_profile_data_exists(comp_profile_dir) is False:
            raise ValueError(f'[FAIL] No such computation profile data for {socket.gethostname()} '
                            f'in {comp_profile_dir}')
        if check_server_profile_data_exists(memory_profile_dir) is False:
            raise ValueError(f'[FAIL] No such memory profile data for {socket.gethostname()} '
                            f'in {memory_profile_dir}')
    for available_server in config_params['available_servers']:
        if check_server_profile_data_exists(comp_profile_dir, available_server) is False:
            raise ValueError(f'[FAIL] No such computation profile data for available server: ```{available_server}``` '
                            f'in {comp_profile_dir} => Registered available_servers: {config_params["available_servers"]}')
        if check_server_profile_data_exists(memory_profile_dir, available_server) is False:
            raise ValueError(f'[FAIL] No such memory profile data for available server: ```{available_server}``` '
                            f'in {memory_profile_dir} => Registered available_servers: {config_params["available_servers"]}')
    print_one_rank(f'[PASS] \"{comp_profile_dir}\" and \"{memory_profile_dir}\" has the right structure for configuration solver')

    # [CHECK 4] GPU cluster info JSON data
    gpu_cluster_info_file = config_params['gpu_cluster_info']
    gpu_cluster_info = read_json(gpu_cluster_info_file)
    if socket.gethostname() not in gpu_cluster_info.keys():
        raise ValueError(
            f'Current server: {socket.gethostname()} is not registered '
            f'in gpu cluster info json file: {gpu_cluster_info_file}'
        )
    if torch.cuda.get_device_name() != gpu_cluster_info[socket.gethostname()]['type']:
        raise ValueError(
            f'Registerd GPU type in server {socket.gethostname()} in {gpu_cluster_info_file} '
            f'```{gpu_cluster_info[socket.gethostname()]["type"]}``` is not equal to '
            f'real GPU type in server: ```{torch.cuda.get_device_name()}```'
        )
    print_one_rank(f'[PASS] \"{gpu_cluster_info_file}\" registers the right GPU hardware for configuration solver')
