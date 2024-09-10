import argparse
import os

from iidp.utils.json_utils import read_json
from iidp.profiler import MAX_MEM_PROFILE_FILE_NAME


parser = argparse.ArgumentParser(description='Memory Profile Data Summary API')
parser.add_argument('--profile-dir', '-d', type=str, default=None,
                    help='Directory of profile data file.')
parser.add_argument('--config-file', '-c', type=str, default=None,
                    help='Configuration file path (json) - Deprecated')


def main():
    args = parser.parse_args()

    if args.config_file is not None and args.profile_dir is not None:
        raise ValueError(f'Not support both setup of --config-file (-c) and --profile-dir (-d)')
    if args.config_file is None and args.profile_dir is None:
        raise ValueError(f'One of --config-file (-c) and --profile-dir (-d) must be setup')

    summary_str = ''
    col_str = '---------------------------------------------------'

    if args.config_file is not None:
        print(f'[WARNING][Memory profile data summary] Argument --config-file is deprecated.')
        config_params = read_json(args.config_file)
        gpu_cluster_info = read_json(config_params["gpu_cluster_info"])
        memory_profile_dir = config_params['memory_profile_dir']

        for lbs in sorted(os.listdir(memory_profile_dir), key=lambda x: int(x)):
            static_lbs_profile_dir = os.path.join(memory_profile_dir, lbs)
            for server_name in os.listdir(static_lbs_profile_dir):
                gpu_type = gpu_cluster_info[server_name]['type']
                max_memory_profile_file = os.path.join(
                    static_lbs_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
                memory_profile_json_data = read_json(max_memory_profile_file)
                max_num_models = memory_profile_json_data['max_num_models']
                summary_str += f'   {lbs}\t|\t{gpu_type}\t|    {max_num_models} \n'
            summary_str += col_str+'\n'
    else:
        memory_profile_dir = args.profile_dir
        for lbs in sorted(os.listdir(memory_profile_dir), key=lambda x: int(x)):
            static_lbs_profile_dir = os.path.join(memory_profile_dir, lbs)
            for server_name in os.listdir(static_lbs_profile_dir):
                max_memory_profile_file = os.path.join(
                    static_lbs_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
                memory_profile_json_data = read_json(max_memory_profile_file)
                max_num_models = memory_profile_json_data['max_num_models']
                gpu_type = memory_profile_json_data['gpu_type']
                summary_str += f'   {lbs}\t|\t{gpu_type}\t|    {max_num_models} \n'
            summary_str += col_str+'\n'

    print('======== Memory Profile Data Summary ========')
    print('   LBS\t|\tGPU\t|\tMax number of VSWs')
    print(col_str)
    print(summary_str)


if __name__ == '__main__':
    main()