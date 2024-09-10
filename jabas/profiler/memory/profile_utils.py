import os

from iidp.utils.json_utils import read_json
from iidp.utils.global_vars import MAX_MEM_PROFILE_FILE_NAME
from iidp.profiler.memory.profile_utils import *


def get_mem_profile_data_summary(profile_dir):
    summary_str = ''
    col_str = '---------------------------------------------------'
    for lbs in sorted(os.listdir(profile_dir), key=lambda x: int(x)):
        static_lbs_profile_dir = os.path.join(profile_dir, lbs)
        for server_name in os.listdir(static_lbs_profile_dir):
            max_memory_profile_file = os.path.join(
                static_lbs_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
            memory_profile_json_data = read_json(max_memory_profile_file)
            max_num_models = memory_profile_json_data['max_num_models']
            gpu_type = memory_profile_json_data['gpu_type']
            summary_str += f'   {lbs}\t|\t{gpu_type}\t|    {max_num_models} \n'
        summary_str += col_str+'\n'
    return summary_str