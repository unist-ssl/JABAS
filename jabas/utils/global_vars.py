from iidp.utils.global_vars import *


REQUIRED_CONFIG_JSON_KEYS = [
    'memory_profile_dir',
    'comp_profile_dir',
    'comm_profile_dir',
    'bucket_profile_dir',
    'gpu_cluster_info',
    'available_servers',
    'batch_size_lower_bound',
    'batch_size_upper_bound',
    'similarity_target'
]

DEFAULT_RESOURCE_ALLOC_UNIT = '4gpu'