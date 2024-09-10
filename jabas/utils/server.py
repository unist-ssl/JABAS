from iidp.utils.server import *


def get_resource_info_dict(global_server_info):
    resource_info_dict = {}
    resource_info_dict['total_num_gpus'] = global_server_info.total_num_gpus
    for server_info in global_server_info:
        if server_info.resource_info.device_name in resource_info_dict.keys():
            resource_info_dict[server_info.resource_info.device_name] += server_info.resource_info.num_gpus_in_server
        else:
            resource_info_dict[server_info.resource_info.device_name] = server_info.resource_info.num_gpus_in_server

    return resource_info_dict