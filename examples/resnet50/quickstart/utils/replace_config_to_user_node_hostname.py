import argparse
import os
import copy

from jabas.utils.json_utils import read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--node0','-n0', type=str, default=None, required=True,
                    help='Replace node0 name')
parser.add_argument('--node1','-n1', type=str, default=None, required=True,
                    help='Replace node1 name')
parser.add_argument('--initialize', action='store_true',
                    help='Rollback to an initial given example')


COMP_PROFILE_DATA = 'quickstart/cluster_comp_profile_data'
MEM_PROFILE_DATA = 'quickstart/cluster_mem_profile_data'
CONFIG_FILE = 'quickstart/config.json'
CLUSTER_INFO_FILE = 'quickstart/cluster_info.json'


def check_exists(path):
    if not os.path.exists(path):
        raise ValueError(
            f'[FAIL] Path \"{path}\" must exist')


def rename_profile_dir(path, args):
    for (root, dirs, files) in os.walk(path):
        if len(dirs) == 0: # reach to the last direcotry
            src_root, dst_root = root, ''
            if args.src_node0 in root:
                src_root = copy.deepcopy(root)
                dst_root = src_root.replace(args.src_node0, args.dst_node0)
            elif args.src_node1 in root:
                src_root = copy.deepcopy(root)
                dst_root = src_root.replace(args.src_node1, args.dst_node1)
            else:
                raise NotADirectoryError(f'Not support such directory path: {root}')
            if dst_root != '':
                print(f'[INFO] Replace directory name to {dst_root}')
                os.rename(src_root, dst_root)


def main():
    args = parser.parse_args()

    check_exists(COMP_PROFILE_DATA)
    check_exists(MEM_PROFILE_DATA)
    check_exists(CONFIG_FILE)
    check_exists(CLUSTER_INFO_FILE)

    if args.initialize:
        args.src_node0 = args.node0
        args.src_node1 = args.node1
        args.dst_node0 = 'node0'
        args.dst_node1 = 'node1'
    else:
        args.src_node0 = 'node0'
        args.src_node1 = 'node1'
        args.dst_node0 = args.node0
        args.dst_node1 = args.node1

    rename_profile_dir(COMP_PROFILE_DATA, args)

    rename_profile_dir(MEM_PROFILE_DATA, args)

    config_params = read_json(CONFIG_FILE)
    config_params['available_servers'] = [args.dst_node0,args.dst_node1]
    print(f"[INFO] Replace JABAS configuration parameter ```available_servers``` to {config_params['available_servers']}")
    write_json(CONFIG_FILE, config_params)

    cluster_info = read_json(CLUSTER_INFO_FILE)
    cluster_info[args.dst_node0] = cluster_info[args.src_node0]
    del cluster_info[args.src_node0]
    cluster_info[args.dst_node1] = cluster_info[args.src_node1]
    del cluster_info[args.src_node1]
    print(f'[INFO] Replace cluster info to {cluster_info}')
    write_json(CLUSTER_INFO_FILE, cluster_info)


if __name__ == '__main__':
    main()