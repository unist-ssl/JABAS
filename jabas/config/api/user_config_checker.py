from iidp.config.config_utils import check_user_config_is_valid

import argparse

parser = argparse.ArgumentParser(description='Configuration Checker API')
parser.add_argument('--config-file', '-c', type=str, required=True,
                    help='Configuration file path (json)')


def main():
    args = parser.parse_args()

    print(f'===================== Checking config JSON file: {args.config_file} ===============================')
    check_user_config_is_valid(args.config_file)


if __name__ == '__main__':
    main()