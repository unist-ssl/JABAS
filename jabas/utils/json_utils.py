import json

from iidp.utils.json_utils import read_json

def write_json(json_file_path, data):
    try:
        with open(json_file_path, 'w') as jf:
            json.dump(data, jf, ensure_ascii=False, indent=2)
    except IOError as e:
        print("[json_utils][write_json] I/O error({0}): {1} - file path: {2} data: {3}".format(e.errno, e.strerror, json_file_path, data))
        exit(1)