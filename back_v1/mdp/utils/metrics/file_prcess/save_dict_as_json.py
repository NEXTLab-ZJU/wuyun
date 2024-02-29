from collections import defaultdict, OrderedDict
import json


def save_dict_as_json(dict_data,  dst_path):
    json_str = json.dumps(dict_data)
    with open(dst_path, 'w') as json_file:
        json_file.write(json_str)

def print_dict(dict):
    for key, value in dict.items():
        print(key, ":", value)

