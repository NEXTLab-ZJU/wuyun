import argparse
import yaml

def get_args():
    yaml_path = './configs/config.yaml'
    print(f"WuYun: load the version of config file = {yaml_path}")
    with open(yaml_path,'r') as file:
        opt = argparse.Namespace(**yaml.load(file.read(), Loader=yaml.FullLoader))
    return opt