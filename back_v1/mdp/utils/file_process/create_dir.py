import os
import subprocess


def create_dir(path):
    if os.path.exists(path):
        subprocess.check_call(f'rm -rf "{path}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(path)
        print("recreate dir success")
    else:
        os.makedirs(path)


def create_dirlist(path_list):
    for path in path_list:
        if os.path.exists(path):
            subprocess.check_call(f'rm -rf "{path}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
            os.makedirs(path)
            print("recreate dir success")
        else:
            os.makedirs(path)