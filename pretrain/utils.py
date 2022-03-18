"""
@Time   :   2021-01-12 15:10:43
@File   :   utils.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import json
import os
import sys


def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)


def dump_json(obj, fp):
    try:
        fp = os.path.abspath(fp)
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        with open(fp, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False,
                      indent=4, separators=(',', ':'))
        print(f'json文件保存成功，{fp}')
        return True
    except Exception as e:
        print(f'json文件{obj}保存失败, {e}')
        return False


def get_main_dir():
    # 如果是使用pyinstaller打包后的执行文件，则定位到执行文件所在目录
    if hasattr(sys, 'frozen'):
        return os.path.join(os.path.dirname(sys.executable))
    # 其他情况则定位至项目根目录
    return os.path.join(os.path.dirname(__file__), '..')


def get_abs_path(*name):
    return os.path.abspath(os.path.join(get_main_dir(), *name))


def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    return lines
