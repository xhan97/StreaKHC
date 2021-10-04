'''
Author: your name
Date: 2020-11-12 11:41:39
LastEditTime: 2021-01-16 16:08:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /StreamHC/Code/utils/file_utils.py
'''
# coding: utf-8

import errno
import os

import numpy as np


def mkdir_p_safe(dir):
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def remove_dirs(exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name)
    if os.path.exists(file_path):
        os.removedirs(file_path)


# def load_data(file_path):
#     if file_path[-3:] == "tsv":
#         data = pd.read_csv(file_path, delimiter='\t')
#     elif file_path[-3:] == "csv":
#         data = pd.read_csv(file_path)
#     else:
#         return None
#     data = data.dropna(axis=1, how='all')
#     for item in data.values:
#         yield([item[2:], item[1], item[0]])

def load_data(filename):
    if filename.endswith(".csv"):
        split_sep = ","
    elif filename.endswith(".tsv"):
        split_sep = '\t'
    with open(filename, 'r') as f:
        for line in f:
            splits = line.strip().split(sep=split_sep)
            pid, l, vec = splits[0], splits[1], np.array([float(x)
                                                          for x in splits[2:]])
            yield ((vec, l, pid))