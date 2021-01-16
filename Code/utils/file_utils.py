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

import pandas as pd


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

def load_data(file_path):
    #data = pd.read_csv(file_path, delimiter='\t')
    data = pd.read_csv(file_path)
    data = data.dropna(how='all')
    data = data.dropna(axis=1, how='all')
    # scaler = MinMaxScaler()
    # data.iloc[:, 2:] = scaler.fit_transform(data.iloc[:, 2:])
    for item in data.values:
        yield([item[2:], item[1], item[0]])