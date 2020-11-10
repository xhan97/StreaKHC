'''
Author: your name
Date: 2020-11-11 01:41:31
LastEditTime: 2020-11-11 01:53:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \StreamHC\Code\utils\file_utils.py
'''
import errno
import os

import pandas as pd


def mkdir_p_safe(dir):
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def load_data(file_name):
    data = pd.read_csv(file_name, delimiter='\t')
    data = data.dropna(how='all')
    data = data.dropna(axis=1, how='all')
    # scaler = MinMaxScaler()
    # data.iloc[:, 2:] = scaler.fit_transform(data.iloc[:, 2:])
    for item in data.values:
        yield([item[2:], item[1], item[0]])


def remove_file(exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name+'.tsv')
    if os.path.exists(file_path):
        os.remove(file_path)
