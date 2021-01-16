'''
Author: your name
Date: 2021-01-16 16:10:48
LastEditTime: 2021-01-16 16:47:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \StreamHC\Code\test\runTest.py
'''
from copy import deepcopy
import time

from Code.test.testINode import grid_search_inode
from  Code.test.testPNode import grid_research_pnode
import numpy as np
import pandas as pd
from Code.utils.file_utils import load_data, mkdir_p_safe, remove_dirs

import os
# Inode parameter
psi = [3, 5, 7, 13, 15]
rates = [0.6, 0.7, 0.8, 0.9, 1]
t = 300

remove = False
inputdir = "Code\\data\\addData"
exp_dir_base = './Code/testResult/All/'
dati = time.strftime("%Y%m%d%H%M%S", time.localtime())
exp_dir_base = exp_dir_base+dati
shuffle_times = 10


for parents, dirnames, filenames in os.walk(inputdir):
    for filename in filenames:
        data = list(load_data("Code\\data\\addData\\"+filename))
        n = int(len(data)/4)
        f = filename[:-4]
        if remove:
            remove_dirs(file_name=f, exp_dir_base=exp_dir_base)
        for i in range(shuffle_times):
            np.random.shuffle(data)
            P_data = deepcopy(data)
            grid_search_inode(dataset=data, n=n, t=t, psi=psi, rates=rates,
                                file_name=f, exp_dir_base=exp_dir_base, shuffle_index=i)
            grid_research_pnode(dataset=P_data, file_name=f,
                            exp_dir_base=exp_dir_base, shuffle_index=i)
