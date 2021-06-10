'''
Author: your name
Date: 2021-01-16 16:10:48
LastEditTime: 2021-06-10 13:59:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \StreamHC\Code\test\runTest.py
'''
from copy import deepcopy
import time

from Code.test.testINode import grid_search_inode
from Code.test.testPNode import grid_research_pnode
import numpy as np
import pandas as pd
from Code.utils.file_utils import load_data, mkdir_p_safe, remove_dirs
import os

# Inode parameter
# psi = [64, 100, 256, 500, 1000, 2000]
psi = [4] #[5, 7, 15, 17]
rates =[0.6]  #[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t = 300

remove = False
inputdir = "Code/data/originalData"
exp_dir_base = 'Code/testResult/All/'
dati = time.strftime("%Y%m%d%H%M%S", time.localtime())
exp_dir_base = exp_dir_base+dati
shuffle_times = 1

for parents, dirnames, filenames in os.walk(inputdir):
    print(filenames)
    for filename in filenames:
        data = list(load_data(inputdir+"/"+filename))
        # for slice in list(range(len(data)))[10000::20000]:
        #     data = data[:slice]
        n = int(1/4*len(data))
        exp_dir_base = exp_dir_base + "_"
        # n = len(data)
        n = 5000
        print(n)
        f = filename[:-4]
        print(f)
        if remove:
            remove_dirs(file_name=f, exp_dir_base=exp_dir_base)
        for i in range(shuffle_times):
            np.random.shuffle(data)
            P_data = deepcopy(data)
            grid_search_inode(dataset=data, n=n, t=t, psi=psi, rates=rates,
                            file_name=f, exp_dir_base=exp_dir_base, shuffle_index=i)
            # grid_research_pnode(dataset=P_data, file_name=f,
            #                    exp_dir_base=exp_dir_base, shuffle_index=i, use_ik=False)
