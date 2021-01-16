'''
Author: your name
Date: 2021-01-16 15:13:32
LastEditTime: 2021-01-16 15:41:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Code\data\dataProcess.py
'''
import pandas as pd
import os




inputdir = r'data'
fileList = os.listdir(inputdir)

for parents, dirnames, filenames in os.walk(inputdir):
    for filename in filenames:
        data = pd.read_csv('data\\'+filename,header=None)
        data.to_csv(filename,header=0)
