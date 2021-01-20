'''
Author: your name
Date: 2021-01-17 11:36:04
LastEditTime: 2021-01-17 12:03:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \StreamHC\Code\testResult\collectResult.py
'''
import pandas as pd
import numpy as np
import os

inputdir = r'Code\\testResult\\extraAll'
data = pd.DataFrame()

for parents, dirnames, filenames in os.walk(inputdir):
    for filename in filenames:
        if filename == "allResult.tsv":
            df = pd.read_csv(os.path.join(parents, filename),sep="\t")
            data = data.append(df, ignore_index=True)
print(data)

data.to_csv("extraalldata.csv",sep="\t",index=False)