import pandas as pd
import numpy as np
import time
from Code.utils.file_utils import load_data, mkdir_p_safe, remove_dirs
from scipy.cluster.hierarchy import linkage, dendrogram

dataset = list(load_data("./Code/data/addData/split1/aloi.tsv"))
linkMethod = ['single','complete','average']
start_time = time.time()
ti = []
for m in linkMethod:
    run_time = []
    start_time = time.time()
    for index,item in enumerate(dataset):
        d = dataset[:index]
        if index >= 2:
            X = linkage(dataset[:index], method=m, metric='euclidean')
        if (index % 100 == 0):
            tree_mi_time = time.time()
            run_time.append((index, tree_mi_time - start_time))
            print(index, tree_mi_time - start_time)
            if tree_mi_time - start_time >= 20:
                break
    with open(m+".tsv","a") as f:
                for item in run_time:
                        f.write("%s\t%s\n" %(
                            item[0],
                            item[1]
                        ))
    ti.append(run_time)
    