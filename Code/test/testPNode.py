'''
@Author: Xin Han
@Date: 2020-06-07 11:24:57
LastEditTime: 2021-01-16 16:44:04
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@file_path: \StreamHC\Code\testPNode.py
'''

import os
import time
from copy import deepcopy

import pandas as pd
import numpy as np
from Code.models.PNode import PNode
from Code.utils.deltasep_utils import create_dataset
from Code.utils.dendrogram_purity import expected_dendrogram_purity
from Code.utils.file_utils import load_data, remove_dirs, mkdir_p_safe


def create_p_tree(dataset):
    """Create trees over the same points.

    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.  After each insertion, verify that the dendrogram purity
    is still 1.0 (perfect).

    Args:
        dataset - a list of points with which to build the tree.

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """

    root = PNode(exact_dist_thres=10)

    run_time = []
    tree_st_time = time.time()
    for i, pt in enumerate(dataset):
        root = root.insert(pt, collapsibles=None, L=float('inf'))
        if (i % 300 == 0): 
            tree_mi_time = time.time()
            run_time.append((i, tree_mi_time - tree_st_time))
    return root,run_time


def save_data_Pnode(args, exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name+'.tsv')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'purity',
                'clustering_time_elapsed',
            ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%f\t%f\n' % (
            args['dataset'],
            args['algorithm'],
            args['purity'],
            args['clustering_time_elapsed']
        ))

def save_all_data(args, exp_dir_base, filename):
    file_path = os.path.join(exp_dir_base, 'allResult.tsv')
    mkdir_p_safe(exp_dir_base)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\n' % (
                'algorithm',
                'dataset',
                'purity',
                ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%f\n' % (
            args['algorithm'],
            filename,
            args['purity'],
        ))


def grid_research_pnode(dataset, file_name, exp_dir_base, shuffle_index):
    ti = 0
    purity = 0
    data = deepcopy(dataset)
    sts = time.time()
    root,run_time = create_p_tree(data)
    print(run_time)
    ets = time.time()
    dendrogram_purity = expected_dendrogram_purity(root)
    ti += ets-sts
    if dendrogram_purity > purity:
        purity = dendrogram_purity
    del data
    args = {'dataset': file_name+"_"+"shuffle"+"_"+str(shuffle_index),
            'algorithm': "PERCH",
            'purity': purity,
            'clustering_time_elapsed': ti,
            }
    save_data_Pnode(args, exp_dir_base, file_name)
    save_all_data(args,exp_dir_base,file_name)


if __name__ == "__main__":

    remove = False
    file_name = "spambase"
    exp_dir_base = './Code/testResult/Pnode'
    # mkdir_p_safe(exp_dir_base)

    dataset = list(load_data("./Code/data/originalData/"+file_name+".tsv"))
    if remove:
        remove_dirs(file_name=file_name, exp_dir_base=exp_dir_base)
    for i in range(1):
        np.random.shuffle(dataset)
        grid_research_pnode(dataset=dataset, file_name=file_name,
                            exp_dir_base=exp_dir_base, shuffle_index=i)
