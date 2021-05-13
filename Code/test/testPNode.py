'''
@Author: Xin Han
@Date: 2020-06-07 11:24:57
LastEditTime: 2021-05-13 15:55:56
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
from Code.utils.dendrogram_purity import dendrogram_purity, expected_dendrogram_purity
from Code.utils.file_utils import load_data, remove_dirs, mkdir_p_safe
from Code.utils.Graphviz import Graphviz
from Code.utils.anne_no_pool import aNNE_similarity
from scipy.spatial.distance import cdist

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
        root = root.insert(pt, collapsibles=None, L=5000)
        if (i % 5000 == 0): 
            tree_mi_time = time.time()
            run_time.append((i, tree_mi_time - tree_st_time))
    return root,run_time


def save_data_Pnode(args, exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name+'.tsv')
    mkdir_p_safe(exp_dir_base)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'purity',
                'psi',
                'clustering_time_elapsed',
            ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%f\t%f\t%f\n' % (
            args['dataset'],
            args['algorithm'],
            args['purity'],
            args['psi'],
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


def grid_research_pnode(dataset, file_name, exp_dir_base, shuffle_index,use_ik=False):
    ti = 0
    purity = 0
    if (use_ik):
        psi = [3, 5, 7, 13, 15, 17, 21, 25]
        data = deepcopy(dataset)
        met = [pt[0] for pt in data]
        x = cdist(met, met, 'euclidean')
        for pi in psi:
            oneHot, subIndexSet, aNNEMetrix = aNNE_similarity(x, pi, 300)
            for i, pt in enumerate(data):
                pt[0] = aNNEMetrix[i]
            sts = time.time()
            root,run_time = create_p_tree(data)
            #Graphviz.write_tree("ptree.dot",root)
            #print(run_time)
            ets = time.time()
            dendrogram_purity = expected_dendrogram_purity(root)
            #dendrogram_purity = 0
            ti += ets-sts
            if dendrogram_purity > purity:
                purity = dendrogram_purity
            args = {'dataset': file_name+"_"+"shuffle"+"_"+str(shuffle_index),
                'algorithm': "PERCH",
                'purity': dendrogram_purity,
                'psi':pi,
                'clustering_time_elapsed': ti,
                }
            save_data_Pnode(args,exp_dir_base,file_name)
    else:
        data = deepcopy(dataset)
        root,run_time = create_p_tree(data)
        with open("pnodeTime.tsv","a") as f:
            for item in run_time:
                    f.write("%s\t%s\n" %(
                        item[0],
                        item[1]
                    ))
        #Graphviz.write_tree("ptree.dot",root)
        #print(run_time)
        ets = time.time()
        purity = expected_dendrogram_purity(root)
        sts = time.time()
        #dendrogram_purity = 0
        ti += ets-sts

    args = {'dataset': file_name+"_"+"shuffle"+"_"+str(shuffle_index),
            'algorithm': "PERCH",
            'purity': purity,
            'clustering_time_elapsed': ti,
            }
    # save_data_Pnode(args, exp_dir_base, file_name)
    save_all_data(args,exp_dir_base,file_name)


if __name__ == "__main__":

    remove = False
    file_name = "wine"
    exp_dir_base = './Code/testResult/Pnode'
    # mkdir_p_safe(exp_dir_base)

    dataset = list(load_data("./Code/data/addData/split5/"+file_name+".csv"))

    
    if remove:
        remove_dirs(file_name=file_name, exp_dir_base=exp_dir_base)
    for i in range(1):
        np.random.shuffle(dataset)
        grid_research_pnode(dataset=dataset, file_name=file_name,
                            exp_dir_base=exp_dir_base, shuffle_index=i, use_ik=False)
