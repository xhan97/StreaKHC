'''
@Author: Xin Han
@Date: 2020-06-07 11:24:57
LastEditTime: 2021-06-09 17:08:42
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@file_path: \StreamHC\Code\testPNode.py
'''

import os
import time
from copy import deepcopy

import numpy as np
from src.models.PNode import PNode
from src.utils.dendrogram_purity import  expected_dendrogram_purity,dendrogram_purity
from src.utils.file_utils import load_data, remove_dirs, mkdir_p_safe
from src.utils.Graphviz import Graphviz
from src.models.streKhc.IKMapper import IKMapper
from src.utils.serialize_trees import serliaze_tree_to_file, serliaze_collapsed_tree_to_file_with_point_ids

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
    L = 5000
    collapsibles = [] if L < float("Inf") else None
    for i, pt in enumerate(dataset):
        root = root.insert(pt, collapsibles=collapsibles, L=L)
        if (i % 10000 == 0): 
            tree_mi_time = time.time()
            run_time.append((i, tree_mi_time - tree_st_time))
            print(run_time)
    return root,run_time


def save_data_pnode(args, exp_dir_base, file_name):
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
        psi = [3, 5, 7, 13, 15]
        data = deepcopy(dataset)
        met = np.array([pt[0] for pt in data])
        for pi in psi:
            ik_mapper = IKMapper(t=200, psi=pi)
            ik_mapper = ik_mapper.fit(met)
            #oneHot, subIndexSet, aNNEMetrix = isolation_kernel_map(x, pi, 300)
            for i, pt in enumerate(data):
                pt[0] = ik_mapper.embeding_mat[i]
            sts = time.time()
            root,run_time = create_p_tree(data)
            #Graphviz.write_tree("ptree.dot",root)
            print(run_time)
            ets = time.time()
            save_tree_filename = "_".join(
                [file_name,"shuffle_index", str(shuffle_index),str(pi), "tree.txt"])
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
            save_data_pnode(args,exp_dir_base,file_name)
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
        # purity = expected_dendrogram_purity(root)
        purity = 0
        sts = time.time()
        #dendrogram_purity = 0
        ti += ets-sts

    args = {'dataset': file_name+"_"+"shuffle"+"_"+str(shuffle_index),
            'algorithm': "PERCH",
            'purity': purity,
            'clustering_time_elapsed': ti,
            }
    # save_data_node(args, exp_dir_base, file_name)
    save_all_data(args,exp_dir_base,file_name)


if __name__ == "__main__":

    remove = False
    file_name = "covtype"
    exp_dir_base = './exp_out/purity_test/Pnode/'
    start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_dir_base = os.path.join(exp_dir_base,"use_ik" ,start_time)
    mkdir_p_safe(exp_dir_base)
    dataset = list(load_data("./data/raw/"+file_name+".tsv"))
    if remove:
        remove_dirs(file_name=file_name, exp_dir_base=exp_dir_base)
    for i in range(5):
        np.random.shuffle(dataset)
        grid_research_pnode(dataset=dataset, file_name=file_name,
                            exp_dir_base=exp_dir_base, shuffle_index=i, use_ik=True)
