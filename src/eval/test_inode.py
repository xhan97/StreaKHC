'''
@Author: Xin Han
@Date: 2020-06-07 11:24:57
LastEditTime: 2021-06-10 14:15:09
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@file_path: \StreamHC\Code\testINode.py
'''
# coding: utf-8

import os
from posixpath import join
import time
from copy import deepcopy
import numpy as np
import pandas as pd

from src.models.INode import INode
from src.utils.anne_no_pool import add_nne, isolation_kernel_map
from src.utils.dendrogram_purity import expected_dendrogram_purity
from src.utils.file_utils import load_data, mkdir_p_safe, remove_dirs
from src.utils.Graphviz import Graphviz
from src.utils.serialize_trees import serliaze_tree_to_file


def add_nne_data(dataset, n, psi, t):
    """Add ik value to dataset.
    Args:
      dataset - a list of points with which to build the tree.
      n - the number of dataset to build aNNE metrix
      psi - parameter of ik
      t - paremeter of ik
    Return:
      dataset with ik value

    """
    met = [pt[0] for pt in dataset[:n]]
    one_hot_encoder, sub_index_set, anne_metrix = isolation_kernel_map(
        met, psi, t)
    for i, pt in enumerate(dataset[:n]):
        pt.append(anne_metrix[i])
    return one_hot_encoder, sub_index_set, dataset


def create_streKhc_tree(dataset, n, psi, t, rate):
    """Create trees over the same points.
    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.

    Args:
        dataset - a list of points with which to build the tree.
        n - numuber of point to intitial ik metrix
        psi - parameter of ik
        t - parameter of ik
        w - windows size
        l - ik metrix update length

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """

    train_dataset = np.array([pt[0] for pt in dataset[:n]])
    nne_st = time.time()
    one_hot_encoder, center_index_set, data = add_nne_data(dataset, n, psi, t)
    unique_index = list(set.union(*map(set, center_index_set)))
    center_data = train_dataset[unique_index]
    nne_ed = time.time()
    print(nne_ed - nne_st)

    root = INode(exact_dist_thres=10)
    run_time = []
    tree_st_time = time.time()

    # history = []
    for i, pt in enumerate(data):
        if len(pt) == 3:
            #st = time.time()
            ikv = add_nne(unique_index, center_data, pt[0],
                          one_hot_encoder, center_index_set)
            #et = time.time()
            #print("add time:%s"%(et-st))
            pt.append(ikv)
        if (i % 5000 == 0):
            tree_mi_time = time.time()
            run_time.append((i, tree_mi_time - tree_st_time))

        root = root.insert(pt, delete_node=True,
                           L=5000, t=300, rate=rate)
    return root, run_time


def save_data(args, exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name+'.tsv')
    mkdir_p_safe(exp_dir_base)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'purity',
                'clustering_time_elapsed',
                "max_psi",
                "max_rt"))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%f\t%f\t%s\t%s\n' % (
            args['dataset'],
            args['algorithm'],
            args['purity'],
            args['clustering_time_elapsed'],
            args["max_psi"],
            args["max_rt"]))


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


def save_grid_search_data(args, exp_dir_base, filename):
    file_path = os.path.join(exp_dir_base, 'gridSearch.tsv')
    mkdir_p_safe(exp_dir_base)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\t%s\n' % (
                'algorithm',
                'dataset',
                'psi',
                'beta',
                'purity',
            ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%d\t%f\t%f\n' % (
            args['algorithm'],
            filename,
            args["psi"],
            args["beta"],
            args['purity'],
        ))


def grid_search_inode(dataset, psi, t, n, rates, file_name, exp_dir_base, shuffle_index):
    ti = 0
    purity = 0
    max_ps = psi[0]
    max_rt = rates[0]
    for ps in psi:
        for rt in rates:
            print(ps, rt)
            data = deepcopy(dataset)
            sts = time.time()
        #     try:
            root, runTime = create_streKhc_tree(
                data, n, ps, t, rate=rt)
        #     except:
        #         continue
            ets = time.time()
            print("time of build tree: %s" % (ets-sts))
            ti += ets-sts
            with open("InodeTime.tsv", "a") as f:
                for item in runTime:
                    f.write("%s\t%s\n" % (
                        item[0],
                        item[1]
                    ))
            print(runTime)
            save_tree_filename = "_".join([file_name, str(ps), str(rt), "tree.csv"])
            serliaze_tree_to_file(root, os.join(exp_dir_base, save_tree_filename))
            # Graphviz.write_tree("tree.dot",root)
            # print(root.point_counter)
            # print(runTime)

            pu_sts = time.time()
            # dendrogram_purity = expected_dendrogram_purity(root)
            dendrogram_purity = 0
            pu_ets = time.time()
            print("purity time: %s" % (pu_ets - pu_sts))
            #dendrogram_purity = 0
            print("dendrogram_purity: %s" % (dendrogram_purity))
            args = {
                'dataset': '_'.join([file_name, "shuffle", shuffle_index]),
                'algorithm': "IKSHC",
                'purity': dendrogram_purity,
                "psi": ps,
                "beta": rt}
            save_grid_search_data(
                args, exp_dir_base=exp_dir_base, filename=file_name)
            #pu_ets = time.time()
            #print("time of calcute purity: %s" % (pu_ets-pu_sts))
            if dendrogram_purity > purity:
                max_ps = ps
                max_rt = rt
                purity = dendrogram_purity
            del data
    tim = ti/(len(psi)*len(rates))
    args = {
        'dataset': '_'.join([file_name, "shuffle", shuffle_index]),
        'algorithm': "IKSHC",
        'purity': purity,
        'clustering_time_elapsed': tim,
        "max_psi": max_ps,
        "max_rt": max_rt}
    save_data(args, exp_dir_base, file_name)
    save_all_data(args=args, exp_dir_base=exp_dir_base, filename=file_name)


if __name__ == "__main__":
    psi = [5, 15, 17, 21, 25]
    rates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    t = 300
    remove = False
    file_name = "covtype"
    input_data_dir = 'data/raw'
    exp_dir_base_inode = 'exp_out/purity_test/streaKHC'
    start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_dir_base_inode = os.path.join(exp_dir_base_inode, start_time)

    shuffle_times = 10
    file_path = os.path.join(input_data_dir, file_name+".tsv")
    dataset = list(load_data(file_path))
    # n = int(len(dataset)/4)
    n = 25000

    if remove:
        remove_dirs(file_name=file_name, exp_dir_base=exp_dir_base_inode)
    grid_search_inode(dataset=dataset, n=n, t=t, psi=psi, rates=rates,
                      file_name=file_name, exp_dir_base=exp_dir_base_inode, shuffle_index=i)
