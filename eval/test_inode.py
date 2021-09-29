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
import time

from src.models.INode import INode
from src.utils.anne_no_pool import add_nne, isolation_kernel_map
from src.utils.dendrogram_purity import expected_dendrogram_purity
from src.utils.file_utils import load_data, mkdir_p_safe, remove_dirs
from src.utils.Graphviz import Graphviz
from src.utils.serialize_trees import serliaze_tree_to_file


def create_streKhc_tree(data_path, n, psi, t, rate):
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
    root = INode()
    run_time = []
    tree_st_time = time.time()

    i = 0
    train_dataset = []
    for pt in load_data(data_path):
        if i <= n:
            train_dataset.append(pt)
            if i == n:
                train_dataset_feature = [pt[0] for pt in train_dataset]
                one_hot_encoder, center_index_set, anne_metrix = isolation_kernel_map(
                    train_dataset_feature, psi, t)
                unique_index = list(set.union(*map(set, center_index_set)))
                center_data = train_dataset[unique_index]
                j = 0
                for train_pt in train_dataset:
                    train_pt.append(anne_metrix[j])
                    root = root.insert(train_pt, rate=rate)
                    j += 1
        else:
            ikv = add_nne(unique_index, center_data, pt[0],
                          one_hot_encoder, center_index_set)
            pt.insert(ikv)
            root = root.insert(pt, rate=rate)
        i += 1
        if (i % 5000 == 0):
            tree_mi_time = time.time()
            run_time.append((i, tree_mi_time - tree_st_time))
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


def grid_search_inode(data_path, psi, t, n, rates, file_name, exp_dir_base):
    ti = 0
    purity = 0
    max_ps = psi[0]
    max_rt = rates[0]

    for ps in psi:
        for rt in rates:
            print(ps, rt)
            sts = time.time()
        #     try:
            root, runTime = create_streKhc_tree(
                data_path, n, ps, t, rate=rt)
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
            save_tree_filename = "_".join(
                [file_name, str(ps), str(rt), "tree.csv"])
            serliaze_tree_to_file(root, os.join(
                exp_dir_base, save_tree_filename))
            # Graphviz.write_tree("tree.dot",root)
            # print(root.point_counter)
            # print(runTime)

            pu_sts = time.time()
            dendrogram_purity = expected_dendrogram_purity(root)
            dendrogram_purity = 0
            pu_ets = time.time()
            print("purity time: %s" % (pu_ets - pu_sts))
            #dendrogram_purity = 0
            print("dendrogram_purity: %s" % (dendrogram_purity))
            args = {
                'dataset': '_'.join([file_name, "shuffle",]),
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
    tim = ti/(len(psi)*len(rates))
    args = {
        'dataset': '_'.join([file_name, "shuffle"]),
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
    file_name = "wine.csv"
    input_data_dir = 'data/raw'
    exp_dir_base_inode = 'exp_out/purity_test/streaKHC'
    start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_dir_base_inode = os.path.join(exp_dir_base_inode, start_time)

    file_path = os.path.join(input_data_dir, file_name)
    #n = int(len(dataset)/4)
    n = 68

    grid_search_inode(data_path=file_path, n=n, t=t, psi=psi, rates=rates,
                      file_name=file_name[:-3], exp_dir_base=exp_dir_base_inode)
