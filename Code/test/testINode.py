'''
@Author: Xin Han
@Date: 2020-06-07 11:24:57
LastEditTime: 2020-11-11 16:13:52
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@file_path: \StreamHC\Code\testINode.py
'''
# coding: utf-8

import os
from random import shuffle
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from Code.models.INode import INode
from Code.utils.anne import add_nne_data, addNNE
from Code.utils.deltasep_utils import create_dataset
from Code.utils.dendrogram_purity import (dendrogram_purity,
                                          expected_dendrogram_purity)
from Code.utils.Graphviz import Graphviz
from Code.utils.file_utils import load_data,remove_file
#from graphviz import Source
from sklearn.preprocessing import MinMaxScaler


def create_i_tree(dataset, n, psi, t, rate):
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

    met = [pt[0] for pt in dataset[:n]]
    oneHot, subIndexSet, data = add_nne_data(dataset, n, psi, t)
    root = INode(exact_dist_thres=10)

    history = []
    for i, pt in enumerate(data):
        if len(pt) == 3:
            ikv = addNNE(met, pt[0], oneHot, subIndexSet)
            pt.append(ikv)

        history.append(pt[0])

        # if i > w:
        #     history.pop(0)
        # if i>n and (i-n) % l == 0 :
        #     met = history
        #     x = cdist(met,met, 'euclidean')
        #     oneHot, subIndexSet, _= aNNE_similarity(x, psi, t)

        root = root.insert(pt, collapsibles=None,
                           L=float('inf'), t=300, rate=rate)
        # if i%10 == 0:
        # gv = Graphviz()
        # tree = gv.graphviz_tree(root)
        # src = Source(tree)
        # src.render('treeResult\\'+'tree'+str(i)+'.gv', view=True,format='png')
    return root

# def load_data(file_name):
#     with open(file_name, 'r') as f:
#         for line in f:
#             splits = line.strip().split('\t')
#             pid, l, vec = splits[0], splits[1], np.array([float(x)
#                                                           for x in splits[2:]])
#             yield ([vec, l, pid])

def save_data(args, exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name+'.tsv')
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
        # fout.write('%s-pick-k\t%s\t%f\n' % (
        #     args.algorithm, args.dataset,
        #     clustering_time_elapsed + pick_k_time))


def grid_search_inode(dataset, psi, t, n, rates, file_name, exp_dir_base, shuffle_index):
    ti = 0
    purity = 0
    max_ps = psi[0]
    max_rt = rates[0]
    for ps in psi:
        for rt in rates:
            data = deepcopy(dataset)
            sts = time.time()
            root = create_i_tree(
                data, n, ps, t, rate=rt)
            ets = time.time()
            print("time of build tree: %s" %(ets-sts))
            ti += ets-sts
            pu_sts = time.time()
            dendrogram_purity = expected_dendrogram_purity(root)
            pu_ets = time.time()
            print("time of calcute purity: %s" %(pu_ets-pu_sts))
            if dendrogram_purity > purity:
                max_ps = ps
                max_rt = rt
                purity = dendrogram_purity
            del data
    tim = ti/(len(psi)*len(rates))
    args = {'dataset': file_name+"_"+"shuffle"+"_"+str(shuffle_index),
            'algorithm': "IKSHC",
            'purity': purity,
            'clustering_time_elapsed': tim,
            "max_psi": max_ps,
            "max_rt": max_rt}
    save_data(args, exp_dir_base, file_name)


if __name__ == "__main__":
    # def load_df(df):
    #     for item in df.values:
    #         yield([item[:-2], item[-2], item[-1]])

    # dimensions = [10]
    # size = 1000
    # num_clus = 5
    # for dim in dimensions:
    #   print("TESTING DIMENSIONS == %d" % dim)
    #   dataset = create_dataset(dim, size, num_clusters=num_clus)

    # dataset = pd.DataFrame(dataset)
    # scaler = MinMaxScaler()
    # dataset.iloc[:,:-2] = scaler.fit_transform(dataset.iloc[:,:-2])
    # dataset = list(load_df(dataset))

    n = 500
    psi = [3,5,7,13,15]
    rates = [0.6,0.7,0.8]
    t = 200
    w = 100
    l = w * 0.5
    remove = True
    file_name = "spambase"
    exp_dir_base = './Code/testResult/Inode'

    dataset = list(load_data("./Code/data/spambase.tsv"))
    if remove:
        remove_file(file_name=file_name, exp_dir_base=exp_dir_base)
    for i in range(5):
        np.random.shuffle(dataset)
        grid_search_inode(dataset=dataset, n=n, t=t, psi=psi, rates=rates,
                            file_name=file_name, exp_dir_base=exp_dir_base, shuffle_index=i)
