'''
@Author: Xin Han
@Date: 2020-06-07 11:24:57
LastEditTime: 2020-11-10 16:56:28
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \StreamHC\Code\testINode.py
'''
# coding: utf-8

import datetime
import time

import numpy as np
import pandas as pd
#from graphviz import Source
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from Code.models.INode import INode
from Code.utils.anne import add_nne_data, addNNE, aNNE_similarity
from Code.utils.deltasep_utils import create_dataset
from Code.utils.dendrogram_purity import (dendrogram_purity,
                                          expected_dendrogram_purity)
from Code.utils.Graphviz import Graphviz

def create_trees_w_purity_check(dataset, n, psi, t, w, l, rate):
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

        root = root.insert(pt, collapsibles=None, L=float('inf'), t=300, rate= rate)
        # if i%10 == 0:
        # gv = Graphviz()
        # tree = gv.graphviz_tree(root)
        # src = Source(tree)
        # src.render('treeResult\\'+'tree'+str(i)+'.gv', view=True,format='png')
    return root

# def load_data(filename):
#     with open(filename, 'r') as f:
#         for line in f:
#             splits = line.strip().split('\t')
#             pid, l, vec = splits[0], splits[1], np.array([float(x)
#                                                           for x in splits[2:]])
#             yield ([vec, l, pid])

def save_data(data, exp_dir_base, fileName):
    import os
    with open(os.path.join(exp_dir_base, fileName+'.tsv'), 'a') as fout:
        fout.write('%s\t%s\t%f\t%f\t%s\t%s\n' % (
            data['dataset'],data['algorithm'],
            data['purity'],
            data['clustering_time_elapsed'],
            data["max_psi"],
            data["max_rt"]))
        # fout.write('%s-pick-k\t%s\t%f\n' % (
        #     args.algorithm, args.dataset,
        #     clustering_time_elapsed + pick_k_time))


def load_data(filename):
    data = pd.read_csv(filename, delimiter='\t')
    data = data.dropna(how='all')
    data = data.dropna(axis=1, how='all')
    # scaler = MinMaxScaler()
    # data.iloc[:, 2:] = scaler.fit_transform(data.iloc[:, 2:])
    for item in data.values:
        yield([item[2:], item[1], item[0]])


def load_df(df):
    for item in df.values:
        yield([item[:-2], item[-2], item[-1]])


if __name__ == "__main__":

    from copy import deepcopy
    dimensions = [10]
    size = 1000
    num_clus = 5

    dataset = list(load_data("./Code/data/glass.tsv"))
    # for dim in dimensions:
    #   print("TESTING DIMENSIONS == %d" % dim)
    #   dataset = create_dataset(dim, size, num_clusters=num_clus)

    # dataset = pd.DataFrame(dataset)
    # scaler = MinMaxScaler()
    # dataset.iloc[:,:-2] = scaler.fit_transform(dataset.iloc[:,:-2])
    # dataset = list(load_df(dataset))

    n = 50
    psi = [3,5,7,13,15,17,20]
    rates = [0.6,0.7,0.8,0.9]
    t = 200
    w = 100
    l = w * 0.5

    fileName = "glass"
    
    exp_dir_base = './Code/testResult/'
    
    for i in range(10):
        np.random.shuffle(dataset)
        ti = 0
        purity = 0
        max_ps = psi[0]
        max_rt = rates[0]
        for ps in psi:
            for rt in rates:
                data = deepcopy(dataset)
                sts = time.time()
                root = create_trees_w_purity_check(data, n, ps, t, w, l, rate = rt)
                ets = time.time()
                dendrogram_purity = expected_dendrogram_purity(root)
                ti += ets-sts
                if dendrogram_purity > purity:
                    max_ps = ps
                    max_rt = rt
                    purity =  dendrogram_purity
                del data
        tim = ti/(len(psi)*len(rates))

        args = {'dataset':fileName+"_"+"shuffle"+"_"+str(i),
        'algorithm':"IKSHC",
        'purity':purity,
        'clustering_time_elapsed':tim,
        "max_psi": max_ps,
        "max_rt":max_rt}

        save_data(args,exp_dir_base,fileName)
    
