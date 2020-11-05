'''
@Author: Xin Han
@Date: 2020-06-07 11:24:57
@LastEditTime: 2020-07-03 17:33:32
@LastEditors: Please set LastEditors
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

from models.INode import INode
from utils.anne import add_nne_data, addNNE, aNNE_similarity
from utils.deltasep_utils import create_dataset
from utils.dendrogram_purity import (dendrogram_purity,
                                     expected_dendrogram_purity)
from utils.Graphviz import Graphviz


def create_trees_w_purity_check(dataset, n, psi, t, w, l):
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
    rate = 0.7
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

        root = root.insert(pt, collapsibles=None, L=float('inf'), t=300)
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

    from copy import copy, deepcopy
    dimensions = [10]
    size = 100
    num_clus = 5

    dataset = list(load_data("./Code/data/glass.tsv"))

    # for dim in dimensions:
    #   print("TESTING DIMENSIONS == %d" % dim)
    #   dataset = create_dataset(dim, size, num_clusters=num_clus)

    # dataset = pd.DataFrame(dataset)
    # scaler = MinMaxScaler()
    # dataset.iloc[:,:-2] = scaler.fit_transform(dataset.iloc[:,:-2])
    # dataset = list(load_df(dataset))

    n = 100
    psi = 3
    t = 200
    w = 100
    l = w * 0.5
    np.random.shuffle(dataset)
    data = deepcopy(dataset)
    sts = time.time()
    root = create_trees_w_purity_check(data, n, psi, t, w, l)
    ets = time.time()
    print(ets-sts)
    dendrogram_purity = expected_dendrogram_purity(root)
    print(dendrogram_purity)
