'''
@Author: your name
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
from scipy.spatial.distance import cdist

from graphviz import Source
from models.INode import INode
from utils.add_nne import add_nne_data, addNNE, aNNE_similarity
from utils.deltasep_utils import create_dataset
from utils.dendrogram_purity import (dendrogram_purity,
                                     expected_dendrogram_purity)
from utils.Graphviz import Graphviz


def create_trees_w_purity_check(n,psi,t,dataset,L):
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
    
    met = [pt[0] for pt in dataset[:n]]
    
    oneHot,subIndexSet,data = add_nne_data(dataset,n,psi,t)
    root = INode(exact_dist_thres=10)
    

    history = []
    for i, pt in enumerate(data):
        if len(pt)==3:
          ikv = addNNE(met,pt[0],oneHot,subIndexSet)
          pt.append(ikv)
        history.append[pt[0]]

        if len(history) % L == 0:
            met = history
            x = cdist(met,met, 'euclidean')
            oneHot, subIndexSet, _= aNNE_similarity(x, psi, t)
            history = []

        root = root.insert(pt, collapsibles=None, L= float('inf'))
        #if i%10 == 0:
        # gv = Graphviz()
        # tree = gv.graphviz_tree(root)
        # src = Source(tree)
        # src.render('treeResult\\'+'tree'+str(i)+'.gv', view=True,format='png')
    return root



def load_data(filename):
    with open(filename, 'r') as f:
        for line in f:
            splits = line.strip().split('\t')
            pid, l, vec = splits[0], splits[1], np.array([float(x)
                                                          for x in splits[2:]])
            yield ([vec, l, pid])



if __name__ == "__main__":

    from copy import copy, deepcopy
    dimensions = [5]
    size = 200
    num_clus = 3
    
    dataset = list(load_data("./StreamHC/Code/data/spambase.tsv"))
    # for dim in dimensions:
    #   print("TESTING DIMENSIONS == %d" % dim)
    #   dataset = create_dataset(dim, size, num_clusters=num_clus)  

    n = 200
    psi = 10
    t = 300

    np.random.shuffle(dataset)
    data = deepcopy(dataset)
    sts = time.time()
    root = create_trees_w_purity_check(n,psi,t,data)
    ets = time.time() 
    dendrogram_purity = expected_dendrogram_purity(root)
    print(dendrogram_purity)
    print(ets-sts)
