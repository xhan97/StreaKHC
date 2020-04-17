
# coding: utf-8

import pandas as pd
import numpy as np
from numba import jit
import math
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix


@jit(nopython=True)
def _fast_norm(x):
    """Compute the number of x using numba.

    Args:
    x - a numpy vector (or list).

    Returns:
    The 2-norm of x.
    """
    s = 0.0
    for i in range(len(x)):
        s += x[i] ** 2
    return math.sqrt(s)


@jit(nopython=True)
def _fast_norm_diff(x, y):
    """Compute the norm of x - y using numba.

    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).

    Returns:
    The 2-norm of x - y.
    """
    return _fast_norm(x - y)


def aNNE_similarity(m_distance, psi, t):
    
    aNNEMetrix = []
    subIndexSet = np.array([])
    
    n = np.array(range(len(m_distance)))
    one_hot = preprocessing.OneHotEncoder(sparse=False)
    
    psi_t = np.array(range(psi)).reshape(psi,1)
    oneHot = one_hot.fit(psi_t)
    
    for i in range(t):
        subIndex = np.random.choice(n, size=psi,replace=False)
        subIndexSet = np.append(subIndexSet,subIndex)
        centerIdx = np.argmin(m_distance[subIndex],0)
        centerIdxT = centerIdx.reshape(len(centerIdx),1)
        embedIdex = oneHot.transform(centerIdxT)
        if len(aNNEMetrix)==0:
            aNNEMetrix = embedIdex
        else:
            aNNEMetrix = np.concatenate((aNNEMetrix,embedIdex),axis=1)
            
    #aNNEMetrix = csr_matrix(aNNEMetrix)
    return oneHot,subIndexSet.reshape(t,psi),aNNEMetrix


def addNNE(met,x,oneHot,subIndexSet):
    
    ind = list(set(subIndexSet.reshape(-1)))
    ind = [int(x) for x in ind]
    met = np.array(met)
    indData = met[ind]
    d = [_fast_norm_diff(x,y) for y in indData]
    disDict = dict(zip(ind,d))

    # embSet = np.array([]) 
    # for item in subIndexSet:
    #     nn = min(item, key=lambda x: disDict[x])
    #     emb = oneHot.transform(list(item).index(nn))
    #     embSet = np.append(embSet, emb)

    nn = [min(item, key=lambda x: disDict[x]) for item in subIndexSet]
    ind = [[list(subIndexSet[i]).index(nn[i])] for i in range(len(nn))]
    embSet = oneHot.transform(ind)
    embSet  = embSet.reshape(-1)
    #aNNEMetrix = np.concatenate((aNNEMetrix,embSet),axis=0)

    return embSet



if __name__ == '__main__':
  
    from deltasep_utils import create_dataset

    dataset = create_dataset(3, 20, num_clusters=3)
    np.random.shuffle(dataset)
    met = [pt[0] for pt in dataset][:-1]
    addx = [pt[0] for pt in dataset][-1]
    
    x = cdist(met,met, 'euclidean') 
    oneHot,subIndexSet,aNNEMetrix = aNNE_similarity(x,3,3)
    
    test = addNNE(met,addx,oneHot,subIndexSet)
    print(test)
    
    
    
