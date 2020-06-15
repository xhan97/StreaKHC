
# coding: utf-8

import pandas as pd
import numpy as np
from numba import jit
import math
from sklearn import preprocessing
from scipy.spatial.distance import cdist
#from scipy.sparse import csr_matrix
import pathos.pools as pp


#@jit(nopython=True)
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


#@jit(nopython=True)
def _fast_norm_diff(x, y):
    """Compute the norm of x - y using numba.

    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).

    Returns:
    The 2-norm of x - y.
    """
    s = 0.0
    d = x-y
    for i in range(len(d)):
        s += d[i] ** 2
    
    return s


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
        if len(aNNEMetrix) == 0:
            aNNEMetrix = embedIdex
        else:
            aNNEMetrix = np.concatenate((aNNEMetrix,embedIdex),axis=1)
            

    return oneHot,subIndexSet.reshape(t,psi),aNNEMetrix


def addNNE(met,x,oneHot,subIndexSet):
    
    ind = [int(x) for x in list(set(subIndexSet.reshape(-1)))]
    met = np.array(met)
    indData = met[ind]
    
    x = [x]*len(indData)
    p = pp.ProcessPool(4)
    d = np.sqrt(p.map(_fast_norm_diff, x, indData))
    disDict = dict(zip(ind,d))
    #disDict = {k:v for k,v in zip(ind,d)}
    
    

    # embSet = np.array([]) 
    # for item in subIndexSet:
    #     nn = min(item, key=lambda x: disDict[x])
    #     emb = oneHot.transform(list(item).index(nn))
    #     embSet = np.append(embSet, emb)
    
    ind = [[list(item).index(min(item, key=lambda x: disDict[x]))] 
            for item in subIndexSet]

    embSet = oneHot.transform(ind)
    embSet  = embSet.reshape(-1)

    return embSet



if __name__ == '__main__':
  
    from deltasep_utils import create_dataset
    import time

    dataset = create_dataset(3, 20, num_clusters=3)
    np.random.shuffle(dataset)
    met = [pt[0] for pt in dataset][:-1]
    addx = [pt[0] for pt in dataset][-1]
    
    x = cdist(met,met, 'euclidean') 
   
    oneHot,subIndexSet,aNNEMetrix = aNNE_similarity(x,3,3)
    
    sts = time.time()
    test = addNNE(met,addx,oneHot,subIndexSet)
    ets = time.time()
    print(ets-sts)
     
    
    print(test)
    
    
    
