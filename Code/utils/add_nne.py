
# coding: utf-8

import pandas as pd
import numpy as np
from numba import jit
import math
from sklearn import preprocessing
from scipy.spatial.distance import cdist
#from scipy.sparse import csr_matrix
import pathos.pools as pp

from multiprocessing import Pool
from functools import partial


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


#@jit(nopython=False)
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
    #s = sum([i**2 for i in d ])
    for i in range(len(d)):
         s += d[i] ** 2
    return s


def aNNE_similarity(m_distance, psi, t):
    """[summary]

    Args:
        m_distance ([type]): [description]
        psi ([type]): [description]
        t ([type]): [description]
    """
  
    n = np.array(range(len(m_distance)))
    one_hot = preprocessing.OneHotEncoder(sparse=False)
    
    psi_t = np.array(range(psi)).reshape(psi,1)
    oneHot = one_hot.fit(psi_t)
    
    processor = partial(embeding, 
                        n=n,
                        psi = psi,
                        m_distance = m_distance,
                        oneHot = oneHot)
    with Pool() as pool:
        embeding_set = np.array(pool.map(processor, range(t)))
    
#    embeding_set = np.array([embeding(n, psi, m_distance, oneHot) 
#                    for i in range(t)])
  
    subIndexSet  = np.concatenate(embeding_set[:,0], axis=0)
    aNNEMetrix = np.concatenate(embeding_set[:,1],axis=1)
    
    return oneHot,subIndexSet,aNNEMetrix


def embeding(_, n, psi, m_distance, oneHot):
    
    subIndex = np.random.choice(n, size=psi,replace=False)
    centerIdx = np.argmin(m_distance[subIndex],0)
    centerIdx_t = centerIdx.reshape(len(centerIdx),1)
    embedIdex = oneHot.transform(centerIdx_t)
    return ([subIndex], embedIdex)


#def aNNE_similarity(m_distance, psi, t):
#   
#    aNNEMetrix = []
#    subIndexSet = np.array([])
#   
#    n = np.array(range(len(m_distance)))
#    
#    one_hot = preprocessing.OneHotEncoder(sparse=False)
#   
#    psi_t = np.array(range(psi)).reshape(psi,1)
#    oneHot = one_hot.fit(psi_t)
#   
#
#
# #    embeding_set = [embeding(n, psi, m_distance, oneHot) for i in range(t)]
# #
# #    subIndexSet, aNNEMetrix = embeding_set[:][0], embeding_set[:][1]
#
#    for i in range(t):
#         subIndex = np.random.choice(n, size=psi,replace=False)
#         subIndexSet = np.append(subIndexSet,subIndex)
#         centerIdx = np.argmin(m_distance[subIndex],0)
#         centerIdxT = centerIdx.reshape(len(centerIdx),1)
#         embedIdex = oneHot.transform(centerIdxT)
#         if len(aNNEMetrix) == 0:
#             aNNEMetrix = embedIdex
#         else:
#             aNNEMetrix = np.concatenate((aNNEMetrix,embedIdex),axis=1)
#
#    return oneHot,subIndexSet.reshape(t,psi),aNNEMetrix

def addNNE(met,x,oneHot,subIndexSet):
    """Calcute the aNNE value to a new point x.

    Args:
        met (2D list): distance matrix
        x (list): a new x present by vecture
        oneHot (oneHot): the used encoding rule
        subIndexSet (2D numpy array): the index of point used to build a Voronoi diagram

    Returns:
        [numpy array]: the aNNE value of x. 
    """
    
    ind = list(set(subIndexSet.reshape(-1).astype(int)))
    met = np.array(met)
    indData = met[ind]
    
    p = pp.ProcessPool(4)
    d = np.sqrt(p.map(_fast_norm_diff, [x]*len(indData), indData))
    disDict = dict(zip(ind,d))
    
    ind = [[list(item).index(min(item, key=lambda x: disDict[x]))] 
            for item in subIndexSet]
    embSet = oneHot.transform(ind).reshape(-1)
    return embSet



if __name__ == '__main__':
  
    from deltasep_utils import create_dataset
    import time

    dataset = create_dataset(3, 100, num_clusters=3)
    np.random.shuffle(dataset)
    met = [pt[0] for pt in dataset][:-1]
    addx = [pt[0] for pt in dataset][-1]
    
    x = cdist(met,met, 'euclidean') 
   
    sts = time.time()
    oneHot,subIndexSet,aNNEMetrix = aNNE_similarity(x,5,100)
    ets = time.time()
    test = addNNE(met,addx,oneHot,subIndexSet)
    print(ets-sts)
    print(len(test))
