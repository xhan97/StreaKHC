# coding: utf-8

import pandas as pd
import numpy as np
from numba import jit
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import pathos.pools as pp
from multiprocessing import Pool
from functools import partial

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
    #s = sum([i**2 for i in d ])
    for i in range(len(d)):
         s += d[i] ** 2
    return s


def embeding(_, n, psi, m_distance, oneHot):
    """ help function for aNNE_similarity
    """
    subIndex = np.random.choice(n, size=psi,replace=False)
    centerIdx = np.argmin(m_distance[subIndex],0)
    centerIdx_t = centerIdx.reshape(len(centerIdx),1)
    embedIdex = oneHot.transform(centerIdx_t)
    return ([subIndex], embedIdex)


def aNNE_similarity(m_distance, psi, t):
    """Get aNNE metrix of given distance matrix

    Args:
        m_distance (2D array): distance matrix
        psi (int): parameter of aNNE
        t (int): parameter of aNNE
    Returns:
        oneHot (oneHot): the used encoding rule
        subIndexSet (): the set of index of point used to build Voronoi diagram
        aNNEMetrix (2D array): aNNE value of given distance matrix
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

    subIndexSet  = np.concatenate(embeding_set[:,0], axis=0)
    aNNEMetrix = np.concatenate(embeding_set[:,1],axis=1)
    
    return oneHot,subIndexSet,aNNEMetrix


def addNNE(met,x,oneHot,subIndexSet):
    """Calcute the aNNE value to a new point x.

    Args:
        met (2D array): distance matrix
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


def add_nne_data(dataset,n,psi,t):
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
  
  x = cdist(met,met, 'euclidean') 
  oneHot,subIndexSet,aNNEMetrix = aNNE_similarity(x,psi,t)
  for i, pt in enumerate(dataset[:n]):
      pt.append(aNNEMetrix[i])
      
  return oneHot,subIndexSet,dataset



if __name__ == '__main__':
    import time
    from deltasep_utils import create_dataset
    
    dataset = create_dataset(3, 100, num_clusters=3)
    np.random.shuffle(dataset)
    met = [pt[0] for pt in dataset][:-1]
    addx = [pt[0] for pt in dataset][-1]
    
    x = cdist(met,met, 'euclidean') 
   
    sts = time.time()
    oneHot,subIndexSet,aNNEMetrix = aNNE_similarity(x,5,200)
    ets = time.time()
    test = addNNE(met,addx,oneHot,subIndexSet)
    print(ets-sts)
    print(len(test))
