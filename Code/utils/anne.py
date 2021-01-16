# coding: utf-8

from functools import partial
from multiprocessing import Pool
import multiprocessing

import numpy as np
import pandas as pd
import pathos.pools as pp
from numba import jit
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from itertools import chain 

from sklearn.neighbors import BallTree
import numpy as np
import time

def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    # st = time.time()
    tree = BallTree(candidates, leaf_size=15, metric='euclidean')
    # et = time.time()
    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)
    #print("search time: %s"%(et-st))

    distance_dict = dict(zip(indices[0], distances[0])) 
    # Transpose to get distances and indices into arrays
    #distances = distances.transpose()
    #indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    #closest = indices
    #closest_dist = distances

    # Return indices and distances
    return distance_dict

#@jit(nopython=True)
def _fast_norm_diff(x, y):
    """Compute the norm of x - y using numba.

    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).

    Returns:
    The 2-norm of x - y.

    """
    # s = 0.0
    d = x-y
    # #s = sum([i**2 for i in d ])
    # for i in range(len(d)):
    #      s += d[i] ** 2
    # return s

    return sum([i**2 for i in d ])


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
                        n = n,
                        psi = psi,
                        m_distance = m_distance,
                        oneHot = oneHot)
    with Pool(processes=2) as pool:
        embeding_set = np.array(pool.map(processor, range(t)))

    subIndexSet  = np.concatenate(embeding_set[:,0], axis=0)
    aNNEMetrix = np.concatenate(embeding_set[:,1],axis=1)
    
    return oneHot,subIndexSet,aNNEMetrix


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
  #d = len(dataset[0][0])
  met = [pt[0] for pt in dataset[:n]]
  #met = np.random.random_sample(size =(n,d))
  x = cdist(met,met, 'euclidean') 
  oneHot,subIndexSet,aNNEMetrix = aNNE_similarity(x,psi,t)
  for i, pt in enumerate(dataset[:n]):
      pt.append(aNNEMetrix[i])
      
  return oneHot,subIndexSet,dataset


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
    #st = time.time()
    ind = list(set(chain(*subIndexSet))) 
    #ind = list(set(subIndexSet.reshape(-1).astype(int)))
    met = np.array(met)
    indData = met[ind]
    

    # p = pp.ProcessPool(8)
    # print(len(indData))
    #with Pool() as pool:
    #st = time.time()
    #d = np.sqrt(p.map(_fast_norm_diff, [x]*len(indData), indData))
    #print(d)
    #et = time.time()
    #st = time.time()
    distance_dict = get_nearest([x],indData,k_neighbors=len(indData))
    distDict = {}
    for i in range(len(ind)):
        distDict[ind[i]] = distance_dict[i]
    #disDict = dict(zip(ind,d))
    #print(distDict)
    #print(disDict)
    ind = [[list(item).index(min(item, key=lambda x: distDict[x]))] 
            for item in subIndexSet]
    #print(ind)
    embSet = oneHot.transform(ind).reshape(-1)
    #et = time.time()
    #print("maptime:%s"%(et-st))

    return embSet


if __name__ == '__main__':
    import time

    from Code.utils.deltasep_utils import create_dataset
    
    dataset = create_dataset(5, 500, num_clusters=3)
    #np.random.shuffle(dataset)
    met = [pt[:3] for pt in dataset[:400]]
    addx = dataset[401][:3]
    
    x = cdist(met,met, 'euclidean') 
    
    sts = time.time()
    oneHot,subIndexSet,aNNEMetrix = aNNE_similarity(x,5,10)
    ets = time.time()
    test = addNNE(met,addx,oneHot,subIndexSet)
    add_end = time.time()

    print("build time:%s"%(ets-sts))
    print("add time:%s" %(add_end-ets))
