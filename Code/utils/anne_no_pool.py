# coding: utf-8

import pandas as pd
import numpy as np
from numba import jit
import math
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
import numpy as np
import time


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


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""
    # Create tree from the candidate points
    # st = time.time()
    tree = BallTree(candidates, leaf_size=30, metric='euclidean')
    # et = time.time()
    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)
    #print("search time: %s"%(et-st))
    distance_dict = dict(zip(indices[0], distances[0]))

    return distance_dict


def add_nne_data(dataset, n, psi, t):
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
    x = cdist(met, met, 'euclidean')

    oneHot, subIndexSet, aNNEMetrix = aNNE_similarity(x, psi, t)
    for i, pt in enumerate(dataset[:n]):
        pt.append(aNNEMetrix[i])

    return oneHot, subIndexSet, dataset


def aNNE_similarity(m_distance, psi, t):

    aNNEMetrix = []
    subIndexSet = np.array([])
    #n = np.array(range(len(m_distance)))
    one_hot = preprocessing.OneHotEncoder(sparse=False)
    psi_t = np.array(range(psi)).reshape(psi, 1)
    oneHot = one_hot.fit(psi_t)

    for i in range(t):
        subIndex = np.random.choice(len(m_distance), size=psi, replace=False)
        subIndexSet = np.append(subIndexSet, subIndex)
        centerIdx = np.argmin(m_distance[subIndex], 0)
        centerIdxT = centerIdx.reshape(len(centerIdx), 1)
        embedIdex = oneHot.transform(centerIdxT)
        if len(aNNEMetrix) == 0:
            aNNEMetrix = embedIdex
        else:
            aNNEMetrix = np.concatenate((aNNEMetrix, embedIdex), axis=1)
        subIndexSet = subIndexSet.astype(int)
    #aNNEMetrix = csr_matrix(aNNEMetrix)
    return oneHot, subIndexSet.reshape(t, psi).tolist(), aNNEMetrix


def addNNE(ind, indData, x, oneHot, subIndexSet):
    """Calcute the aNNE value to a new point x.

    Args:
        met (2D array): distance matrix
        x (list): a new x present by vecture
        oneHot (oneHot): the used encoding rule
        subIndexSet (2D numpy array): the index of point used to build a Voronoi diagram

    Returns:
        [numpy array]: the aNNE value of x.
    """

    distance = [_fast_norm_diff(x, y) for y in indData]
    disDict = dict(zip(ind, distance))
    ind = [[item.index(min(item, key=lambda x: disDict[x]))]
           for item in subIndexSet]
    embSet = oneHot.transform(ind).reshape(-1)
    return embSet


if __name__ == '__main__':

    import time
    from Code.utils.deltasep_utils import create_dataset
    dataset = create_dataset(5, 500, num_clusters=3)
    # np.random.shuffle(dataset)
    met = np.array([pt[:3] for pt in dataset[:400]])
    addx = dataset[401][:3]
    x = cdist(met, met, 'euclidean')
    sts = time.time()
    oneHot, subIndexSet, aNNEMetrix = aNNE_similarity(x, 13, 200)
    ets = time.time()
    test = addNNE(met, addx, oneHot, subIndexSet)
    add_end = time.time()
    print("build time:%s" % (ets-sts))
    print("add time:%s" % (add_end-ets))
