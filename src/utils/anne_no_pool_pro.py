# Copyright 2021 Administrator
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def isolation_kernel_map(data, psi, t):
    distance_matrix = cdist(data, data, 'euclidean')
    embeding_metrix = []
    center_index_set = np.array([])
    one_hot = preprocessing.OneHotEncoder(sparse=False)
    psi_t = np.array(range(psi)).reshape(psi, 1)
    one_hot_encoder = one_hot.fit(psi_t)

    for i in range(t):
        center_index = np.random.choice(
            len(distance_matrix), size=psi, replace=False)
        center_index_set = np.append(center_index_set, center_index)
        nearest_center_index = np.argmin(distance_matrix[center_index], 0)
        nearest_center_index_trans = nearest_center_index.reshape(
            len(nearest_center_index), 1)
        ik_value = one_hot_encoder.transform(nearest_center_index_trans)
        if len(embeding_metrix) == 0:
            embeding_metrix = ik_value
        else:
            embeding_metrix = np.concatenate(
                (embeding_metrix, ik_value), axis=1)
        center_index_set = center_index_set.astype(int)
    #aNNEMetrix = csr_matrix(aNNEMetrix)
    return one_hot_encoder, center_index_set.reshape(t, psi).tolist(), embeding_metrix


def add_nne(ind, indData, new_point, oneHot, subIndexSet):
    """Calcute the aNNE value to a new point x.

    Args:
        met (2D array): distance matrix
        new_point (list): a new x present by vecture
        oneHot (oneHot): the used encoding rule
        subIndexSet (2D numpy array): the index of point used to build a Voronoi diagram

    Returns:
        [numpy array]: the aNNE value of x.
    """

    distance = [_fast_norm_diff(new_point, item) for item in indData]
    # distance = map(lambda y: _fast_norm_diff(x,y), indData)
    disDict = dict(zip(ind, distance))
#     
    ind = [[item.index(min(item, key=lambda i: disDict[i]))]
           for item in subIndexSet]
    #st_time = time.time()
    ik_value = oneHot.transform(ind).reshape(-1)
    #end_time = time.time()
    #print(end_time-st_time)
    return ik_value


if __name__ == '__main__':

    from src.utils.deltasep_utils import create_dataset
    dataset = create_dataset(5, 5000, num_clusters=3)
    # np.random.shuffle(dataset)
    data = np.array([pt[:3] for pt in dataset[:200]])
    x = cdist(data, data, 'euclidean')
    sts = time.time()
    oneHot, subIndexSet, aNNEMetrix = isolation_kernel_map(x, 13, 200)
    
    unique_index = list(set.union(*map(set, subIndexSet)))
    center_data = data[unique_index]
    ets = time.time()
    for dt in dataset[200:]:
        addx = dt[:3]
        test = add_nne(unique_index, center_data, addx,  oneHot, subIndexSet)
    add_end = time.time()
    #print("build time:%s" % (ets-sts))
    print("add time:%s" % (add_end-ets))