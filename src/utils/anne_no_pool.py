# Copyright 2021 Xin Han
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

import math
import time

import numpy as np
from numba import jit
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree


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
    distances, indices = tree.query(src_points, k=k_neighbors)
    distance_dict = dict(zip(indices[0], distances[0]))
    return distance_dict


def isolation_kernel_map(data, psi, t):
    distance_matrix = cdist(data, data, 'euclidean')
    for i in range(t):
        center_index = np.random.choice(
            len(data), size=psi, replace=False)
        if i == 0:
            center_index_set = np.array([center_index])
        else:
            center_index_set = np.append(
                center_index_set, np.array([center_index]), axis=0)
        nearest_center_index = np.argmin(distance_matrix[center_index], 0)
        ik_value = np.eye(psi, dtype=int)[nearest_center_index]
        if i == 0:
            embeding_metrix = ik_value
        else:
            embeding_metrix = np.concatenate(
                (embeding_metrix, ik_value), axis=1)
    return center_index_set, embeding_metrix



def add_nne(ind, indData, new_point, subIndexSet):
    """Calcute the aNNE value to a new point x.

    Args:
        met (2D array): distance matrix
        new_point (list): a new x present by vecture
        oneHot (oneHot): the used encoding rule
        subIndexSet (2D numpy array): the index of point used to build a Voronoi diagram

    Returns:
        [numpy array]: the aNNE value of x.
    """
    st = time.time()
    distance = [_fast_norm_diff(new_point, item) for item in indData]
    dist_diction = dict(zip(ind, distance))
    ind = [[item.index(min(item, key=lambda i: dist_diction[i]))]
           for item in subIndexSet]
    psi = len(subIndexSet[0])
    ik_value = np.eye(psi, dtype=int)[ind].reshape(-1)
    et = time.time()
    print(et - st)
    return ik_value


if __name__ == '__main__':

    from src.utils.deltasep_utils import create_dataset
    dataset = create_dataset(128, 108000, num_clusters=1000)
    # np.random.shuffle(dataset)
    data = np.array([pt[:3] for pt in dataset[:20000]])
    x = cdist(data, data, 'euclidean')
    sts = time.time()
    center_index_set, embeding_matrix = isolation_kernel_map(x, 25, 200)
    unique_index = np.unique(center_index_set).tolist()
    center_index_set_list = center_index_set.tolist()
    index_data = data[unique_index]
    ets = time.time()
    for dt in dataset[20000:]:
        addx = dt[:3]
        test= add_nne(unique_index, index_data, addx, center_index_set_list)
    add_end = time.time()
    print("add time:%s" % (add_end-ets))
