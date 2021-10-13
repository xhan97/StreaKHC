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

import math
import time

import numpy as np
from numba import jit
from numpy.core.defchararray import center
from scipy.spatial.distance import cdist


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


class IKMapper():

    def __init__(self,
                 t,
                 psi,
                 ) -> None:
        self.t = t
        self.psi = psi
        self.embeding_metrix = None
        self.center_index_set = None
        self.center_data = None
        self.unique_index = None

    def fit(self, X):
        """ Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances. 
        Returns
        -------
        self : object
        """
        # correct sample size
        n = X.shape[0]
        self.psi = min(self.psi, n)
        distance_matrix = cdist(X, X, 'euclidean')
        for i in range(self.t):
            center_index = np.random.choice(n, self.psi, replace=False)
            if i == 0:
                self.center_index_set = np.array([center_index])
            else:
                self.center_index_set = np.append(
                    self.center_index_set, np.array([center_index]), axis=0)
            nearest_center_index = np.argmin(distance_matrix[center_index], 0)
            ik_value = np.eye(self.psi, dtype=int)[nearest_center_index]
            if i == 0:
                self.embeding_metrix = ik_value
            else:
                self.embeding_metrix = np.concatenate(
                    (self.embeding_metrix, ik_value), axis=1)
        assert self.embeding_metrix.shape == (
            n, self.psi * self.t), "Shape of ik_value is incorrect ! Except: %s, Get:%s" % ((n, self.psi * self.t), self.embeding_metrix.shape)
        self.unique_index = np.unique(self.center_index_set)
        self.center_data = X[self.unique_index]
        return self

    def embeding_mat(self):
        """Get the isolation kernel map feature of fit dataset.
        """
        return self.embeding_metrix

    def transform(self, x):
        """ Compute the isolation kernel map feature of x.

        Parameters
        ----------
        x: array-like of shape (1, n_features)
            The input instances.

        Returns
        -------
        ik_value: np.array of shape (sample_size times n_members,)
            The isolation kernel map of the input instance.
        """
        #st = time.time()
        x_dists = np.array([_fast_norm_diff(x, center)
                           for center in self.center_data])
        #et = time.time()
        mapping_array = np.zeros(self.unique_index.max()+1,
                                 dtype=x_dists.dtype)
        mapping_array[self.unique_index] = x_dists
        x_center_dist_mat = mapping_array[self.center_index_set]

        # nearest center index
        nearest_center_index = np.argmin(x_center_dist_mat, axis=1)
        ik_value = np.eye(self.psi, dtype=int)[
            nearest_center_index].flatten()
        assert ik_value.shape == (
            self.psi * self.t,), "Shape of ik_value is incorrect ! Except: %s, Get:%s" % ((self.psi * self.t,), ik_value.shape)

        #print(et - st)
        return ik_value


if __name__ == '__main__':

    from src.utils.deltasep_utils import create_dataset
    dataset = create_dataset(128, 5000, num_clusters=3)
    n = 10000
    # np.random.shuffle(dataset)
    data = np.array([pt[:3] for pt in dataset[:n]])
    sts = time.time()
    ikm = IKMapper(t=200, psi=13)
    print("start train")
    ets = time.time()
    ik_maper = ikm.fit(data)
    f_ets = time.time()
    cal_time = 0
    for dt in dataset[n:]:
        addx = dt[:3]
        test = ik_maper.transform(addx)
    add_end = time.time()
    print("add time:%s" % (add_end-f_ets))
    print("fit time: %s" % (f_ets-ets))
