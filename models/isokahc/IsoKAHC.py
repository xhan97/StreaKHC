# Copyright 2022 Xin Han
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

import numpy as np
from IsoKernel import IsolationKernel
from scipy.cluster.hierarchy import linkage
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler


class IsoKAHC(BaseEstimator):
    def __init__(self, n_estimators=200, max_samples="auto", method='single', ik=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.isokernel = ik
        self.method = method

    def _get_ikfeture(self, X):
        self.isokernel = IsolationKernel(
            X, self.n_estimators, self.max_samples)
        X = self.isokernel.fit_transform(X)
        return X

    def fit(self, X) -> 'IsoKAHC':
        # Check data
        X = self._validate_data(X, accept_sparse=False)
        # scaler = MinMaxScaler()
        # X = scaler.fit_transform(X)
        if isinstance(self.isokernel, IsolationKernel):
            X = self.isokernel.transform(X)
        else:
            X = self._get_ikfeture(X)
        similarity_matrix = np.inner(X, X) / self.n_estimators
        self.dendrogram_ = linkage(1 - similarity_matrix, method=self.method)
        return self

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to data and return the dendrogram. Same parameters as the ``fit`` method.
        Returns
        -------
        dendrogram : np.ndarray
            Dendrogram.
        """
        self.fit(*args, **kwargs)
        return self.dendrogram_
