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

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line                

from IsoKAHC import IsoKAHC
from sklearn.datasets import load_wine
from utils import metrics

def test_isokahc():
    test_idk = IsoKAHC(t=200, psi=8)
    X, y = load_wine(return_X_y=True)
    ik_den = test_idk.fit_transform(X)
    dp = metrics.dendrogram_purity(ik_den, y)
    print(dp)
