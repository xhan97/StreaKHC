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

import errno
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def mkdir_p_safe(dir):
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def remove_dirs(exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name)
    if os.path.exists(file_path):
        os.removedirs(file_path)


def load_static_data(filename):
    if filename.endswith(".csv"):
        split_sep = ","
    elif filename.endswith(".tsv"):
        split_sep = "\t"
    data = pd.read_csv(filename, sep=split_sep, header=None)
    data = data.to_numpy()
    pid, l, vec = data[:, 0], data[:, 1], data[:, 2:]
    return pid, l, vec


def load_data_stream(filename):
    if filename.endswith(".csv"):
        split_sep = ","
    elif filename.endswith(".tsv"):
        split_sep = '\t'
    with open(filename, 'r') as f:
        for line in f:
            splits = line.strip().split(sep=split_sep)
            pid, l, vec = splits[0], splits[1], np.array([float(x)
                                                          for x in splits[2:]])
            yield ((l, pid, vec))


def load_npy_stream(filename, is_scale=False, is_shuffle=False):
    data = np.load(filename, allow_pickle=True)
    X, y = data["X"], data["y"]
    if is_scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    pid_list = np.array(range(len(y)))
    y = y[..., np.newaxis]
    pid_list = pid_list[..., np.newaxis]
    concat_data = np.concatenate((pid_list, y, X), axis=1)
    if is_shuffle:
        rng = np.random.default_rng()
        rng.shuffle(concat_data)
    for pt in concat_data:
        yield ((int(pt[1]), int(pt[0]), pt[2:]))
