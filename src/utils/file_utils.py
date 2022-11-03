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


# def load_nn_stream(filename, nbrs, X_query):
#     pid, l, vec = load_static_data(filename=filename)
#     for i in range(len(pid)):
#         nn_index = get_nn_index(nbrs=nbrs, X_query=X_query)
#         yield ((l[nn_index], pid[nn_index], vec[nn_index]))
