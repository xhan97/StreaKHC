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
        split_sep = "\t"
    with open(filename, "r") as f:
        for line in f:
            splits = line.strip().split(sep=split_sep)
            pid, l, vec = (
                int(float(splits[0])),
                int(float(splits[1])),
                np.array([float(x) for x in splits[2:]]),
            )
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


def format_value(v):
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def process_dict(d: dict):
    return list(d.keys()), [format_value(v) for v in d.values()]


def save_results(
    data_info: dict,
    algorithm_info: dict,
    metrics_info: dict,
    exp_dir_base: str,
):
    """Saves grid search result to a CSV file."""
    if os.path.isdir(exp_dir_base):
        os.makedirs(exp_dir_base, exist_ok=True)
        file_path = os.path.join(
            exp_dir_base, "{}.csv".format(data_info.get("dataset"))
        )
    else:
        os.makedirs(os.path.dirname(exp_dir_base), exist_ok=True)
        file_path = exp_dir_base
    data_info_header, data_info_values = process_dict(data_info)
    algorithm_info_header, algorithm_info_values = process_dict(algorithm_info)
    metrics_header, metrics_values = process_dict(metrics_info)

    header = data_info_header + algorithm_info_header + metrics_header
    values = data_info_values + algorithm_info_values + metrics_values

    # Write header if file does not exist
    write_header = not os.path.exists(file_path)
    with open(file_path, "a") as fout:
        if write_header:
            fout.write(",".join(header) + "\n")
        fout.write(",".join(values) + "\n")
    print(f"Results saved to {file_path}")
