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


import argparse
import os

import numpy as np
from sklearn import tree

from src.IKMapper import IKMapper
from src.INode import INode
from src.utils.dendrogram_purity import (
    dendrogram_purity, expected_dendrogram_purity)
from src.utils.file_utils import load_data
from src.utils.Graphviz import Graphviz
from src.utils.serialize_trees import serliaze_tree_to_file


def record_build_tree(data_path, m, psi, t):
    """Create trees over the points from input data path.
    Return pointers to the roots of all trees for evaluation.  
    The trees will be created via the insert methods passed in.

    Args:
        data_path - path to dataset.
        m - numuber of point to intitial ik metrix
        psi - particial size  to build isolation kernel mapper
        t - sample size to build isolation kernel mapper

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """
    root = INode()
    train_dataset = []
    L = 5000
    tree_list = []
    for i, pt in enumerate(load_data(data_path), start=1):
        if i <= m:
            train_dataset.append(pt)
            if i == m:
                ik_mapper = IKMapper(t=t, psi=psi)
                ik_mapper = ik_mapper.fit(np.array(
                    [pt[2] for pt in train_dataset]))
                for j, train_pt in enumerate(train_dataset, start=1):
                    l, pid, ikv = train_pt[0], train_pt[1], ik_mapper.embeding_mat[j-1]
                    root = root.insert((l, pid, ikv), L=L,
                                       t=t, delete_node=True)
                    if j % 5 == 0:
                        tree_list.append((j, dendrogram_purity(root)))
        else:
            l, pid = pt[:2]
            root = root.insert((l, pid, ik_mapper.transform(
                pt[2])), L=L, t=t, delete_node=True)
            if i % 5 == 0:
                tree_list.append((i, dendrogram_purity(root)))
    #tree_list.append((i, dendrogram_purity(root)))
    return tree_list


def save_dp(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, 'dp_result.csv')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\n' % (
                'Points',
                "Dendrogram purity",
            ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%.2f\n' % (
            args['points'],
            args['dp'],
        ))


def get_dp(data_path, psi, t, m, file_name, exp_dir_base):
    tree_list = record_build_tree(data_path=data_path, m=m, psi=psi, t=t)
    #dp_res = cal_dp(tree_list, dendrogram_purity)
    file_path = os.path.join(exp_dir_base, 'dp_result.csv')
    with open(file_path, 'a') as fout:
        for item in tree_list:
            fout.write('%s\t%s\t%.2f\n' % (
                file_name,
                item[0],
                item[1]))


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate StreaKHC clustering.')
    parser.add_argument('--input', '-i', type=str,
                        help='<Required> Path to the dataset.', required=True)
    parser.add_argument('--outdir', '-o', type=str,
                        help='<Required> The output directory', required=True)
    parser.add_argument('--dataset', '-n', type=str,
                        help='<Required> The name of the dataset', required=True)
    parser.add_argument('--sample_size', '-t', type=int, default=300,
                        help='<Required> Sample size for isolation kernel mapper')
    parser.add_argument('--psi', '-p', type=int, required=True,
                        help='<Required> Particial size for isolation kernel mapper')
    parser.add_argument('--train_size', '-m', type=int, required=True,
                        help='<Required> Initial used data size to build Isolation Kernel Mapper')
    args = parser.parse_args()
    get_dp(data_path=args.input, m=args.train_size, t=args.sample_size, psi=args.psi,
           file_name=args.dataset, exp_dir_base=args.outdir)


if __name__ == "__main__":
    #main()
    data_path = "data/shuffle_data/2022-01-22-02-13-54-336/wine_8.csv"
    m = 44
    t = 300
    psi = 7
    file_name = "wine_3"
    exp_dir_base = "exp_out/test"
    get_dp(data_path=data_path, m=m, t=t, psi=psi,
                      file_name=file_name, exp_dir_base=exp_dir_base)
