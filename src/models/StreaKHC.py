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
import datetime
import os
import time

import numpy as np
from src.utils.dendrogram_purity import expected_dendrogram_purity,dendrogram_purity
from src.utils.file_utils import load_data, mkdir_p_safe
from src.utils.Graphviz import Graphviz
from src.utils.serialize_trees import serliaze_tree_to_file

from IKMapper import IKMapper
from INode import INode

# process = psutil.Process(os.getpid())


def build_streKhc_tree(data_path, m, psi, t, rate):
    """Create trees over the same points.
    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.

    Args:
        data_path - a list of points with which to build the tree.
        n - numuber of point to intitial ik metrix
        psi - particial size  to build isolation kernel mapper
        t - sample size to build isolation kernel mapper
        rate - windows size

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """
    root = INode()
    i = 0
    train_dataset = []
    L = 5000
    for pt in load_data(data_path):
        if i <= m:
            train_dataset.append(pt)
            if i == m:
                train_dataset_vec = np.array(
                    [pt[2] for pt in train_dataset])
                ik_mapper = IKMapper(t=t, psi=psi)
                ik_mapper = ik_mapper.fit(train_dataset_vec)
                j = 0
                for train_pt in train_dataset:
                    l, pid, ikv = train_pt[0], train_pt[1], ik_mapper.embeding_mat[j]
                    root = root.insert((l, pid, ikv), L=L,
                                       rate=rate, delete_node=True)
                    j += 1
        else:
            l, pid = pt[:2]
            root = root.insert((l, pid, ik_mapper.transform(
                pt[2])), L=L, rate=rate, delete_node=True)
        i += 1
        if i % 5000 == 0:
            print(i)
    return root


def save_data(args, exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name+'.tsv')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'purity',
                "max_psi",
                "max_rt"))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%.2f\t%s\t%s\n' % (
            args['dataset'],
            args['algorithm'],
            args['purity'],
            args["max_psi"],
            args["max_rt"]))


def grid_search_inode(data_path, psi, t, m, rates, file_name, exp_dir_base_data):
    max_purity = 0
    max_ps = psi[0]
    max_rt = rates[0]
    for ps in psi:
        for rt in rates:
            root = build_streKhc_tree(
                data_path, m, ps, t, rate=rt)
            purity = expected_dendrogram_purity(root)
            if purity > max_purity:
                max_ps = ps
                max_rt = rt
                max_root = root
                max_purity = purity
    serliaze_tree_to_file(max_root, os.path.join(
        exp_dir_base_data, 'tree.tsv'))
    Graphviz.write_tree(os.path.join(
        exp_dir_base_data, "tree.dot"), max_root)
    args = {
        'dataset': file_name,
        'algorithm': "StreaKHC",
        'purity': max_purity,
        "max_psi": max_ps,
        "max_rt": max_rt}
    save_data(args, exp_dir_base_data, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate StreaKHC clustering.')
    parser.add_argument('--input', '-i', type=str,
                        help='<Required> Path to the dataset.', required=True)
    parser.add_argument('--outdir', '-o', type=str,
                        help='<Required> The output directory', required=True)
    parser.add_argument('--dataset', '-n', type=str,
                        help='<Required> The name of the dataset', required=True)
    parser.add_argument('--rates', '-r', nargs='+',
                        type=float, help="<Required> rates")
    parser.add_argument('--sample_size', '-t', type=int,
                        help='<Required> Sample size for isolation kernel mapper', default=300)
    parser.add_argument('--psi', '-p', nargs='+', type=int,
                        help='<Required> Particial size for isolation kernel mapper', required=True)
    parser.add_argument('--train_size', '-m', type=int,
                        help='<Required> Initial used data size to build Isolation Kernel Mapper')
    parser.add_argument('--suffix', '-suf', type=str,
                        help='<Required> suffix of dataset')

    # parser.add_argument('--serliaze_tree', action='store_true', default=False)
    # parser.set_defaults(serliaze_tree=False)
    # parser.add_argument('--plot_tree',
    #                     dest='plot_tree', action='store_true')
    # parser.set_defaults(plot_tree=False)

    args = parser.parse_args()
    print(parser.parse_args())
    # ts = time.time()
    # st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    # start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    mkdir_p_safe(args.outdir)
    grid_search_inode(data_path=args.input, m=args.train_size, t=args.sample_size, psi=args.psi,
                      rates=args.rates, file_name=args.dataset, exp_dir_base_data=args.outdir)

    # psi = [7, 15, 17, 21, 25]
    # #psi = [7]
    # rates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # #rates = [0.7]
    # t = 200
    # file_name = "wine.csv"
    # input_data_dir = 'data\\raw'
    # exp_dir_base_inode = 'exp_out\\purity_test\\streaKHC'
    # start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # exp_dir_base_inode = os.path.join(exp_dir_base_inode, start_time)
    # mkdir_p_safe(exp_dir_base_inode)
    #
    # n = 30
    # grid_search_inode(data_path=file_path, m=n, t=t, psi=psi, rates=rates,
    #                   file_name=file_name[:-4], exp_dir_base=exp_dir_base_inode)
