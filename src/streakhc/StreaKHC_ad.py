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

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import argparse
import numpy as np

from INode import INode
from src.utils.IsoKernel import IsolationKernel
from src.utils.file_utils import load_npy_stream
from src.utils.metrics import ad_metric


def streKHC(data_path, m, psi, t):
    """Create trees over the same points.
    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.

    Args:
        data_path - path to dataset.
        m - number of point to initial ik matrix
        psi - partial size  to build isolation kernel mapper
        t - sample size to build isolation kernel mapper

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """
    root = INode()
    train_dataset = []
    L = 5000
    for i, pt in enumerate(load_npy_stream(data_path, is_scale=True, is_shuffle=True), start=1):
        if i <= m:
            train_dataset.append(pt)
            if i == m:
                ik = IsolationKernel(n_estimators=t, max_samples=psi)
                ik = ik.fit(np.array(
                    [pt[2] for pt in train_dataset]))
                for j, train_pt in enumerate(train_dataset, start=1):
                    l, pid, ikv = train_pt[0], train_pt[1], ik.transform([train_pt[2]])[0]
                    root = root.grow((l, pid, ikv), L=L, delete_node=True)
        else:
            l, pid = pt[:2]
            root = root.grow((l, pid, ik.transform(
                [pt[2]])[0]), L=L, delete_node=True)
    return root


def save_data(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, 'score.tsv')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'aucroc',
                "max_psi",
            ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%.2f\t%s\n' % (
            args['dataset'],
            args['algorithm'],
            args['aucroc'],
            args["max_psi"],
        ))


def save_grid_data(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, 'grid_score.tsv')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'aucroc',
                "psi",
            ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%.2f\t%s\n' % (
            args['dataset'],
            args['algorithm'],
            args['aucroc'],
            args["psi"],
        ))


def grid_search_inode(data_path, psi, t, m, file_name, exp_dir_base):
    alg = 'StreaKHC'
    max_score = 0
    for ps in psi:
        root = streKHC(
            data_path, m, ps, t)
        
        ad_score = ad_metric(root)["aucroc"]
        if ad_score > max_score:
            max_ps = ps
            max_root = root
            max_score = ad_score
        res = {'dataset': file_name,
               'algorithm': alg,
               'aucroc': ad_score,
               "psi": ps,
               }
        save_grid_data(res, exp_dir_base)

    args = {'dataset': file_name,
            'algorithm': alg,
            'aucroc': max_score,
            "max_psi": max_ps,
            }
    save_data(args, exp_dir_base)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate StreaKHC clustering.')
    parser.add_argument('--input', '-i', type=str,
                        help='<Required> Path to the dataset.', required=True)
    parser.add_argument('--outdir', '-o', type=str,
                        help='<Required> The output directory', required=True)
    parser.add_argument('--dataset', '-n', type=str,
                        help='<Required> The name of the dataset', required=True)
    parser.add_argument('--sample_size', '-t', type=int, default=200,
                        help='<Required> Sample size for isolation kernel mapper')
    parser.add_argument('--psi', '-p', nargs='+', type=int, required=True,
                        help='<Required> Particial size for isolation kernel mapper')
    args = parser.parse_args()
    
    data = np.load(args.input, allow_pickle=True)
    train_size = min(int(len(data["y"])/4), 5000)
    
    grid_search_inode(data_path=args.input, m=train_size, t=args.sample_size, psi=args.psi,
                      file_name=args.dataset, exp_dir_base=args.outdir)


if __name__ == "__main__":
    main()
    # data_path = "/home/hanxin/project/StreamHC/data/anomaly/Classical/33_skin.npz"
    # m = 1000
    # t = 200
    # psi = [2, 4, 6]
    # file_name = "wine"
    # exp_dir_base = "./exp_out/test"
    # grid_search_inode(data_path=data_path, m=m, t=t, psi=psi,
    #                   file_name=file_name, exp_dir_base=exp_dir_base)
