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

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import argparse
import warnings
import time

import numpy as np
from sklearn.kernel_approximation import Nystroem

from src.streakhc_snstr.INode_snstr import INode_snstr
from src.utils.dendrogram_purity import expected_dendrogram_purity
from src.utils.file_utils import load_data_stream
from src.utils.Graphviz import Graphviz
from src.utils.serialize_trees import serliaze_tree_to_file

warnings.filterwarnings("ignore", category=FutureWarning)


def streKHC_snstr(data_path, m, sig, n_components, window_size=5000):
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
    root = INode_snstr()
    train_dataset = []
    L = 5000
    st = time.time()
    for i, pt in enumerate(load_data_stream(data_path), start=1):
        if i <= m:
            train_dataset.append(pt)
            if i == m:
                gk = Nystroem(kernel="rbf", n_components=n_components, gamma=sig ** -2)
                gk = gk.fit(np.array([pt[2] for pt in train_dataset]))
                for _, train_pt in enumerate(train_dataset, start=1):
                    l, pid, gkv = (
                        train_pt[0],
                        train_pt[1],
                        gk.transform([train_pt[2]])[0]
                    )
                    root = root.grow((l, pid, gkv), L=L, delete_node=True)

        else:
            l, pid = pt[:2]
            root = root.grow((l, pid, gk.transform([train_pt[2]])[0]), L=L, delete_node=True)

        if i % window_size == 0:
            print("Finish %d points in %.2f seconds." % (i, time.time() - st))
    return root


def save_data(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, "score.tsv")
    if not os.path.exists(file_path):
        with open(file_path, "w") as fout:
            fout.write(
                "%s\t%s\t%s\t%s\n" % ("dataset", "algorithm", "purity", "max_sig",)
            )
    with open(file_path, "a") as fout:
        fout.write(
            "%s\t%s\t%.2f\t%s\n"
            % (args["dataset"], args["algorithm"], args["purity"], args["max_sig"],)
        )


def save_grid_data(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, "grid_score.tsv")
    if not os.path.exists(file_path):
        with open(file_path, "w") as fout:
            fout.write("%s\t%s\t%s\t%s\n" % ("dataset", "algorithm", "purity", "sig",))
    with open(file_path, "a") as fout:
        fout.write(
            "%s\t%s\t%.2f\t%s\n"
            % (args["dataset"], args["algorithm"], args["purity"], args["sig"],)
        )


def grid_search_gnode(data_path, sig_list, n_components, m, file_name, exp_dir_base):
    alg = "StreaKHC_nystr"
    max_purity = 0
    for sig in sig_list:
        root = streKHC_snstr(data_path, m, sig, n_components)
        purity = 1
        # purity = expected_dendrogram_purity(root)
        if purity > max_purity:
            max_sig = sig
            max_root = root
            max_purity = purity
        res = {
            "dataset": file_name,
            "algorithm": alg,
            "purity": purity,
            "sig": sig,
        }
        save_grid_data(res, exp_dir_base)

    args = {
        "dataset": file_name,
        "algorithm": alg,
        "purity": max_purity,
        "max_sig": max_sig,
    }
    save_data(args, exp_dir_base)
    # serliaze_tree_to_file(max_root, os.path.join(exp_dir_base, "tree.tsv"))
    # Graphviz.write_tree(os.path.join(exp_dir_base, "tree.dot"), max_root)


def main():
    parser = argparse.ArgumentParser(description="Evaluate StreaKHC clustering.")
    parser.add_argument(
        "--input", "-i", type=str, help="<Required> Path to the dataset.", required=True
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        help="<Required> The output directory",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        "-n",
        type=str,
        help="<Required> The name of the dataset",
        required=True,
    )
    parser.add_argument(
        "--n_components",
        "-t",
        type=int,
        default=400,
        help="<Required> n_components for Gaussian kernel mapper",
    )
    parser.add_argument(
        "--sig",
        "-s",
        nargs="+",
        type=int,
        required=True,
        help="<Required> sig for Gaussian kernel mapper",
    )
    parser.add_argument(
        "--train_size",
        "-m",
        type=int,
        required=True,
        help="<Required> Initial used data size to build Gaussian kernel  Mapper",
    )
    parser.add_argument(
        "--data_feature",
        "-nf",
        type=int,
        required=True,
        help="<Required> n_feature for data set",
    )
    args = parser.parse_args()
    sig_list = [2 ** s for s in args.sig] + [
        args.data_feature * (2 ** s) for s in args.sig
    ]

    grid_search_gnode(
        data_path=args.input,
        m=args.train_size,
        n_components=args.n_components,
        sig_list=sig_list,
        file_name=args.dataset,
        exp_dir_base=args.outdir,
    )


if __name__ == "__main__":
    # main()
    data_path = "./data/raw/aloi_1.tsv"
    m = 44
    t = 200

    #sig = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    sig = [1]
    sig_list = [2 ** s for s in sig] + [128 * (2 ** s) for s in sig]
    file_name = "aloi"
    exp_dir_base = "./exp_out/test"
    grid_search_gnode(
        data_path=data_path,
        m=m,
        sig_list=sig_list,
        n_components=400,
        file_name=file_name,
        exp_dir_base=exp_dir_base,
    )

