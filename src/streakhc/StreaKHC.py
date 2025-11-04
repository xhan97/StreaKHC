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
import numpy as np
import time

from INode import INode
from src.utils.IsoKernel import IsolationKernel
from src.utils.file_utils import load_data_stream, save_results
from src.utils.Graphviz import Graphviz
from src.utils.dendrogram_purity import expected_dendrogram_purity
from src.utils.flat_evaluate import (
    cut_tree,
    get_contingency_matrix,
    purity_score,
    nmi_score,
    accuracy_score,
    ari_score,
    rand_index_score,
)
from src.utils.serialize_trees import serialize_tree_to_file


print("StreaKHC module loaded.")


def StreaKHC(data_path, m, psi, t, window_size=5000):
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
    st = time.time()
    for i, pt in enumerate(load_data_stream(data_path), start=1):
        if i <= m:
            train_dataset.append(pt)
            if i == m:
                ik = IsolationKernel(n_estimators=t, max_samples=psi)
                ik = ik.fit(np.array([pt[2] for pt in train_dataset]))
                for j, train_pt in enumerate(train_dataset, start=1):
                    l, pid, ikv = (
                        train_pt[0],
                        train_pt[1],
                        ik.transform([train_pt[2]])[0],
                    )
                    root = root.grow((l, pid, ikv), L=L, delete_node=True)
        else:
            l, pid = pt[:2]
            root = root.grow((l, pid, ik.transform([pt[2]])[0]), L=L, delete_node=True)

        if i % window_size == 0:
            print("Finish %d points in %.2f seconds." % (i, time.time() - st))
    return root


def grid_search_inode(data_path, psi, t, m, file_name, exp_dir_base):
    alg = "StreaKHC"
    best_acc, best_purity, best_nmi, best_ari, best_ri = 0, 0, 0, 0, 0
    data_info = {"dataset": file_name}
    algorithm_info = {"algorithm": alg}
    for ps in psi:
        root = StreaKHC(data_path, m, ps, t)
        # print(root.get_sibings_is_internal)
        # denpurity = expected_dendrogram_purity(root)
        y_hat, y_true = cut_tree(root)
        contingency_matrix = get_contingency_matrix(y_true, y_hat)
        acc = accuracy_score(y_true, y_hat, contingency_matrix)
        purity = purity_score(y_true, y_hat, contingency_matrix)
        nmi = nmi_score(y_true, y_hat)
        ari = ari_score(y_true, y_hat)
        ri = rand_index_score(y_true, y_hat)
        if acc > best_acc:
            max_ps = ps
            max_root = root
            best_acc = acc
        if purity > best_purity:
            best_purity = purity
        if nmi > best_nmi:
            best_nmi = nmi
        if ari > best_ari:
            best_ari = ari
        if ri > best_ri:
            best_ri = ri
        metrics_info = {
            "psi": ps,
            "purity": purity,
            "nmi": nmi,
            "accuracy": acc,
            # "denpurity": denpurity,
            "ari": ari,
            "ri": ri,
        }
        save_results(
            data_info=data_info,
            algorithm_info=algorithm_info,
            metrics_info=metrics_info,
            exp_dir_base=os.path.join(exp_dir_base, "grid_search.csv"),
        )

    args = {
        "max_psi": max_ps,
        "best_acc": best_acc,
        "best_nmi": best_nmi,
        "best_purity": best_purity,
        "best_ari": best_ari,
        "best_ri": best_ri,
        # "best_denpurity": best_denpurity,
    }
    save_results(
        data_info=data_info,
        algorithm_info=algorithm_info,
        metrics_info=args,
        exp_dir_base=os.path.join(exp_dir_base, "best_results.csv"),
    )
    # serialize_tree_to_file(max_root, os.path.join(exp_dir_base, "tree.tsv"))
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
        "--sample_size",
        "-t",
        type=int,
        default=300,
        help="<Required> Sample size for isolation kernel mapper",
    )
    parser.add_argument(
        "--psi",
        "-p",
        nargs="+",
        type=int,
        required=True,
        help="<Required> Particial size for isolation kernel mapper",
    )
    parser.add_argument(
        "--train_size",
        "-m",
        type=int,
        required=True,
        help="<Required> Initial used data size to build Isolation Kernel Mapper",
    )
    args = parser.parse_args()
    grid_search_inode(
        data_path=args.input,
        m=args.train_size,
        t=args.sample_size,
        psi=args.psi,
        file_name=args.dataset,
        exp_dir_base=args.outdir,
    )


if __name__ == "__main__":
    main()
    # # data_path = "./data/shuffle_data/2023-03-24-15-44-58-392/45_wine_2.csv"
    # data_path = "data/shuffle_data/2025-10-30-18-27-08-990/wine_3.csv"
    # m = 44
    # t = 200
    # psi = [3, 5, 7, 13, 15, 17, 21, 25]
    # file_name = "Wine"
    # exp_dir_base = "./exp_out/test"
    # grid_search_inode(
    #     data_path=data_path,
    #     m=m,
    #     t=t,
    #     psi=psi,
    #     file_name=file_name,
    #     exp_dir_base=exp_dir_base,
    # )
