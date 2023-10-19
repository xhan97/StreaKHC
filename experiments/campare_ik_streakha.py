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

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.isokahc.IsoKAHC import IsoKAHC
from src.utils.IsoKernel import IsolationKernel
from src.streakhc.INode import INode
from src.streakhc.utils.dendrogram_purity import expected_dendrogram_purity
from src.utils.file_utils import load_data_stream, load_static_data

from src.isokahc.utils import metrics


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
    return root, ik


def save_data(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, "score.tsv")
    if not os.path.exists(file_path):
        with open(file_path, "w") as fout:
            fout.write(
                "%s\t%s\t%s\t%s\n" % ("dataset", "algorithm", "purity", "max_psi",)
            )
    with open(file_path, "a") as fout:
        fout.write(
            "%s\t%s\t%.2f\t%s\n"
            % (args["dataset"], args["algorithm"], args["purity"], args["max_psi"],)
        )


def save_grid_data(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, "grid_score.tsv")
    if not os.path.exists(file_path):
        with open(file_path, "w") as fout:
            fout.write("%s\t%s\t%s\t%s\n" % ("dataset", "algorithm", "purity", "psi",))
    with open(file_path, "a") as fout:
        fout.write(
            "%s\t%s\t%.2f\t%s\n"
            % (args["dataset"], args["algorithm"], args["purity"], args["psi"],)
        )


def grid_search_inode(data_path, psi, t, m, file_name, exp_dir_base):

    max_purity = 0
    isokhc_dp_max = 0
    X, y = load_static_data(data_path)
    for p in psi:
        root, ik = streKHC(data_path, m, p, t)
        purity = expected_dendrogram_purity(root)
        if purity > max_purity:
            max_ps = p
            max_purity = purity
        res = {
            "dataset": file_name,
            "algorithm": "StreaKHC",
            "purity": purity,
            "psi": p,
        }
        save_grid_data(res, exp_dir_base)

        isokhc = IsoKAHC(method="single", iso_kernel=ik)
        Z = isokhc.fit_transform(X)
        isokhc_dp = metrics.dendrogram_purity(Z, y)

        res = {
            "dataset": file_name,
            "algorithm": "isokhc_single",
            "purity": isokhc_dp,
            "psi": p,
        }
        if isokhc_dp > isokhc_dp_max:
            isokhc_psi_max = p
            isokhc_dp_max = isokhc_dp
        save_grid_data(res, exp_dir_base)

    args = {
        "dataset": file_name,
        "algorithm": "StreaKHC",
        "purity": max_purity,
        "max_psi": max_ps,
    }
    save_data(args, exp_dir_base)

    args = {
        "dataset": file_name,
        "algorithm": "isokhc_average",
        "purity": isokhc_dp_max,
        "max_psi": isokhc_psi_max,
    }
    save_data(args, exp_dir_base)


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
    # data_path = "./data/shuffle_data/2022-07-21-13-16-57-560/ALLAML_5.csv"
    # m = 18
    # t = 200
    # psi = [3, 5, 10, 17, 21, 25]
    # file_name = "ALLAML"
    # exp_dir_base = "./exp_out/test"
    # grid_search_inode(data_path=data_path, m=m, t=t, psi=psi,
    #                   file_name=file_name, exp_dir_base=exp_dir_base)
