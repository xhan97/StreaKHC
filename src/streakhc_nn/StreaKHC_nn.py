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
from INode import INode
from src.utils.IsoKernel import IsolationKernel
from src.utils.file_utils import load_static_data
from src.utils.Graphviz import Graphviz
from src.utils.dendrogram_purity import expected_dendrogram_purity, dendrogram_purity
from src.utils.serialize_trees import serliaze_tree_to_file

# from memory_profiler import profile


def get_nn_index(q, mat, mask_index):
    x_sim = np.inner(q, mat)
    x_sim[mask_index] = -np.inf
    nn_ind = np.argmax(x_sim)
    return nn_ind


# @profile
def streKHC_nn(data_path, psi, t):
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
    L = 5000
    pid, l, vec = load_static_data(data_path)
    ik = IsolationKernel(n_estimators=t, max_samples=psi)
    ik = ik.fit(vec)
    ikv = ik.transform(vec)
    sim = np.inner(ikv, ikv)
    del ik
    np.fill_diagonal(sim, -np.inf)
    x_ind, y_ind = np.unravel_index(np.argmax(sim, axis=None), sim.shape)
    num_samples = len(pid)
    mask_index = []
    for i in range(num_samples):
        if i == 0:
            insert_index = x_ind
        elif i == 1:
            insert_index = y_ind
        else:
            insert_index = get_nn_index(root.ikv, ikv, mask_index)
        root = root.grow(
            (l[insert_index], pid[insert_index], ikv[insert_index]),
            L=L,
            delete_node=True,
        )
        mask_index.append(insert_index)

        # if i % 200 == 0 and i != 0:
        #     #serliaze_tree_to_file(root, os.path.join('./exp_out/test/nn', 'tree_{}_{}.tsv'.format(psi, i)))
        #     Graphviz.write_tree(os.path.join('./exp_out/test/Synthetic/nn', 'tree_{}_{}.dot'.format(psi, i)), root)

    return root, mask_index


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


def save_mask_index(mask_index, file_path):
    with open(file_path, "w") as fout:
        for ind in mask_index:
            fout.write("{}\n".format(ind))


def load_mask_index(file_path):
    mask_index = []
    with open(file_path, "r") as fin:
        for line in fin:
            mask_index.append(int(line.strip()))
    return mask_index


# @profile
def grid_search_inode(data_path, psi, t, file_name, exp_dir_base):
    alg = "StreaKHC_nn"
    max_purity = 0
    max_mask_index = []
    for ps in psi:
        root, mask_index = streKHC_nn(data_path, ps, t)
        purity = dendrogram_purity(root)
        if purity > max_purity:
            max_ps = ps
            # max_root = root
            max_purity = purity
            max_mask_index = mask_index
        res = {
            "dataset": file_name,
            "algorithm": alg,
            "purity": purity,
            "psi": ps,
        }
        save_grid_data(res, exp_dir_base)

    args = {
        "dataset": file_name,
        "algorithm": alg,
        "purity": max_purity,
        "max_psi": max_ps,
    }
    save_data(args, exp_dir_base)
    save_mask_index(
        max_mask_index,
        os.path.join(exp_dir_base, "index.tsv"),
    )
    # serliaze_tree_to_file(max_root, os.path.join(
    #     exp_dir_base, 'tree.tsv'))
    # Graphviz.write_tree(os.path.join(
    #     exp_dir_base, 'tree.dot'), max_root)


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
        t=args.sample_size,
        psi=args.psi,
        file_name=args.dataset,
        exp_dir_base=args.outdir,
    )


if __name__ == "__main__":
    main()
    # data_path = "data/shuffle_data/2022-11-03-17-17-20-762/Synthetic_1.csv"
    # m = 44
    # t = 200
    # psi = [3, 5, 10, 17, 21, 25]
    # file_name = "Synthetic"
    # exp_dir_base = "./exp_out/test"
    # grid_search_inode(data_path=data_path, t=t, psi=psi,
    #                   file_name=file_name, exp_dir_base=exp_dir_base)
