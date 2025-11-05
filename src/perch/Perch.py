# Copyright 2025 Xin Han
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
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import time

from copy import deepcopy
import numpy as np
from src.perch.PNode import PNODE
from src.utils.dendrogram_purity_pool import (
    expected_dendrogram_purity,
    dendrogram_purity,
)
from src.utils.file_utils import save_results, load_static_data
from src.perch.utils.exact_cluster import cut_tree
from src.perch.utils.file import load_data_stream
from src.utils.flat_evaluate import (
    get_contingency_matrix,
    purity_score,
    nmi_score,
    accuracy_score,
    ari_score,
    rand_index_score,
)

# from src.utils.Graphviz_pnode import Graphviz
from src.utils.IsoKernel import IsolationKernel

# from src.utils.serialize_trees import serliaze_tree_to_file, serliaze_collapsed_tree_to_file_with_point_ids

# import psutil
# from memory_profiler import profile
# process = psutil.Process(os.getpid())
# @profile


def create_p_tree(data):
    """Create trees over the same points.

    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.  After each insertion, verify that the dendrogram purity
    is still 1.0 (perfect).

    Args:
        dataset - a list of points with which to build the tree.

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """

    root = PNODE(exact_dist_thres=10)
    run_time = []
    # mem_used = []
    L = 5000
    collapsibles = [] if L < float("Inf") else None
    i = 0
    tree_st_time = time.time()
    for pt in data:
        root = root.insert(pt, collapsibles=collapsibles, L=L)
        if i % 5000 == 0:
            tree_mi_time = time.time()
            run_time.append((i, tree_mi_time - tree_st_time))
            print(run_time)
            # mem_used.append(process.memory_info().rss / 1024 ** 2)
        i += 1
    # print(mem_used)
    return root, run_time


def create_p_tree_path(data_path):
    """Create trees over the same points.

    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.  After each insertion, verify that the dendrogram purity
    is still 1.0 (perfect).

    Args:
        dataset - a list of points with which to build the tree.

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """

    root = PNODE(exact_dist_thres=30)
    run_time = []
    inter_purity_list = []
    # mem_used = []
    L = 5000
    collapsibles = [] if L < float("Inf") else None
    # i = 0
    tree_st_time = time.time()
    for pt in load_data_stream(data_path):
        root = root.insert(pt, collapsibles=collapsibles, L=L)
        # if ((i > 1) and (i % 5 == 0)):
        #     inter_purity_list.append((i, dendrogram_purity(root)))
        # tree_mi_time = time.time()
        # run_time.append((i, tree_mi_time - tree_st_time))
        # print(run_time)
        # mem_used.append(process.memory_info().rss / 1024 ** 2)
        # i += 1
    # print(mem_used)
    return root


def grid_research_pnode(data_path, file_name, exp_dir_base, use_ik=False):
    ti = 0
    tree_purity = 0
    alg = "PERCH"
    data_info = {"dataset": file_name}
    algorithm_info = {"algorithm": alg}
    if use_ik:
        psi = [13, 15, 17, 21, 25]
        data = list(load_static_data(data_path))
        met = np.array([pt[0] for pt in data])
        for pi in psi:
            ik = IsolationKernel(n_estimators=300, max_samples=pi)
            iks = ik.fit_transform(met)
            for i, pt in enumerate(data):
                pt[0] = iks[i]
            root, run_time = create_p_tree(data)
            # with open("pnodeikTime.tsv", "a") as f:
            #     for item in run_time:
            #         f.write("%.2f\t%.2f\n" % (
            #             item[0],
            #             item[1]
            #         ))
            # save_tree_filename = "_".join(
            #     [file_name, "shuffle_index", str(shuffle_index), str(pi), "tree.txt"])
            tree_purity = expected_dendrogram_purity(root)
            # dendrogram_purity = 0
            if tree_purity > tree_purity:
                tree_purity = tree_purity
            args = {
                "dataset": file_name,
                "algorithm": "PERCH-ik",
                "purity": tree_purity,
                "psi": pi,
            }
            # save_data_pnode(args, exp_dir_base, file_name)
        args = {
            "dataset": file_name,
            "algorithm": "PERCH-ik",
            "purity": tree_purity,
        }
        # save_all_data(args, exp_dir_base, file_name)
    else:
        root = create_p_tree_path(data_path=data_path)
        # tree_purity = expected_dendrogram_purity(root)
        y_hat, y_true = cut_tree(root)
        contingency_matrix = get_contingency_matrix(y_true, y_hat)
        acc = accuracy_score(y_true, y_hat, contingency_matrix)
        purity = purity_score(y_true, y_hat, contingency_matrix)
        nmi = nmi_score(y_true, y_hat)
        ari = ari_score(y_true, y_hat)
        ri = rand_index_score(y_true, y_hat)
        metrics_info = {
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
            exp_dir_base=os.path.join(exp_dir_base, "best_results.csv"),
        )
        # st = time.time()
        # et = time.time()
        # print(et - st)
        # print(purity)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PERCH clustering.")
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
        "--use_ik",
        "-k",
        type=bool,
        help="Whether to use Isolation Kernel",
    )

    args = parser.parse_args()
    grid_research_pnode(
        data_path=args.input,
        file_name=args.dataset,
        exp_dir_base=args.outdir,
        use_ik=False,
    )


if __name__ == "__main__":
    main()
    # exp_dir_base_pnode = "./exp_out/purity_test/Pnode/"
    # start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # exp_dir_base_pnode = os.path.join(exp_dir_base_pnode, start_time)

    # grid_research_pnode(
    #     data_path="/home/xinhan/project/code/StreaKHC/data/shuffle_data/2025-11-04-16-28-55-224/ALLAML_0.csv",
    #     file_name="ALLAML_0",
    #     exp_dir_base=exp_dir_base_pnode,
    #     use_ik=False,
    # )
    # os.makedirs(exp_dir_base_pnode, exist_ok=True)
    # for file_name in os.listdir(input_data_dir):
    #     data_path = os.path.join(input_data_dir, file_name)
    #     grid_research_pnode(
    #         data_path=data_path,
    #         file_name=file_name[:-3],
    #         exp_dir_base=exp_dir_base_pnode,
    #         use_ik=False,
    #     )
