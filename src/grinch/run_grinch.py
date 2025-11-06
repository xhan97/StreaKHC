import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import argparse
import time

import numpy as np
from src.grinch.grinch import Grinch

from src.utils.dendrogram_purity import expected_dendrogram_purity
from src.utils.file_utils import load_data_stream, save_results
from src.utils.flat_evaluate import (
    accuracy_score,
    ari_score,
    get_contingency_matrix,
    nmi_score,
    purity_score,
    rand_index_score,
)
from src.utils.Graphviz import Graphviz
from src.utils.serialize_trees import serialize_tree_to_file
from utils.exact_cluster import cut_tree_grinch

print("Grinch module loaded.")


def create_g_tree_path(data_path):
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

    g1 = Grinch(norm="l2")

    run_time = []
    # mem_used = []
    L = 5000
    collapsibles = [] if L < float("Inf") else None
    i = 0
    tree_st_time = time.time()
    for pt in load_data_stream(data_path):
        g1.insert(pt[1], point_vec=pt[2], point_label=pt[0])
        if i % 5000 == 0:
            tree_mi_time = time.time()
            run_time.append((i, tree_mi_time - tree_st_time))
            print(run_time)
            # mem_used.append(process.memory_info().rss / 1024 ** 2)
        i += 1
    return g1.root_node


def grid_research_grinch(data_path, file_name, exp_dir_base, use_ik=False):
    ti = 0
    tree_purity = 0
    alg = "Grinch"
    data_info = {"dataset": file_name}
    algorithm_info = {"algorithm": alg}
    root = create_g_tree_path(data_path=data_path)
    # tree_purity = expected_dendrogram_purity(root)
    y_hat, y_true = cut_tree_grinch(root)
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
    grid_research_grinch(
        data_path=args.input,
        file_name=args.dataset,
        exp_dir_base=args.outdir,
        use_ik=False,
    )


if __name__ == "__main__":
    # main()
    exp_dir_base_gnode = "./exp_out/purity_test/Gnode/"
    start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_dir_base_gnode = os.path.join(exp_dir_base_gnode, start_time)
    os.makedirs(exp_dir_base_gnode, exist_ok=True)
    grid_research_grinch(
        data_path="/home/xinhan/project/code/StreaKHC/data/shuffle_data/2025-11-04-16-28-55-224/ALLAML_0.csv",
        file_name="ALLAML_0",
        exp_dir_base=exp_dir_base_gnode,
        use_ik=False,
    )

    # point_labels = np.random.random_integers(0, 10, 100)
    # vectors = np.random.random((100, 5)).astype(np.float32)
    # grinch = Grinch(points=vectors)
    # grinch.build_dendrogram()
    # grinch.write_tree("tmp.tree.out", point_labels)
