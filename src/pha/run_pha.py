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

from pha import pha_cluster_from_points

from src.utils.file_utils import load_static_data, save_results
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import fcluster
from src.utils.flat_evaluate import (
    purity_score,
    nmi_score,
    accuracy_score,
    ari_score,
    rand_index_score,
)


def get_labels(X, S, n_clusters):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Z, _, _ = pha_cluster_from_points(X, S=S)
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    return labels


def run(data_path, file_name, S, exp_dir_base):
    _, y, X = load_static_data(data_path)
    n_clusters = len(set(y))
    best_acc, best_purity, best_nmi, best_ari, best_rand = 0, 0, 0, 0, 0

    for s_i in S:
        try:
            print(f"Running PHA with S={s_i} on dataset {file_name}...")
            labels = get_labels(X, s_i, n_clusters)
            purity = purity_score(y, labels)
            nmi = nmi_score(y, labels)
            accuracy = accuracy_score(y, labels)
            ari = ari_score(y, labels)
            rand_index = rand_index_score(y, labels)

            best_acc = max(best_acc, accuracy)
            best_purity = max(best_purity, purity)
            best_nmi = max(best_nmi, nmi)
            best_ari = max(best_ari, ari)
            best_rand = max(best_rand, rand_index)
        except Exception as e:
            print(f"Error with S={s_i} on dataset {file_name}: {e}")
            continue

    data_info = {
        "dataset": file_name,
    }
    algorithm_info = {
        "algorithm": "PHA",
    }
    res_info = {
        "best_purity": best_purity,
        "best_nmi": best_nmi,
        "best_acc": best_acc,
        "best_ari": best_ari,
        "best_ri": best_rand,
    }

    save_results(
        data_info,
        algorithm_info,
        res_info,
        exp_dir_base=os.path.join(exp_dir_base, "best_results.csv"),
    )


def main():

    parser = argparse.ArgumentParser(description="Run PHA clustering algorithm")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument(
        "--file_name", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument("--S", type=int, nargs="+", help="Number of nearest neighbors")
    parser.add_argument(
        "--exp_dir_base",
        type=str,
        default="experiments/pha_results",
        help="Base directory to save experiment results",
    )

    args = parser.parse_args()

    run(
        data_path=args.data_path,
        file_name=args.file_name,
        S=args.S,
        exp_dir_base=args.exp_dir_base,
    )


if __name__ == "__main__":
    main()
