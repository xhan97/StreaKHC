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

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import warnings

warnings.filterwarnings("ignore")

import argparse
from sklearn.cluster import AgglomerativeClustering
from src.utils.file_utils import load_static_data, save_results
from sklearn.preprocessing import MinMaxScaler

from src.utils.flat_evaluate import (
    purity_score,
    nmi_score,
    accuracy_score,
    ari_score,
    rand_index_score,
)


def get_labels(X, n_clusters, linkage):
    agc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    labels = agc.fit_predict(X)
    return labels


def run(data_path, file_name, linkage, exp_dir_base):
    _, y, X = load_static_data(data_path)
    n_clusters = len(set(y))
    labels = get_labels(X, n_clusters, linkage)

    purity = purity_score(y, labels)
    nmi = nmi_score(y, labels)
    accuracy = accuracy_score(y, labels)
    ari = ari_score(y, labels)
    rand_index = rand_index_score(y, labels)

    data_info = {
        "dataset": file_name,
    }

    algorithm_info = {
        "linkage": linkage,
    }

    res_info = {
        "purity": purity,
        "nmi": nmi,
        "accuracy": accuracy,
        "ari": ari,
        "rand_index": rand_index,
    }

    save_results(
        data_info,
        algorithm_info,
        res_info,
        exp_dir_base=os.path.join(exp_dir_base, "best_results.csv"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data file"
    )
    parser.add_argument(
        "--file_name", type=str, required=True, help="Name of the dataset"
    )

    parser.add_argument(
        "--linkage",
        type=str,
        default="single",
        choices=["complete", "average", "single"],
        help="Linkage criterion",
    )
    parser.add_argument(
        "--exp_dir_base",
        type=str,
        required=True,
        help="Directory to save experiment results",
    )

    args = parser.parse_args()

    run(
        data_path=args.data_path,
        file_name=args.file_name,
        linkage=args.linkage,
        exp_dir_base=args.exp_dir_base,
    )


if __name__ == "__main__":
    main()
