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

import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings

warnings.filterwarnings("ignore")

import argparse
from IsoKAHC import IsoKAHC
from src.utils.file_utils import load_static_data
from sklearn.preprocessing import MinMaxScaler
from utils import metrics


def get_purity(X, n_estimators, max_samples, method):
    idk = IsoKAHC(n_estimators=n_estimators, max_samples=max_samples, method=method)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    ik_den = idk.fit_transform(X)
    return ik_den


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


def run(data_path, file_name, n_estimators, max_samples_list, method, exp_dir_base):
    pid, y, X = load_static_data(data_path)
    max_purity = 0.0
    for m_sp in max_samples_list:
        ik_den = get_purity(X, n_estimators, m_sp, method)
        purity_tmp = metrics.dendrogram_purity(ik_den, y)
        if purity_tmp > max_purity:
            max_purity = purity_tmp
            max_psi = m_sp
        max_purity = max(max_purity, purity_tmp)
        res = {
            "dataset": file_name,
            "algorithm": method,
            "purity": purity_tmp,
            "psi": m_sp,
        }
        save_grid_data(res, exp_dir_base)

    args = {
        "dataset": file_name,
        "algorithm": method,
        "purity": max_purity,
        "max_psi": max_psi,
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
        default=200,
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
        "--method", "-m", type=str, required=True, help="<Required> Linkage function"
    )
    args = parser.parse_args()
    run(
        data_path=args.input,
        file_name=args.dataset,
        n_estimators=args.sample_size,
        max_samples_list=args.psi,
        method=args.method,
        exp_dir_base=args.outdir,
    )


if __name__ == "__main__":
    main()
    # data_path = "./data/runned/wine.csv"
    # t = 200
    # psi = [3, 5, 10, 17, 21, 25]
    # file_name = "wine"
    # exp_dir_base = "./exp_out/test"
    # run(data_path=data_path, method='average', n_estimators=t,
    #     max_samples_list=psi, file_name=file_name, exp_dir_base=exp_dir_base)
