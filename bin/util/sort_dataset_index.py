# Copyright (C) 2023 Xin Han
#
# This file is part of StreamHC.
#
# StreamHC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# StreamHC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StreamHC.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import pandas as pd


def load_index(file_path):
    mask_index = []
    with open(file_path, "r") as fin:
        for line in fin:
            mask_index.append(int(line.strip()))
    return mask_index


def load_data(file_name):
    if file_name.endswith(".csv"):
        split_sep = ","
    elif file_name.endswith(".tsv"):
        split_sep = "\t"
    data = pd.read_csv(file_name, sep=split_sep, header=None)
    return data


def sort_file(data_path, index_path, exp_dir_base):
    data = load_data(data_path)
    index = load_index(index_path)
    data = data.iloc[index]
    data.to_csv(
        exp_dir_base,
        index=False,
        header=False,
    )


if __name__ == "__main__":
    # sort_file(  )
    sort_file(
        data_path=sys.argv[1],  # "/home/hanxin/project/StreamHC/data/raw/LSVT.csv",
        index_path=sys.argv[2],  # "/home/hanxin/project/StreamHC/exp_out/2023-10-19-14-27-38-026/LSVT/run_1/mask_index_LSVT.tsv",
        exp_dir_base=sys.argv[3],  # "/home/hanxin/project/StreamHC/exp_out/test/sort_file",
    )

