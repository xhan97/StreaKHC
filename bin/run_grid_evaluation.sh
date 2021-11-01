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

#!/usr/bin/env bash

set -exu

output_dir=${1:-"exp_out"}
run_id=${2:-1}

dataset_path=$STREASKH_ROOT/data/raw/
shuffle_data_path=$STREASKH_ROOT/data/shuffle_data/
dataset_runned_path=$STREASKH_ROOT/data/runned/

num_runs=5
TIME=`(date +%Y-%m-%d-%H-%M-%S-%3N)`

output_dir="${output_dir}/$TIME"
suffix=".csv"

mkdir -p $output_dir
expected_dp_point_file="None"
data_files=$(ls $dataset_path*$suffix)

for dataset_file in $data_files
    do
        sh bin/util/shuffle_dataset.sh $dataset_file $num_runs $shuffle_data_path
        dataset_name=`basename -s $suffix  $dataset_file`

        for i in `seq 1 $num_runs`
            do
                shuffled_data="${shuffle_data_path}/${dataset_name}_${i}$suffix"
                exp_output_dir="$output_dir/$dataset_name/run_$i"

                python3 StreaKHC.py --input   $shuffled_data\
                                               --outdir  $exp_output_dir \
                                               --dataset ${dataset_name}\
                                               --rates 0.4 0.5 0.6 0.7 0.8 0.9 1  \
                                               --psi 3 5 7 13 15 17 21 25  \
                                               --train_size 30
                dot -Kdot -Tpng $exp_output_dir/tree.dot  -o $exp_output_dir/tree.png
            done
        #mv $dataset_file $dataset_runned_path
    done

