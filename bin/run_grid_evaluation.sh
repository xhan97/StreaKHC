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
TIME=$( (date +%Y-%m-%d-%H-%M-%S-%3N))
output_dir="${output_dir}/$TIME"
shuffle_data_path="$STREASKH_DATA_SHUFFLE/$TIME"

num_runs=10
for suffix in '.csv' '.tsv'; do
    if [ -z "$(ls $STREASKH_DATA*$suffix)" ]; then
        echo "No dataset endwith ${suffix} in ${STREASKH_DATA}"
        break
    fi
    data_files=$(ls $STREASKH_DATA*$suffix)
    for dataset_file in ${data_files}; do
        sh bin/util/shuffle_dataset.sh $dataset_file $num_runs $shuffle_data_path
        dataset_name=$(basename -s $suffix $dataset_file)
        data_size=$(wc -l <$dataset_file)
        t_size=$(expr ${data_size} / 4)
        for i in $(seq 1 $num_runs); do
            shuffled_data="${shuffle_data_path}/${dataset_name}_${i}$suffix"
            for beta in $(seq 0.5 0.1 1); do
                (
                    beta_str=$(echo ${beta}*10 | bc)
                    exp_output_dir="${output_dir}/${dataset_name}/run_$i/${beta_str%.*}"
                    mkdir -p ${exp_output_dir}
                    python3 StreaKHC.py --input ${shuffled_data} \
                        --outdir ${exp_output_dir} \
                        --dataset ${dataset_name} \
                        --beta ${beta} \
                        --psi 3 5 7 13 15 17 21 25 \
                        --train_size ${t_size}
                    #dot -Kdot -Tpng $exp_output_dir/tree.dot -o $exp_output_dir/tree.png
                ) &
            done
            wait
        done
        #mv $dataset_file $STREASKH_DATA_RUNNED
    done
done
sh bin/util/collect_and_format_results.sh $output_dir
