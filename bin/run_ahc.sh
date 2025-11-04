#!/usr/bin/env bash

set -exu

output_dir=${1:-"exp_out"}
TIME=$( (date +%Y-%m-%d-%H-%M-%S-%3N))
output_dir="${output_dir}/$TIME"

num_runs=1

for suffix in '.csv' '.tsv'; do
    if [ -z "$(ls $STREASKH_DATA*$suffix)" ]; then
        echo "No dataset endwith ${suffix} in ${STREASKH_DATA}"
        continue
    fi
    data_files=$(ls $STREASKH_DATA*$suffix)
    for dataset_file in ${data_files}; do
        dataset_name=$(basename -s $suffix $dataset_file)
        data_size=$(wc -l <$dataset_file)
        t_size=$(expr ${data_size} / 4)

        for lk in 'single' 'complete' 'average'; do
            (
                exp_output_dir="${output_dir}/${dataset_name}_lk_${lk}/run_1"
                mkdir -p ${exp_output_dir}
                python3 src/ahc/run_ahc.py --data_path ${dataset_file} \
                    --file_name ${dataset_name} \
                    --linkage ${lk} \
                    --exp_dir_base ${exp_output_dir}
            ) &
        done
        wait
        #mv $dataset_file $STREASKH_DATA_RUNNED
    done
done
sh bin/util/collect_and_format_results.sh $output_dir
