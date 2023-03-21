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
        continue
    fi
    data_files=$(ls $STREASKH_DATA*$suffix)
    for dataset_file in ${data_files}; do
        sh bin/util/shuffle_dataset.sh $dataset_file $num_runs $shuffle_data_path
        dataset_name=$(basename -s $suffix $dataset_file)
        data_size=$(wc -l <$dataset_file)
        t_size=$(expr ${data_size} / 4)
        #t_size=${data_size}
        for i in $(seq 1 1 $num_runs); do
            (
                shuffled_data="${shuffle_data_path}/${dataset_name}_${i}$suffix"
                #shuffled_data="${dataset_file}"
                exp_output_dir="${output_dir}/${dataset_name}/run_$i"
                mkdir -p ${exp_output_dir}
                python3 src/streakhc/StreaKHC.py --input ${shuffled_data} \
                    --outdir ${exp_output_dir} \
                    --dataset ${dataset_name} \
                    --psi 70 100 200 \
                    --train_size ${t_size}
                dot -Kdot -Tpng $exp_output_dir/tree.dot -o $exp_output_dir/tree.png
            ) &
        done
        wait
        #mv $dataset_file $STREASKH_DATA_RUNNED
    done
done
sh bin/util/collect_and_format_results.sh $output_dir
