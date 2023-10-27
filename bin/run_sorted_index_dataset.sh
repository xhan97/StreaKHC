#!/usr/bin/env bash

set -exu

index_dir=$1
if index_dir; then
	echo "Index dir not found"
	exit 1
fi

output_dir=${2:-"exp_out"}
TIME=$( (date +%Y-%m-%d-%H-%M-%S-%3N))
output_dir="${output_dir}/$TIME"
shuffle_data_path="$STREASKH_DATA_SHUFFLE/$TIME"

num_runs=5
for suffix in '.csv' '.tsv'; do
	if [ -z "$(ls $STREASKH_DATA*$suffix)" ]; then
		echo "No dataset endwith ${suffix} in ${STREASKH_DATA}"
		continue
	fi
	data_files=$(ls $STREASKH_DATA*$suffix)
	for dataset_file in ${data_files}; do
		sh bin/util/sort_dataset_index.sh $dataset_file $index_dir $num_runs $shuffle_data_path
	done
done
# sh bin/util/collect_and_format_results.sh $output_dir
