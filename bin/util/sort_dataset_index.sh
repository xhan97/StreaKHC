set -exu

data_path=$1
index_dir=$2
num_shuffles=$3
sorted_data_dir=$4

mkdir -p $sorted_data_dir

data_name=$(basename $data_path)
data_name=${data_name%%.*}

for i in $(seq 1 $num_shuffles); do
	sorted_data_path="${sorted_data_dir}/${data_name}_${i}.csv"
	index_path="${index_dir}/${data_name}/run_${i}/index.tsv"
	echo "Sorting $data_name > $sorted_data_path"
	python3 bin/util/sort_dataset_index.py $data_path $index_path $sorted_data_path
done