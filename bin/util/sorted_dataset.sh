set -exu

dataset=$1
num_shuffles=$2
shuffle_data_path=$3

mkdir -p $shuffle_data_path

data_name=`basename $dataset`

# sorted dataset
sorted_data="${shuffle_data_path}/${data_name%%.*}_0.${data_name#*.}"
sort -k 2,2 -k 1,1  -t','  -g  $dataset > $sorted_data
