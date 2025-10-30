#!/usr/bin/env bash

set -xu

exp_dir=$1

# Find all best_results.csv files once and combine them
# Store file list in a temporary variable
csv_files=$(find "$exp_dir" -name best_results.csv | sort)

if [ -z "$csv_files" ]; then
    echo "Error: No best_results.csv files found in $exp_dir" >&2
    exit 1
fi

{
    # Extract header from first file
    echo "$csv_files" | head -n 1 | xargs head -n 1
    # Extract data rows from all files
    echo "$csv_files" | xargs tail -n +2 -q
} > "$exp_dir/all_scores.csv"

# python3 bin/util/format_result_table.py "$exp_dir/all_scores.txt" > "$exp_dir/dendrogram_purity.tex"

# cat $exp_dir/dendrogram_purity.tex

# echo "Dendrogram Purity Result table saved here: $exp_dir/dendrogram_purity.tex"