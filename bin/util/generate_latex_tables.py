#!/usr/bin/env python3
"""
Generate professional LaTeX tables from clustering results.

For each metric, creates a table showing mean ± std for each algorithm on each dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def format_mean_std(mean, std, decimals=2):
    """Format mean ± std in LaTeX format."""
    format_str = f"{{:.{decimals}f}}"
    mean_str = format_str.format(mean)
    std_str = format_str.format(std)
    return f"${mean_str} \\pm {std_str}$"


def bold_text(text):
    """Wrap text in LaTeX bold."""
    return f"\\textbf{{{text}}}"


def generate_metric_table(df, metric_col, metric_name, output_file=None):
    """
    Generate a LaTeX table for a specific metric using pandas.

    Args:
        df: pandas DataFrame with clustering results
        metric_col: Column name of the metric (e.g., 'best_acc')
        metric_name: Display name of the metric (e.g., 'Accuracy')
        output_file: Optional file path to save the table

    Returns:
        LaTeX table string
    """
    # Group by dataset and algorithm, compute mean and std
    grouped = (
        df.groupby(["dataset", "algorithm"])[metric_col]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Pivot to get algorithms as columns
    pivot_mean = grouped.pivot(index="dataset", columns="algorithm", values="mean")
    pivot_std = grouped.pivot(index="dataset", columns="algorithm", values="std")

    # Fill NaN std with 0 (for single runs)
    pivot_std = pivot_std.fillna(0)

    # Build LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(
        "\\caption{" + metric_name + " results: Mean $\\pm$ Standard Deviation}"
    )
    latex.append("\\label{tab:" + metric_col + "}")

    # Determine column format
    algorithms = pivot_mean.columns.tolist()
    col_format = "l" + "c" * len(algorithms)
    latex.append("\\begin{tabular}{" + col_format + "}")
    latex.append("\\toprule")

    # Header row
    header = "Dataset & " + " & ".join(algorithms) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Data rows
    for dataset in pivot_mean.index:
        row_values = []
        row_means = []

        # Get means for this row
        for algo in algorithms:
            if pd.notna(pivot_mean.loc[dataset, algo]):
                row_means.append(pivot_mean.loc[dataset, algo])
            else:
                row_means.append(None)

        # Find best value(s) in this row
        valid_means = [m for m in row_means if m is not None]
        max_mean = max(valid_means) if valid_means else None

        # Format cells
        for i, algo in enumerate(algorithms):
            if row_means[i] is not None:
                mean = pivot_mean.loc[dataset, algo]
                std = pivot_std.loc[dataset, algo]
                cell = format_mean_std(mean, std)

                # Bold the best value(s)
                if max_mean is not None and abs(mean - max_mean) < 1e-10:
                    cell = bold_text(cell)
            else:
                cell = "---"

            row_values.append(cell)

        row = f"{dataset} & " + " & ".join(row_values) + " \\\\"
        latex.append(row)

    # Compute overall statistics (average across datasets)
    latex.append("\\midrule")
    avg_means = pivot_mean.mean(axis=0)
    avg_row_values = []

    # Find best overall
    max_avg_mean = avg_means.max() if not avg_means.empty else None

    for algo in algorithms:
        if pd.notna(avg_means[algo]):
            mean_val = avg_means[algo]
            cell = f"${mean_val:.2f}$"

            if max_avg_mean is not None and abs(mean_val - max_avg_mean) < 1e-10:
                cell = bold_text(cell)
        else:
            cell = "---"

        avg_row_values.append(cell)

    avg_row = "\\textit{Average} & " + " & ".join(avg_row_values) + " \\\\"
    latex.append(avg_row)

    # Table footer
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(latex_str)
        print(f"✓ Saved {metric_name} table to: {output_file}")

    return latex_str


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_latex_tables.py <csv_file> [output_dir]")
        print("\nExample:")
        print("  python generate_latex_tables.py exp_out/2025-10-30/all_scores.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else Path(csv_file).parent

    # Read data
    print(f"\n{'='*70}")
    print(f"Reading data from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # Define metrics
    metrics = [
        ("best_acc", "Clustering Accuracy"),
        ("best_nmi", "Normalized Mutual Information"),
        ("best_purity", "Purity Score"),
        ("best_ari", "Adjusted Rand Index"),
        ("best_ri", "Rand Index"),
    ]

    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check available columns
    available_cols = set(df.columns)

    # Generate table for each metric
    print(f"{'='*70}\n")
    for metric_col, metric_name in metrics:
        if metric_col in available_cols:
            print(f"Generating table for: {metric_name}")
            output_file = output_path / f"{metric_col}_table.tex"

            latex_table = generate_metric_table(
                df, metric_col, metric_name, output_file
            )

            # Also print to console
            print("\n" + latex_table)
            print("\n" + "-" * 70 + "\n")
        else:
            print(f"⚠ Warning: Column '{metric_col}' not found in data\n")

    # Generate a combined file with all tables
    combined_file = output_path / "all_metrics_tables.tex"
    with open(combined_file, "w") as f:
        f.write("% LaTeX tables for clustering metrics\n")
        f.write("% Generated automatically\n")
        f.write("% Required packages: \\usepackage{booktabs}\n\n")

        for metric_col, metric_name in metrics:
            if metric_col in available_cols:
                latex_table = generate_metric_table(df, metric_col, metric_name)
                f.write(latex_table)
                f.write("\n\n")

    print(f"{'='*70}")
    print(f"✓ All tables combined in: {combined_file}")
    print(f"{'='*70}\n")

    # Print summary statistics
    print("Summary Statistics:")
    print("=" * 70)

    # Count runs per dataset
    dataset_counts = df.groupby("dataset").size()
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} runs")

    # Count runs per algorithm
    print("\nRuns per algorithm:")
    algo_counts = df.groupby("algorithm").size()
    for algo, count in algo_counts.items():
        print(f"  {algo}: {count} runs")

    print(f"\nTotal datasets: {len(dataset_counts)}")
    print(f"Total algorithms: {len(algo_counts)}")
    print(f"Total runs: {len(df)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
