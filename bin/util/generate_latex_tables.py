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
    Supports multiple algorithms comparison across datasets.

    Args:
        df: pandas DataFrame with clustering results
        metric_col: Column name of the metric (e.g., 'best_acc')
        metric_name: Display name of the metric (e.g., 'Accuracy')
        output_file: Optional file path to save the table

    Returns:
        LaTeX table string
    """
    # Determine algorithm column - try 'algorithm' first, then 'linkage'
    algo_col = "algorithm" if "algorithm" in df.columns else "linkage"

    # Group by dataset and algorithm, compute mean and std
    grouped = (
        df.groupby(["dataset", algo_col])[metric_col].agg(["mean", "std"]).reset_index()
    )

    # Pivot to get algorithms as columns
    pivot_mean = grouped.pivot(index="dataset", columns=algo_col, values="mean")
    pivot_std = grouped.pivot(index="dataset", columns=algo_col, values="std")

    # Fill NaN std with 0 (for single runs)
    pivot_std = pivot_std.fillna(0)

    # Get sorted algorithms and datasets for consistent ordering
    algorithms = sorted(pivot_mean.columns.tolist())
    datasets = sorted(pivot_mean.index.tolist())

    # Build LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(
        "\\caption{" + metric_name + " results: Mean $\\pm$ Standard Deviation}"
    )
    latex.append("\\label{tab:" + metric_col + "}")

    # Determine column format - use smaller font if many algorithms
    if len(algorithms) > 4:
        latex.append("\\footnotesize")

    col_format = "l" + "c" * len(algorithms)
    latex.append("\\begin{tabular}{" + col_format + "}")
    latex.append("\\toprule")

    # Header row - break long algorithm names if needed
    formatted_algos = []
    for algo in algorithms:
        if len(algo) > 10:
            # Break long names
            formatted_algo = algo.replace("_", "\\_").replace(
                "StreaKHC", "\\texttt{StreaKHC}"
            )
        else:
            formatted_algo = algo.replace("_", "\\_")
        formatted_algos.append(formatted_algo)

    header = "Dataset & " + " & ".join(formatted_algos) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Track statistics for ranking algorithms
    algorithm_wins = {algo: 0 for algo in algorithms}
    algorithm_scores = {algo: [] for algo in algorithms}

    # Data rows
    for dataset in datasets:
        row_values = []
        row_means = []

        # Get means for this row
        for algo in algorithms:
            if algo in pivot_mean.columns and dataset in pivot_mean.index:
                if pd.notna(pivot_mean.loc[dataset, algo]):
                    mean_val = pivot_mean.loc[dataset, algo]
                    row_means.append(mean_val)
                    algorithm_scores[algo].append(mean_val)
                else:
                    row_means.append(None)
            else:
                row_means.append(None)

        # Find best value(s) in this row (handle ties)
        valid_means = [m for m in row_means if m is not None]
        if valid_means:
            max_mean = max(valid_means)
            # Count wins for ranking
            for i, algo in enumerate(algorithms):
                if row_means[i] is not None and abs(row_means[i] - max_mean) < 1e-6:
                    algorithm_wins[algo] += 1
        else:
            max_mean = None

        # Format cells
        for i, algo in enumerate(algorithms):
            if row_means[i] is not None:
                mean = row_means[i]
                std = pivot_std.loc[dataset, algo] if dataset in pivot_std.index else 0
                cell = format_mean_std(mean, std)

                # Bold the best value(s) - allow for small floating point differences
                if max_mean is not None and abs(mean - max_mean) < 1e-6:
                    cell = bold_text(cell)
            else:
                cell = "---"

            row_values.append(cell)

        # Format dataset name
        formatted_dataset = dataset.replace("_", "\\_")
        row = f"{formatted_dataset} & " + " & ".join(row_values) + " \\\\"
        latex.append(row)

    # Compute overall statistics (average across datasets)
    latex.append("\\midrule")
    avg_means = []
    avg_row_values = []

    for algo in algorithms:
        if algorithm_scores[algo]:
            avg_mean = np.mean(algorithm_scores[algo])
            avg_means.append(avg_mean)
        else:
            avg_means.append(None)

    # Find best overall average
    valid_avg_means = [m for m in avg_means if m is not None]
    max_avg_mean = max(valid_avg_means) if valid_avg_means else None

    for i, algo in enumerate(algorithms):
        if avg_means[i] is not None:
            mean_val = avg_means[i]
            cell = f"${mean_val:.3f}$"

            if max_avg_mean is not None and abs(mean_val - max_avg_mean) < 1e-6:
                cell = bold_text(cell)
        else:
            cell = "---"

        avg_row_values.append(cell)

    avg_row = "\\textit{Average} & " + " & ".join(avg_row_values) + " \\\\"
    latex.append(avg_row)

    # Add wins row for comparison
    if len(algorithms) > 1:
        latex.append("\\midrule")
        wins_row_values = []
        max_wins = max(algorithm_wins.values()) if algorithm_wins.values() else 0

        for algo in algorithms:
            wins = algorithm_wins[algo]
            cell = f"{wins}"
            if wins == max_wins and max_wins > 0:
                cell = bold_text(cell)
            wins_row_values.append(cell)

        wins_row = "\\textit{Wins} & " + " & ".join(wins_row_values) + " \\\\"
        latex.append(wins_row)

    # Table footer
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    # Add note about wins if multiple algorithms
    if len(algorithms) > 1:
        latex.append("\\\\[0.5em]")
        latex.append(
            "\\footnotesize \\textit{Wins}: Number of datasets where algorithm achieved best performance."
        )

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

    # Check available columns
    available_cols = set(df.columns)

    # Auto-detect metrics from CSV columns
    # Try common metric column patterns
    potential_metrics = [
        ("accuracy", "Clustering Accuracy"),
        ("best_acc", "Clustering Accuracy"),
        ("nmi", "Normalized Mutual Information"),
        ("best_nmi", "Normalized Mutual Information"),
        ("purity", "Purity Score"),
        ("best_purity", "Purity Score"),
        ("ari", "Adjusted Rand Index"),
        ("best_ari", "Adjusted Rand Index"),
        ("rand_index", "Rand Index"),
        ("best_ri", "Rand Index"),
    ]

    # Filter to only available columns
    metrics = [(col, name) for col, name in potential_metrics if col in available_cols]

    if not metrics:
        print(f"Warning: No recognized metric columns found!")
        print(f"Available columns: {list(available_cols)}")
        # Use all numeric columns except dataset and algorithm
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        excluded_cols = {"dataset", "algorithm", "linkage", "max_psi"}
        metrics = [
            (col, col.title().replace("_", " "))
            for col in numeric_cols
            if col not in excluded_cols
        ]

    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    # Print comprehensive summary statistics
    print("Summary Statistics:")
    print("=" * 70)

    # Count runs per dataset
    dataset_counts = df.groupby("dataset").size()
    print("Runs per dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count} runs")

    # Determine algorithm column
    algo_col = "algorithm" if "algorithm" in df.columns else "linkage"

    # Count runs per algorithm
    print(f"\nRuns per {algo_col}:")
    algo_counts = df.groupby(algo_col).size()
    for algo, count in sorted(algo_counts.items()):
        print(f"  {algo}: {count} runs")

    # Dataset-Algorithm matrix
    print(f"\nDataset-{algo_col.title()} coverage:")
    coverage = df.groupby(["dataset", algo_col]).size().unstack(fill_value=0)
    print(coverage)

    print(f"\nTotal datasets: {len(dataset_counts)}")
    print(f"Total algorithms: {len(algo_counts)}")
    print(f"Total runs: {len(df)}")

    # Algorithm comparison summary if multiple algorithms
    if len(algo_counts) > 1:
        print(f"\n{'Algorithm Comparison Summary'}")
        print("-" * 40)

        # For each metric, show which algorithm wins most often
        for metric_col, metric_name in metrics:
            if metric_col in available_cols:
                # Calculate wins per algorithm
                wins_per_algo = {}
                datasets_in_metric = df["dataset"].unique()

                for dataset in datasets_in_metric:
                    dataset_data = df[df["dataset"] == dataset]
                    if len(dataset_data) > 0:
                        # Get mean performance per algorithm for this dataset
                        algo_means = dataset_data.groupby(algo_col)[metric_col].mean()
                        if len(algo_means) > 0:
                            best_score = algo_means.max()
                            winners = algo_means[
                                abs(algo_means - best_score) < 1e-6
                            ].index.tolist()
                            for winner in winners:
                                wins_per_algo[winner] = wins_per_algo.get(
                                    winner, 0
                                ) + 1 / len(winners)

                if wins_per_algo:
                    print(f"\n{metric_name} - Dataset wins:")
                    for algo, wins in sorted(
                        wins_per_algo.items(), key=lambda x: x[1], reverse=True
                    ):
                        print(f"  {algo}: {wins:.1f} wins")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
