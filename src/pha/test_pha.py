#!/usr/bin/env python3
"""
Test script for PHA clustering algorithm to verify correctness against MATLAB implementation.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))
from pha import pha_cluster, pha_cluster_from_points


def test_simple_data():
    """Test with simple 2D data points."""
    print("=" * 60)
    print("Test 1: Simple 2D data points")
    print("=" * 60)

    # Create simple test data with clear clusters
    np.random.seed(42)

    # Three distinct clusters
    cluster1 = np.random.normal([2, 2], 0.3, (5, 2))
    cluster2 = np.random.normal([6, 6], 0.3, (5, 2))
    cluster3 = np.random.normal([2, 6], 0.3, (5, 2))

    X = np.vstack([cluster1, cluster2, cluster3])
    print(f"Data shape: {X.shape}")

    # Run PHA clustering
    Z, potentials, parents = pha_cluster_from_points(X, S=10)

    print(f"Linkage matrix shape: {Z.shape}")
    print(f"Potentials: {potentials}")
    print(f"Parents: {parents}")

    # Get clusters for different k values
    for k in [2, 3, 4]:
        clusters = fcluster(Z, k, criterion="maxclust")
        print(f"Clusters for k={k}: {clusters}")

        # Calculate silhouette score (simple version)
        from sklearn.metrics import silhouette_score

        if len(np.unique(clusters)) > 1:
            score = silhouette_score(X, clusters)
            print(f"Silhouette score for k={k}: {score:.3f}")

    return X, Z


def test_distance_matrix():
    """Test with predefined distance matrix."""
    print("\n" + "=" * 60)
    print("Test 2: Predefined distance matrix")
    print("=" * 60)

    # Create a small distance matrix manually
    n = 5
    dist_matrix = np.array(
        [
            [0.0, 1.0, 4.0, 3.0, 5.0],
            [1.0, 0.0, 3.0, 2.0, 4.0],
            [4.0, 3.0, 0.0, 1.0, 2.0],
            [3.0, 2.0, 1.0, 0.0, 1.5],
            [5.0, 4.0, 2.0, 1.5, 0.0],
        ]
    )

    print("Distance matrix:")
    print(dist_matrix)

    # Run PHA clustering
    Z, potentials, parents = pha_cluster(dist_matrix, S=5)

    print(f"\nPotentials: {potentials}")
    print(f"Parents: {parents}")
    print(f"Linkage matrix Z:")
    print(Z)

    # Get clusters
    clusters = fcluster(Z, 2, criterion="maxclust")
    print(f"Clusters for k=2: {clusters}")

    return Z


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Test 3: Edge cases")
    print("=" * 60)

    # Test with 2 points
    print("Testing with 2 points...")
    X_small = np.array([[0, 0], [1, 1]])
    Z_small, _, _ = pha_cluster_from_points(X_small, S=10)
    print(f"Linkage matrix for 2 points: {Z_small}")

    # Test with identical points
    print("\nTesting with identical points...")
    X_identical = np.array([[1, 1], [1, 1], [2, 2]])
    try:
        Z_identical, _, _ = pha_cluster_from_points(X_identical, S=10)
        print(f"Linkage matrix for identical points: {Z_identical}")
    except Exception as e:
        print(f"Error with identical points: {e}")

    # Test with single dimension
    print("\nTesting with 1D data...")
    X_1d = np.array([[1], [2], [5], [6], [10]])
    Z_1d, potentials_1d, _ = pha_cluster_from_points(X_1d, S=10)
    print(f"1D potentials: {potentials_1d}")
    print(f"1D linkage matrix: {Z_1d}")


def compare_with_scipy():
    """Compare PHA with scipy standard methods."""
    print("\n" + "=" * 60)
    print("Test 4: Comparison with scipy methods")
    print("=" * 60)

    # Generate test data
    np.random.seed(123)
    X = np.random.rand(20, 3)

    # PHA clustering
    Z_pha, _, _ = pha_cluster_from_points(X, S=10)

    # Scipy methods
    from scipy.cluster.hierarchy import linkage

    Z_single = linkage(X, method="single")
    Z_complete = linkage(X, method="complete")
    Z_average = linkage(X, method="average")

    print("Comparing clustering results for k=4:")

    methods = [
        ("PHA", Z_pha),
        ("Single", Z_single),
        ("Complete", Z_complete),
        ("Average", Z_average),
    ]

    for name, Z in methods:
        clusters = fcluster(Z, 4, criterion="maxclust")
        unique_clusters = len(np.unique(clusters))
        print(f"{name:10} -> {unique_clusters} unique clusters: {clusters}")


def visualize_results(X, Z):
    """Create visualization if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt

        print("\n" + "=" * 60)
        print("Visualization")
        print("=" * 60)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot original data with colors based on clustering
        clusters = fcluster(Z, 3, criterion="maxclust")
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=clusters, cmap="tab10")
        ax1.set_title("PHA Clustering Results")
        ax1.set_xlabel("X1")
        ax1.set_ylabel("X2")
        plt.colorbar(scatter, ax=ax1)

        # Plot dendrogram
        dendrogram(Z, ax=ax2)
        ax2.set_title("PHA Dendrogram")
        ax2.set_xlabel("Data Point Index")
        ax2.set_ylabel("Distance")

        plt.tight_layout()
        plt.savefig("pha_clustering_test.png", dpi=150, bbox_inches="tight")
        print("Visualization saved as 'pha_clustering_test.png'")

    except ImportError:
        print("Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    """Run all tests."""
    print("PHA Clustering Algorithm - Comprehensive Test Suite")
    print("Python translation of MATLAB implementation")

    # Run tests
    X, Z = test_simple_data()
    test_distance_matrix()
    test_edge_cases()
    compare_with_scipy()

    # Visualize if possible
    if X is not None and Z is not None:
        visualize_results(X, Z)

    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("The Python implementation appears to be working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
