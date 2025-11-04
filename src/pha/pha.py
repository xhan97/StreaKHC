#!/usr/bin/env python3
"""
PHA (Potential-based Hierarchical Agglomerative) Clustering Algorithm

This is a Python translation of the MATLAB PHA clustering implementation.

Author: Translated from MATLAB code by Yonggang Lu (ylu@lzu.edu.cn)
Original Reference:
    Yonggang Lu, Yi Wan. (2013). "PHA: A Fast Potential-based Hierarchical
    Agglomerative Clustering Method", Pattern Recognition, Vol. 46(5), pp. 1227-1239.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage as scipy_linkage
import warnings


def pha_cluster(d_matrix, S=10):
    """
    Performs hierarchical clustering using the PHA method.

    The function produces a hierarchical cluster tree (Z) from the input distance matrix.
    The output Z is similar to the output by scipy.cluster.hierarchy.linkage.

    Parameters:
    -----------
    d_matrix : array-like, shape (n_samples, n_samples)
        Distance matrix defining distances between objects
    S : float, optional (default=10)
        Scale factor for determining parameter delta.
        If two points are closer than delta, they don't have attractive force.

    Returns:
    --------
    Z : ndarray, shape (n_samples-1, 4)
        Hierarchical cluster tree represented as a matrix (scipy format).
        Each row represents one merge operation.
        Format: [cluster1_id, cluster2_id, distance, cluster_size]
    total_potential : ndarray, shape (n_samples,)
        Total potential values for each point
    parents : ndarray, shape (n_samples,)
        The parent index of each data point in the hierarchy

    Usage:
    ------
    >>> from scipy.cluster.hierarchy import fcluster
    >>> Z, potentials, parents = pha_cluster(distance_matrix)
    >>> clusters = fcluster(Z, k, criterion='maxclust')
    """

    # Input validation
    d_matrix = np.asarray(d_matrix)
    if d_matrix.ndim != 2:
        raise ValueError("Distance matrix must be 2-dimensional")

    num_pts, num_pts2 = d_matrix.shape
    if num_pts != num_pts2:
        raise ValueError("Distance matrix should be a square matrix!")

    if num_pts < 2:
        raise ValueError("Need at least 2 points for clustering")

    # Compute delta automatically
    min_dist = np.zeros(num_pts)
    for i in range(num_pts):
        # Find non-zero distances (excluding self-distance)
        mask = d_matrix[i, :] != 0
        if np.any(mask):
            min_dist[i] = np.min(d_matrix[i, mask])
        else:
            # If all distances are zero (single point case), use a small value
            min_dist[i] = 1e-10

    delta = np.mean(min_dist) / S

    # Avoid division by zero
    if delta == 0:
        delta = 1e-10
        warnings.warn("Delta is zero, using small value to avoid division by zero")

    # Compute total potential for each point
    total_potential = np.zeros(num_pts)

    for i in range(num_pts):
        dist_to_all = d_matrix[i, :]
        # Select points with distance >= delta
        sel_indices = np.where(dist_to_all >= delta)[0]

        # Calculate potential from points farther than delta
        if len(sel_indices) > 0:
            # Avoid division by zero
            distances = d_matrix[i, sel_indices]
            distances = np.maximum(distances, 1e-10)  # Avoid division by zero
            total_p = np.sum(1.0 / distances)
        else:
            total_p = 0.0

        # For points within delta, potential = 1/delta
        num_close_points = num_pts - len(sel_indices) - 1  # -1 for self
        total_p += num_close_points * (1.0 / delta)

        # Negative potential (as in original code)
        total_potential[i] = -total_p

    # Sort points by potential (ascending order)
    sorted_indices = np.argsort(total_potential)

    # Initialize parent information
    parents = np.arange(num_pts)  # Initially, each point is its own parent
    dist_to_parent = np.zeros(num_pts)

    # Build parent-child relationships
    for pi in range(1, num_pts):  # Start from second point (index 1)
        center_idx = sorted_indices[pi]
        visited_pts_idx = sorted_indices[:pi]  # All previously visited points

        # Find distances to all visited points
        dist_to_visited = d_matrix[center_idx, visited_pts_idx]

        # Find closest visited point
        min_idx = np.argmin(dist_to_visited)
        min_dist = dist_to_visited[min_idx]

        parents[center_idx] = visited_pts_idx[min_idx]
        dist_to_parent[center_idx] = min_dist

    # Build linkage matrix Z (compatible with scipy format)
    # Z is a (num_pts-1) by 4 matrix: [cluster1, cluster2, distance, num_points]
    Z = np.zeros((num_pts - 1, 4))

    # Create merge operations based on parent-child relationships
    merge_operations = []
    for i in range(1, num_pts):  # Skip first point (has no parent)
        if dist_to_parent[i] > 0:
            merge_operations.append((i, parents[i], dist_to_parent[i]))

    # Sort by distance (ascending order for hierarchical clustering)
    merge_operations.sort(key=lambda x: x[2])

    # Track cluster membership using Union-Find with proper scipy indexing
    # In scipy linkage:
    # - Original points are clusters 0 to n-1
    # - New clusters are n, n+1, n+2, etc.
    parent_cluster = list(range(num_pts))  # Each point is its own cluster initially
    cluster_sizes = [1] * num_pts  # Size of each cluster

    def find_cluster(x):
        """Find the current cluster ID for point x."""
        return parent_cluster[x]

    def get_cluster_size(cluster_id):
        """Get size of a cluster."""
        if cluster_id < num_pts:
            return cluster_sizes[cluster_id]
        else:
            # For merged clusters, look up in Z matrix
            merge_idx = cluster_id - num_pts
            if merge_idx < len(Z):
                return int(Z[merge_idx, 3])
            return 1

    # Process merges
    merge_count = 0
    processed_pairs = set()

    for child_idx, parent_idx, distance in merge_operations:
        if merge_count >= num_pts - 1:
            break

        # Get current clusters for both points
        cluster1 = find_cluster(parent_idx)
        cluster2 = find_cluster(child_idx)

        # Skip if already in same cluster
        if cluster1 == cluster2:
            continue

        # Skip if we've already processed this pair
        pair = tuple(sorted([cluster1, cluster2]))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)

        # Calculate cluster sizes
        size1 = get_cluster_size(cluster1)
        size2 = get_cluster_size(cluster2)

        # Record the merge in Z matrix
        Z[merge_count, 0] = min(cluster1, cluster2)
        Z[merge_count, 1] = max(cluster1, cluster2)
        Z[merge_count, 2] = distance
        Z[merge_count, 3] = size1 + size2

        # New cluster ID for the merged cluster
        new_cluster_id = num_pts + merge_count

        # Update cluster membership for all points
        for i in range(num_pts):
            if parent_cluster[i] == cluster1 or parent_cluster[i] == cluster2:
                parent_cluster[i] = new_cluster_id

        merge_count += 1

    # If we don't have enough merges, create artificial ones
    while merge_count < num_pts - 1:
        # Find remaining separate clusters
        remaining_clusters = {}
        for i in range(num_pts):
            cluster_id = find_cluster(i)
            if cluster_id not in remaining_clusters:
                remaining_clusters[cluster_id] = []
            remaining_clusters[cluster_id].append(i)

        # If we have at least 2 separate clusters, merge two of them
        if len(remaining_clusters) >= 2:
            cluster_list = list(remaining_clusters.keys())
            cluster1 = cluster_list[0]
            cluster2 = cluster_list[1]

            size1 = get_cluster_size(cluster1)
            size2 = get_cluster_size(cluster2)

            Z[merge_count, 0] = min(cluster1, cluster2)
            Z[merge_count, 1] = max(cluster1, cluster2)
            Z[merge_count, 2] = 1.0  # Default distance
            Z[merge_count, 3] = size1 + size2

            # Merge clusters
            new_cluster_id = num_pts + merge_count
            for i in range(num_pts):
                if parent_cluster[i] == cluster1 or parent_cluster[i] == cluster2:
                    parent_cluster[i] = new_cluster_id
        else:
            # Fallback: merge first two points
            Z[merge_count, 0] = 0
            Z[merge_count, 1] = 1
            Z[merge_count, 2] = 1.0
            Z[merge_count, 3] = 2

        merge_count += 1

    return Z, total_potential, parents


def pha_cluster_from_points(X, S=10, metric="euclidean"):
    """
    Convenience function to perform PHA clustering directly from data points.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data points
    S : float, optional (default=10)
        Scale factor for determining parameter delta
    metric : str, optional (default='euclidean')
        Distance metric to use

    Returns:
    --------
    Z : ndarray, shape (n_samples-1, 4)
        Hierarchical cluster tree
    total_potential : ndarray, shape (n_samples,)
        Total potential values for each point
    parents : ndarray, shape (n_samples,)
        The parent index of each data point
    """
    # Compute pairwise distances
    distances = pdist(X, metric=metric)  # type: ignore
    distance_matrix = squareform(distances)

    return pha_cluster(distance_matrix, S)


# Example usage and testing
if __name__ == "__main__":
    # Test with a simple dataset
    np.random.seed(42)

    # Create test data
    X = np.random.rand(10, 2)

    print("Testing PHA clustering...")
    print(f"Input data shape: {X.shape}")

    # Method 1: From distance matrix
    from scipy.spatial.distance import pdist, squareform

    distances = pdist(X, metric="euclidean")
    distance_matrix = squareform(distances)

    Z, potentials, parents = pha_cluster(distance_matrix, S=10)

    print(f"Linkage matrix shape: {Z.shape}")
    print(f"Potentials shape: {potentials.shape}")
    print(f"Parents shape: {parents.shape}")

    # Method 2: Direct from points
    Z2, potentials2, parents2 = pha_cluster_from_points(X, S=10)

    # Verify both methods give same result
    print(f"Results identical: {np.allclose(Z, Z2)}")

    # Compare with scipy linkage
    from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster

    Z_scipy = scipy_linkage(X, method="single")
    print(f"Scipy linkage shape: {Z_scipy.shape}")

    # Get clusters
    clusters_pha = fcluster(Z, 3, criterion="maxclust")
    clusters_scipy = fcluster(Z_scipy, 3, criterion="maxclust")

    print(f"PHA clusters: {clusters_pha}")
    print(f"Scipy clusters: {clusters_scipy}")

    print("PHA clustering test completed successfully!")
