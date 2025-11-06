"""
Grinch clustering utilities for tree cutting and evaluation.
"""

import heapq
from typing import Optional, Tuple, List
import numpy as np
from src.grinch.grinch import Grinch


def get_children_similarity_grinch(grinch: Grinch, node_id: int) -> float:
    """Get the similarity between the two children of a node in Grinch.

    Args:
        grinch: The Grinch clustering object.
        node_id: The node ID to get the similarity from.
    Returns:
        The similarity between the two children.
    """
    if grinch.is_leaf(node_id):
        return 0.0

    # Get the linkage score (similarity) for this internal node
    similarity = grinch.get_score(node_id)
    return similarity


def cut_tree_grinch(
    grinch: Grinch, data_list: list, n_cluster: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """Extract subtree from Grinch with n_cluster clusters.

    Args:
        grinch: The Grinch clustering object.
        data_list: The original data list with labels.
        n_cluster: The number of clusters to extract.
    Returns:
        Tuple of (predicted_labels, true_labels).
    """
    # Get true labels from data
    y_true = [
        pt[1] for pt in data_list
    ]  # Extract labels from [vector, label, point_id]

    if n_cluster is None:
        # If n_cluster is None, determine automatically from true labels
        n_cluster = len(set(y_true))

    if grinch.use_gnodes:
        # Use GNode-based tree structure
        root_gnode = grinch.get_gnode_root()
        if root_gnode is None:
            y_hat = list(range(len(y_true)))
            return y_hat, y_true

        # Try different thresholds to get approximately n_cluster clusters
        best_threshold = 0.5
        best_assignments = None
        best_diff = float("inf")

        # Try a range of thresholds
        for threshold in np.linspace(0.1, 0.9, 20):
            assignments = grinch.gnode_flat_clustering(threshold)
            if assignments:
                n_clusters_found = len(set(assignments.values()))
                diff = abs(n_clusters_found - n_cluster)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold
                    best_assignments = assignments

        if best_assignments is None:
            # Fallback: each point is its own cluster
            y_hat = list(range(len(y_true)))
        else:
            # Convert assignments dictionary to list
            y_hat = []
            for i in range(len(y_true)):
                y_hat.append(
                    best_assignments.get(i, i)
                )  # Default to point ID if not found

        return y_hat, y_true

    else:
        # Original array-based implementation
        root_id = grinch.root()

        # Collect all internal node scores
        all_scores = []
        for node_id in range(grinch.max_num_points, grinch.next_node_id):
            if grinch.parent[node_id] != -2:  # Not deleted
                if not grinch.is_leaf(node_id):
                    score = grinch.get_score(node_id)
                    if np.isfinite(score):
                        all_scores.append(score)

        if len(all_scores) == 0:
            # Fallback: each point is its own cluster
            y_hat = list(range(len(y_true)))
            return y_hat, y_true

        # Sort scores and find a threshold that gives approximately n_cluster clusters
        all_scores.sort(reverse=True)  # Higher scores first

        best_threshold = None
        best_diff = float("inf")

        # Try different thresholds
        for i in range(len(all_scores)):
            threshold = all_scores[i]
            assignments = grinch.flat_clustering(threshold)
            if isinstance(assignments, np.ndarray):
                valid_assignments = assignments[assignments >= 0]
                n_clusters_found = len(set(valid_assignments))

                diff = abs(n_clusters_found - n_cluster)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold

        # If no good threshold found, use median
        if best_threshold is None:
            best_threshold = np.median(all_scores) if all_scores else 0.5

        # Get final cluster assignments
        assignments = grinch.flat_clustering(best_threshold)

        if isinstance(assignments, np.ndarray):
            y_hat = assignments.copy()

            # Handle case where some points are not assigned (assignment == -1)
            max_cluster_id = int(max(y_hat)) if len(y_hat) > 0 else 0
            for i in range(len(y_hat)):
                if y_hat[i] == -1:
                    max_cluster_id += 1
                    y_hat[i] = max_cluster_id

            # Ensure we have the right number of points
            if len(y_hat) != len(y_true):
                # Pad or truncate as needed
                if len(y_hat) < len(y_true):
                    # Add missing assignments
                    for i in range(len(y_hat), len(y_true)):
                        max_cluster_id += 1
                        y_hat = np.append(y_hat, max_cluster_id)
                else:
                    # Truncate
                    y_hat = y_hat[: len(y_true)]

            return y_hat.tolist(), y_true
        else:
            # Fallback
            y_hat = list(range(len(y_true)))
            return y_hat, y_true


def extract_grinch_dendrogram(grinch: Grinch) -> Tuple[np.ndarray, List[int]]:
    """Extract dendrogram information from Grinch tree.

    Args:
        grinch: The Grinch clustering object.

    Returns:
        Tuple of (linkage_matrix, leaf_labels) compatible with scipy.hierarchy format.
    """
    # This is a more complex function that would convert Grinch's tree structure
    # to a scipy-compatible linkage matrix. For now, we'll use a placeholder.
    # TODO: Implement proper dendrogram extraction
    n_points = grinch.point_counter
    if n_points < 2:
        return np.array([]), []

    # Create a simple linkage matrix as placeholder
    linkage_matrix = np.zeros((n_points - 1, 4))
    for i in range(n_points - 1):
        linkage_matrix[i] = [i, i + 1, 1.0, 2]

    leaf_labels = list(range(n_points))
    return linkage_matrix, leaf_labels


def get_grinch_tree_stats(grinch: Grinch) -> dict:
    """Get statistics about the Grinch tree.

    Args:
        grinch: The Grinch clustering object.

    Returns:
        Dictionary with tree statistics.
    """
    stats = {
        "num_points": grinch.point_counter,
        "num_internal_nodes": grinch.next_node_id - grinch.max_num_points,
        "max_nodes": grinch.max_nodes,
        "tree_height": 0,  # Would need to calculate
        "total_nodes": grinch.next_node_id,
    }

    return stats
