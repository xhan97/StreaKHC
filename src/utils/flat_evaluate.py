# Copyright 2025 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional
import numpy as np
import math
from src.streakhc.INode import INode
import heapq
import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment
from numba import jit


@jit(nopython=True)
def _fast_dot(x, y):
    """Compute the dot product of x and y using numba.

    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).

    Returns:
    x_T.y
    """
    return np.dot(x, y)


@jit(nopython=True)
def _fast_normalize_dot(x, y):
    """Compute the dot product of x and y using numba.

    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).
    t - an integel

    Returns:
    Normalized x_T.y
    """

    return _fast_dot(x, y) / (math.sqrt(_fast_dot(x, x)) * (math.sqrt(_fast_dot(y, y))))


def get_children_similarity(node: INode) -> float:
    """Get the similarity between the two children of a node.

    Args:
        node: The node to get the similarity from.
    Returns:
        The similarity between the two children.
    """
    if node.is_leaf():
        return 0.0
    left_child_ikv = node.children[0].ikv
    right_child_ikv = node.children[1].ikv
    curr_similarity = _fast_normalize_dot(left_child_ikv, right_child_ikv)

    if curr_similarity == 0.0:
        return 0.0
    return curr_similarity


def cut_tree(root: INode, n_cluster: Optional[int] = None) -> tuple[list, list]:
    """Extract subtree from root with n_cluster clusters.

    Args:
        root: The root node of the tree.
        n_cluster: The number of clusters to extract.
    Returns:
        The extracted labels and ground_truth labels.
    """
    curr_root = root.root()

    if n_cluster is None:
        # If n_cluster is None, return all leaves as clusters
        gt_labels = []
        for leaf in curr_root.leaves():
            gt_labels.append(leaf.pts[0][0])  # True label
        n_cluster = len(set(gt_labels))

    # Use a min-heap to track internal nodes by their similarity
    # (similarity, node_id, node) - lower similarity = higher priority to split
    heap = []

    if not curr_root.is_leaf():
        similarity = get_children_similarity(curr_root)
        heapq.heappush(heap, (similarity, curr_root.id, curr_root))

    # Keep track of cluster nodes (initially just the root)
    cluster_nodes = [curr_root]

    # Iteratively split clusters until we reach n_cluster
    while len(cluster_nodes) < n_cluster and heap:
        # Pop the node with lowest children similarity (most different children)
        _, _, node_to_split = heapq.heappop(heap)

        # Remove the node from cluster_nodes
        cluster_nodes.remove(node_to_split)

        # Add its children as new clusters
        left_child = node_to_split.children[0]
        right_child = node_to_split.children[1]
        cluster_nodes.extend([left_child, right_child])

        # If children are internal nodes, add them to the heap
        if not left_child.is_leaf():
            left_similarity = get_children_similarity(left_child)
            heapq.heappush(heap, (left_similarity, left_child.id, left_child))

        if not right_child.is_leaf():
            right_similarity = get_children_similarity(right_child)
            heapq.heappush(heap, (right_similarity, right_child.id, right_child))

    # Extract labels from the cluster nodes
    labels = []
    ground_truth = []

    for cluster_id, cluster_node in enumerate(cluster_nodes):
        # Get all leaf nodes under this cluster
        leaves = cluster_node.leaves()
        for leaf in leaves:
            ground_truth.append(leaf.pts[0][0])  # True label
            labels.append(cluster_id)  # Assigned cluster

    return labels, ground_truth


def get_contingency_matrix(y_true, y_pred):
    """Helper method to calculate contingency matrix once.

    Returns:
        Contingency matrix of shape (n_classes_true, n_classes_pred)
        where element [i, j] is the number of samples with true label i
        that are assigned to cluster j.
    """
    return metrics.cluster.contingency_matrix(y_true, y_pred)


def accuracy_score(y_true, y_pred, contingency_matrix=None):
    """Calculate clustering accuracy using Hungarian algorithm.

    Finds the best one-to-one mapping between true labels and predicted clusters,
    then computes accuracy based on this optimal assignment.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
        contingency_matrix: Optional pre-computed contingency matrix

    Returns:
        Clustering accuracy in [0, 1]
    """
    if contingency_matrix is None:
        contingency_matrix = get_contingency_matrix(y_true, y_pred)

    # Ensure it's a dense array for compatibility
    contingency_matrix = np.asarray(contingency_matrix)

    # Use Hungarian algorithm to find optimal assignment
    # Maximize the sum by using negative values
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Sum the correctly assigned samples and divide by total
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def purity_score(y_true, y_pred, contingency_matrix=None):
    """Calculate purity score for clustering.

    For each predicted cluster, finds the most common true label and
    computes the fraction of correctly assigned samples.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
        contingency_matrix: Optional pre-computed contingency matrix

    Returns:
        Purity score in [0, 1]
    """
    if contingency_matrix is None:
        contingency_matrix = get_contingency_matrix(y_true, y_pred)

    # Ensure it's a dense array for compatibility
    contingency_matrix = np.asarray(contingency_matrix)

    # For each cluster (column), take the max count across all true labels
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def nmi_score(y_true, y_pred):
    """Calculate Normalized Mutual Information score.

    Measures the mutual information between true labels and predicted clusters,
    normalized to [0, 1] range.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels

    Returns:
        NMI score in [0, 1]
    """
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def ari_score(y_true, y_pred):
    """Calculate Adjusted Rand Index score.

    Measures similarity between true labels and predicted clusters,
    adjusted for chance grouping.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
    Returns:
        ARI score in [-1, 1]
    """
    return metrics.adjusted_rand_score(y_true, y_pred)


def rand_index_score(y_true, y_pred):
    """Calculate Rand Index score.

    Measures similarity between true labels and predicted clusters.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
    Returns:
        RI score in [0, 1]
    """
    return metrics.rand_score(y_true, y_pred)
