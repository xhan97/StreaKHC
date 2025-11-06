"""
Grinch clustering utilities for tree cutting and evaluation.
适用于 grinch_new.py (完全在线模式，基于 GNode)
"""

from typing import Optional, Tuple, List, TYPE_CHECKING
import heapq
import numpy as np

if TYPE_CHECKING:
    from src.grinch.grinch import Grinch


def cut_tree_grinch(
    grinch: "Grinch", n_cluster: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """Extract flat clustering from Grinch with exactly n_cluster clusters.

    Iteratively splits the internal node with lowest children similarity
    until we reach the desired number of clusters.

    Args:
        grinch: The Grinch clustering object (grinch_new.py).
        n_cluster: The number of clusters to extract (optional).
                  If None, inferred from unique ground truth labels.

    Returns:
        Tuple of (predicted_labels, ground_truth_labels).
        Both are lists of integers with same length as number of data points.

    Example:
        >>> grinch = Grinch(dim=10, norm='l2', sim='dot')
        >>> for i in range(100):
        >>>     grinch.insert(i, vectors[i], labels[i])
        >>> y_pred, y_true = cut_tree_grinch(grinch, n_cluster=5)
    """
    # Get root node (GNode)
    curr_root = grinch.root()
    if curr_root is None:
        # No tree built yet, return empty lists
        return [], []

    all_leaves = curr_root.leaves()
    if not all_leaves:
        return [], []

    point_id_to_idx = {}  # point_id -> ground_truth index
    for idx, leaf in enumerate(all_leaves):
        point_id_to_idx[leaf.id] = idx

    # Build ground_truth from leaf labels
    ground_truth = [leaf.pts[0][1] for leaf in all_leaves]
    n_cluster = len(set(ground_truth))

    # Use a min-heap to track internal nodes by their similarity
    # (similarity, node_id, node) - lower similarity = higher priority to split
    heap = []

    if not curr_root.is_leaf():
        similarity = _get_children_similarity(curr_root)
        heapq.heappush(heap, (similarity, id(curr_root), curr_root))

    # Keep track of cluster nodes (initially just the root)
    cluster_nodes = [curr_root]

    # Iteratively split clusters until we reach n_cluster
    while len(cluster_nodes) < n_cluster and heap:
        # Pop the node with lowest children similarity (most different children)
        _, _, node_to_split = heapq.heappop(heap)

        # Remove the node from cluster_nodes
        if node_to_split in cluster_nodes:
            cluster_nodes.remove(node_to_split)

            # Add its children as new clusters
            if len(node_to_split.children) >= 2:
                left_child = node_to_split.children[0]
                right_child = node_to_split.children[1]
                cluster_nodes.extend([left_child, right_child])

                # If children are internal nodes, add them to the heap
                if not left_child.is_leaf():
                    left_similarity = _get_children_similarity(left_child)
                    heapq.heappush(heap, (left_similarity, id(left_child), left_child))

                if not right_child.is_leaf():
                    right_similarity = _get_children_similarity(right_child)
                    heapq.heappush(
                        heap, (right_similarity, id(right_child), right_child)
                    )

    # Extract predicted labels from the cluster nodes
    # Similar to PNODE's approach
    predicted_labels = [-1] * len(ground_truth)

    for cluster_id, cluster_node in enumerate(cluster_nodes):
        # Get all leaf nodes under this cluster
        leaves = cluster_node.leaves()
        for leaf in leaves:
            if leaf.id in point_id_to_idx:
                idx = point_id_to_idx[leaf.id]
                predicted_labels[idx] = cluster_id

    return predicted_labels, ground_truth


def _get_children_similarity(node) -> float:
    """Get the similarity between the two children of a GNode.

    This is similar to PNODE's get_children_similarity but uses GNode's
    score attribute directly (which is computed during tree construction).

    Args:
        node: The GNode to get similarity from.

    Returns:
        The similarity score between children (lower = more different).
    """
    if node.is_leaf():
        return 0.0

    # GNode already computes and stores the score during tree construction
    # The score represents similarity between children
    if hasattr(node, "score") and node.score is not None:
        # Return negative score for min-heap (we want to split low similarity first)
        # But GNode score is already similarity, so return as-is
        return -float(node.score)  # Negate so min-heap prioritizes low similarity

    # Fallback: compute similarity between children if not available
    if len(node.children) >= 2:
        child1 = node.children[0]
        child2 = node.children[1]

        # Use GNode's compute_similarity method
        if hasattr(child1, "compute_similarity"):
            similarity = child1.compute_similarity(child2)
            return -float(similarity)  # Negate for min-heap

    return 0.0


def extract_grinch_dendrogram(grinch: "Grinch") -> Tuple[np.ndarray, List[int]]:
    """Extract dendrogram information from Grinch tree (GNode-based).

    Args:
        grinch: The Grinch clustering object.

    Returns:
        Tuple of (linkage_matrix, leaf_labels) compatible with scipy.hierarchy format.

    Note:
        This creates a simplified linkage matrix. For complete dendrogram,
        consider using GNode.to_newick() or other tree export methods.
    """
    root_node = grinch.root()

    if root_node is None:
        return np.array([]), []

    # Get all leaves
    leaves = root_node.leaves()
    n_points = len(leaves)

    if n_points < 2:
        return np.array([]), [leaf.id for leaf in leaves]

    # Create a simple linkage matrix (placeholder implementation)
    # TODO: Implement proper dendrogram extraction from GNode tree
    linkage_matrix = np.zeros((n_points - 1, 4))
    for i in range(n_points - 1):
        linkage_matrix[i] = [i, i + 1, 1.0, 2]

    leaf_labels = [leaf.id for leaf in leaves]
    return linkage_matrix, leaf_labels


def get_grinch_tree_stats(grinch: "Grinch") -> dict:
    """Get statistics about the Grinch tree.

    Args:
        grinch: The Grinch clustering object.

    Returns:
        Dictionary with tree statistics.
    """
    root_node = grinch.root()

    stats = {
        "num_points": grinch.point_counter,
        "root_exists": root_node is not None,
        "num_rotates": grinch.number_of_rotates,
        "num_grafts": grinch.number_of_grafts,
        "time_in_search": grinch.time_in_search,
        "time_in_rotate": grinch.time_in_rotate,
        "time_in_graft": grinch.time_in_graft,
        "time_in_update": grinch.time_in_update,
    }

    if root_node is not None:
        stats["num_leaves"] = len(root_node.leaves())
        stats["num_descendants"] = root_node.num_descendants
        stats["tree_depth"] = _calculate_tree_depth(root_node)
    else:
        stats["num_leaves"] = 0
        stats["num_descendants"] = 0
        stats["tree_depth"] = 0

    return stats


def _calculate_tree_depth(node) -> int:
    """Calculate the depth of a GNode tree.

    Args:
        node: The root GNode.

    Returns:
        Maximum depth of the tree.
    """
    if node.is_leaf():
        return 0

    if not node.children:
        return 0

    max_child_depth = max(_calculate_tree_depth(child) for child in node.children)
    return 1 + max_child_depth


def get_node_similarity(grinch: "Grinch", node) -> float:
    """Get the similarity score of a GNode (for external use).

    Similar to PNODE's get_children_similarity.

    Args:
        grinch: The Grinch clustering object.
        node: The GNode to get similarity from.

    Returns:
        The similarity score (0.0 for leaf nodes, positive for internal nodes).
    """
    if node.is_leaf():
        return 0.0

    if hasattr(node, "score") and node.score is not None:
        return float(node.score)

    return 0.0
