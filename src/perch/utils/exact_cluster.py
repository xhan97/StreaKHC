import heapq
from typing import Optional
import numpy as np
from numba import jit
from PNode import PNODE


@jit(nopython=True)
def fast_avg_distance(pts1, pts2):
    """Compute the average distance between two sets of points.

    Args:
        pts1: First set of points as numpy array.
        pts2: Second set of points as numpy array.
    Returns:
        The average distance between the two sets of points.
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0

    n1 = len(pts1)
    n2 = len(pts2)
    total_distance = 0.0

    for i in range(n1):
        for j in range(n2):
            # Compute squared Euclidean distance
            dist_sq = 0.0
            for d in range(len(pts1[i])):
                diff = pts1[i][d] - pts2[j][d]
                dist_sq += diff * diff
            # Use sqrt only once per pair instead of inside the loop
            total_distance += dist_sq**0.5

    # Pre-compute denominator
    avg = total_distance / (n1 * n2)
    return avg


def get_children_similarity(node: PNODE) -> float:
    """Get the similarity between the two children of a node.

    Args:
        node: The node to get the similarity from.
    Returns:
        The similarity between the two children.
    """
    if node.is_leaf():
        return 0.0
    lt_pts = np.array([p[0] for l in node.children[0].leaves() for p in l.pts])
    rt_pts = np.array([p[0] for l in node.children[1].leaves() for p in l.pts])
    curr_similarity = fast_avg_distance(lt_pts, rt_pts)
    return curr_similarity


def cut_tree(root: PNODE, n_cluster: Optional[int] = None) -> tuple[list, list]:
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
            gt_labels.append(leaf.pts[0][1])  # True label
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
            ground_truth.append(leaf.pts[0][1])  # True label
            labels.append(cluster_id)  # Assigned cluster

    return labels, ground_truth
