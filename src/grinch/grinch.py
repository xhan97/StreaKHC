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

"""Grinch - Online hierarchical clustering using GNode tree structure."""

import logging
import time

import numpy as np
from GNode import GNode

logging.basicConfig(level=logging.INFO)


class Grinch:
    """Grinch: Greedy Incremental Clustering for Hierarchies.

    Online hierarchical clustering algorithm with adaptive tree restructuring.

    Core operations:
    - Insert: Add new points to the tree
    - Rotate: Find optimal insertion position by traversing upward
    - Graft: Optimize tree structure by reorganizing nodes
    """

    def __init__(
        self,
        dim=None,
        rotate_cap=100,
        graft_cap=100,
        norm="l2",
        sim="dot",
        canopies=None,
    ):
        """Initialize Grinch clusterer.

        Args:
            dim: Data dimension (auto-detected from first point if None)
            rotate_cap: Capacity limit for rotate operation
            graft_cap: Capacity limit for graft operation
            norm: Centroid normalization ("l2", "l_inf", "none")
            sim: Similarity metric ("dot", "l2", "sql2")
            canopies: Canopy clustering results
        """
        self.point_counter = 0
        self.norm = norm
        self.sim_type = sim
        self.rotate_cap = rotate_cap
        self.graft_cap = graft_cap
        self.canopies = canopies

        self.dim = dim
        self.root_node = None

        # Statistics
        self.time_in_search = 0
        self.time_in_rotate = 0
        self.time_in_update = 0
        self.time_in_graft = 0
        self.time_in_lca = 0

        self.number_of_rotates = 0
        self.number_of_grafts = 0
        self.number_of_grafts_considered = 0
        self.number_of_rotates_considered = 0

        self.this_number_of_rotates = 0
        self.this_number_of_grafts = 0
        self.this_number_of_grafts_considered = 0
        self.this_number_of_rotates_considered = 0

        logging.debug("Grinch initialized with GNode structure")

    def _normalize_vector(self, vec):
        """Normalize vector based on self.norm setting."""
        if self.norm == "l2":
            norm_val = np.linalg.norm(vec)
            if norm_val > 0:
                return vec / norm_val
            return vec
        elif self.norm == "l_inf":
            max_val = np.max(np.abs(vec))
            if max_val > 0:
                return vec / max_val
            return vec
        else:
            return vec

    def clear_stats(self):
        """Clear statistics counters."""
        self.this_number_of_rotates = 0
        self.this_number_of_grafts = 0
        self.this_number_of_grafts_considered = 0
        self.this_number_of_rotates_considered = 0

    def stats_string(self):
        """Generate statistics string."""
        r = (
            "search_time=%s\trotate_time=%s\tgraft_time=%s\tupdate_time=%s\t"
            "num_rotate=%s\tnum_graft=%s\tnum_rotate_considered=%s\tnum_graft_considered=%s\n"
            % (
                self.time_in_search,
                self.time_in_rotate,
                self.time_in_graft,
                self.time_in_update,
                self.number_of_rotates,
                self.number_of_grafts,
                self.number_of_rotates_considered,
                self.number_of_grafts_considered,
            )
        )
        self.clear_stats()
        return r

    def insert(self, point_id, point_vec, point_label=None):
        """Insert a new data point into the clustering tree.

        Args:
            point_id: Point ID
            point_vec: Feature vector (will be normalized based on self.norm)
            point_label: Point label (optional)

        Raises:
            ValueError: If point_vec is None
        """
        if point_vec is None:
            raise ValueError("point_vec is required in online mode")

        point_vec = self._normalize_vector(point_vec)

        start_time = time.time()
        logging.debug("[insert] Inserting point %s", point_id)

        if self.point_counter == 0:
            self._create_root_node(point_id, point_vec, point_label)
            self.point_counter += 1
            return

        self._insert_into_tree(point_id, point_vec, point_label)
        self.point_counter += 1

        elapsed = time.time() - start_time
        logging.debug("[insert] Finished inserting point %s (%.4fs)", point_id, elapsed)

    def _create_root_node(self, point_id, point_vec, point_label=None):
        """Create root node."""
        if self.dim is None and point_vec is not None:
            self.dim = len(point_vec)
            logging.debug("[Grinch] Auto-detected dimension: %d", self.dim)

        self.root_node = GNode(node_id=point_id, dim=self.dim)
        self.root_node.add_pt(point_id, label=point_label, point_vector=point_vec)

        if point_vec is not None:
            self.root_node.centroid = point_vec.copy()
            self.root_node.sum_vector = point_vec.copy()

        self.root_node.num_descendants = 1
        logging.debug("[insert] Created root node for point %s", point_id)

    def _insert_into_tree(self, point_id, point_vec, point_label=None):
        """Insert point into existing tree using GNode's split_down method."""
        nearest_node = self._find_and_time_nearest_neighbor(point_vec)
        if nearest_node is None:
            logging.warning("No nearest neighbor found for point %s", point_id)
            return

        logging.debug(
            "[insert] Nearest neighbor for point %s is %s", point_id, nearest_node.id
        )

        sibling = self._find_and_time_rotate_with_vec(point_vec, nearest_node)
        new_point_data = (point_id, point_label, point_vec)
        new_leaf = sibling.split_down(new_point_data)
        parent = new_leaf.parent

        if sibling.parent == parent and parent.parent is None:
            self.root_node = parent

        self._update_parent_attributes(parent, sibling, new_leaf)

        start = time.time()
        parent.update_recursively(norm=self.norm, sim_type=self.sim_type)
        self.time_in_update += time.time() - start

        self._time_graft(parent)

    def _find_and_time_nearest_neighbor(self, point_vec):
        """Find and time nearest neighbor search."""
        start = time.time()
        nearest_node, _ = self._find_nearest_neighbor(point_vec)
        self.time_in_search += time.time() - start
        return nearest_node

    def _find_and_time_rotate_with_vec(self, point_vec, nearest_node):
        """Find and time rotate operation with vector."""
        start = time.time()
        sibling = self._find_rotate_with_vec(point_vec, nearest_node)
        self.time_in_rotate += time.time() - start
        return sibling

    def _time_graft(self, parent):
        """Execute and time graft operation."""
        start = time.time()
        self._graft(parent)
        self.time_in_graft += time.time() - start

    def _find_nearest_neighbor(self, query_vec):
        """Find the leaf node most similar to query vector."""
        if self.root_node is None:
            return None, -np.inf

        leaves = self.root_node.leaves()
        best_node = None
        best_sim = -np.inf

        for leaf in leaves:
            if leaf.centroid is None:
                continue

            sim = self._compute_similarity(query_vec, leaf.centroid)
            if sim > best_sim:
                best_sim = sim
                best_node = leaf

        return best_node, best_sim

    def _compute_similarity(self, vec1, vec2):
        """Compute similarity between two vectors."""
        if self.sim_type == "dot":
            return np.dot(vec1, vec2)
        elif self.sim_type == "l2":
            dist = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1 + dist)
        elif self.sim_type == "sql2":
            dist_sq = np.sum((vec1 - vec2) ** 2)
            return 1.0 / (1 + dist_sq)
        else:
            return np.dot(vec1, vec2)

    def _find_rotate_with_vec(self, point_vec, nearest_node):
        """Find optimal insertion position via rotate operation.

        Traverse upward to find the most similar position for the new point.
        """
        self.number_of_rotates_considered += 1
        self.this_number_of_rotates_considered += 1

        if nearest_node.centroid is None:
            return nearest_node

        current = nearest_node
        current_score = self._compute_similarity(point_vec, current.centroid)

        while current.parent is not None:
            parent = current.parent

            if parent.num_descendants >= self.rotate_cap:
                break

            if parent.centroid is None:
                break

            parent_score = self._compute_similarity(point_vec, parent.centroid)

            if current_score < parent_score:
                current = parent
                current_score = parent_score
                self.number_of_rotates += 1
                self.this_number_of_rotates += 1
            else:
                break

        return current

    def _update_parent_attributes(self, parent, child1, child2):
        """Update parent node attributes."""
        parent.update_as_parent_of(child1, child2, self.norm, self.sim_type)

    def _graft(self, node):
        """Optimize tree structure via graft operation.

        Reorganize tree by moving nodes to more similar positions.
        """
        start_time = time.time()
        logging.debug("[graft] graft(%s)", node.id)

        if not self._validate_node_for_graft(node):
            return

        offlimits = self._get_offlimits_nodes(node)
        nearest_neighbor = self._find_graft_target(node, offlimits)
        if nearest_neighbor is None:
            logging.debug("[graft] No valid graft target found")
            return

        graft_paths = self._collect_graft_paths(node, nearest_neighbor)
        if not graft_paths:
            logging.debug("[graft] No valid graft paths")
            return

        self._evaluate_and_execute_graft(node, graft_paths)
        self.time_in_graft += time.time() - start_time

    def _validate_node_for_graft(self, node):
        """Validate if node can be grafted."""
        if node.centroid is None:
            logging.debug("[graft] Node has no centroid, skipping")
            return False

        if self.root_node is None:
            logging.debug("[graft] No root node")
            return False

        return True

    def _get_offlimits_nodes(self, node):
        """Get set of nodes that cannot be graft targets.

        Includes descendants (avoid cycles) and sibling leaf nodes.
        """
        offlimits = set()
        descendants = node.descendants()
        offlimits.update(descendants)

        if node.parent is not None:
            siblings = node.siblings()
            if siblings:
                sibling = siblings[0]
                if sibling.is_leaf():
                    offlimits.add(sibling)

        logging.debug("[graft] Offlimits size: %s", len(offlimits))
        return offlimits

    def _find_graft_target(self, node, offlimits):
        """Find best graft target node among all leaves (excluding offlimits)."""
        if self.root_node is None:
            return None

        search_start = time.time()
        best_target = None
        best_similarity = -np.inf

        all_leaves = self.root_node.leaves()
        for leaf in all_leaves:
            if leaf in offlimits:
                continue

            if leaf.centroid is None:
                continue

            similarity = node.compute_similarity(leaf, sim_type=self.sim_type)
            if similarity > best_similarity:
                best_similarity = similarity
                best_target = leaf

        self.time_in_graft_search = time.time() - search_start

        if best_target:
            logging.debug(
                "[graft] Best target: %s (sim=%.4f)", best_target.id, best_similarity
            )

        return best_target

    def _collect_graft_paths(self, node, target):
        """Collect possible graft paths from both nodes to their LCA."""
        lca = node.lca(target)
        if lca is None:
            logging.debug("[graft] No LCA found")
            return None

        node_ancestors = self._collect_ancestors_to_lca(node, lca)
        target_ancestors = self._collect_ancestors_to_lca(target, lca)

        logging.debug(
            "[graft] LCA: %s, Node ancestors: %s, Target ancestors: %s",
            lca.id,
            len(node_ancestors),
            len(target_ancestors),
        )

        if not node_ancestors or not target_ancestors:
            return None

        return {
            "lca": lca,
            "node_ancestors": node_ancestors,
            "target_ancestors": target_ancestors,
        }

    def _collect_ancestors_to_lca(self, node, lca):
        """Collect ancestor path from node to LCA (limited by graft_cap)."""
        ancestors = []
        current = node

        while current != lca and current is not None:
            if current.num_descendants < self.graft_cap:
                ancestors.append(current)
            current = current.parent

        return ancestors

    def _evaluate_and_execute_graft(self, node, graft_paths):
        """Evaluate all possible graft pairs and execute the best one."""
        node_ancestors = graft_paths["node_ancestors"]
        target_ancestors = graft_paths["target_ancestors"]

        score_matrix = self._build_graft_score_matrix(node_ancestors, target_ancestors)
        best_pair = self._find_best_graft_pair(
            node_ancestors, target_ancestors, score_matrix
        )

        num_considered = len(node_ancestors) * len(target_ancestors)
        self.number_of_grafts_considered += num_considered
        self.this_number_of_grafts_considered = num_considered

        if best_pair:
            self._execute_graft(node, best_pair)
        else:
            logging.debug("[graft] No beneficial graft found")

    def _build_graft_score_matrix(self, node_ancestors, target_ancestors):
        """Build graft score matrix.

        Matrix element [i,j] represents benefit of grafting
        node_ancestors[i] with target_ancestors[j].
        """
        M = len(node_ancestors)
        N = len(target_ancestors)

        graft_scores = np.zeros((M, N), dtype=np.float32)
        for i, n1 in enumerate(node_ancestors):
            for j, n2 in enumerate(target_ancestors):
                graft_scores[i, j] = n1.compute_similarity(n2, sim_type=self.sim_type)

        node_parent_scores = self._get_parent_scores(node_ancestors)
        target_parent_scores = self._get_parent_scores(target_ancestors)

        return {
            "graft_scores": graft_scores,
            "node_parent_scores": node_parent_scores,
            "target_parent_scores": target_parent_scores,
        }

    def _get_parent_scores(self, ancestors):
        """Get parent scores for ancestor nodes."""
        scores = np.full(len(ancestors), -np.inf, dtype=np.float32)

        for i, node in enumerate(ancestors):
            if node.parent and len(node.parent.children) == 2:
                if hasattr(node.parent, "score") and node.parent.score is not None:
                    scores[i] = node.parent.score

        return scores

    def _find_best_graft_pair(self, node_ancestors, target_ancestors, score_matrix):
        """Find best graft pair.

        Graft condition: new pairing score must be better than both current parent scores.
        """
        graft_scores = score_matrix["graft_scores"]
        node_parent_scores = score_matrix["node_parent_scores"]
        target_parent_scores = score_matrix["target_parent_scores"]

        M, N = graft_scores.shape

        node_parent_matrix = node_parent_scores.reshape(M, 1)
        target_parent_matrix = target_parent_scores.reshape(1, N)

        better_than_node_parent = graft_scores > node_parent_matrix
        better_than_target_parent = graft_scores > target_parent_matrix
        is_beneficial = better_than_node_parent & better_than_target_parent

        valid_scores = graft_scores.copy()
        valid_scores[~is_beneficial] = 0

        best_idx = np.argmax(valid_scores)
        best_i = best_idx // N
        best_j = best_idx % N
        best_score = valid_scores[best_i, best_j]

        if best_score > 0 and is_beneficial[best_i, best_j]:
            return {
                "node1": node_ancestors[best_i],
                "node2": target_ancestors[best_j],
                "score": best_score,
                "node_parent_score": node_parent_scores[best_i],
                "target_parent_score": target_parent_scores[best_j],
            }

        return None

    def _execute_graft(self, original_node, graft_pair):
        """Execute graft operation."""
        node1 = graft_pair["node1"]
        node2 = graft_pair["node2"]

        self.number_of_grafts += 1
        self.this_number_of_grafts += 1

        logging.debug(
            "[graft] Grafting %s to %s (score=%.4f > max(%.4f, %.4f))",
            node1.id,
            node2.id,
            graft_pair["score"],
            graft_pair["node_parent_score"],
            graft_pair["target_parent_score"],
        )

        update_starts = self._collect_update_starts(node2, original_node)
        self._perform_graft_operation(node1, node2)
        self._update_affected_subtrees(update_starts)

    def _collect_update_starts(self, node2, original_node):
        """Collect starting nodes for update."""
        update_starts = []

        if node2.parent and node2.parent.parent:
            update_starts.append(node2.parent.parent)

        if original_node.parent:
            update_starts.append(original_node.parent)

        return update_starts

    def _update_affected_subtrees(self, update_starts):
        """Update subtrees affected by graft using GNode's update_recursively."""
        for start in update_starts:
            if start is not None:
                start.update_recursively(norm=self.norm, sim_type=self.sim_type)

    def _perform_graft_operation(self, node1, node2):
        """Perform actual graft operation by reorganizing tree structure.

        Implements make_sibling logic from original Grinch.
        """
        logging.debug(
            "_perform_graft_operation(node1=%s, node2=%s)", node1.id, node2.id
        )

        node2_parent = node2.parent
        node2_sibling = None
        if node2_parent and len(node2_parent.children) == 2:
            siblings = node2.siblings()
            if siblings:
                node2_sibling = siblings[0]

        if node2_parent and node2_sibling:
            node2_grandparent = node2_parent.parent

            node2_parent.remove_child(node2_sibling)
            node2_parent.remove_child(node2)

            if node2_grandparent:
                node2_grandparent.remove_child(node2_parent)
                node2_grandparent.add_child(node2_sibling)
            else:
                node2_sibling.relink_parent(None)
                if self.root_node == node2_parent:
                    self.root_node = node2_sibling

        new_parent = GNode(dim=self.dim)
        node1_parent = node1.parent

        if node1_parent:
            node1_parent.remove_child(node1)
            node1_parent.add_child(new_parent)
            new_parent.relink_parent(node1_parent)
        else:
            new_parent.relink_parent(None)
            self.root_node = new_parent

        new_parent.add_child(node1)
        new_parent.add_child(node2)

        new_parent.num_descendants = node1.num_descendants + node2.num_descendants

        if node1.sum_vector is not None and node2.sum_vector is not None:
            new_parent.sum_vector = node1.sum_vector + node2.sum_vector
            new_parent.update_centroid(norm=self.norm)

        new_parent.score = node1.compute_similarity(node2, sim_type=self.sim_type)
        new_parent.needs_update_model = True
        new_parent.needs_update_desc = True

    def root(self):
        """Return root node of the tree."""
        return self.root_node
