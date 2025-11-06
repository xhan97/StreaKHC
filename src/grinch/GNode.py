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

import random
import string
from collections import defaultdict
from queue import Queue

import numpy as np


class GNode:
    """Grinch hierarchical clustering node."""

    def __init__(self, node_id=None, dim=None):
        self.id = (
            node_id
            if node_id is not None
            else "gid"
            + "".join(
                random.choice(string.ascii_uppercase + string.digits) for _ in range(12)
            )
        )
        self.numeric_id = node_id  # For compatibility with array-based operations

        # Tree structure
        self.children = []
        self.parent = None
        self.descendants_list = []

        # Data storage
        self.pts = []  # List of (point_id, label) tuples
        self.point_counter = 0
        self.num_descendants = 0

        # Centroid and sum for computing similarity
        self.centroid = None
        self.sum_vector = None
        if dim is not None:
            self.centroid = np.zeros(dim, dtype=np.float32)
            self.sum_vector = np.zeros(dim, dtype=np.float32)

        # Scoring and update flags
        self.score = -np.inf
        self.needs_update_model = False
        self.needs_update_desc = False
        self.new_node = True

    def __lt__(self, other):
        """An arbitrary way to determine an order when comparing 2 nodes."""
        return self.id < other.id

    def add_child(self, new_child):
        """Add a GNode as a child of this node.

        Args:
        new_child - a GNode.

        Returns:
        A pointer to self with modifications to self and new_child.
        """
        new_child.parent = self
        self.children.append(new_child)
        return self

    def remove_child(self, child):
        """Remove a child from this node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def add_pt(self, point_id, label=None, point_vector=None):
        """Add a data point to this node.

        Args:
        point_id - the ID of the point
        label - (optional) class label of the point
        point_vector - (optional) feature vector of the point

        Returns:
        A pointer to this node (i.e., self). Self now "contains" the point.
        """
        self.point_counter += 1
        if label is not None:
            self.pts.append((point_id, label))
        else:
            self.pts.append(point_id)

        if point_vector is not None and self.sum_vector is not None:
            self.sum_vector += point_vector

        self.needs_update_model = True
        self.needs_update_desc = True
        return self

    def update_centroid(self, norm="l2"):
        """Update the centroid based on the sum vector and number of descendants."""
        if self.sum_vector is None or self.num_descendants == 0:
            return

        # Reset centroid if not a new node
        if not self.new_node and self.centroid is not None:
            self.centroid *= 0

        if self.centroid is not None:
            self.centroid += self.sum_vector
            self.centroid /= self.num_descendants

        # Apply normalization
        if self.centroid is not None:
            if norm == "l2":
                norm_val = np.linalg.norm(self.centroid)
                if norm_val > 0:
                    self.centroid /= norm_val
            elif norm == "l_inf":
                norm_val = np.linalg.norm(self.centroid, np.inf)
                if norm_val > 0:
                    self.centroid /= norm_val
            # For norm == "none", no normalization applied

    def update_descendants(self):
        """Update the descendants list for this node."""
        self.descendants_list.clear()
        if self.is_leaf():
            # Leaf nodes contain themselves in descendants
            if hasattr(self, "numeric_id") and self.numeric_id is not None:
                self.descendants_list.append(self.numeric_id)
        else:
            # Internal nodes collect descendants from children
            for child in self.children:
                if child.needs_update_desc:
                    child.update_descendants()
                self.descendants_list.extend(child.descendants_list)

        self.needs_update_desc = False

    def update_from_children(self):
        """Update this node's properties based on its children."""
        if len(self.children) < 2:
            return

        # Update number of descendants
        self.num_descendants = sum(child.num_descendants for child in self.children)

        # Update sum vector
        if self.sum_vector is not None:
            self.sum_vector *= 0  # Reset
            for child in self.children:
                if child.sum_vector is not None:
                    self.sum_vector += child.sum_vector

        # Mark for model update
        self.score = -np.inf
        self.needs_update_model = True
        self.needs_update_desc = True

    def compute_similarity(self, other, sim_type="dot"):
        """Compute similarity between this node and another node.

        Args:
        other - another GNode
        sim_type - type of similarity ("dot", "l2", "sql2")

        Returns:
        Similarity score between the two nodes.
        """
        if self.centroid is None or other.centroid is None:
            return 0.0

        if sim_type == "dot":
            return np.dot(self.centroid, other.centroid)
        elif sim_type == "l2":
            dist = np.linalg.norm(self.centroid - other.centroid)
            return 1.0 / (1 + dist)
        elif sim_type == "sql2":
            dist_sq = np.sum((self.centroid - other.centroid) ** 2)
            return 1.0 / (1 + dist_sq)
        else:
            return np.dot(self.centroid, other.centroid)

    def find_nearest_neighbor(self, candidate_nodes, sim_type="dot"):
        """Find the nearest neighbor among candidate nodes.

        Args:
        candidate_nodes - list of GNode objects to consider
        sim_type - similarity metric to use

        Returns:
        The GNode with highest similarity to this node.
        """
        if not candidate_nodes:
            return None

        best_node = None
        best_sim = -np.inf

        for node in candidate_nodes:
            if node == self:
                continue
            sim = self.compute_similarity(node, sim_type)
            if sim > best_sim:
                best_sim = sim
                best_node = node

        return best_node

    def split_down(self, new_point_data):
        """Create a new node for the new point and restructure the tree.

        Args:
        new_point_data - tuple of (point_id, label, point_vector)

        Returns:
        A pointer to the new leaf node containing the new point.
        """
        # Create new internal node
        new_internal = GNode(
            dim=len(self.centroid) if self.centroid is not None else None
        )
        new_internal.pts = self.pts[:]  # Copy points to the new internal node
        new_internal.point_counter = self.point_counter

        # Handle parent relationship
        if self.parent:
            self.parent.add_child(new_internal)
            self.parent.remove_child(self)

        new_internal.add_child(self)

        # Create new leaf for the new point
        point_id, label, point_vector = new_point_data
        new_leaf = GNode(
            node_id=point_id,
            dim=len(point_vector) if point_vector is not None else None,
        )
        new_leaf.add_pt(point_id, label, point_vector)
        if point_vector is not None:
            new_leaf.centroid = point_vector.copy()
            new_leaf.sum_vector = point_vector.copy()
            new_leaf.num_descendants = 1

        new_internal.add_child(new_leaf)
        return new_leaf

    def update_recursively(self, norm="l2", sim_type="dot"):
        """Update this node and propagate updates up the tree.

        Updates this node first, then all ancestors up to the root.

        Args:
            norm: Normalization type for centroid ("l2", "l_inf", "none")
            sim_type: Similarity type for computing score ("dot", "l2", "sql2")

        Returns:
            The root node after updates
        """
        # First update this node itself
        self.update_from_children()
        self.update_centroid(norm=norm)

        # Recompute score for binary nodes
        if len(self.children) == 2:
            self.score = self.children[0].compute_similarity(
                self.children[1], sim_type=sim_type
            )

        # Then update all ancestors
        current_node = self
        while current_node.parent:
            parent = current_node.parent
            parent.update_from_children()
            parent.update_centroid(norm=norm)

            # Recompute score for binary nodes
            if len(parent.children) == 2:
                parent.score = parent.children[0].compute_similarity(
                    parent.children[1], sim_type=sim_type
                )

            current_node = parent
        return current_node

    # Tree navigation methods (similar to INode)
    def siblings(self):
        """Return a list of my siblings."""
        if self.parent and hasattr(self.parent, "children"):
            return [child for child in self.parent.children if child != self]
        else:
            return []

    def aunts(self):
        """Return a list of all of my aunts."""
        if self.parent and self.parent.parent:
            grandparent = self.parent.parent
            aunts_list = []
            for child in grandparent.children:  # type: ignore
                if child != self.parent:
                    aunts_list.append(child)
            return aunts_list
        return []

    def _ancestors(self):
        """Return all of this node's ancestors in order to the root."""
        anc = []
        curr = self
        while curr.parent:
            anc.append(curr.parent)
            curr = curr.parent
        return anc

    def depth(self):
        """Return the number of ancestors on the root to leaf path."""
        return len(self._ancestors())

    def height(self):
        """Return the height of this node."""
        if self.is_leaf():
            return 0
        return max(child.height() for child in self.children) + 1

    def descendants(self):
        """Return all descendants of the current node."""
        d = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            d.append(n)
            for c in n.children:
                queue.put(c)
        return d

    def leaves(self):
        """Return the list of leaves under this node."""
        lvs = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            if n.children:
                for c in n.children:
                    queue.put(c)
            else:
                lvs.append(n)
        return lvs

    def lca(self, other):
        """Compute the lowest common ancestor between this node and other.

        Args:
        other - a GNode in the tree.

        Returns:
        A GNode that is the lowest common ancestor between self and other.
        """
        ancestors = set(self._ancestors())
        ancestors.add(self)

        curr_node = other
        while curr_node not in ancestors:
            curr_node = curr_node.parent
            if curr_node is None:
                return None
        return curr_node

    def root(self):
        """Return the root of the tree."""
        curr_node = self
        while curr_node.parent:
            curr_node = curr_node.parent
        return curr_node

    def is_leaf(self):
        """Returns true if self is a leaf, else false."""
        return len(self.children) == 0

    def is_internal(self):
        """Returns false if self is a leaf, else true."""
        return not self.is_leaf()

    # Clustering evaluation methods
    def purity(self, cluster=None):
        """Compute the purity of this node.

        Args:
        cluster - (optional) str, compute purity with respect to this cluster.

        Returns:
        A float representing the purity of this node.
        """
        if cluster:
            pts = [p for l in self.leaves() for p in l.pts if len(p) > 1]
            if not pts:
                return 1.0
            return float(len([pt for pt in pts if pt[1] == cluster])) / len(pts)
        else:
            label_to_count = self.class_counts()
            if not label_to_count:
                return 1.0
        return max(label_to_count.values()) / sum(label_to_count.values())

    def class_counts(self):
        """Produce a map from label to the # of descendant points with label."""
        label_to_count = defaultdict(float)
        pts = []
        for leaf in self.leaves():
            for pt in leaf.pts:
                if isinstance(pt, tuple) and len(pt) > 1:
                    pts.append(pt)

        for pt in pts:
            if len(pt) > 1:
                label = pt[1]
                label_to_count[label] += 1.0
        return label_to_count

    def pure_class(self):
        """If this node has purity 1.0, return its label; else return None."""
        cc = self.class_counts()
        if len(cc) == 1:
            return list(cc.keys())[0]
        else:
            return None

    def flat_clustering(self, threshold):
        """Extract flat clustering from tree using similarity threshold.

        Args:
        threshold - similarity threshold for cutting the tree

        Returns:
        List of cluster assignments for leaf nodes.
        """
        frontier = [self.root()]
        clusters = []

        while frontier:
            node = frontier.pop(0)
            if (
                node.children
                and len(node.children) == 2
                and node.children[0].compute_similarity(node.children[1]) < threshold
            ):
                frontier.extend(node.children)
            else:
                clusters.append(node)

        # Assign cluster IDs to leaf nodes
        assignments = {}
        for cluster_id, cluster_node in enumerate(clusters):
            for leaf in cluster_node.leaves():
                for pt in leaf.pts:
                    point_id = pt[0] if isinstance(pt, tuple) else pt
                    assignments[point_id] = cluster_id

        return assignments

    # Helper methods for tree construction
    def has_valid_centroid(self):
        """Check if this node has a valid centroid.

        Returns:
            bool: True if centroid exists and is not None
        """
        return self.centroid is not None

    def relink_parent(self, new_parent):
        """Handle parent relationship restructuring.

        Updates this node's parent reference when tree structure changes.

        Args:
            new_parent: The new parent GNode (or None if becoming root)
        """
        self.parent = new_parent

    def update_as_parent_of(self, child1, child2, sim_type="dot", norm="l2"):
        """Update this node's attributes as the parent of two children.

        This is used when creating a new internal node that joins two nodes.

        Args:
            child1: First child node
            child2: Second child node
            sim_type: Similarity type for scoring
            norm: Normalization method for centroid
        """
        # Update number of descendants
        self.num_descendants = child1.num_descendants + child2.num_descendants

        # Update sum vector and centroid
        if child1.sum_vector is not None and child2.sum_vector is not None:
            if self.sum_vector is None:
                self.sum_vector = np.zeros_like(child1.sum_vector)
            self.sum_vector = child1.sum_vector + child2.sum_vector
            self.update_centroid(norm=norm)

        # Compute similarity score between children
        self.score = child1.compute_similarity(child2, sim_type=sim_type)
