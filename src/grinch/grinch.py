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

"""
Grinch - 使用 GNode 构建层次聚类树
基于原始 Grinch 算法，但使用 GNode 对象来表示树结构
"""

import logging
import time

import numpy as np
from GNode import GNode

# 配置logging
logging.basicConfig(level=logging.INFO)


class Grinch:
    """基于 GNode 的 Grinch 层次聚类算法

    Grinch (Greedy Incremental Clustering for Hierarchies) 是一个在线层次聚类算法。

    核心操作：
    1. Insert: 将新点插入到树中
    2. Rotate: 向上遍历找到最佳插入位置
    3. Graft: 通过重组树结构来优化聚类质量

    算法特点：
    - 在线学习：可以逐个处理数据点
    - 自适应：通过graft操作不断优化树结构
    - 高效：基于对象的树结构，内存使用最优

    使用示例：
        >>> points = np.random.random((100, 10))
        >>> grinch = Grinch(points=points)
        >>> grinch.build_dendrogram()
        >>> clusters = grinch.flat_clustering(threshold=0.5)
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
        """初始化 Grinch 聚类器（完全在线模式）

        Args:
            dim: 数据维度（可选，将从第一个点自动推断）
            rotate_cap: rotate 操作的容量限制
            graft_cap: graft 操作的容量限制
            norm: 质心归一化方式 ("l2", "l_inf", "none")
            sim: 相似度度量方式 ("dot", "l2", "sql2")
            canopies: Canopy聚类结果
        """
        self.point_counter = 0
        self.norm = norm
        self.sim_type = sim
        self.rotate_cap = rotate_cap
        self.graft_cap = graft_cap
        self.canopies = canopies

        # dim 可以从第一个点自动推断，如果未指定
        self.dim = dim

        # GNode 树的根节点
        self.root_node = None

        # 统计信息
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
        """根据self.norm参数归一化向量

        Args:
            vec: 输入向量

        Returns:
            归一化后的向量
        """
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
        else:  # norm == "none"
            return vec

    def clear_stats(self):
        """清除统计信息"""
        self.this_number_of_rotates = 0
        self.this_number_of_grafts = 0
        self.this_number_of_grafts_considered = 0
        self.this_number_of_rotates_considered = 0

    def stats_string(self):
        """生成统计信息字符串"""
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
        """插入一个新的数据点到聚类树中

        核心插入流程：
        1. 创建新的叶子节点
        2. 找到最近邻节点
        3. 通过Rotate找到最佳插入位置
        4. 创建新的父节点连接
        5. 更新祖先节点
        6. 执行Graft优化

        Args:
            point_id: 点的ID
            point_vec: 点的特征向量（必须提供，将根据self.norm自动归一化）
            point_label: 点的标签（可选）

        Raises:
            ValueError: 如果 point_vec 为 None
        """
        if point_vec is None:
            raise ValueError("point_vec is required in online mode")

        # 归一化输入向量（根据self.norm参数）
        point_vec = self._normalize_vector(point_vec)

        start_time = time.time()
        logging.debug("[insert] Inserting point %s", point_id)

        # 处理第一个点
        if self.point_counter == 0:
            self._create_root_node(point_id, point_vec, point_label)
            self.point_counter += 1
            return

        # 执行插入流程
        self._insert_into_tree(point_id, point_vec, point_label)
        self.point_counter += 1

        elapsed = time.time() - start_time
        logging.debug("[insert] Finished inserting point %s (%.4fs)", point_id, elapsed)

    def _create_root_node(self, point_id, point_vec, point_label=None):
        """创建根节点

        Args:
            point_id: 点的ID
            point_vec: 点的特征向量
            point_label: 点的标签（可选）
        """
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
        """将点插入到已存在的树中

        使用GNode的split_down方法来建树

        Args:
            point_id: 点的ID
            point_vec: 点的特征向量
            point_label: 点的标签（可选）
        """
        # 步骤1: 找到最近邻
        nearest_node = self._find_and_time_nearest_neighbor(point_vec)
        if nearest_node is None:
            logging.warning("No nearest neighbor found for point %s", point_id)
            return

        logging.debug(
            "[insert] Nearest neighbor for point %s is %s", point_id, nearest_node.id
        )

        # 步骤2: 执行Rotate找到最佳位置（直接使用point_vec）
        sibling = self._find_and_time_rotate_with_vec(point_vec, nearest_node)

        # 步骤3: 使用split_down创建新节点和重组树
        new_point_data = (point_id, point_label, point_vec)
        new_leaf = sibling.split_down(new_point_data)

        # 获取新创建的父节点
        parent = new_leaf.parent

        # 如果sibling之前是根节点，更新root_node
        if sibling.parent == parent and parent.parent is None:
            self.root_node = parent

        # 步骤4: 更新新父节点的属性
        self._update_parent_attributes(parent, sibling, new_leaf)

        # 步骤5: 更新祖先（使用GNode的update_recursively）
        start = time.time()
        parent.update_recursively(norm=self.norm, sim_type=self.sim_type)
        self.time_in_update += time.time() - start

        # 步骤6: 执行Graft优化
        self._time_graft(parent)

    def _find_and_time_nearest_neighbor(self, point_vec):
        """查找最近邻并计时

        Args:
            point_vec: 查询向量

        Returns:
            最近邻节点
        """
        start = time.time()
        nearest_node, _ = self._find_nearest_neighbor(point_vec)
        self.time_in_search += time.time() - start
        return nearest_node

    def _find_and_time_rotate(self, new_leaf, nearest_node):
        """执行Rotate并计时

        Args:
            new_leaf: 新叶子节点
            nearest_node: 最近邻节点

        Returns:
            最佳兄弟节点
        """
        start = time.time()
        sibling = self._find_rotate(new_leaf, nearest_node)
        self.time_in_rotate += time.time() - start
        return sibling

    def _find_and_time_rotate_with_vec(self, point_vec, nearest_node):
        """使用点向量执行Rotate并计时

        Args:
            point_vec: 新点的特征向量
            nearest_node: 最近邻节点

        Returns:
            最佳兄弟节点
        """
        start = time.time()
        sibling = self._find_rotate_with_vec(point_vec, nearest_node)
        self.time_in_rotate += time.time() - start
        return sibling

    def _time_graft(self, parent):
        """执行Graft并计时

        Args:
            parent: 父节点
        """
        start = time.time()
        self._graft(parent)
        self.time_in_graft += time.time() - start

    def _find_nearest_neighbor(self, query_vec):
        """找到与查询向量最相似的叶子节点

        Args:
            query_vec: 查询向量

        Returns:
            (nearest_node, similarity): 最近的节点和相似度
        """
        if self.root_node is None:
            return None, -np.inf

        # 获取所有叶子节点
        leaves = self.root_node.leaves()

        best_node = None
        best_sim = -np.inf

        for leaf in leaves:
            if leaf.centroid is None:
                continue

            # 计算相似度
            sim = self._compute_similarity(query_vec, leaf.centroid)

            if sim > best_sim:
                best_sim = sim
                best_node = leaf

        return best_node, best_sim

    def _compute_similarity(self, vec1, vec2):
        """计算两个向量之间的相似度

        Args:
            vec1: 第一个向量
            vec2: 第二个向量

        Returns:
            相似度分数
        """
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

    def _find_rotate(self, new_node, nearest_node):
        """找到通过rotate操作的最佳插入位置

        Rotate操作：向上遍历树，找到与新节点最相似的位置。
        这样可以确保新节点被插入到语义最接近的位置。

        Args:
            new_node: 新插入的节点
            nearest_node: 最近邻节点

        Returns:
            GNode: 最佳的兄弟节点位置
        """
        # 更新统计
        self.number_of_rotates_considered += 1
        self.this_number_of_rotates_considered += 1

        # 计算初始相似度
        if not self._has_valid_centroid(new_node, nearest_node):
            return nearest_node

        current = nearest_node
        current_score = new_node.compute_similarity(current, sim_type=self.sim_type)

        # 向上遍历寻找更好的位置
        while current.parent is not None:
            parent = current.parent

            # 检查容量限制
            if parent.num_descendants >= self.rotate_cap:
                break

            # 计算与父节点的相似度
            if parent.centroid is None:
                break

            parent_score = new_node.compute_similarity(parent, sim_type=self.sim_type)

            # 如果父节点更相似，继续向上
            if current_score < parent_score:
                current = parent
                current_score = parent_score
                self.number_of_rotates += 1
                self.this_number_of_rotates += 1
            else:
                # 找到最佳位置
                break

        return current

    def _find_rotate_with_vec(self, point_vec, nearest_node):
        """使用点向量找到通过rotate操作的最佳插入位置

        Rotate操作：向上遍历树，找到与新点向量最相似的位置。

        Args:
            point_vec: 新点的特征向量
            nearest_node: 最近邻节点

        Returns:
            GNode: 最佳的兄弟节点位置
        """
        # 更新统计
        self.number_of_rotates_considered += 1
        self.this_number_of_rotates_considered += 1

        # 检查最近邻节点是否有效
        if nearest_node.centroid is None:
            return nearest_node

        current = nearest_node
        current_score = self._compute_similarity(point_vec, current.centroid)

        # 向上遍历寻找更好的位置
        while current.parent is not None:
            parent = current.parent

            # 检查容量限制
            if parent.num_descendants >= self.rotate_cap:
                break

            # 计算与父节点的相似度
            if parent.centroid is None:
                break

            parent_score = self._compute_similarity(point_vec, parent.centroid)

            # 如果父节点更相似，继续向上
            if current_score < parent_score:
                current = parent
                current_score = parent_score
                self.number_of_rotates += 1
                self.this_number_of_rotates += 1
            else:
                # 找到最佳位置
                break

        return current

    def _has_valid_centroid(self, *nodes):
        """检查所有节点是否都有有效的centroid

        Args:
            *nodes: 要检查的节点

        Returns:
            bool: 是否所有节点都有centroid
        """
        return all(node.has_valid_centroid() for node in nodes)

    def _update_parent_attributes(self, parent, child1, child2):
        """更新父节点的属性

        Args:
            parent: 父节点
            child1: 第一个子节点
            child2: 第二个子节点
        """
        parent.update_as_parent_of(child1, child2, self.norm, self.sim_type)

    def _graft(self, node):
        """执行graft操作优化树结构

        Graft操作通过重新组织树结构来优化聚类质量。
        基本思想：将节点移动到更相似的位置。

        Args:
            node: 要graft的节点
        """
        start_time = time.time()
        logging.debug("[graft] graft(%s)", node.id)

        # 验证节点有效性
        if not self._validate_node_for_graft(node):
            return

        # 获取禁止区域（不能graft的节点）
        offlimits = self._get_offlimits_nodes(node)

        # 找到最佳的graft目标
        nearest_neighbor = self._find_graft_target(node, offlimits)
        if nearest_neighbor is None:
            logging.debug("[graft] No valid graft target found")
            return

        # 收集可能的graft路径
        graft_paths = self._collect_graft_paths(node, nearest_neighbor)
        if not graft_paths:
            logging.debug("[graft] No valid graft paths")
            return

        # 评估并执行最佳graft
        self._evaluate_and_execute_graft(node, graft_paths)

        self.time_in_graft += time.time() - start_time

    def _validate_node_for_graft(self, node):
        """验证节点是否可以进行graft操作

        Args:
            node: 待验证的节点

        Returns:
            bool: 是否可以graft
        """
        if node.centroid is None:
            logging.debug("[graft] Node has no centroid, skipping")
            return False

        if self.root_node is None:
            logging.debug("[graft] No root node")
            return False

        return True

    def _get_offlimits_nodes(self, node):
        """获取禁止graft的节点集合

        Offlimits包括：
        1. 当前节点的所有后代（避免创建环）
        2. 当前节点的兄弟节点（避免无意义的操作）

        Args:
            node: 当前节点

        Returns:
            set: 禁止graft的节点集合
        """
        offlimits = set()

        # 添加所有后代节点
        descendants = node.descendants()
        offlimits.update(descendants)

        # 添加兄弟节点
        if node.parent is not None:
            siblings = node.siblings()
            if siblings:
                sibling = siblings[0]
                if sibling.is_leaf():
                    offlimits.add(sibling)

        logging.debug("[graft] Offlimits size: %s", len(offlimits))
        return offlimits

    def _find_graft_target(self, node, offlimits):
        """找到最佳的graft目标节点

        在所有叶子节点中（排除offlimits），找到与当前节点最相似的节点。

        Args:
            node: 当前节点
            offlimits: 禁止的节点集合

        Returns:
            GNode: 最佳目标节点，如果没有则返回None
        """
        if self.root_node is None:
            return None

        search_start = time.time()
        best_target = None
        best_similarity = -np.inf

        all_leaves = self.root_node.leaves()
        for leaf in all_leaves:
            # 跳过禁止的节点
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
        """收集从两个节点到它们LCA的所有可能路径

        Args:
            node: 当前节点
            target: 目标节点

        Returns:
            dict: 包含路径信息的字典，如果无效则返回None
        """
        # 找到最低公共祖先
        lca = node.lca(target)
        if lca is None:
            logging.debug("[graft] No LCA found")
            return None

        # 收集从node到lca的路径
        node_ancestors = self._collect_ancestors_to_lca(node, lca)

        # 收集从target到lca的路径
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
        """收集从节点到LCA的祖先路径（受graft_cap限制）

        Args:
            node: 起始节点
            lca: 最低公共祖先

        Returns:
            list: 祖先节点列表
        """
        ancestors = []
        current = node

        while current != lca and current is not None:
            # 只收集后代数小于graft_cap的节点
            if current.num_descendants < self.graft_cap:
                ancestors.append(current)
            current = current.parent

        return ancestors

    def _evaluate_and_execute_graft(self, node, graft_paths):
        """评估所有可能的graft配对并执行最佳的一个

        Args:
            node: 当前节点
            graft_paths: graft路径信息
        """
        node_ancestors = graft_paths["node_ancestors"]
        target_ancestors = graft_paths["target_ancestors"]

        # 构建graft评分矩阵
        score_matrix = self._build_graft_score_matrix(node_ancestors, target_ancestors)

        # 找到最佳graft配对
        best_pair = self._find_best_graft_pair(
            node_ancestors, target_ancestors, score_matrix
        )

        # 更新统计信息
        num_considered = len(node_ancestors) * len(target_ancestors)
        self.number_of_grafts_considered += num_considered
        self.this_number_of_grafts_considered = num_considered

        # 执行graft（如果找到有效的配对）
        if best_pair:
            self._execute_graft(node, best_pair)
        else:
            logging.debug("[graft] No beneficial graft found")

    def _build_graft_score_matrix(self, node_ancestors, target_ancestors):
        """构建graft评分矩阵

        评分矩阵的每个元素 [i,j] 表示将 node_ancestors[i] 和
        target_ancestors[j] 进行graft的收益。

        Args:
            node_ancestors: 节点侧的祖先列表
            target_ancestors: 目标侧的祖先列表

        Returns:
            dict: 包含评分矩阵和相关信息
        """
        M = len(node_ancestors)
        N = len(target_ancestors)

        # 计算如果graft的相似度分数
        graft_scores = np.zeros((M, N), dtype=np.float32)
        for i, n1 in enumerate(node_ancestors):
            for j, n2 in enumerate(target_ancestors):
                graft_scores[i, j] = n1.compute_similarity(n2, sim_type=self.sim_type)

        # 获取当前父节点的分数（baseline）
        node_parent_scores = self._get_parent_scores(node_ancestors)
        target_parent_scores = self._get_parent_scores(target_ancestors)

        return {
            "graft_scores": graft_scores,
            "node_parent_scores": node_parent_scores,
            "target_parent_scores": target_parent_scores,
        }

    def _get_parent_scores(self, ancestors):
        """获取祖先节点的父节点分数

        Args:
            ancestors: 祖先节点列表

        Returns:
            np.ndarray: 父节点分数数组
        """
        scores = np.full(len(ancestors), -np.inf, dtype=np.float32)

        for i, node in enumerate(ancestors):
            if node.parent and len(node.parent.children) == 2:
                if hasattr(node.parent, "score") and node.parent.score is not None:
                    scores[i] = node.parent.score

        return scores

    def _find_best_graft_pair(self, node_ancestors, target_ancestors, score_matrix):
        """找到最佳的graft配对

        Graft条件：新的配对分数必须同时优于两个节点当前的父节点分数

        Args:
            node_ancestors: 节点侧祖先列表
            target_ancestors: 目标侧祖先列表
            score_matrix: 评分矩阵

        Returns:
            dict: 最佳配对信息，如果没有则返回None
        """
        graft_scores = score_matrix["graft_scores"]
        node_parent_scores = score_matrix["node_parent_scores"]
        target_parent_scores = score_matrix["target_parent_scores"]

        M, N = graft_scores.shape

        # 转换为矩阵格式以便广播
        node_parent_matrix = node_parent_scores.reshape(M, 1)
        target_parent_matrix = target_parent_scores.reshape(1, N)

        # 检查graft条件
        # 只有当graft分数同时优于两个父节点分数时才有效
        better_than_node_parent = graft_scores > node_parent_matrix
        better_than_target_parent = graft_scores > target_parent_matrix
        is_beneficial = better_than_node_parent & better_than_target_parent

        # 将非有益的graft设为0
        valid_scores = graft_scores.copy()
        valid_scores[~is_beneficial] = 0

        # 找到最佳配对
        best_idx = np.argmax(valid_scores)
        best_i = best_idx // N
        best_j = best_idx % N
        best_score = valid_scores[best_i, best_j]

        # 检查是否真的有有益的graft
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
        """执行graft操作

        Args:
            original_node: 原始节点（用于记录）
            graft_pair: graft配对信息
        """
        node1 = graft_pair["node1"]
        node2 = graft_pair["node2"]

        # 更新统计
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

        # 收集需要更新的起点
        update_starts = self._collect_update_starts(node2, original_node)

        # 执行实际的树重组
        self._perform_graft_operation(node1, node2)

        # 更新受影响的子树
        self._update_affected_subtrees(update_starts)

    def _collect_update_starts(self, node2, original_node):
        """收集需要更新的起点节点

        Args:
            node2: graft的目标节点
            original_node: 原始节点

        Returns:
            list: 需要更新的起点列表
        """
        update_starts = []

        if node2.parent and node2.parent.parent:
            update_starts.append(node2.parent.parent)

        if original_node.parent:
            update_starts.append(original_node.parent)

        return update_starts

    def _update_affected_subtrees(self, update_starts):
        """更新受graft影响的子树

        使用GNode的update_recursively方法更新所有受影响的节点。

        Args:
            update_starts: 更新起点列表
        """
        for start in update_starts:
            if start is not None:
                start.update_recursively(norm=self.norm, sim_type=self.sim_type)

    def _perform_graft_operation(self, node1, node2):
        """实际执行graft操作，重新组织树结构

        这个方法实现了原始grinch中的make_sibling逻辑

        Args:
            node1: 第一个要graft的节点
            node2: 第二个要graft的节点
        """
        logging.debug(
            "_perform_graft_operation(node1=%s, node2=%s)", node1.id, node2.id
        )

        # 保存node2的旧父节点和祖父节点
        node2_parent = node2.parent
        node2_sibling = None
        if node2_parent and len(node2_parent.children) == 2:
            siblings = node2.siblings()
            if siblings:
                node2_sibling = siblings[0]

        # 如果node2有父节点，需要处理node2的兄弟节点
        if node2_parent and node2_sibling:
            node2_grandparent = node2_parent.parent

            # 将node2的兄弟节点提升到node2父节点的位置
            node2_parent.remove_child(node2_sibling)
            node2_parent.remove_child(node2)

            if node2_grandparent:
                node2_grandparent.remove_child(node2_parent)
                node2_grandparent.add_child(node2_sibling)
            else:
                # node2_parent是根节点
                node2_sibling.relink_parent(None)
                if self.root_node == node2_parent:
                    self.root_node = node2_sibling

        # 创建新的父节点连接node1和node2
        new_parent = GNode(dim=self.dim)

        # 保存node1的旧父节点
        node1_parent = node1.parent

        # 设置新父节点的位置
        if node1_parent:
            node1_parent.remove_child(node1)
            node1_parent.add_child(new_parent)
            new_parent.relink_parent(node1_parent)
        else:
            # node1是根节点
            new_parent.relink_parent(None)
            self.root_node = new_parent

        # 将node1和node2添加为新父节点的子节点
        new_parent.add_child(node1)
        new_parent.add_child(node2)

        # 更新新父节点的属性
        new_parent.num_descendants = node1.num_descendants + node2.num_descendants

        if node1.sum_vector is not None and node2.sum_vector is not None:
            new_parent.sum_vector = node1.sum_vector + node2.sum_vector
            new_parent.update_centroid(norm=self.norm)

        new_parent.score = node1.compute_similarity(node2, sim_type=self.sim_type)
        new_parent.needs_update_model = True
        new_parent.needs_update_desc = True

    def root(self):
        """返回树的根节点"""
        return self.root_node
