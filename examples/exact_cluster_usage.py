"""
exact_cluster 工具的使用示例
展示如何使用重构后的工具函数
"""

import numpy as np
import sys

sys.path.insert(0, "/home/xinhan/project/code/StreaKHC")

from src.grinch.grinch_new import Grinch
from src.grinch.utils.exact_cluster import (
    cut_tree_grinch,
    get_grinch_tree_stats,
    get_node_similarity,
)


def example_basic_clustering():
    """基础聚类示例"""
    print("=" * 60)
    print("示例1: 基础聚类")
    print("=" * 60)

    # 创建数据：3个明显的簇
    np.random.seed(42)
    data_list = []

    # 簇1: 围绕 [1, 0, 0]
    for i in range(10):
        vec = np.array([1.0, 0.0, 0.0]) + np.random.randn(3) * 0.1
        data_list.append([vec, 0, i])  # [vector, label, point_id]

    # 簇2: 围绕 [0, 1, 0]
    for i in range(10, 20):
        vec = np.array([0.0, 1.0, 0.0]) + np.random.randn(3) * 0.1
        data_list.append([vec, 1, i])

    # 簇3: 围绕 [0, 0, 1]
    for i in range(20, 30):
        vec = np.array([0.0, 0.0, 1.0]) + np.random.randn(3) * 0.1
        data_list.append([vec, 2, i])

    # 构建 Grinch 树（完全在线模式）
    grinch = Grinch(dim=3, norm="l2", sim="dot")

    for vec, label, point_id in data_list:
        grinch.insert(point_id, vec, f"point_{point_id}")

    # 切割树以获得3个簇
    y_hat, y_true = cut_tree_grinch(grinch, data_list, n_cluster=3)

    # 评估结果
    print(f"数据点数: {len(y_true)}")
    print(f"真实簇数: {len(set(y_true))}")
    print(f"预测簇数: {len(set(y_hat))}")

    # 计算每个真实簇的预测情况
    for true_label in sorted(set(y_true)):
        indices = [i for i, label in enumerate(y_true) if label == true_label]
        predicted_labels = [y_hat[i] for i in indices]
        print(f"  真实簇 {true_label}: 预测标签 {set(predicted_labels)}")


def example_tree_statistics():
    """树统计信息示例"""
    print("\n" + "=" * 60)
    print("示例2: 树统计信息")
    print("=" * 60)

    # 创建并构建树
    grinch = Grinch(dim=10, norm="l2", sim="dot", rotate_cap=1000)

    print("逐步插入点并观察统计信息:\n")

    for n_points in [10, 50, 100]:
        # 重新创建树
        grinch = Grinch(dim=10, norm="l2", sim="dot")

        for i in range(n_points):
            vec = np.random.randn(10)
            grinch.insert(i, vec, f"p{i}")

        # 获取统计信息
        stats = get_grinch_tree_stats(grinch)

        print(f"插入 {n_points} 个点后:")
        print(f"  叶子数: {stats['num_leaves']}")
        print(f"  树深度: {stats['tree_depth']}")
        print(f"  Rotate次数: {stats['num_rotates']}")
        print(f"  Graft次数: {stats['num_grafts']}")
        print(
            f"  构建时间: {stats['time_in_search'] + stats['time_in_rotate'] + stats['time_in_graft']:.4f}s"
        )
        print()


def example_node_similarity():
    """节点相似度示例"""
    print("=" * 60)
    print("示例3: 节点相似度分析")
    print("=" * 60)

    # 创建简单的树
    grinch = Grinch(dim=2, norm="l2", sim="dot")

    # 插入5个点
    points = [
        np.array([1.0, 0.0]),
        np.array([0.9, 0.1]),
        np.array([0.0, 1.0]),
        np.array([0.1, 0.9]),
        np.array([0.5, 0.5]),
    ]

    for i, vec in enumerate(points):
        grinch.insert(i, vec, f"p{i}")

    # 分析树结构
    root = grinch.root()

    if root is not None:
        print(f"根节点相似度: {get_node_similarity(grinch, root):.4f}")
        print(f"根节点后代数: {root.num_descendants}")

        # 遍历内部节点
        def print_tree(node, depth=0):
            indent = "  " * depth
            if node.is_leaf():
                print(f"{indent}叶子 {node.id}")
            else:
                sim = get_node_similarity(grinch, node)
                print(
                    f"{indent}内部节点 (相似度={sim:.4f}, 后代={node.num_descendants})"
                )
                for child in node.children:
                    print_tree(child, depth + 1)

        print("\n树结构:")
        print_tree(root)


def example_dynamic_clustering():
    """动态聚类示例：边插入边聚类"""
    print("\n" + "=" * 60)
    print("示例4: 动态在线聚类")
    print("=" * 60)

    grinch = Grinch(dim=2, norm="l2", sim="dot")

    # 创建数据流
    data_stream = []
    for i in range(20):
        if i < 10:
            vec = np.array([1.0, 0.0]) + np.random.randn(2) * 0.1
            label = 0
        else:
            vec = np.array([0.0, 1.0]) + np.random.randn(2) * 0.1
            label = 1
        data_stream.append([vec, label, i])

    # 动态插入并定期检查聚类
    print("动态聚类过程:\n")

    for i, (vec, label, point_id) in enumerate(data_stream):
        grinch.insert(point_id, vec, f"p{point_id}")

        # 每5个点检查一次聚类
        if (i + 1) % 5 == 0:
            current_data = data_stream[: i + 1]
            y_hat, y_true = cut_tree_grinch(grinch, current_data, n_cluster=2)
            n_clusters = len(set(y_hat))

            print(f"插入 {i+1} 个点: 当前聚类数 = {n_clusters}")


def example_performance_comparison():
    """性能对比示例"""
    print("\n" + "=" * 60)
    print("示例5: 不同参数的性能对比")
    print("=" * 60)

    import time

    test_sizes = [50, 100, 200]

    for n in test_sizes:
        # 生成数据
        data = [(np.random.randn(10), 0, i) for i in range(n)]

        # 测试不同的 rotate_cap
        for rotate_cap in [100, 1000, 10000]:
            grinch = Grinch(dim=10, norm="l2", sim="dot", rotate_cap=rotate_cap)

            start_time = time.time()
            for vec, label, point_id in data:
                grinch.insert(point_id, vec, f"p{point_id}")
            build_time = time.time() - start_time

            stats = get_grinch_tree_stats(grinch)

            print(f"n={n}, rotate_cap={rotate_cap}:")
            print(f"  构建时间: {build_time:.4f}s")
            print(f"  Rotate次数: {stats['num_rotates']}")
            print(f"  树深度: {stats['tree_depth']}")


if __name__ == "__main__":
    print("exact_cluster 工具使用示例\n")

    example_basic_clustering()
    example_tree_statistics()
    example_node_similarity()
    example_dynamic_clustering()
    example_performance_comparison()

    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成")
    print("=" * 60)
