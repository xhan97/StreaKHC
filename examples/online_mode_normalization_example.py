"""
完全在线模式下的归一化示例

展示如何在 Grinch 完全在线模式中正确处理归一化
"""

import numpy as np
import sys

sys.path.insert(0, "/home/xinhan/project/code/StreaKHC")

from src.grinch.grinch_new import Grinch


def example_auto_normalization():
    """示例1: 使用 Grinch 内置的自动归一化（推荐）"""
    print("=" * 60)
    print("示例1: 使用 Grinch 内置的自动归一化")
    print("=" * 60)

    # 创建 Grinch 实例，设置 norm="l2"
    # Grinch 会在 insert() 时自动归一化每个输入向量
    grinch = Grinch(dim=128, sim="dot", norm="l2", rotate_cap=5000)  # 自动 L2 归一化

    print(f"Grinch 配置: norm={grinch.norm}, sim_type={grinch.sim_type}")

    # 插入数据 - 不需要手动归一化
    for i in range(100):
        # 生成随机向量（未归一化）
        point_vec = np.random.randn(128)

        # 直接插入 - Grinch 会自动归一化
        grinch.insert(point_id=i, point_vec=point_vec, point_label=f"point_{i}")

        if i == 0:
            # 验证第一个点已被归一化
            root_norm = np.linalg.norm(grinch.root_node.centroid)
            print(f"\n第一个点插入后:")
            print(f"  原始向量范数: {np.linalg.norm(point_vec):.6f}")
            print(f"  存储的centroid范数: {root_norm:.6f}")
            print(f"  已归一化: {np.isclose(root_norm, 1.0)}")

    # 获取聚类结果
    assignments = grinch.flat_clustering(threshold=0.5)
    num_clusters = len(set(assignments))

    print(f"\n聚类结果:")
    print(f"  处理了 {grinch.point_counter} 个点")
    print(f"  形成了 {num_clusters} 个聚类")
    print(f"  Rotate次数: {grinch.number_of_rotates}")
    print(f"  Graft次数: {grinch.number_of_grafts}")


def example_manual_normalization():
    """示例2: 手动归一化（性能优化）"""
    print("\n" + "=" * 60)
    print("示例2: 手动归一化（适合大规模数据）")
    print("=" * 60)

    from sklearn.preprocessing import normalize

    # 设置 norm="none" 因为我们会手动归一化
    grinch = Grinch(dim=128, sim="dot", norm="none", rotate_cap=5000)  # 禁用自动归一化

    print(f"Grinch 配置: norm={grinch.norm}, sim_type={grinch.sim_type}")
    print("手动归一化: 使用 sklearn.preprocessing.normalize")

    for i in range(100):
        # 生成随机向量
        point_vec = np.random.randn(128)

        # 手动 L2 归一化
        point_vec = normalize(point_vec.reshape(1, -1), norm="l2")[0]

        # 插入已归一化的向量
        grinch.insert(point_id=i, point_vec=point_vec, point_label=f"point_{i}")

        if i == 0:
            # 验证
            root_norm = np.linalg.norm(grinch.root_node.centroid)
            print(f"\n第一个点插入后:")
            print(f"  手动归一化后的向量范数: {np.linalg.norm(point_vec):.6f}")
            print(f"  存储的centroid范数: {root_norm:.6f}")

    assignments = grinch.flat_clustering(threshold=0.5)
    num_clusters = len(set(assignments))

    print(f"\n聚类结果:")
    print(f"  处理了 {grinch.point_counter} 个点")
    print(f"  形成了 {num_clusters} 个聚类")


def example_no_normalization():
    """示例3: 不使用归一化"""
    print("\n" + "=" * 60)
    print("示例3: 不使用归一化（原始向量）")
    print("=" * 60)

    # 设置 norm="none" 直接使用原始向量
    grinch = Grinch(dim=128, sim="dot", norm="none", rotate_cap=5000)  # 不归一化

    print(f"Grinch 配置: norm={grinch.norm}, sim_type={grinch.sim_type}")

    for i in range(100):
        # 生成随机向量（范围在 [-1, 1] 之间）
        point_vec = np.random.randn(128)

        # 直接插入原始向量
        grinch.insert(point_id=i, point_vec=point_vec, point_label=f"point_{i}")

        if i == 0:
            root_norm = np.linalg.norm(grinch.root_node.centroid)
            print(f"\n第一个点插入后:")
            print(f"  原始向量范数: {np.linalg.norm(point_vec):.6f}")
            print(f"  存储的centroid范数: {root_norm:.6f}")
            print(f"  未归一化（范数不等于1）")

    assignments = grinch.flat_clustering(threshold=0.5)
    num_clusters = len(set(assignments))

    print(f"\n聚类结果:")
    print(f"  处理了 {grinch.point_counter} 个点")
    print(f"  形成了 {num_clusters} 个聚类")


def example_l_inf_normalization():
    """示例4: L-infinity 归一化"""
    print("\n" + "=" * 60)
    print("示例4: L-infinity 归一化")
    print("=" * 60)

    # 使用 L-infinity 归一化
    grinch = Grinch(
        dim=128, sim="dot", norm="l_inf", rotate_cap=5000  # L-infinity 归一化
    )

    print(f"Grinch 配置: norm={grinch.norm}, sim_type={grinch.sim_type}")

    for i in range(100):
        point_vec = np.random.randn(128)

        grinch.insert(point_id=i, point_vec=point_vec, point_label=f"point_{i}")

        if i == 0:
            max_val = np.max(np.abs(grinch.root_node.centroid))
            print(f"\n第一个点插入后:")
            print(f"  原始向量最大绝对值: {np.max(np.abs(point_vec)):.6f}")
            print(f"  centroid最大绝对值: {max_val:.6f}")
            print(f"  已L-inf归一化: {np.isclose(max_val, 1.0)}")

    assignments = grinch.flat_clustering(threshold=0.5)
    num_clusters = len(set(assignments))

    print(f"\n聚类结果:")
    print(f"  处理了 {grinch.point_counter} 个点")
    print(f"  形成了 {num_clusters} 个聚类")


def comparison_example():
    """示例5: 对比不同归一化方式的效果"""
    print("\n" + "=" * 60)
    print("示例5: 对比不同归一化方式")
    print("=" * 60)

    # 生成相同的测试数据
    np.random.seed(42)
    test_data = [np.random.randn(64) for _ in range(200)]

    results = {}

    for norm_type in ["none", "l2", "l_inf"]:
        grinch = Grinch(dim=64, sim="dot", norm=norm_type, rotate_cap=5000)

        for i, point_vec in enumerate(test_data):
            grinch.insert(point_id=i, point_vec=point_vec.copy())

        assignments = grinch.flat_clustering(threshold=0.5)

        results[norm_type] = {
            "num_clusters": len(set(assignments)),
            "rotates": grinch.number_of_rotates,
            "grafts": grinch.number_of_grafts,
        }

    print("\n归一化方式对比:")
    print(f"{'Norm':>10} {'聚类数':>10} {'Rotate':>10} {'Graft':>10}")
    print("-" * 44)
    for norm_type, stats in results.items():
        print(
            f"{norm_type:>10} {stats['num_clusters']:>10} {stats['rotates']:>10} {stats['grafts']:>10}"
        )


if __name__ == "__main__":
    # 运行所有示例
    example_auto_normalization()
    example_manual_normalization()
    example_no_normalization()
    example_l_inf_normalization()
    comparison_example()

    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print(
        """
推荐使用方式:
1. 【推荐】自动归一化: norm="l2"
   - 代码简洁，Grinch 自动处理
   - 适合大多数场景

2. 手动归一化: norm="none" + sklearn.normalize
   - 性能略好（避免重复归一化）
   - 适合大规模数据或对性能敏感的场景

3. 不归一化: norm="none"
   - 适合已经归一化的数据
   - 或者数据本身尺度一致的情况

归一化选择:
- norm="l2": L2 归一化（最常用）
- norm="l_inf": L-infinity 归一化
- norm="none": 不归一化
    """
    )
