from src.grinch.grinch_new import Grinch
import numpy as np

# 测试更大的数据集
np.random.seed(42)
points = np.random.random((100, 10))
labels = np.random.randint(0, 5, 100)

g = Grinch(points=points, norm="l2", sim="dot")
g.build_dendrogram()

print(f"\n============= 修复后的 Grinch 统计 =============")
print(f"总点数: {g.num_points}")
print(f"Rotate次数: {g.number_of_rotates}")
print(f"Graft次数: {g.number_of_grafts}")
print(f"Rotate考虑次数: {g.number_of_rotates_considered}")
print(f"Graft考虑次数: {g.number_of_grafts_considered}")
print(f"============================================")
