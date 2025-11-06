# Grinch (GNode版本) 使用说明

## 概述

`grinch_new.py` 是基于 GNode 对象实现的 Grinch 层次聚类算法。与原始的数组版本（`grinch.py`）不同，这个版本使用面向对象的树结构，提供了更清晰的代码组织和更灵活的扩展性。

## 主要特性

### 1. 基于GNode的树结构
- 使用 `GNode` 类表示树中的每个节点
- 每个节点维护自己的质心、子节点、父节点等信息
- 更符合面向对象的设计原则

### 2. 核心算法保持一致
- **插入 (Insert)**: 逐点插入构建层次聚类树
- **旋转 (Rotate)**: 找到最佳插入位置
- **嫁接 (Graft)**: 优化树结构（当前版本使用简化实现）

### 3. 支持的参数
```python
Grinch(
    points=None,          # 数据点数组 (num_points, dim)
    num_points=None,      # 点的数量
    dim=None,             # 数据维度
    rotate_cap=100,       # rotate操作的容量限制
    graft_cap=100,        # graft操作的容量限制
    norm="l2",           # 质心归一化: "l2", "l_inf", "none"
    sim="dot",           # 相似度度量: "dot", "l2", "sql2"
    pids=None,           # 点的ID列表
    canopies=None,       # Canopy聚类结果
)
```

## 使用示例

### 基本用法

```python
import numpy as np
from src.grinch.grinch_new import Grinch

# 准备数据
num_points = 100
dim = 5
vectors = np.random.random((num_points, dim)).astype(np.float32)
labels = np.random.randint(0, 10, num_points)

# 创建Grinch实例
grinch = Grinch(points=vectors, norm="l2", sim="dot")

# 构建层次聚类树
grinch.build_dendrogram()

# 写入树文件
grinch.write_tree("output.tree", labels)

# 获取平坦聚类
assignments = grinch.flat_clustering(threshold=0.5)
```

### 增量插入

```python
# 逐个插入点
grinch = Grinch(num_points=100, dim=5)
grinch.points = vectors

for i in range(num_points):
    grinch.insert(i)
```

### 访问树结构

```python
# 获取根节点
root = grinch.root()

# 获取所有叶子节点
leaves = root.leaves()

# 计算树高度
height = root.height()

# 获取节点的后代
descendants = root.descendants()
```

## 与原始版本的区别

| 特性 | 原始版本 (grinch.py) | GNode版本 (grinch_new.py) |
|------|---------------------|--------------------------|
| 数据结构 | NumPy数组 | GNode对象 |
| 树表示 | parent数组 + children列表 | GNode树结构 |
| 内存管理 | 预分配固定大小数组 | 动态创建节点 |
| 代码复杂度 | 较高（索引操作） | 较低（对象操作） |
| 扩展性 | 较难 | 较易 |
| 性能 | 可能更快（数组操作） | 可能稍慢（对象开销） |

## 测试

运行测试套件：

```bash
conda activate shc
python test_grinch_new.py
```

运行演示：

```bash
python demo_grinch_new.py
```

## 算法详情

### 1. 插入过程 (Insert)

```
对于每个新点 p:
  1. 如果是第一个点，创建根节点
  2. 否则：
     a. 找到最近邻节点 nn
     b. 执行 rotate 找到最佳插入位置 sib
     c. 创建新的父节点连接 sib 和 p
     d. 更新从父节点到根的路径
     e. (可选) 执行 graft 优化树结构
```

### 2. 旋转 (Rotate)

```
找到最佳插入位置：
  从最近邻开始，向上遍历祖先
  选择与新点相似度最高的位置
  受 rotate_cap 容量限制
```

### 3. 嫁接 (Graft)

```
优化树结构：
  找到可能的重组位置
  计算所有候选配对的分数
  评估graft条件（必须优于当前父节点）
  执行最优的嫁接操作
  更新受影响的子树

Graft改进：
  - 完整实现原始Grinch的graft逻辑
  - 支持graft_cap容量限制
  - 维护树的二叉结构
  - 保证父子关系一致性
```

## 输出格式

### 树文件格式 (write_tree)

```
node_id    parent_id    label
0          123          class_0
1          123          class_1
...
123        456          None
...
-1         None         None
```

- 前 num_points 行是叶子节点（数据点）
- 后续行是内部节点
- 最后一行标记结束

### 平坦聚类 (flat_clustering)

返回一个数组，每个元素是对应点的聚类ID：

```python
assignments = [0, 0, 1, 1, 2, 2, ...]
```

## 性能统计

Grinch会跟踪以下统计信息：

- `search_time`: 搜索最近邻的时间
- `rotate_time`: 执行rotate操作的时间
- `graft_time`: 执行graft操作的时间
- `update_time`: 更新节点的时间
- `number_of_rotates`: rotate操作次数
- `number_of_grafts`: graft操作次数

使用 `grinch.stats_string()` 查看统计信息。

## 未来改进

1. ~~**完整的Graft实现**~~ ✅ **已完成！**
2. **性能优化**: 使用更高效的最近邻搜索（如KD树）
3. **并行化**: 支持多线程/多进程加速
4. **更多相似度度量**: 支持自定义相似度函数
5. **可视化**: 添加树结构可视化工具

## 参考文献

基于原始 Grinch 算法：
- Monath et al. "Gradient-based Hierarchical Clustering using Continuous Representations of Trees in Hyperbolic Space" (KDD 2019)

## 许可证

Apache License 2.0
