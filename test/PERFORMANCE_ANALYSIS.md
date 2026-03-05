# AkazeAligner 性能分析

## 问题：为何 test_akaze_aligner 每张 ~1248ms，而 cuakaze 每张 ~60ms？

### 原因分析

| 阶段 | cuakaze (C++) | AkazeAligner (Python) |
|------|---------------|------------------------|
| 图像内存 | 预分配 GPU，复用 | 每次调用 cudaMalloc/cudaFree |
| 检测 | 2×fastDetect ~74ms/对 | 2×fast_detect ~80ms/对 |
| 匹配 | cuMatch ~0.02ms | match ~5ms |
| **NNDR 过滤** | **无** | **~1700ms (naive) / ~300ms (scipy)** |
| RANSAC | 无 | findHomography ~1ms |

### 瓶颈：NNDR 过滤

`_nndr_filter` 对每对 (template, image) 计算完整距离矩阵：
- 3000×3000 特征 → 9M 元素
- 朴素实现 `(desc_q - desc_t)**2` 创建 (N,M,61) 数组，~1700ms
- 使用 `scipy.spatial.distance.cdist` 优化后 ~300ms

cuakaze 不做 NNDR，只做 detect + match，因此快。

### 解决方案

1. **`--no-nndr`**：跳过 NNDR，仅用 RANSAC 过滤离群点，速度接近 cuakaze
2. **安装 scipy**：`pip install scipy`，NNDR 从 ~1700ms 降至 ~300ms
3. **C++ 端 NNDR**：在 match 中返回 top-2 距离，可进一步加速（需改 bindings）

### 实测耗时（1000×1000 图，~3000 特征）

| 模式 | 每张耗时 |
|------|----------|
| use_nndr=True, 无 scipy | ~1200ms |
| use_nndr=True, 有 scipy | ~400ms |
| **use_nndr=False** | **~86ms**（接近 cuakaze） |
