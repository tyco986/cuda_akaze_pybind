# 可重复性测试报告 (Reproducibility Test Report)

## 测试结果摘要

**结论: 检测到可重复性问题**

- `num_pts` 在两次运行中**一致** (3631)
- 关键点**集合**相同，但**顺序**在两次运行中**不一致**
- 相同的关键点 (x, y, 特征等) 出现在两次运行中，但索引位置不同

## 输出不一致的函数 (粗定位)

根据测试和代码分析，以下函数导致输出顺序不一致：

### 1. **gNmsRNaive** (CUDA kernel) - 主要根源
- **位置**: `akazed.cu` 约 1568 行 (akaze 命名空间)
- **原因**: 使用 `atomicInc(&d_point_counter, 0x7fffffff)` 分配输出索引
- **说明**: 多个线程并行发现局部极大值时，通过原子操作竞争获取写入槽位，执行顺序非确定性，导致关键点写入顺序随运行变化

### 2. **hNmsR** (Host 函数)
- **位置**: `akazed.cu` 约 2629 行
- **原因**: 调用 `gNmsRNaive` kernel，其输出顺序由 kernel 决定

### 3. **gNmsR** (CUDA kernel，当前未使用)
- **位置**: `akazed.cu` 约 1446 行
- **说明**: 注释中显示已被 gNmsRNaive 替代，同样使用 `atomicInc` (约 1435 行)

### 4. **gCalcOrient** (CUDA kernel) - 可能影响
- **位置**: `akazed.cu` 约 1715-1718 行
- **原因**: 使用 `atomicAdd` 累加方向直方图，可能影响角度计算的一致性
- **说明**: 若仅关注关键点顺序，主要问题在 NMS；若角度也有差异，需检查此处

## 其他使用原子操作的函数 (潜在影响)

| 函数 | 原子操作 | 影响 |
|------|----------|------|
| gScharrContrast / 相关 kernel | atomicMax | 影响 kcontrast |
| gCalcExtremaMap 相关 | atomicAdd (直方图) | 可能影响阈值 |
| fastakaze::gNmsRNaive | atomicInc | Fast 版本同样存在顺序问题 |

## 修复建议

1. **对关键点排序**: 在 `hNmsR` 返回后或 `detectAndCompute` 结束前，按 (x, y) 或 (octave, response) 等对关键点排序，使输出顺序确定
2. **替代 atomicInc**: 使用两阶段方法——先收集候选，再顺序写入，避免原子竞争
3. **固定随机性**: 若存在其他非确定性来源，考虑固定 CUDA 随机数种子或执行策略

## 测试脚本使用

```bash
# 运行完整可重复性测试
./run_reproducibility_test.sh

# 仅运行两次并比较 (无 GDB)
./build/repro_test repro_results/run1_dump.txt
./build/repro_test repro_results/run2_dump.txt
diff repro_results/run1_dump.txt repro_results/run2_dump.txt
```

## 输出文件

- `repro_results/run1_dump.txt`, `run2_dump.txt`: 完整检测输出
- `repro_results/run1_checkpoints.txt`, `run2_checkpoints.txt`: GDB 断点追踪 (num_pts 等)
