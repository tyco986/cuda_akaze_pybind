# Initcheck 修改方案重新评估

> 基于方案 A 实施后的 initcheck 结果（错误从 219200 降至 40672）重新评估。

---

## 1. 当前状态

| 阶段 | 错误数 | 根因 | 状态 |
|------|--------|------|------|
| 原始 | 219200 | `detect` 中 cudaMemcpy 的 source（tmem）未初始化 | ✅ 已修复（方案 A） |
| 方案 A 后 | 40672 | `sortAkazePoints` 中 thrust::sort 读取 `d_data[num_pts, max_pts)` | ❌ 待修复 |

---

## 2. 剩余问题分析：sortAkazePoints

### 调用链
```
detect() → hNmsR() 写入 [0, num_pts) → sortAkazePoints(d_data, num_pts)
```

### 根因
- `thrust::sort(ptr, ptr + num_pts)` 只对前 `num_pts` 个元素排序
- CUB 内部用 **256 元素块** 做 BlockLoad，块边界可能超出 `num_pts`
- 例：`num_pts=3631`，最后一块从 3584 起，会读到 3584–3839，其中 3631–3839 未初始化

### 数据流
- `initAkazeData`: `cudaMalloc(d_data, max_pts * sizeof(AkazePoint))`，未初始化
- `hNmsR`: 只写入 `[0, num_pts)`
- `[num_pts, max_pts)` 从未写入，但 CUB 会读到

---

## 3. 修订后的修改方案

### 方案 A（已完成）✅
**位置**: `akaze.cpp` → `allocMemory()`
```cpp
CHECK(cudaMemset(*addr, 0, offsets[noctaves] * sizeof(float)));
```
**作用**: 修复 tmem 未初始化导致的 cudaMemcpy source 问题

---

### 方案 B（新增，建议优先）— 修复 sortAkazePoints

**目标**: 在 `sortAkazePoints` 前，保证 `d_data[num_pts, max_pts)` 已初始化

#### 选项 B1：在 `initAkazeData` 中初始化（推荐）
**位置**: `akaze.cpp` → `initAkazeData()`
```cpp
if (dev) {
    CHECK(cudaMalloc((void**)&data.d_data, size));
    CHECK(cudaMemset(data.d_data, 0, size));  // 新增
}
```
**优点**: 一次性初始化，所有使用 `d_data` 的路径都受益  
**缺点**: 每次分配都做一次 memset，开销略增（约 40KB × 4 = 160KB）

#### 选项 B2：在 sort 前按需 memset
**位置**: `akaze.cpp` → `detect()` 和 `fastDetect()` 中，`sortAkazePoints` 调用前
```cpp
if (result.num_pts < result.max_pts) {
    CHECK(cudaMemset(result.d_data + result.num_pts, 0,
        (result.max_pts - result.num_pts) * sizeof(AkazePoint)));
}
sortAkazePoints(result.d_data, result.num_pts);
```
**优点**: 只 memset 尾部，更精确  
**缺点**: 需在 detect 和 fastDetect 两处各加一次

**建议**: 优先 B1，逻辑更简单

---

### 方案 C（可选）— oparams

**位置**: `akaze.cpp` → `detectAndCompute` / `fastDetectAndCompute`  
**现状**: `oparams` 由 `allocMemory` 完全覆盖，initcheck 未报错  
**结论**: 暂不做，保持现状即可

---

### 方案 D（可选）— h_data

**位置**: `akaze.cpp` → `initAkazeData()`  
**现状**: `h_data` 用于 cudaMemcpy2D 的 **destination**（DeviceToHost），不参与 initcheck 当前报错  
**结论**: 暂不做；若后续 initcheck 报告 host 未初始化，可再考虑 `memset(h_data, 0, size)`

---

## 4. 实施顺序建议

1. **方案 B1**：在 `initAkazeData` 中初始化 `d_data`  
2. 重新运行 `bash run_compute_sanitizer.sh initcheck`  
3. 若仍有 initcheck 报错，再按报错位置继续排查

---

## 5. 验证命令

```bash
# 运行 initcheck
bash run_compute_sanitizer.sh initcheck

# 若 initcheck 通过，再跑可重复性测试
./run_reproducibility_test.sh

# 运行主程序
./build/cuakaze 0 1
```
