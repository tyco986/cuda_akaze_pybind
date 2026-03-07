# Racecheck 修改方案评估

> 基于 Compute Sanitizer racecheck 报告（6 errors + 6 warnings）的详细分析与修复方案。

---

## 1. 问题概览

| 位置 | 类型 | 行号 | 优先级 | 与可重复性关系 |
|------|------|------|--------|----------------|
| **gHammingMatch** | Read-Write | 2245–2248 | **高** | 直接导致 match/dist 非确定性 |
| gDownWithSmooth (float) | Write-Write | 475 vs 505 | 中 | 检测阶段，当前输出稳定 |
| fastakaze::gDownWithSmooth | Write-Write | 3213 vs 3243 | 中 | 同上 |

---

## 2. gHammingMatch 树形归约

### 根因分析
```cpp
// 当前代码 (akazed.cu 约 2240–2249)
if (tid < 8)
{
    volatile int* vsmem = flags;
    vsmem[tid] += vsmem[tid + 8];   // Step 1
    vsmem[tid] += vsmem[tid + 4];   // Step 2: tid 0 读 vsmem[4]，同时 tid 4 写 vsmem[4] → 竞争
    vsmem[tid] += vsmem[tid + 2];   // Step 3
    vsmem[tid] += vsmem[tid + 1];   // Step 4
}
```

- 归约逻辑：将 `flags[0..15]` 求和到 `flags[0]`
- 竞争：Step 2 中 tid 0 读 `vsmem[4]` 时，tid 4 正在写 `vsmem[4]`
- 原因：各步之间无 `__syncthreads()`，上一步写未完成就被下一步读

### 修复方案（推荐）

在每步归约之间插入 `__syncthreads()`。**注意**：`__syncthreads()` 必须被 block 内所有线程执行，需放在 `if (tid < 8)` 之外。

```cpp
// 修复后
if (tid < 8)
{
    volatile int* vsmem = flags;
    vsmem[tid] += vsmem[tid + 8];
}
__syncthreads();
if (tid < 8)
{
    volatile int* vsmem = flags;
    vsmem[tid] += vsmem[tid + 4];
}
__syncthreads();
if (tid < 8)
{
    volatile int* vsmem = flags;
    vsmem[tid] += vsmem[tid + 2];
}
__syncthreads();
if (tid < 8)
{
    volatile int* vsmem = flags;
    vsmem[tid] += vsmem[tid + 1];
}
```

### 备选方案
- 使用 `__shfl_down_sync` 做 warp 内归约（需调整参与线程范围）
- 使用 `atomicAdd` 做归约（性能较差）

---

## 3. gDownWithSmooth（float 与 int 版本）

### 根因分析

`at_edge` 为真时存在两处写 `sdata`：
1. 第一处（约 469/3213）：`sdata[toy][tix]`，`toy = tiy + 2`
2. 第二处（约 505/3243）：`sdata[new_toy][tix]`，`new_toy` 由边界分支计算

**重叠场景**（branch 3: `siy + 4 >= swhp.y`）：
- `new_toy = toy + 2`
- 线程 (tiy=12)：`toy=14`，`new_toy=16` → 写 `sdata[16][tix]`
- 线程 (tiy=14)：`toy=16` → 写 `sdata[16][tix]`
- 同一 `sdata[16][tix]` 被两个线程写入 → Write-Write 竞争

语义上，`at_edge` 的写是镜像边界，应覆盖第一处写的结果。

### 修复方案（推荐）

在第一处写与 `at_edge` 写之间插入 `__syncthreads()`，保证：
1. 所有线程完成第一处写
2. 再由 `at_edge` 线程用正确的镜像值覆盖

```cpp
// 第一处写 (已有)
if (in_bounds)
    sdata[toy][tix] = ...;
else
    sdata[toy][tix] = 0.0f;  // 或 0 (int 版本)

__syncthreads();   // 新增：确保第一处写完成

if (at_edge)
{
    if (in_bounds)
        sdata[new_toy][tix] = ...;
    else
        sdata[new_toy][tix] = 0.0f;  // 或 0 (int 版本)
}
__syncthreads();
```

**需修改位置**：
- `akaze::gDownWithSmooth`：约 476 行后（`else { sdata[toy][tix] = 0.0f; }` 之后、`if (at_edge)` 之前）
- `fastakaze::gDownWithSmooth`：约 3214 行后，同样位置

---

## 4. 实施顺序建议

1. **gHammingMatch**：优先修复，与可重复性直接相关
2. **gDownWithSmooth (float)**：在 `akazed.cu` 约 476 行后添加 `__syncthreads()`
3. **fastakaze::gDownWithSmooth**：在 `akazed.cu` 约 3214 行后添加 `__syncthreads()`

---

## 5. 验证

```bash
# 运行 racecheck
bash run_compute_sanitizer.sh racecheck

# 若 racecheck 通过，运行可重复性测试
./run_reproducibility_test.sh

# 运行主程序
./build/cuakaze 0 1
```
