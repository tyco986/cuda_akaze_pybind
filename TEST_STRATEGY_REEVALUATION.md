# 测试方案重新评估 (Test Strategy Re-evaluation)

> 基于 path 顺序测试、隔离测试与代码审查的结论更新。

---

## 1. 当前证据汇总

| 测试 | 结果 | 说明 |
|------|------|------|
| 完整 pipeline (path 1→2→3→4) | ❌ match/dist 不一致 | 32 次运行输出不同 |
| gHammingMatch 入参 (match_input.bin) | ✅ 32 次完全一致 | 完整 pipeline 中 path 3 前 dump |
| path_order_inv (path 3 先执行) | ❌ 不一致 | 无 path 1/2 污染，仍不一致 |
| path3_only (仅 path 3) | ❌ 不一致 | 完全隔离，仍不一致 |
| 隔离测试 (test_gHammingMatch) | ✅ 32 次一致 | 同一 match_input.bin → 输出一致 |

---

## 2. 关键矛盾

- **完整 pipeline**：入参一致，但 cuMatch 输出不一致
- **隔离测试**：同一入参 → 输出一致
- **path3_only**：无 path 1/2，仍不一致

若 gHammingMatch 在相同输入下确定，则完整 pipeline 中 cuMatch 输出应一致，与事实不符。

---

## 3. 重要发现：test_gHammingMatch 与 akazed.cu 实现不同

**test_gHammingMatch.cu**（约 84–89 行）的归约逻辑：

```cpp
if (tid < 8) {
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
}
```

- 无 `__syncthreads()` 步间同步
- 与 akazed.cu 中已修复的版本不同

**akazed.cu** 中已修复版本：每步单独 `if`，步间有 `__syncthreads()`。

**结论**：隔离测试实际运行的是 test_gHammingMatch 的旧实现，而非 akazed.cu 中修复后的 gHammingMatch。因此「gHammingMatch 本身确定」的结论需重新审视。

---

## 4. 修正后的假设

| 假设 | 状态 | 说明 |
|------|------|------|
| path 1/2 污染 path 3 | ❌ 已排除 | path_order_inv、path3_only 均不一致 |
| gHammingMatch 在相同输入下确定 | ✅ 已确认 | 同步后 test_gHammingMatch 隔离测试 32 次一致 |
| detectAndCompute 输出一致 | ✅ 完整 pipeline 中成立 | match_input 32 次一致 |
| path3_only 中 detectAndCompute 输出一致 | ✅ 已确认 | 32 次 match_input.bin 完全一致 |

---

## 5. 建议的测试方案（按优先级）

### 优先级 1：统一 gHammingMatch 实现并重测 ✅ 已完成

1. **同步 test_gHammingMatch 与 akazed.cu** ✅
   - 已将 test_gHammingMatch 的归约逻辑改为与 akazed.cu 一致（步间 `__syncthreads()`）

2. **重新做隔离测试** ✅
   - 用 match_input.bin 跑 32 次更新后的 test_gHammingMatch
   - **结果**：32 次输出完全一致
   - **结论**：gHammingMatch（与 akazed.cu 同步后）在相同输入下确定

### 优先级 2：验证 path3_only 的输入 ✅ 已完成

3. **path3_only 下 dump match_input** ✅
   - 运行 `./run_path3_only_match_input_test.sh`
   - **结果**：32 次 match_input.bin 完全一致
   - **结论**：detectAndCompute 在 path3_only 中确定；问题在 **cuMatch**（gHammingMatch 在 detectAndCompute 之后执行时非确定）

### 优先级 3：detectAndCompute 与 cuMatch 之间

4. **在 cuMatch 前加 cudaDeviceSynchronize**
   - 在 path 3 内、cuMatch 之前插入 `cudaDeviceSynchronize()`
   - 跑 32 次 path3_only，看 match/dist 是否稳定

5. **在 cuMatch 前对 match/dist 做 cudaMemset**
   - 对 data1.d_data、data2.d_data 中 match/dist 区域做 `cudaMemset`
   - 排除未初始化写入导致的不一致

### 优先级 4：工具与版本

6. **initcheck shared**（需 CUDA 2025.4+）
   - 当前 Compute Sanitizer 2025.1 不支持
   - 升级后可检查 `__shared__` 未初始化读取

7. **Path 1 vs Path 2 隔离**（优先级较低）
   - path 1+3 only vs path 2+3 only
   - 已确认 path 1/2 非主因，可作为补充验证

---

## 6. 测试执行顺序建议

```
1. 同步 test_gHammingMatch 与 akazed.cu
2. 重新跑隔离测试（32 次）
3. path3_only 下 dump match_input 并比较
4. 若 2、3 均通过，再试 cudaDeviceSynchronize / cudaMemset
```

---

## 7. 当前待办更新

| 序号 | 任务 | 状态 |
|------|------|------|
| 1 | 同步 test_gHammingMatch 归约逻辑与 akazed.cu | ✅ 已完成 |
| 2 | 重新做隔离测试（32 次） | ✅ 32 次输出一致 |
| 3 | path3_only 下 dump match_input 并比较 | ✅ match_input 一致，dump 不一致 → 问题在 cuMatch |
| 4 | cuMatch 前加 cudaDeviceSynchronize 并重测 | 待做 |
| 5 | initcheck shared（需 CUDA 2025.4+） | 暂缓 |
