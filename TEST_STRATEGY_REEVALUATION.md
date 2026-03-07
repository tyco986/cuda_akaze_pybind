# 测试方案重新评估 (Test Strategy Re-evaluation)

> 基于 path 顺序测试、隔离测试与代码审查的结论更新。

---

## 0. run_reproducibility_test.sh 覆盖范围与 diff 分析（2025-03 确认）

### 0.1 覆盖的函数 / 路径

| Path | 函数 | 覆盖的 kernel/逻辑 |
|------|------|-------------------|
| 启动 | setHistogram | 仅 repro_test 调用 |
| Path 1 | detectAndCompute (float) | akaze 命名空间：gConv2d, gDownWithSmooth, gFlow, gNldStep, gDerivate, gHessianDeterminant, gCalcExtremaMap, gNms/gNmsR, gRefine, gCalcOrient, gDescribe/gDescribe2, gBuildDescriptor 等 |
| Path 2 | fastDetectAndCompute (uchar) | fastakaze 命名空间：gConv2d, gDownWithSmooth, gFlowNaive, gNldStepNaive, gDerivate, gHessianDeterminant, gCalcExtremaMap, gNmsRNaive, gRefine, gCalcOrient, gDescribe2, gBuildDescriptor 等 |
| Path 3 | detectAndCompute + cuMatch | 同上 + cuMatch → **gHammingMatch** |
| Path 4 | fastDetectAndCompute + cuMatch | 同上 + cuMatch → **gHammingMatch** |

**结论**：`run_reproducibility_test.sh` 覆盖了 pipeline 中所有主要函数（setHistogram、detectAndCompute、fastDetectAndCompute、cuMatch/gHammingMatch）。

### 0.2 diff 按 section 拆分（run1 vs run2）

对 `diff repro_results/run1_dump.txt repro_results/run2_dump.txt` 按行号映射到 dump 的 section：

| 行号 | Section | 说明 |
|------|---------|------|
| 124, 133, 136, 142–144, 147 | [detectAndCompute_float_match1] | **Path 3** cuMatch 输出，match/dist 不同 |
| 220, 226, 229, 231–232, 239, 242, 255, 259 | [fastDetectAndCompute_uchar_match1] | **Path 4** cuMatch 输出，match/dist 不同 |

- Path 1 [detectAndCompute_float]、Path 2 [fastDetectAndCompute_uchar]：无 match/dist（均为 0），diff 中**无差异**
- Path 3、Path 4 的 match1 区：**均有 diff**，差异仅在 match/dist 字段

### 0.3 不一致来源结论

| 问题 | 结论 |
|------|------|
| 不一致是否仅来自 gHammingMatch？ | **是**。Path 3 和 Path 4 都调用 cuMatch → gHammingMatch，两处 match/dist 均不一致 |
| 其他函数（detect、describe 等）是否也有不一致？ | **否**。Path 1/2 的 detect 输出、desc_checksum 在 diff 中无差异；match_input.bin 32 次一致，说明 detectAndCompute 输出确定 |
| 之前是否漏排查？ | Path 4 的 cuMatch 输出之前未单独拆分 diff，现已确认 **Path 4 也有不一致**，与 Path 3 同源（gHammingMatch） |

---

## 1. 当前证据汇总

| 测试 | 结果 | 说明 |
|------|------|------|
| 完整 pipeline (path 1→2→3→4) | ❌ match/dist 不一致 | 32 次运行输出不同 |
| gHammingMatch 入参 (match_input.bin) | ✅ 32 次完全一致 | 完整 pipeline 中 path 3 前 dump |
| path_order_inv (path 3 先执行) | ❌ 不一致 | 无 path 1/2 污染，仍不一致 |
| path3_only (仅 path 3) | ❌ 不一致 | 完全隔离，仍不一致 |
| path4_only (仅 path 4) | ❌ 不一致 | 完全隔离，仍不一致；与 path3_only 同源（gHammingMatch） |
| 隔离测试 (test_gHammingMatch) | ✅ 32 次一致 | 同一 match_input.bin → 输出一致 |

---

## 2. 关键矛盾

- **完整 pipeline**：入参一致，但 cuMatch 输出不一致
- **隔离测试**：同一入参 → 输出一致
- **path3_only**：无 path 1/2，仍不一致

若 gHammingMatch 在相同输入下确定，则完整 pipeline 中 cuMatch 输出应一致，与事实不符。

---

## 3. 重要发现：test_gHammingMatch 与 akazed.cu 实现不同（已修复）

> **注**：以下为历史发现。test_gHammingMatch 已与 akazed.cu 同步（§7 任务 1 ✅）。

**原 test_gHammingMatch.cu**（约 84–89 行）的归约逻辑：

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

### 优先级 3：detectAndCompute 与 cuMatch 之间 ✅ 已试

4. ~~**在 cuMatch 前加 cudaDeviceSynchronize**~~ ✅ 已试，无效（已保留在 akaze.cpp 中作为正确同步）

5. ~~**在 cuMatch 前对 match/dist 做 cudaMemset**~~（未单独试；GPU 状态清洗 cudaMemset 1MB 已试，无效）

### 优先级 4：工具与版本

6. **initcheck shared**（需 CUDA 2025.4+）
   - 当前 Compute Sanitizer 2025.1 不支持
   - 升级后可检查 `__shared__` 未初始化读取
   - Docker 用户：改用 `nvidia/cuda:12.9.0-devel-ubuntu22.04` 等镜像

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
| 4 | cuMatch 前加 cudaDeviceSynchronize 并重测 | ✅ 已试，无效；h_data→d_data 刷新也无效 |
| 5 | initcheck shared（需 CUDA 2025.4+） | 暂缓 |
| 6 | cuMatch 前 GPU 状态清洗（cudaMemset 1MB） | ✅ 已试，无效 |

---

## 8. 测试方案重新评估（2025-03 完整版）

### 8.1 测试脚本总览

| 脚本 | 用途 | 覆盖范围 |
|------|------|----------|
| `run_reproducibility_test.sh` | 主测试：32 次完整 pipeline + match_input 比较 + 隔离测试 | Path 1–4、gHammingMatch 隔离 |
| `run_path_order_tests.sh` | path_order_inv + path3_only + path4_only 各 32 次 | 验证 path 1/2 是否污染 |
| `run_path3_only_match_input_test.sh` | path3_only + dump match_input | 验证 detectAndCompute 在 path3 中是否确定 |
| `run_match_repro_test.sh` | `--dump-match-input` 生成 32 份输入 + 隔离测试 | 简化版入参比较（仅 detectAndCompute，无 path 1/2/3/4） |
| `run_compute_sanitizer.sh` | initcheck / racecheck / memcheck | 未初始化内存、数据竞争、越界 |

### 8.2 覆盖缺口

| 缺口 | 说明 | 建议 |
|------|------|------|
| **path4_only** | ✅ 已实现 | repro_test 支持 `path4_only`；run_path_order_tests.sh 已包含，32 次 → DIFFER |
| **GDB trace** | `gdb_trace_functions.gdb` 不存在，run_reproducibility_test.sh 中 GDB 段会失败 | 创建该文件或移除/跳过 GDB 段 |
| **diff 按 section 输出** | 当前 diff 混在一起，不便于快速定位 | 可增加按 section 拆分的 diff（如只 diff match1 区） |

### 8.3 冗余与简化

| 项目 | 说明 |
|------|------|
| run_match_repro_test vs run_reproducibility_test | `run_match_repro_test.sh` 用 `--dump-match-input` 只跑 detectAndCompute，生成 match_input；`run_reproducibility_test.sh` 用完整 4 path 并 dump path 3 前的 match_input。两者入参来源不同（前者无 path 1/2，后者有）。若需「完整 pipeline 下的入参」，应优先用 run_reproducibility_test。 |
| path_order_tests vs path3_only_match_input | path_order_tests 含 path3_only；path3_only_match_input 额外 dump match_input。后者更细，可保留；path_order_tests 可只保留 path_order_inv。 |

### 8.4 根因与下一步

**当前根因**：gHammingMatch 在「紧接 detectAndCompute 后」执行时非确定；同一输入在隔离进程中确定。

**已尝试无效**：cudaDeviceSynchronize、h_data→d_data 刷新。

**建议下一步（按优先级）**：

1. **path4_only 验证** ✅ 已完成：path4_only 32 次 → **DIFFER**，与 path3_only 一致，确认 path 4 单独时 cuMatch 也不一致。
2. **cuMatch 前 GPU 状态清洗** ✅ 已试：cuMatch 前 cudaMalloc + cudaMemset(1MB) + cudaFree **无效**，path3_only/path4_only 仍 DIFFER。
3. **initcheck shared**：升级到 CUDA 2025.4+ 后，检查 gHammingMatch 的 `__shared__` 未初始化读取。
4. **修复 GDB 段**：创建 `gdb_trace_functions.gdb` 或改为 `[ -f gdb_trace_functions.gdb ]` 再执行，避免静默失败。
