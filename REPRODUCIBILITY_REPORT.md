# 可重复性测试报告 (Reproducibility Test Report)

> **新 Agent 上下文恢复**：本文档包含完整排查历程与当前状态。新 Agent 请先阅读「快速上下文」和「当前待办」，再按需查阅各章节。

---

## 快速上下文 (Quick Context for New Agent)

### 项目
- **cuda_akaze_pybind**：基于 CUDA 的 AKAZE 特征检测与匹配
- **问题**：32 次完整 pipeline 运行后，match/dist 结果不一致

### 当前状态一览

| 项目 | 状态 | 说明 |
|------|------|------|
| 检测阶段 (detectAndCompute) | ✅ 已解决 | x, y, octave, response, size, angle 及描述子 32 次完全一致 |
| gHammingMatch 入参 | ✅ 已确认一致 | 32 次完整运行中，传给 gHammingMatch 的 points1/points2 完全相同 |
| gHammingMatch 本身 | ✅ 已确认确定 | 隔离测试：同一份 match_input.bin 在 32 个独立进程中运行 → 输出完全一致 |
| 匹配阶段 (cuMatch 输出) | ❌ 待解决 | match/dist 在 32 次运行间有差异 |
| **根因** | **运行时状态污染** | path 1、path 2 在 path 3 之前执行，其 GPU 状态影响 path 3 的 gHammingMatch 行为 |

### 关键结论
- **问题不在** gHammingMatch 算法或上游输入顺序（两者均已排除）
- **问题在** path 1/2 的 GPU 状态污染 path 3 的 gHammingMatch 执行环境

### 当前待办 (Next Steps)
1. ~~**Compute Sanitizer**~~：✅ 已执行 initcheck/racecheck，定位到 gHammingMatch 树形归约竞争、gDownWithSmooth shared 写冲突
2. **修复 gHammingMatch**：在 akazed.cu 约 2241–2249 的归约循环各步之间添加 `__syncthreads()`
3. **修复 gDownWithSmooth**：解决 `sdata[toy]` 与 `sdata[new_toy]` 的 Write-Write 竞争

### 关键文件
| 文件 | 用途 |
|------|------|
| `repro_test.cpp` | 完整 repro 测试，支持 `--dump-match-input` 和第二个参数 dump path 3 的 match 入参 |
| `run_reproducibility_test.sh` | 32 次 repro_test，比较入参和输出，并做 gHammingMatch 隔离测试 |
| `run_match_repro_test.sh` | 32 次 dump 入参并比较 |
| `run_compute_sanitizer.sh` | 调用 Compute Sanitizer（initcheck/racecheck/memcheck） |
| `test_gHammingMatch.cu` | gHammingMatch 独立测试 |
| `akazed.cu` | gHammingMatch 约 2170 行，使用 `__shared__` 数组 |

---

## 测试结果摘要 (32 次运行)

| 阶段 | 状态 | 说明 |
|------|------|------|
| 检测 (detectAndCompute / fastDetectAndCompute) | ✅ 已解决 | x, y, octave, response, size, angle 及描述子 32 次完全一致 |
| 匹配 (cuMatch) | ❌ 待解决 | match、dist 字段在不同运行间有差异 |

---

## 已排除的假设 (Resolved / Ruled Out)

### ✅ 假设 1：gHammingMatch 本身非确定
- **验证**：用同一份 `match_input.bin` 在 32 个独立进程中运行 `test_gHammingMatch` → 输出**完全一致**
- **结论**：gHammingMatch 在相同输入下是确定的

### ✅ 假设 2：上游输入（关键点顺序）在多次运行间不同
- **验证**：32 次完整 repro_test 中，path 3 的 cuMatch 前 dump 的 gHammingMatch 入参（points1、points2 等）**完全一致**
- **结论**：上游输入已排除，问题不在关键点顺序

### ✅ 假设 3：CUDA kernel 启动调度导致非确定性
- **验证**：已测试 `CUDA_LAUNCH_BLOCKING=1`，32 次运行仍不一致
- **结论**：排除 kernel 启动调度，问题在 kernel 内部逻辑或 GPU 状态

---

## 当前根因：运行时状态污染

### 现象
- **完整 4 路径场景**：path 1、path 2、path 3、path 4 依次执行
- **path 3 的 gHammingMatch**：入参 32 次一致，但输出 32 次不一致
- **隔离场景**：仅运行 path 3 的 gHammingMatch（无 path 1/2 干扰）→ 输出 32 次一致

### 结论
path 1、path 2 在 path 3 之前执行，其 GPU 状态（如 shared memory、寄存器残留、L2 缓存等）影响 path 3 的 gHammingMatch 行为。

### 数据流
```
path 1, path 2 执行 → GPU 状态被修改
    ↓
path 3: detectAndCompute (输入一致) → cuMatch → gHammingMatch
    ↓
gHammingMatch 收到相同的 points1/points2，但受 path 1/2 残留状态影响 → 输出不一致
```

---

## Compute Sanitizer 排查

### 目的
用 NVIDIA Compute Sanitizer 检测未初始化内存、数据竞争，定位污染源。

### 工具状态汇总
| 工具 | 状态 | 说明 |
|------|------|------|
| initcheck | ✅ 0 errors | 方案 A + B1 已修复 tmem、d_data 未初始化 |
| memcheck | ✅ 0 errors | 无越界/非法访问 |
| racecheck | ❌ 6 errors + 6 warnings | gHammingMatch、gDownWithSmooth 待修复 |

### 排查结果（已执行）

#### 1. initcheck（未初始化内存）✅ 已解决
- **原状态**：219200 个错误（tmem 未初始化）→ 40672 个错误（d_data 未初始化）
- **修复**：方案 A（`allocMemory` 中 cudaMemset tmem）+ 方案 B1（`initAkazeData` 中 cudaMemset d_data）
- **当前状态**：**0 errors**

#### 2. memcheck（内存访问错误）✅ 已解决
- **状态**：**0 errors**
- **说明**：无越界、非法访问等问题

#### 3. racecheck（共享内存数据竞争）❌ 待修复
- **状态**：检测到 **6 个 errors + 6 个 warnings**

| 位置 | 类型 | 行号 | 说明 |
|------|------|------|------|
| `akaze::gDownWithSmooth` | Write-Write | 475 vs 505 | `sdata[toy][tix]` 与 `sdata[new_toy][tix]` 可能被不同线程写入同一 shared 位置，`at_edge` 分支导致 `new_toy` 与 `toy` 重叠 |
| `fastakaze::gDownWithSmooth` | Write-Write | 3213 vs 3243 | 同上，int 版本 |
| **`akaze::gHammingMatch`** | **Read-Write** | **2245–2248** | **树形归约缺少 `__syncthreads()`**：`vsmem[tid] += vsmem[tid+8/4/2/1]` 各步之间无同步，导致 tid 读 `vsmem[tid+4]` 时 tid+4 正在写 |

- **与可重复性问题的关系**：`gHammingMatch` 的 shared memory 竞争直接导致 match/dist 非确定性，与报告中「path 3 的 cuMatch 输出不一致」相符。

### 当前限制
- 若在 Docker + WSL2 中报 `Device not supported`，可尝试：
  ```bash
  docker run -it --gpus all \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    <镜像名> \
    /bin/bash
  ```

### 使用方式
```bash
# 未初始化内存检测
bash run_compute_sanitizer.sh initcheck

# 共享内存数据竞争检测（推荐，已定位 gHammingMatch 问题）
bash run_compute_sanitizer.sh racecheck

# 内存访问错误检测
bash run_compute_sanitizer.sh memcheck
```

### 新版 CUDA
支持 `--initcheck-address-space shared` 检查 shared memory（gHammingMatch 使用 `__shared__` 数组）。

---

## 历史记录：检测阶段 (已解决)

以下问题已通过 `sortAkazePoints` 等修复解决，检测阶段当前可重复。

### gNmsRNaive (akaze 命名空间)
- **位置**: `akazed.cu` 约 1568 行
- **原因**: `atomicInc(&d_point_counter, 0x7fffffff)` 导致关键点写入顺序非确定性
- **状态**: ✅ 已解决

### fastakaze::gNmsRNaive
- 同上，Fast 版本存在相同顺序问题
- **状态**: ✅ 已解决

### 其他使用原子操作的函数（检测路径，当前输出稳定）
| 函数 | 原子操作 | 说明 |
|------|----------|------|
| gScharrContrast 相关 | atomicMax | 影响 kcontrast，当前输出稳定 |
| gCalcExtremaMap 相关 | atomicAdd | 直方图累加，当前输出稳定 |
| gCalcOrient | atomicAdd | 方向直方图，当前输出稳定 |

---

## 修复建议（匹配阶段）

### 优先修复（Compute Sanitizer 已定位）

1. **gHammingMatch 树形归约**（akazed.cu 约 2241–2249）  
   - **问题**：`vsmem[tid] += vsmem[tid+8/4/2/1]` 各步之间无 `__syncthreads()`，导致 Read-Write 竞争  
   - **修复**：在每步归约之间插入 `__syncthreads()`，或改用 warp shuffle / 原子操作等无竞争实现

2. **gDownWithSmooth**（akazed.cu 约 467–512）  
   - **问题**：`at_edge` 为真时，`sdata[toy][tix]` 与 `sdata[new_toy][tix]` 可能被不同线程写入同一 shared 位置  
   - **修复**：保证 `(toy,tix)` 与 `(new_toy,tix)` 不重叠，或对同一位置只由一个线程写入

3. **fastakaze::gDownWithSmooth**（akazed.cu 约 3205–3250）  
   - 同上，int 版本需做相同修复

### 备选方案

4. **显式重置 GPU 状态**：在 path 3 的 cuMatch 之前，`cudaDeviceSynchronize`、`cudaMemset` 未使用 buffer  
5. **隔离 path 3**：使 path 3 在单独进程或 `cudaDeviceReset` 后运行，验证是否消除污染

---

## 测试脚本使用

```bash
# 运行完整可重复性测试（32 次，含入参比较与隔离测试）
./run_reproducibility_test.sh

# 仅运行两次并比较 (无 GDB)
./build/repro_test repro_results/run1_dump.txt
./build/repro_test repro_results/run2_dump.txt
diff repro_results/run1_dump.txt repro_results/run2_dump.txt
```

### gHammingMatch 独立测试
```bash
# 方式 A：run_reproducibility_test.sh 已包含隔离测试（用 run1_match_input.bin 跑 32 次 test_gHammingMatch）
./run_reproducibility_test.sh

# 方式 B：run_match_repro_test.sh 单独跑（Phase 1 生成 32 份 match_input，Phase 3 用 match_input_1.bin 跑 32 次 test_gHammingMatch）
./run_match_repro_test.sh repro_results

# 方式 C：手动隔离测试（需先有 match_input.bin）
./build/repro_test repro_results/dump.txt repro_results/match_input.bin   # 生成 match_input.bin
for i in $(seq 1 32); do ./build/test_gHammingMatch repro_results/match_input.bin repro_results/iso_${i}.txt; done
diff repro_results/iso_1.txt repro_results/iso_2.txt   # 验证 32 次输出一致
```

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `repro_results/run*_dump.txt` | 完整检测输出 |
| `repro_results/run*_match_input.bin` | path 3 的 gHammingMatch 入参（用于比较与隔离测试） |
| `repro_results/isolate_run*.txt` | gHammingMatch 隔离测试输出（32 次独立进程，无 path 1/2 干扰） |
| `repro_results/run*_checkpoints.txt` | GDB 断点追踪 (num_pts 等) |
| `repro_results/run*_stdout.txt` | 每次运行的 stdout |
