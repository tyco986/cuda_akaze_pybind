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
| gHammingMatch 本身 | ✅ 已确认确定 | test_gHammingMatch 已与 akazed.cu 同步；隔离测试 32 次输出一致 |
| 匹配阶段 (cuMatch 输出) | ❌ 待解决 | match/dist 在 32 次运行间有差异 |
| **根因** | **cuMatch/gHammingMatch 非确定性** | path 3、path 4 单独运行均不一致，非 path 1/2 污染 |

### 关键结论
- **问题不在** gHammingMatch 算法（隔离测试：同一输入 → 输出一致）
- **问题不在** path 1/2 污染（path_order_inv、path3_only 均不一致）
- **问题在** cuMatch/gHammingMatch：path 3、path 4 单独运行均不一致；cuMatch 在「紧接 detect 后」执行时非确定

### 当前待办 (Next Steps)
1. ~~**Compute Sanitizer**~~：✅ initcheck/racecheck/memcheck 均已 0 errors
2. ~~**修复 gHammingMatch**~~：✅ 归约各步之间添加 `__syncthreads()`
3. ~~**修复 gDownWithSmooth**~~：✅ 在 branch 3 第一次写入与 `if (at_edge)` 之间插入 `__syncthreads()`
4. **排查 cuMatch 非确定性**：已试 cudaDeviceSynchronize、h_data→d_data 刷新、GPU 状态清洗（cudaMemset 1MB），均无效；待试 initcheck shared（需 CUDA 2025.4+）

### 关键文件
| 文件 | 用途 |
|------|------|
| `repro_test.cpp` | 完整 repro 测试，支持 `--dump-match-input`、第三参数 `path_order_inv`/`path3_only`/`path4_only` |
| `run_reproducibility_test.sh` | 32 次 repro_test，比较入参和输出，并做 gHammingMatch 隔离测试 |
| `run_path_order_tests.sh` | path_order_inv + path3_only + path4_only 各 32 次，验证 path 1/2 是否污染 |
| `run_path3_only_match_input_test.sh` | path3_only + dump match_input，验证 detectAndCompute 是否确定 |
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

### 最新验证（racecheck 修复后，2025-03-07）

| 项目 | 结果 |
|------|------|
| gHammingMatch 入参 | ✅ 32 次完全一致 |
| 完整 pipeline 输出 | ❌ Run 2–32 均与 run1 不同（match/dist 差异） |
| 隔离测试 | ✅ 32 次 gHammingMatch 输出完全一致 |

### Path 顺序测试（2025-03-07）

| 测试 | 命令 | 结果 | 说明 |
|------|------|------|------|
| **path_order_inv** | `repro_test out.txt match.bin path_order_inv` | ❌ DIFFER | path 3 先执行（无 path 1/2 污染），32 次输出仍不一致 |
| **path3_only** | `repro_test out.txt "" path3_only` | ❌ DIFFER | 仅运行 path 3，32 次输出仍不一致 |
| **path4_only** | `repro_test out.txt "" path4_only` | ❌ DIFFER | 仅运行 path 4，32 次输出仍不一致 |

**结论**：path 1/2 **不是**污染源。path 3、path 4 单独运行均不一致，说明非确定性来自 **cuMatch/gHammingMatch**（在 detect 之后执行时的行为）。

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

## 当前根因：path 3 内部非确定性

### 现象
- **完整 4 路径场景**：path 3 的 cuMatch 输出 32 次不一致
- **path_order_inv**：path 3 先执行（无 path 1/2）→ 仍不一致
- **path3_only**：仅 path 3 → 仍不一致
- **path4_only**：仅 path 4 → 仍不一致
- **隔离场景**：test_gHammingMatch 用同一 match_input.bin → 32 次一致

### 结论（已修正）
**path 1/2 不是污染源**。path3_only 测试进一步确认：
- **detectAndCompute**：32 次 match_input.bin 完全一致（确定）
- **cuMatch**：相同输入下，紧接 detectAndCompute 后执行时输出不一致（非确定）
- **gHammingMatch 隔离**：从文件加载相同输入 → 32 次一致（确定）

根因：**cuMatch/gHammingMatch 在「紧接 detectAndCompute 后」执行时存在非确定性**，可能为 GPU 缓存、内存访问顺序或 detectAndCompute 残留状态影响。

### 数据流
```
path 3: detectAndCompute → [残留状态?] → cuMatch → gHammingMatch
                              ↓
         同一输入下 test_gHammingMatch 确定，但 path 3 内 cuMatch 非确定
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
| racecheck | ✅ 0 errors | gHammingMatch、gDownWithSmooth 已修复 |

### 排查结果（已执行）

#### 1. initcheck（未初始化内存）✅ 已解决
- **原状态**：219200 个错误（tmem 未初始化）→ 40672 个错误（d_data 未初始化）
- **修复**：方案 A（`allocMemory` 中 cudaMemset tmem）+ 方案 B1（`initAkazeData` 中 cudaMemset d_data）
- **当前状态**：**0 errors**

#### 2. memcheck（内存访问错误）✅ 已解决
- **状态**：**0 errors**
- **说明**：无越界、非法访问等问题

#### 3. racecheck（共享内存数据竞争）✅ 已解决
- **原状态**：6 个 errors + 6 个 warnings（gHammingMatch Read-Write、gDownWithSmooth Write-Write）
- **修复**：
  - **gHammingMatch**：归约各步仅让参与写入的线程执行（tid < 8/4/2、tid == 0），步间插入 `__syncthreads()`（放在 `if` 外，保证 block 内所有线程执行，避免死锁）
  - **gDownWithSmooth**（float + int）：在 branch 3 第一次写入 `sdata[toy][tix]` 与 `if (at_edge)` 之间插入 `__syncthreads()`，保证所有线程完成写入后再读取 `sdata[new_toy][tix]`
- **当前状态**：**0 errors, 0 warnings**

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

# 共享内存数据竞争检测（已修复，0 errors）
bash run_compute_sanitizer.sh racecheck

# 内存访问错误检测
bash run_compute_sanitizer.sh memcheck
```

### initcheck shared memory（需 CUDA 2025.4+）

```bash
bash run_compute_sanitizer.sh initcheck_shared
```

检查 gHammingMatch 的 `__shared__` 未初始化读取。**需 Compute Sanitizer 2025.4+**，当前 2025.1 会报 `unrecognised option '--initcheck-address-space'`。升级 CUDA 后可执行此检查；**Docker 用户**：改用 `nvidia/cuda:12.9.0-devel-ubuntu22.04` 等镜像。

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

## 修复记录（匹配阶段）

### 已修复（Compute Sanitizer 已定位）

1. **gHammingMatch 树形归约**（akazed.cu 约 2241–2249）✅  
   - **问题**：`vsmem[tid] += vsmem[tid+8/4/2/1]` 各步之间无 `__syncthreads()`，导致 Read-Write 竞争  
   - **修复**：每步仅让参与写入的线程执行，步间插入 `__syncthreads()`（放在 `if` 外，避免死锁）

2. **gDownWithSmooth**（akazed.cu 约 467–512，float 版本）✅  
   - **问题**：`at_edge` 为真时，`sdata[toy][tix]` 与 `sdata[new_toy][tix]` 可能被不同线程写入同一 shared 位置  
   - **修复**：在第一次写入与 `if (at_edge)` 之间插入 `__syncthreads()`

3. **fastakaze::gDownWithSmooth**（akazed.cu 约 3205–3250，int 版本）✅  
   - 同上，已做相同修复

### 下一步排查方向

4. ~~**detectAndCompute 与 cuMatch 之间**~~：已试 `cudaDeviceSynchronize()`、h_data→d_data 刷新，均无效  
5. ~~**GPU 状态清洗**~~：已试 cuMatch 前 cudaMalloc + cudaMemset(1MB) + cudaFree，无效  
6. **initcheck shared**：`--initcheck-address-space shared` 检查 gHammingMatch 的 `__shared__` 未初始化读取（需 CUDA 2025.4+，Docker 需用 `nvidia/cuda:12.9.0-devel-ubuntu22.04` 等镜像）  
7. **深入 gHammingMatch**：在相同输入、不同 GPU 状态下仍非确定，需排查 kernel 内是否有未覆盖的竞争或未初始化读取

---

## 测试脚本使用

```bash
# 运行完整可重复性测试（32 次，含入参比较与隔离测试）
./run_reproducibility_test.sh

# Path 顺序测试（path_order_inv + path3_only，各 32 次）
./run_path_order_tests.sh

# path3_only + match_input dump（验证 detectAndCompute 是否确定）
./run_path3_only_match_input_test.sh

# 手动 path 顺序测试
./build/repro_test repro_results/inv_dump.txt repro_results/inv_match.bin path_order_inv   # path 3 先执行
./build/repro_test repro_results/p3only_dump.txt "" path3_only   # 仅 path 3
./build/repro_test repro_results/p4only_dump.txt "" path4_only   # 仅 path 4

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
| `repro_results/inv_run*_dump.txt` | path_order_inv 输出（path 3 先执行） |
| `repro_results/p3only_run*_dump.txt` | path3_only 输出（仅 path 3） |
| `repro_results/p4only_run*_dump.txt` | path4_only 输出（仅 path 4） |
| `repro_results/run*_match_input.bin` | path 3 的 gHammingMatch 入参（用于比较与隔离测试） |
| `repro_results/isolate_run*.txt` | gHammingMatch 隔离测试输出（32 次独立进程，无 path 1/2 干扰） |
| `repro_results/run*_checkpoints.txt` | GDB 断点追踪 (num_pts 等) |
| `repro_results/run*_stdout.txt` | 每次运行的 stdout |
