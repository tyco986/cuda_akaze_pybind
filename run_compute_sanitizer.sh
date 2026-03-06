#!/bin/bash
# Run NVIDIA Compute Sanitizer to detect uninitialized memory and data races.
# Use on a GPU that Compute Sanitizer supports (see "Device not supported" below).
#
# Usage: ./run_compute_sanitizer.sh [initcheck|racecheck|memcheck]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SANITIZER="${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer"
REPRO_BIN="$SCRIPT_DIR/build/repro_test"
OUT_DIR="$SCRIPT_DIR/repro_results"
TOOL="${1:-initcheck}"

if [ ! -f "$SANITIZER" ]; then
    echo "Compute Sanitizer not found at $SANITIZER"
    exit 1
fi
if [ ! -f "$REPRO_BIN" ]; then
    echo "Build repro_test first: cd build && make repro_test"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "===== Compute Sanitizer: $TOOL ====="
echo "Running: $SANITIZER --tool $TOOL ./repro_test ..."
echo ""

case "$TOOL" in
    initcheck)
        # 未初始化内存检测 (global memory; 新版 CUDA 支持 --initcheck-address-space shared)
        $SANITIZER --tool initcheck \
            "$REPRO_BIN" "$OUT_DIR/cs_dump.txt" "$OUT_DIR/cs_match.bin" 2>&1
        ;;
    racecheck)
        # 共享内存数据竞争检测
        $SANITIZER --tool racecheck \
            "$REPRO_BIN" "$OUT_DIR/cs_dump.txt" "$OUT_DIR/cs_match.bin" 2>&1
        ;;
    memcheck)
        # 内存访问错误检测
        $SANITIZER --tool memcheck \
            "$REPRO_BIN" "$OUT_DIR/cs_dump.txt" "$OUT_DIR/cs_match.bin" 2>&1
        ;;
    *)
        echo "Usage: $0 [initcheck|racecheck|memcheck]"
        echo "  initcheck : 未初始化内存读取"
        echo "  racecheck : 共享内存数据竞争"
        echo "  memcheck  : 内存越界/访问错误"
        exit 1
        ;;
esac
