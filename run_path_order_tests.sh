#!/bin/bash
# Path order tests: path_order_inv (path 3 first) and path3_only
# Confirms whether pollution comes from path 1/2

set -e
NUM_RUNS=32
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BUILD_DIR="${BUILD_DIR:-build}"
OUTPUT_DIR="${OUTPUT_DIR:-repro_results}"
OUTPUT_DIR_ABS="$SCRIPT_DIR/$OUTPUT_DIR"
REPRO_BIN="$SCRIPT_DIR/$BUILD_DIR/repro_test"

echo "===== Path Order Tests ($NUM_RUNS runs each) ====="
echo ""

mkdir -p "$OUTPUT_DIR_ABS"

# Build
echo "[1/4] Building repro_test..."
cd "$SCRIPT_DIR/$BUILD_DIR"
make -j$(nproc) repro_test 2>/dev/null || make repro_test
cd "$SCRIPT_DIR"
echo ""

# Test 1: path_order_inv (path 3 runs FIRST, before path 1/2)
echo "[2/4] Running path_order_inv $NUM_RUNS times (path 3 first, no path 1/2 pollution)..."
for i in $(seq 1 $NUM_RUNS); do
    $REPRO_BIN "$OUTPUT_DIR_ABS/inv_run${i}_dump.txt" "$OUTPUT_DIR_ABS/inv_run${i}_match.bin" "path_order_inv" 2>/dev/null | tail -5 || true
done
echo ""

# Compare path_order_inv outputs
echo "[3/4] Comparing path_order_inv outputs..."
INV_IDENTICAL=true
REF_INV="$OUTPUT_DIR_ABS/inv_run1_dump.txt"
for i in $(seq 2 $NUM_RUNS); do
    F="$OUTPUT_DIR_ABS/inv_run${i}_dump.txt"
    if [ ! -f "$F" ] || ! diff -q "$REF_INV" "$F" >/dev/null 2>&1; then
        INV_IDENTICAL=false
        echo "*** path_order_inv run $i DIFFERS from run 1 ***"
        [ -f "$F" ] && diff "$REF_INV" "$F" | head -15
    fi
done
if $INV_IDENTICAL; then
    echo "*** path_order_inv: All $NUM_RUNS outputs IDENTICAL - path 3 is consistent when run first ***"
else
    echo "*** path_order_inv: Some outputs DIFFER ***"
fi
echo ""

# Test 2: path3_only (only path 3, no path 1/2/4)
echo "[4/4] Running path3_only $NUM_RUNS times..."
for i in $(seq 1 $NUM_RUNS); do
    $REPRO_BIN "$OUTPUT_DIR_ABS/p3only_run${i}_dump.txt" "" "path3_only" 2>/dev/null | tail -3 || true
done
echo ""

# Compare path3_only outputs
echo "Comparing path3_only outputs..."
P3ONLY_IDENTICAL=true
REF_P3="$OUTPUT_DIR_ABS/p3only_run1_dump.txt"
for i in $(seq 2 $NUM_RUNS); do
    F="$OUTPUT_DIR_ABS/p3only_run${i}_dump.txt"
    if [ ! -f "$F" ] || ! diff -q "$REF_P3" "$F" >/dev/null 2>&1; then
        P3ONLY_IDENTICAL=false
        echo "*** path3_only run $i DIFFERS from run 1 ***"
        [ -f "$F" ] && diff "$REF_P3" "$F" | head -15
    fi
done
if $P3ONLY_IDENTICAL; then
    echo "*** path3_only: All $NUM_RUNS outputs IDENTICAL - path 3 alone is deterministic ***"
else
    echo "*** path3_only: Some outputs DIFFER ***"
fi
echo ""

echo "===== Summary ====="
echo "path_order_inv (path 3 first): $($INV_IDENTICAL && echo "IDENTICAL" || echo "DIFFER")"
echo "path3_only (path 3 only):      $($P3ONLY_IDENTICAL && echo "IDENTICAL" || echo "DIFFER")"
echo ""
echo "Output files: $OUTPUT_DIR/inv_run*_dump.txt, $OUTPUT_DIR/p3only_run*_dump.txt"
