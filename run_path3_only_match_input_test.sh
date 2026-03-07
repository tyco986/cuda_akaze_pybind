#!/bin/bash
# path3_only + dump match_input: verify if detectAndCompute output is identical across 32 runs

set -e
NUM_RUNS=32
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BUILD_DIR="${BUILD_DIR:-build}"
OUTPUT_DIR="${OUTPUT_DIR:-repro_results}"
OUTPUT_DIR_ABS="$SCRIPT_DIR/$OUTPUT_DIR"
REPRO_BIN="$SCRIPT_DIR/$BUILD_DIR/repro_test"

echo "===== path3_only + match_input dump ($NUM_RUNS runs) ====="
echo ""

mkdir -p "$OUTPUT_DIR_ABS"

echo "[1/3] Running path3_only $NUM_RUNS times (with match_input dump)..."
for i in $(seq 1 $NUM_RUNS); do
    $REPRO_BIN "$OUTPUT_DIR_ABS/p3only_mi_run${i}_dump.txt" \
               "$OUTPUT_DIR_ABS/p3only_mi_run${i}_match_input.bin" \
               "path3_only" 2>/dev/null | tail -2 || true
done
echo ""

echo "[2/3] Comparing match_input.bin (detectAndCompute output before cuMatch)..."
MATCH_INPUT_SAME=true
REF_MI="$OUTPUT_DIR_ABS/p3only_mi_run1_match_input.bin"
for i in $(seq 2 $NUM_RUNS); do
    F="$OUTPUT_DIR_ABS/p3only_mi_run${i}_match_input.bin"
    if [ ! -f "$F" ]; then
        MATCH_INPUT_SAME=false
        echo "*** match_input run $i: file not found ***"
    elif ! cmp -s "$REF_MI" "$F"; then
        MATCH_INPUT_SAME=false
        echo "*** match_input run $i DIFFERS from run 1 ***"
    fi
done
if $MATCH_INPUT_SAME; then
    echo "*** All $NUM_RUNS match_input.bin IDENTICAL - detectAndCompute is deterministic in path3_only ***"
else
    echo "*** Some match_input.bin DIFFER - detectAndCompute is non-deterministic in path3_only ***"
fi
echo ""

echo "[3/3] Comparing full outputs (dump.txt, includes cuMatch result)..."
DUMP_SAME=true
REF_DUMP="$OUTPUT_DIR_ABS/p3only_mi_run1_dump.txt"
for i in $(seq 2 $NUM_RUNS); do
    F="$OUTPUT_DIR_ABS/p3only_mi_run${i}_dump.txt"
    if [ ! -f "$F" ] || ! diff -q "$REF_DUMP" "$F" >/dev/null 2>&1; then
        DUMP_SAME=false
        echo "*** dump run $i DIFFERS from run 1 ***"
        [ -f "$F" ] && diff "$REF_DUMP" "$F" | head -10
    fi
done
if $DUMP_SAME; then
    echo "*** All $NUM_RUNS dump outputs IDENTICAL ***"
else
    echo "*** Some dump outputs DIFFER ***"
fi
echo ""

echo "===== Summary ====="
echo "match_input.bin (detectAndCompute): $($MATCH_INPUT_SAME && echo "IDENTICAL" || echo "DIFFER")"
echo "dump.txt (full path3 output):       $($DUMP_SAME && echo "IDENTICAL" || echo "DIFFER")"
