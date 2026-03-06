#!/bin/bash
# Phase 1: Run full pipeline 32 times, save gHammingMatch inputs from each run.
# Phase 2: Compare the 32 input files - are they identical?
# Phase 3 (optional): Run test_gHammingMatch 32 times with run1's input and compare outputs.
#
# Usage: ./run_match_repro_test.sh [out_dir]
#   out_dir: where to save match_input_1.bin .. match_input_32.bin (default: repro_results)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPRO_BIN="$SCRIPT_DIR/build/repro_test"
TEST_BIN="$SCRIPT_DIR/build/test_gHammingMatch"
OUT_DIR="${1:-repro_results}"
NUM_RUNS=32

mkdir -p "$OUT_DIR"

echo "=== Phase 1: Run full pipeline $NUM_RUNS times, save gHammingMatch inputs ==="
for i in $(seq 1 $NUM_RUNS); do
    echo "Run $i/$NUM_RUNS -> $OUT_DIR/match_input_${i}.bin"
    $REPRO_BIN --dump-match-input "$OUT_DIR/match_input_${i}.bin" 2>/dev/null
done

echo ""
echo "=== Phase 2: Compare the $NUM_RUNS input files ==="
INPUTS_SAME=true
for i in $(seq 2 $NUM_RUNS); do
    if ! cmp -s "$OUT_DIR/match_input_1.bin" "$OUT_DIR/match_input_${i}.bin"; then
        INPUTS_SAME=false
        echo "*** Input run $i DIFFERS from run 1 ***"
    fi
done
if $INPUTS_SAME; then
    echo "*** All $NUM_RUNS gHammingMatch inputs IDENTICAL ***"
else
    echo "*** Some gHammingMatch inputs DIFFER (problem is upstream) ***"
fi

echo ""
echo "=== Phase 3: Run test_gHammingMatch $NUM_RUNS times with run1's input, compare outputs ==="
if [ ! -f "$TEST_BIN" ]; then
    echo "Skipping Phase 3: $TEST_BIN not found (build test_gHammingMatch first)"
else
    for i in $(seq 1 $NUM_RUNS); do
        $TEST_BIN "$OUT_DIR/match_input_1.bin" "$OUT_DIR/match_run${i}.txt" 2>/dev/null
    done
    OUTPUTS_SAME=true
    for i in $(seq 2 $NUM_RUNS); do
        if ! diff -q "$OUT_DIR/match_run1.txt" "$OUT_DIR/match_run${i}.txt" >/dev/null 2>&1; then
            OUTPUTS_SAME=false
            echo "*** gHammingMatch output run $i DIFFERS from run 1 ***"
        fi
    done
    if $OUTPUTS_SAME; then
        echo "*** All $NUM_RUNS gHammingMatch outputs IDENTICAL (with same input) ***"
    else
        echo "*** Some gHammingMatch outputs DIFFER ***"
    fi
fi
