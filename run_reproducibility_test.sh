#!/bin/bash
# Reproducibility test script - runs detection multiple times and compares outputs
# Identifies which functions produce inconsistent output between runs

set -e
NUM_RUNS=32
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BUILD_DIR="${BUILD_DIR:-build}"
OUTPUT_DIR="${OUTPUT_DIR:-repro_results}"
OUTPUT_DIR_ABS="$SCRIPT_DIR/$OUTPUT_DIR"
REPRO_BIN="$SCRIPT_DIR/$BUILD_DIR/repro_test"

echo "===== Reproducibility Test for CUDA-AKAZE ($NUM_RUNS runs) ====="
echo ""

# Ensure data exists
if [ ! -f "data/left.pgm" ]; then
    echo "Error: data/left.pgm not found"
    exit 1
fi

# Build with debug symbols for GDB
mkdir -p "$SCRIPT_DIR/$BUILD_DIR" "$OUTPUT_DIR_ABS"
echo "[1/4] Building project (Debug for GDB)..."
cd "$SCRIPT_DIR/$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE=Debug .. 2>/dev/null || cmake ..
make -j$(nproc) repro_test test_gHammingMatch 2>/dev/null || make repro_test test_gHammingMatch
cd "$SCRIPT_DIR"
echo ""

# Method 1: Simple dump comparison (run repro_test NUM_RUNS times)
# Pass second arg to dump gHammingMatch inputs before path3 cuMatch (full 4-path scenario)
echo "[2/4] Running repro_test $NUM_RUNS times (full 4 paths, dump match inputs before path3 cuMatch)..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS..."
    $REPRO_BIN "$OUTPUT_DIR_ABS/run${i}_dump.txt" "$OUTPUT_DIR_ABS/run${i}_match_input.bin" 2>/dev/null | tee "$OUTPUT_DIR_ABS/run${i}_stdout.txt" || true
done
echo ""

# Compare gHammingMatch inputs (path3, before cuMatch) - same scenario as full repro test
echo "[3/4] Comparing gHammingMatch inputs (32 runs, full 4-path scenario)..."
MATCH_INPUTS_SAME=true
REF_MATCH="$OUTPUT_DIR_ABS/run1_match_input.bin"
if [ -f "$REF_MATCH" ]; then
    for i in $(seq 2 $NUM_RUNS); do
        F="$OUTPUT_DIR_ABS/run${i}_match_input.bin"
        if [ ! -f "$F" ]; then
            MATCH_INPUTS_SAME=false
            echo "*** Match input run $i: file not found ***"
        elif ! cmp -s "$REF_MATCH" "$F"; then
            MATCH_INPUTS_SAME=false
            echo "*** Match input run $i DIFFERS from run 1 ***"
        fi
    done
    if $MATCH_INPUTS_SAME; then
        echo "*** All $NUM_RUNS gHammingMatch inputs IDENTICAL ***"
    else
        echo "*** Some gHammingMatch inputs DIFFER (problem is upstream of gHammingMatch) ***"
    fi
else
    echo "*** run1_match_input.bin not found - skip input comparison ***"
fi
echo ""

# Compare dump files (all against run1)
echo "[4/4] Comparing full outputs..."
ALL_IDENTICAL=true
REF="$OUTPUT_DIR_ABS/run1_dump.txt"
if [ ! -f "$REF" ]; then
    echo "Error: $REF not found"
    exit 1
fi
for i in $(seq 2 $NUM_RUNS); do
    F="$OUTPUT_DIR_ABS/run${i}_dump.txt"
    if [ ! -f "$F" ]; then
        ALL_IDENTICAL=false
        echo "*** Run $i: output file not found ***"
    elif ! diff -q "$REF" "$F" >/dev/null 2>&1; then
        ALL_IDENTICAL=false
        echo "*** Run $i DIFFERS from run 1 ***"
        diff "$REF" "$F" | head -20
        echo ""
    fi
done
if $ALL_IDENTICAL; then
    echo "*** RESULT: All $NUM_RUNS outputs are IDENTICAL - no reproducibility issue detected ***"
else
    echo "*** RESULT: Some outputs DIFFER - reproducibility issue detected! ***"
fi
echo ""

# Isolation test: run test_gHammingMatch 32 times with run1_match_input.bin (no path 1/2 interference)
# Confirms whether gHammingMatch is deterministic when run in clean processes
TEST_BIN="$SCRIPT_DIR/$BUILD_DIR/test_gHammingMatch"
echo "===== Isolation Test: gHammingMatch with run1_match_input.bin (no path 1/2) ====="
if [ -f "$TEST_BIN" ] && [ -f "$REF_MATCH" ]; then
    echo "Running test_gHammingMatch $NUM_RUNS times (each in fresh process)..."
    for i in $(seq 1 $NUM_RUNS); do
        $TEST_BIN "$REF_MATCH" "$OUTPUT_DIR_ABS/isolate_run${i}.txt" 2>/dev/null || true
    done
    ISOLATE_SAME=true
    for i in $(seq 2 $NUM_RUNS); do
        if [ -f "$OUTPUT_DIR_ABS/isolate_run1.txt" ] && [ -f "$OUTPUT_DIR_ABS/isolate_run${i}.txt" ]; then
            if ! diff -q "$OUTPUT_DIR_ABS/isolate_run1.txt" "$OUTPUT_DIR_ABS/isolate_run${i}.txt" >/dev/null 2>&1; then
                ISOLATE_SAME=false
                echo "*** Isolation run $i DIFFERS from run 1 ***"
            fi
        fi
    done
    if $ISOLATE_SAME; then
        echo "*** All $NUM_RUNS isolation outputs IDENTICAL - gHammingMatch is deterministic (problem is runtime state) ***"
    else
        echo "*** Some isolation outputs DIFFER - need to investigate gHammingMatch implementation ***"
    fi
else
    echo "Skipping: $TEST_BIN or $REF_MATCH not found"
fi
echo ""

# Method 2: GDB trace (optional - for function-level isolation)
if command -v gdb >/dev/null 2>&1; then
    echo "===== GDB Function Trace (for detailed analysis) ====="
    echo "Running under GDB to trace function outputs ($NUM_RUNS runs)..."
    
    for i in $(seq 1 $NUM_RUNS); do
        gdb -batch -x gdb_trace_functions.gdb \
            -ex "set args $OUTPUT_DIR_ABS/run${i}_gdb_dump.txt" \
            -ex "run" \
            "$REPRO_BIN" 2>/dev/null | grep -E "CHECKPOINT|num_pts|pt[0-9]" > "$OUTPUT_DIR_ABS/run${i}_checkpoints.txt" || true
    done
    
    GDB_IDENTICAL=true
    for i in $(seq 2 $NUM_RUNS); do
        if [ -f "$OUTPUT_DIR_ABS/run1_checkpoints.txt" ] && [ -f "$OUTPUT_DIR_ABS/run${i}_checkpoints.txt" ]; then
            if ! diff -q "$OUTPUT_DIR_ABS/run1_checkpoints.txt" "$OUTPUT_DIR_ABS/run${i}_checkpoints.txt" >/dev/null 2>&1; then
                GDB_IDENTICAL=false
                echo "GDB checkpoints: Run $i DIFFERENT from run 1"
                diff "$OUTPUT_DIR_ABS/run1_checkpoints.txt" "$OUTPUT_DIR_ABS/run${i}_checkpoints.txt" | head -30
            fi
        fi
    done
    $GDB_IDENTICAL && echo "GDB checkpoints: All $NUM_RUNS IDENTICAL"
else
    echo "GDB not found - skipping function-level trace"
fi

echo ""
echo "Output files saved to: $OUTPUT_DIR/"
echo "  - run1_dump.txt .. run${NUM_RUNS}_dump.txt: Full detection output"
echo "  - run1_match_input.bin .. run${NUM_RUNS}_match_input.bin: gHammingMatch inputs (before path3 cuMatch)"
echo "  - isolate_run1.txt .. isolate_run${NUM_RUNS}.txt: gHammingMatch isolation test (no path 1/2)"
echo "  - run1_checkpoints.txt .. run${NUM_RUNS}_checkpoints.txt: GDB checkpoint traces"
