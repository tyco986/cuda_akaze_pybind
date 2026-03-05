#!/bin/bash
# Reproducibility test script - runs detection twice and compares outputs
# Identifies which functions produce inconsistent output between runs

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BUILD_DIR="${BUILD_DIR:-build}"
OUTPUT_DIR="${OUTPUT_DIR:-repro_results}"
RUN1_DUMP="$OUTPUT_DIR/run1_dump.txt"
RUN2_DUMP="$OUTPUT_DIR/run2_dump.txt"
RUN1_TRACE="$OUTPUT_DIR/run1_gdb_trace.txt"
RUN2_TRACE="$OUTPUT_DIR/run2_gdb_trace.txt"
REPRO_BIN="$BUILD_DIR/repro_test"

echo "===== Reproducibility Test for CUDA-AKAZE ====="
echo ""

# Ensure data exists
if [ ! -f "data/left.pgm" ]; then
    echo "Error: data/left.pgm not found"
    exit 1
fi

# Build with debug symbols for GDB
mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"
echo "[1/4] Building project (Debug for GDB)..."
cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE=Debug .. 2>/dev/null || cmake ..
make -j$(nproc) repro_test 2>/dev/null || make repro_test
cd ..
echo ""

# Method 1: Simple dump comparison (run repro_test twice)
echo "[2/4] Run 1 - executing repro_test..."
$REPRO_BIN "$RUN1_DUMP" 2>/dev/null | tee "$OUTPUT_DIR/run1_stdout.txt" || true
echo ""

echo "[3/4] Run 2 - executing repro_test..."
$REPRO_BIN "$RUN2_DUMP" 2>/dev/null | tee "$OUTPUT_DIR/run2_stdout.txt" || true
echo ""

# Compare dump files
echo "[4/4] Comparing outputs..."
if diff -q "$RUN1_DUMP" "$RUN2_DUMP" >/dev/null 2>&1; then
    echo "*** RESULT: Outputs are IDENTICAL - no reproducibility issue detected ***"
else
    echo "*** RESULT: Outputs DIFFER - reproducibility issue detected! ***"
    echo ""
    echo "Differences:"
    diff "$RUN1_DUMP" "$RUN2_DUMP" || true
fi
echo ""

# Method 2: GDB trace (optional - for function-level isolation)
if command -v gdb >/dev/null 2>&1; then
    echo "===== GDB Function Trace (for detailed analysis) ====="
    echo "Running under GDB to trace function outputs..."
    
    gdb -batch -x gdb_trace_functions.gdb \
        -ex "set args $OUTPUT_DIR/run1_gdb_dump.txt" \
        -ex "run" \
        "$REPRO_BIN" 2>/dev/null | grep -E "CHECKPOINT|num_pts|pt[0-9]" > "$OUTPUT_DIR/run1_checkpoints.txt" || true
    
    gdb -batch -x gdb_trace_functions.gdb \
        -ex "set args $OUTPUT_DIR/run2_gdb_dump.txt" \
        -ex "run" \
        "$REPRO_BIN" 2>/dev/null | grep -E "CHECKPOINT|num_pts|pt[0-9]" > "$OUTPUT_DIR/run2_checkpoints.txt" || true
    
    if [ -f "$OUTPUT_DIR/run1_checkpoints.txt" ] && [ -f "$OUTPUT_DIR/run2_checkpoints.txt" ]; then
        if diff -q "$OUTPUT_DIR/run1_checkpoints.txt" "$OUTPUT_DIR/run2_checkpoints.txt" >/dev/null 2>&1; then
            echo "GDB checkpoints: IDENTICAL"
        else
            echo "GDB checkpoints: DIFFERENT - first inconsistent checkpoint:"
            diff "$OUTPUT_DIR/run1_checkpoints.txt" "$OUTPUT_DIR/run2_checkpoints.txt" | head -30
        fi
    fi
else
    echo "GDB not found - skipping function-level trace"
fi

echo ""
echo "Output files saved to: $OUTPUT_DIR/"
echo "  - run1_dump.txt, run2_dump.txt: Full detection output"
echo "  - run1_checkpoints.txt, run2_checkpoints.txt: GDB checkpoint traces"
