# GDB script to trace function outputs for reproducibility debugging
# Usage: gdb -batch -x gdb_trace_functions.gdb -ex "set args OUTPUT_FILE" -ex "run" ./repro_test
# Run twice with different OUTPUT_FILE (e.g. run1_trace.txt, run2_trace.txt) and compare

set pagination off

# Break at end of detectAndCompute - after copy to host, we have full result
# We break at the line after cudaMemcpy2D (akaze.cpp ~139)
break akaze::Akazer::detectAndCompute
commands
  silent
  finish
  # We're back in runDetectionOnce, akaze_data has the result
  printf "CHECKPOINT detectAndCompute_done: "
  printf "num_pts=%d\n", akaze_data.num_pts
  if akaze_data.num_pts > 0 && akaze_data.h_data != 0
    printf "  pt0: x=%.6f y=%.6f octave=%d response=%.6f size=%.6f\n", akaze_data.h_data[0].x, akaze_data.h_data[0].y, akaze_data.h_data[0].octave, akaze_data.h_data[0].response, akaze_data.h_data[0].size
  end
  if akaze_data.num_pts > 1 && akaze_data.h_data != 0
    printf "  pt1: x=%.6f y=%.6f\n", akaze_data.h_data[1].x, akaze_data.h_data[1].y
  end
  if akaze_data.num_pts > 2 && akaze_data.h_data != 0
    printf "  pt2: x=%.6f y=%.6f\n", akaze_data.h_data[2].x, akaze_data.h_data[2].y
  end
  continue
end

# Break at end of detect() - to check num_pts before hCalcOrient/hDescribe
# The detect() is called from detectAndCompute at akaze.cpp:121
# After detect returns, result.num_pts is set (from GPU). We break in detect at the end.
break akaze::Akazer::detect
commands
  silent
  finish
  # Back in detectAndCompute, result has num_pts from cudaMemcpy at line 450
  # But wait - the cudaMemcpy happens inside detect(). So after finish we're after detect().
  # At that point 'result' is the parameter. In detectAndCompute frame, result is the local var.
  printf "CHECKPOINT after_detect: num_pts=%d\n", result.num_pts
  continue
end

# Break at end of hRefine
break akaze::hRefine
commands
  silent
  finish
  printf "CHECKPOINT after_hRefine: num_pts=%d\n", result.num_pts
  continue
end

# Break at end of hCalcOrient
break akaze::hCalcOrient
commands
  silent
  finish
  printf "CHECKPOINT after_hCalcOrient: num_pts=%d\n", result.num_pts
  continue
end

# Break at end of hDescribe
break akaze::hDescribe
commands
  silent
  finish
  printf "CHECKPOINT after_hDescribe: num_pts=%d\n", result.num_pts
  continue
end

# Break after hNmsR when num_pts is set (akaze.cpp:451)
break akaze.cpp:452
commands
  silent
  printf "CHECKPOINT after_hNmsR: num_pts=%d\n", result.num_pts
  continue
end

run
quit
