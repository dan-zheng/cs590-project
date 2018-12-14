#/usr/bin/env bash

# Copy history file. To be used as base for all transfer learning.
cp ../feature_experiments-ml02/final_results2/conv2d_curve-fix.log conv2d_curve-fix_c12.log

time python3 tune_conv2d_cuda_transfer.py curve-fix c12 c7 conv2d_curve-fix_c12.log 2>&1 | tee transfer_c12_c7_curve-fix_full.txt
time python3 tune_conv2d_cuda_transfer.py curve     c12 c7 conv2d_curve-fix_c12.log 2>&1 | tee transfer_c12_c7_curve_full.txt
time python3 tune_conv2d_cuda_transfer.py itervar   c12 c7 conv2d_curve-fix_c12.log 2>&1 | tee transfer_c12_c7_itervar_full.txt
time python3 tune_conv2d_cuda_transfer.py knob      c12 c7 conv2d_curve-fix_c12.log 2>&1 | tee transfer_c12_c7_knob_full.txt
