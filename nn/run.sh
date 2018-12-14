#/usr/bin/env bash

time python3 tune_conv2d_cuda.py c12 itervar 2000 2>&1   | tee conv2d_c12_itervar_n2000_full.txt
time python3 tune_conv2d_cuda.py c12 knob 2000 2>&1      | tee conv2d_c12_knob_n2000_full.txt
time python3 tune_conv2d_cuda.py c12 curve 2000 2>&1     | tee conv2d_c12_curve_n2000_full.txt
time python3 tune_conv2d_cuda.py c12 curve-fix 2000 2>&1 | tee conv2d_c12_curve-fix_n2000_full.txt

# Evaluate fastest kernel.
python3 tune_conv2d_cuda_test.py c12 conv2d_itervar.log
python3 tune_conv2d_cuda_test.py c12 conv2d_knob.log
python3 tune_conv2d_cuda_test.py c12 conv2d_curve.log
python3 tune_conv2d_cuda_test.py c12 conv2d_curve-fix.log
