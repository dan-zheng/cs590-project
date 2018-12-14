#/usr/bin/env bash
time python3 tune_conv2d_cuda_itervar.py 2>&1 | tee conv2d_itervar_full.txt
time python3 tune_conv2d_cuda_knob.py 2>&1 | tee conv2d_knob_full.txt
time python3 tune_conv2d_cuda_curve.py 2>&1 | tee conv2d_curve_full.txt
