#/usr/bin/env bash
time python3 tune_conv2d_cuda_itervar.py 2>&1 | tee conv2d_itervar_full.txt
time python3 tune_conv2d_cuda_knob.py 2>&1 | tee conv2d_knob_full.txt
time python3 tune_conv2d_cuda_curve.py 2>&1 | tee conv2d_curve_full.txt

# # Old: Evaluate fastest kernel.
# python3 tune_conv2d_cuda_test.py itervar
# python3 tune_conv2d_cuda_test.py knob
# python3 tune_conv2d_cuda_test.py curve

# Evaluate fastest kernel.
python3 tune_conv2d_cuda_test.py c12 conv2d_itervar.log
python3 tune_conv2d_cuda_test.py c12 conv2d_knob.log
python3 tune_conv2d_cuda_test.py c12 conv2d_curve.log
python3 tune_conv2d_cuda_test.py c12 conv2d_curve-fix.log

# Transfer learning.
python3 tune_conv2d_cuda_test.py c7 ../transfer-learning/conv2d_transfer_knob_c12_c7_0_1_2.log
python3 tune_conv2d_cuda_test.py c7 ../feature_experiments_c7/final_results/conv2d_knob_c7.log
