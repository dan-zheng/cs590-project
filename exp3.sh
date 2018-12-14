#/usr/bin/env bash

# Training learning evaluation.
# The following commands print the 

# In-domain for C7.
echo "NOTE: C7 in-domain."
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_knob_c7.log
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_itervar_c7.log
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_curve_c7.log
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_curve-fix_c7.log

# Transfer learning from C12 to C7.
echo "NOTE: C12->C7 transfer."
python3 tune_conv2d_cuda_test.py c7 transfer-learning/conv2d_transfer_knob_c12_c7.log
python3 tune_conv2d_cuda_test.py c7 transfer-learning/conv2d_transfer_itervar_c12_c7.log
python3 tune_conv2d_cuda_test.py c7 transfer-learning/conv2d_transfer_curve_c12_c7.log
python3 tune_conv2d_cuda_test.py c7 transfer-learning/conv2d_transfer_curve-fix_c12_c7.log
