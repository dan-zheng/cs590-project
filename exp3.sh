#/usr/bin/env bash

# Experiment 3: evalute cost model transfer learning.

cd transfer-learning

# Copy history file.
# This file is used as the base for all transfer learning.
cp ../feature_experiments-c12/final_results2/conv2d_curve-fix.log conv2d_curve-fix_c12.log

# Perform transfer learning tuning.
# NOTE: Commented due to long autotuning execution time.
# time python3 tune_conv2d_cuda_transfer.py curve-fix c12 c7 conv2d_curve-fix_c12.log 2>&1 | tee transfer_c12_c7_curve-fix_full.txt
# time python3 tune_conv2d_cuda_transfer.py curve     c12 c7 conv2d_curve-fix_c12.log 2>&1 | tee transfer_c12_c7_curve_full.txt
# time python3 tune_conv2d_cuda_transfer.py itervar   c12 c7 conv2d_curve-fix_c12.log 2>&1 | tee transfer_c12_c7_itervar_full.txt
# time python3 tune_conv2d_cuda_transfer.py knob      c12 c7 conv2d_curve-fix_c12.log 2>&1 | tee transfer_c12_c7_knob_full.txt

# Fastest kernel configurations will be stored in log files.
# Evaluate fastest kernels using 'tune_conv2d_cuda_test.py'.
# Examples using existing log files:

# In-domain tuning for C7.
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_knob_c7.log
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_itervar_c7.log
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_curve_c7.log
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_curve-fix_c7.log

# Transfer learning from C12 to C7.
python3 tune_conv2d_cuda_test.py c7 transfer-learning/conv2d_transfer_knob_c12_c7.log
python3 tune_conv2d_cuda_test.py c7 transfer-learning/conv2d_transfer_itervar_c12_c7.log
python3 tune_conv2d_cuda_test.py c7 transfer-learning/conv2d_transfer_curve_c12_c7.log
python3 tune_conv2d_cuda_test.py c7 transfer-learning/conv2d_transfer_curve-fix_c12_c7.log
