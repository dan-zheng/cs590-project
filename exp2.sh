#/usr/bin/env bash

# Experiment 2: evalute `curve-fix` feature type.

# NOTE: Commented due to long autotuning execution time.
# time python3 tune_conv2d_cuda.py c7 curve-fix 2000 2>&1  | tee conv2d_c7_curve-fix_n2000_full.txt
# time python3 tune_conv2d_cuda.py c12 curve-fix 2000 2>&1 | tee conv2d_c12_curve-fix_n2000_full.txt

# Fastest kernel configurations will be stored in log files.
# Evaluate fastest kernels using 'tune_conv2d_cuda_test.py'.
# Examples using existing log files:
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_curve-fix_c7.log
python3 tune_conv2d_cuda_test.py c12 baseline/conv2d_c12_curve-fix_n2000_0.log
