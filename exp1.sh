#/usr/bin/env bash

# cuDNN baselines.
python3 cudnn/cudnn.py c7
python3 cudnn/cudnn.py c12

# C7 running time.
# 100 iterations.
python3 tune_conv2d_cuda_test.py c7 baseline/conv2d_c7_itervar_n100_1.log
# 1000 iterations.
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_itervar_c7.log
# 2000 iterations.
python3 tune_conv2d_cuda_test.py c7 baseline/conv2d_c7_itervar_n2000_1.log
# Logged results.
grep -nr "Time cost" -C2 feature_experiments_c7/final_results*/*full.txt

# C12 running time.
# 1000 iterations.
python3 feature_experiments-ml02/tune_conv2d_cuda_test.py c12 feature_experiments-ml02/final_results2/conv2d_curve-fix.log
python3 feature_experiments-ml02/tune_conv2d_cuda_test.py c12 feature_experiments-ml02/itervar_final/conv2d_itervar.log
# 2000 iterations.
python3 feature_experiments-ml02/tune_conv2d_cuda_test.py c12 baseline/conv2d_c12_itervar_n2000_0.log
# Logged results.
grep -nr "Time cost" -C2 baseline/*full.txt
