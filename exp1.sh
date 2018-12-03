#/usr/bin/env bash

# Experiment 1: evalute TVM autotuning and cuDNN baseliens.

# Measure cuDNN baselines.
python3 cudnn/cudnn.py c7
python3 cudnn/cudnn.py c12

# Perform tuning using each feature type.
# NOTE: Commented due to long autotuning execution time.
#       It is not recommended to run these commands in succession
#       as they take many hours to complete. (4-10 hours per command)

# time python3 tune_conv2d_cuda.py c7 itervar 2000 2>&1   | tee conv2d_c7_itervar_n2000_full.txt
# time python3 tune_conv2d_cuda.py c7 knob 2000 2>&1      | tee conv2d_c7_knob_n2000_full.txt
# time python3 tune_conv2d_cuda.py c7 curve 2000 2>&1     | tee conv2d_c7_curve_n2000_full.txt
#
# time python3 tune_conv2d_cuda.py c12 itervar 2000 2>&1   | tee conv2d_c12_itervar_n2000_full.txt
# time python3 tune_conv2d_cuda.py c12 knob 2000 2>&1      | tee conv2d_c12_knob_n2000_full.txt
# time python3 tune_conv2d_cuda.py c12 curve 2000 2>&1     | tee conv2d_c12_curve_n2000_full.txt

# Fastest kernel configurations will be stored in log files.
# Evaluate fastest kernels using 'tune_conv2d_cuda_test.py'.
# Examples using existing log files:
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_itervar_c7.log
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_knob_c7.log
python3 tune_conv2d_cuda_test.py c7 feature_experiments_c7/final_results/conv2d_curve_c7.log

python3 tune_conv2d_cuda_test.py c12 baseline/conv2d_c12_itervar_n2000_0.log
python3 tune_conv2d_cuda_test.py c12 baseline/conv2d_c12_knob_n2000_0.log
python3 tune_conv2d_cuda_test.py c12 baseline/conv2d_c12_curve_n2000_0.log
