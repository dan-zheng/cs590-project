#/usr/bin/env bash
time python3 tune_nnvm_cuda_itervar.py 2>&1 | tee nnvm_itervar_full.txt
time python3 tune_nnvm_cuda_knob.py 2>&1    | tee nnvm_knob_full.txt
time python3 tune_nnvm_cuda_curve.py 2>&1   | tee nnvm_curve_full.txt

time python3 ~/cs590-project/resnet18-ablation/tune_nnvm_cuda_curve.py 500 2>&1 | tee resnet18_curve_n500_full.txt
# TODO: Run this with fixed curve sampling.
time python3 ~/cs590-project/resnet18-ablation/tune_nnvm_cuda_curve_fix.py 500 2>&1 | tee resnet18_curve_fix_n500_full.txt

time python3 ~/cs590-project/resnet18-ablation/tune_nnvm_cuda_itervar.py 500 2>&1 | tee resnet18_itervar_n500_full.txt
time python3 ~/cs590-project/resnet18-ablation/tune_nnvm_cuda_knob.py 500 2>&1 | tee resnet18_knob_n500_full.txt

# Parse times.
# python3 ~/cs590-project/parse_times.py resnet18_curve_n500_full.txt
python3 ~/cs590-project/parse_times.py ~/cs590-project/feature_experiments-ml02/conv2d_itervar_full.txt
python3 ~/cs590-project/parse_times.py ~/cs590-project/resnet18-ablation/resnet18_curve_n500_full.txt
