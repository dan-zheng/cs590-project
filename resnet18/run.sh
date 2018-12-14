#/usr/bin/env bash
time python3 tune_nnvm_cuda_itervar.py 2>&1 | tee nnvm_itervar_full.txt
time python3 tune_nnvm_cuda_knob.py 2>&1    | tee nnvm_knob_full.txt
time python3 tune_nnvm_cuda_curve.py 2>&1   | tee nnvm_curve_full.txt
