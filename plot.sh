#/usr/bin/env bash

# Script that generates figures used in paper.

# Experiments 1 and 2: evaluate tuning results by cost model feature type.
python3 plot.py "ResNet C12" feature_experiments_c12/final_results/conv2d_{curve,itervar,knob,curve_fix}_full.txt
# python3 plot.py "ResNet C12" feature_experiments_c12/final_results2/conv2d_{curve,itervar,knob,curve_fix}_full.txt
# python3 plot.py "ResNet C12" baseline/conv2d_c12_{curve,itervar,knob,curve-fix}_n2000_full.txt

# python3 plot.py "ResNet-18 Inference" feature_experiments_resnet18/final_results/resnet18_{curve,itervar,knob,curve_fix}_n500_full.txt
# python3 plot.py "ResNet-18 Inference" feature_experiments_resnet18/final_results2/resnet18_{curve,itervar,knob,curve_fix}_n500_full.txt

# Experiment 3: transfer learning.
python3 plot.py "ResNet C7" feature_experiments_c7/final_results/conv2d_{curve,itervar,knob,curve_fix}_full.txt
python3 plot.py "Transfer Learning (C12 -> C7)" transfer-learning/transfer_c12_c7_{curve,itervar,knob,curve-fix}_full.txt
