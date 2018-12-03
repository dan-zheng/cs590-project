# CS 590: Automated program optimization using ML

Code artifacts and data for CS 590 project.

The project involves several changes to the TVM deep learning compiler.
These changes can be found at [my fork](https://github.com/dan-zheng/tvm/tree/cs590-experiments).
To run experiments, please clone the fork and build from source following [these instructions](https://docs.tvm.ai/install/from_source.html).

## Experiments

Experiments include:
- `exp1.sh`, `exp2.sh`, `exp3.sh`: these scripts summarize paper experiments.
  - Autotuning commands are commented by default due to their long running time.
- Individual scripts were adapted from [AutoTVM tutorials](https://docs.tvm.ai/tutorials/index.html#auto-tuning). Explanations:
  - `tune_conv2d_cuda.py`
    - Tune a specific `conv2d` operator configuration.
  - `tune_conv2d_cuda_test.py`
    - Evaluate performance of tuned `conv2d` configuration stored in log file.
  - `feature_experiments_resnet18/tune_nnvm_cuda.py`
    - Tune ResNet-18 inference (12 `conv2d` configurations total).
  - `transfer-learning/tune_conv2d_cuda_transfer.py`
    - Tune `conv2d` operator using pretrained data for transfer learning.
- Incomplete neural network cost model experiments in `nn-cost-model` and `treernn-cost-model`.

## Data

Autotuning result files are also included, as autotuning execution takes many hours.
- `baseline`, `feature_experiments_c7`, `feature_experiments_c12`: `conv2d` tuning results.
- `feature_experiments_resnet18`: end-to-end ResNet-18 tuning results.
- `transfer-learning`: `conv2d` transfer learning results.

Raw, unpolished data can be found on the `raw-data` branch.

## Plots

Figures in paper are generated via `plot.sh`.
