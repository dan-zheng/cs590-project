# Test cuDNN baseline performance on conv2d configurations from ResNet-18.
# Code adapted from: https://docs.tvm.ai/tutorials/nnvm/using_external_lib.html

import tvm
import numpy as np
from tvm.contrib import graph_runtime as runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing import utils

conv_configs = {
  # Format: N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
  'c7': (1, 28, 28, 128, 256, 3, 3, (2, 2), (1, 1)),
  'c12': (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1))
}

# NOTE: Uncomment to show generated TVM IR.
# import logging
# logging.basicConfig(level=logging.DEBUG)

def main(conv_config):
    # Define conv2d network.
    N, H, W, CO, CI, KH, KW, strides, padding = conv_configs[conv_config]
    batch_size = N
    data_shape = (N, CI, H, W)
    data = sym.Variable(name="data")
    simple_net = sym.conv2d(data=data, kernel_size=(KH, KW), channels=CO, padding=padding)

    # Use cuDNN as conv2d backend.
    net, params = utils.create_workload(simple_net, batch_size, data_shape[1:])
    target = "cuda -libs=cudnn"
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)

    ctx = tvm.context(target, 0)
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out_shape = (batch_size, CO, W, H)
    out = module.get_output(0, tvm.nd.empty(out_shape))
    out_cudnn = out.asnumpy()

    print('Time cost of cuDNN conv2d operator ({}):'.format(conv_config))
    costs = []
    for _ in range(10):
        evaluator = module.module.time_evaluator("run", ctx, number=1000)
        cost = evaluator().mean
        costs.append(cost)
        print('%.8f' % cost)
    print('Mean:', '%.8f' % np.mean(costs))

# Main.
if __name__ == "__main__":
    import sys
    conv_config = sys.argv[1]
    main(conv_config)
