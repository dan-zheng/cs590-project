"""
Using External Libraries in NNVM
================================
**Author**: `Masahiro Masuda <https://github.com/masahi>`_

This is a short tutorial on how to use external libraries such as cuDNN, or cuBLAS with NNVM.

NNVM uses TVM internally to generate target specific code. For example, with cuda backend TVM generates cuda kernels for all layers in the user provided network.
But sometimes it is also helpful to incorporate external libraries developed by various vendors into NNVM.
Luckily, TVM has a mechanism to transparently call into these libraries.
For NNVM users, all we need to do is just to set a target string appropriately.

Before we can use external libraries from NNVM, your TVM needs to be built with libraries you want to use.
For example, to use cuDNN, USE_CUDNN option in tvm/make/config.mk needs to be enabled, and cuDNN include and library directories need to be specified.

To begin with, we import NNVM and TVM.
"""
import tvm
import numpy as np
from tvm.contrib import graph_runtime as runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing import utils

# TODO: Modify this file.

conv_configs = {
  # Format: N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
  'c7': (1, 28, 28, 128, 256, 3, 3, (2, 2), (1, 1)),
  'c12': (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1))
}

def main(conv_config):
    N, H, W, CO, CI, KH, KW, strides, padding = conv_configs[conv_config]
    # N = 100
    # a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    # w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)

    out_channels = CO
    data = sym.Variable(name="data")
    # simple_net = sym.conv2d(data=data, kernel_size=(KH,KW), channels=out_channels, padding = padding, use_bias=True)
    simple_net = sym.conv2d(data=data, kernel_size=(KH,KW), channels=out_channels, padding = padding)

    batch_size = N
    data_shape = (N, CI, H, W)
    net, params = utils.create_workload(simple_net, N, data_shape[1:])

    # NOTE(dan-zheng): Comment out other parts of network (batchnorm and relu).
    # Only measure conv2d running time.

    # out_channels = 16
    # data = sym.Variable(name="data")
    # simple_net = sym.conv2d(data=data, kernel_size=(3,3), channels=out_channels, padding = (1, 1), use_bias=True)
    # simple_net = sym.batch_norm(data=simple_net)
    # simple_net = sym.relu(data=simple_net)
    # batch_size = 1
    # data_shape = (batch_size, 3, 224, 224)
    # net, params = utils.create_workload(simple_net, batch_size, data_shape[1:])

    ######################################################################
    # Build and run with cuda backend
    # -------------------------------
    # We build and run this network with cuda backend, as usual.
    # By setting the logging level to DEBUG, the result of NNVM graph compilation will be dumped as pseudo code.
    import logging
    logging.basicConfig(level=logging.DEBUG) # to dump TVM IR after fusion

    ######################################################################
    # Use cuDNN for a convolutional layer
    # -----------------------------------
    # We can use cuDNN to replace convolution kernels with cuDNN ones.
    # To do that, all we need to do is to append the option " -libs=cudnn" to the target string.
    net, params = utils.create_workload(simple_net, batch_size, data_shape[1:])
    target = "cuda -libs=cudnn" # use cudnn for convolution
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)

    ctx = tvm.context(target, 0)
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    # out_shape = (batch_size, out_channels, 224, 224)
    out_shape = (batch_size, out_channels, W, H)
    out = module.get_output(0, tvm.nd.empty(out_shape))
    out_cudnn = out.asnumpy()

    # evaluator = module.time_evaluator(module.entry_name, ctx, number=400)
    # print('Time cost of this operator: %f' % evaluator(**params).mean)

    # timer = module.module.time_evaluator("run", ctx, number=400)
    for _ in range(10):
        timer = module.module.time_evaluator("run", ctx, number=1000)
        tcost = timer()
        # tvm_output = module.get_output(0)
        # top = np.argmax(tvm_output.asnumpy()[0])
        # tcost = "t={0:.2f}s".format(tcost.mean)
        tcost = "t={0:.10f}s".format(tcost.mean)
        # tcost = "Time cost of this operator={0:.10f}s".format(tcost.mean)
        print(tcost)

    ######################################################################
    # Note that if you use cuDNN, NNVM cannot fuse convolution with layers following it.
    # This is because layer fusion happens at the level of TVM internal representation(IR).
    # NNVM treats external libraries as black box, so there is no way to fuse them with TVM IR.
    #
    # The pseudo code below shows that cuDNN convolution + bias add + batch norm + ReLU turned into two stages of computation, one for cuDNN call and the other for the rest of operations.
    #
    # .. code-block:: text
    #
    #       allocate y[float32 * 1 * 16 * 224 * 224]
    #       produce y {
    #          // attr [0] extern_scope = 0
    #          tvm_call_packed("tvm.contrib.cudnn.conv2d.forward", 1, 0, 1, 1, 1, 1, 1, 1, 1, tvm_stack_make_array(input0, tvm_stack_make_shape(1, 3, 224, 224), 0, 4, 0.000000f, 0), tvm_stack_make_array(input1, tvm_stack_make_shape(16, 3, 3, 3), 0, 4, 0.000000f, 0), tvm_stack_make_array(y, tvm_stack_make_shape(1, 16, 224, 224), 0, 4, 0.000000f, 0))
    #        }
    #       produce compute {
    #          // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 1568
    #          // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 512
    #          compute[((((((blockIdx.x*512) + threadIdx.x)/50176) + ((((blockIdx.x*512) + threadIdx.x)/802816)*16))*50176) + ((((((blockIdx.x*512) + threadIdx.x)/224) % 224)*224) + (((blockIdx.x*64) + threadIdx.x) % 224)))] = max((((y[((((((blockIdx.x*512) + threadIdx.x)/50176) + ((((blockIdx.x*512) + threadIdx.x)/802816)*16))*50176) + ((((((blockIdx.x*512) + threadIdx.x)/224) % 224)*224) + (((blockIdx.x*64) + threadIdx.x) % 224)))] + input2[(((blockIdx.x*512) + threadIdx.x)/50176)])*input3[(((blockIdx.x*512) + threadIdx.x)/50176)]) + input4[(((blockIdx.x*512) + threadIdx.x)/50176)]), 0.000000f)
    #        }
    #

    ######################################################################
    # Verify the result
    # -----------------
    # We can check that the results of two runs match.

    # tvm.testing.assert_allclose(out_cuda, out_cudnn, rtol=1e-5)

#####################################################################
# Conclusion
# ----------
# This tutorial covered the usage of cuDNN with NNVM.
# We also have support for cuBLAS. If cuBLAS is enabled, it will be used inside a fully connected layer (nnvm.symbol.dense).
# To use cuBLAS, set a target string as "cuda -libs=cublas".
# You can use both cuDNN and cuBLAS with "cuda -libs=cudnn,cublas".
#
# For ROCm backend, we have support for MIOpen and rocBLAS.
# They can be enabled with target "rocm -libs=miopen,rocblas".
#
# Being able to use external libraries is great, but we need to keep in mind some cautions.
#
# First, the use of external libraries may restrict your usage of TVM and NNVM.
# For example, MIOpen only supports NCHW layout and fp32 data type at the moment, so you cannot use other layouts or data type in TVM.
#
# Second, and more importantly, external libraries restrict the possibility of operator fusion during graph compilation, as shown above.
# TVM and NNVM aim to achieve the best performance on a variety of hardwares, with joint operator level and graph level optimization.
# To achieve this goal, we should continue developing better optimizations for TVM and NNVM, while using external libraries as a nice way to fall back to existing implementation when necessary.

# Main.
import sys
conv_config = sys.argv[1]
main(conv_config)
