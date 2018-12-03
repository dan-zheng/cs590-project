# Evaluate tuned operator performance.
# NOTE: Autotuning code adapted from:
# - https://docs.tvm.ai/tutorials/autotvm/tune_conv2d_cuda.html
# - https://github.com/dmlc/tvm/blob/master/tutorials/autotvm/tune_conv2d_cuda.py

import logging
import os
import sys
import numpy as np
import tvm
from tvm import autotvm
import topi
from topi.testing import conv2d_nchw_python

import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)

# Define computation and schedule search space.
@autotvm.template
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    data = tvm.placeholder((N, CI, H, W), name='data')
    kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype='float32')
    s = tvm.create_schedule([conv.op])

    ##### Space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    ##### Space definition end #####

    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, 'local')

    # create cache stage
    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])
    AL = s.cache_read(AA, 'local', [OL])
    WL = s.cache_read(WW, 'local', [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    kernel_scope = n  # this is the scope to attach global config inside this kernel

    s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
    s[output].bind(by, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, rym, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxm, rxi = cfg['tile_ry'].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # tune unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    # args = [raw_data, kernel, conv]
    # ir = tvm.lower(s, args, simple_mode=True)
    return s, [raw_data, kernel, conv]

# Search over schedule space.

conv_configs = {
  # Format: N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
  'c7': (1, 28, 28, 128, 256, 3, 3, (2, 2), (1, 1)),
  'c12': (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1))
}

feature_types = ['itervar', 'knob', 'curve']

# Main.

def main(conv_config, log_file):
    assert os.path.exists(log_file), "Log file '{}' does not exist.".format(log_file)

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    N, H, W, CO, CI, KH, KW, strides, padding = conv_configs[conv_config]
    task = autotvm.task.create(conv2d_no_batching,
                               args=(N, H, W, CO, CI, KH, KW, strides, padding),
                               target='cuda')
    print(task.config_space)

    # NOTE(dan-zheng): Don't tune, just test.
    repeat_number = 1000

    # Inspect the best config
    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # Apply history best from config file.
    with autotvm.apply_history_best(log_file):
        with tvm.target.create("cuda"):
            s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
            # ir = tvm.lower(s, arg_bufs, simple_mode=True)
            # print(ir)
            func = tvm.build(s, arg_bufs)

    # Check correctness.
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time.
    # Choose a large repeat number to reduce noise.
    print('Time cost of this operator:')
    costs = []
    for _ in range(10):
        evaluator = func.time_evaluator(func.entry_name, ctx, number=repeat_number)
        cost = evaluator(a_tvm, w_tvm, c_tvm).mean
        costs.append(cost)
        print('%.8f' % cost)
    print('Mean:', '%.8f' % np.mean(costs))

# Main.
if __name__ == "__main__":
    import sys
    conv_config = sys.argv[1]
    log_file = sys.argv[2]
    main(conv_config, log_file)
