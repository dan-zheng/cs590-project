# NOTE: Autotuning code adapted from:
# - https://docs.tvm.ai/tutorials/autotvm/tune_conv2d_cuda.html
# - https://github.com/dmlc/tvm/blob/master/tutorials/autotvm/tune_conv2d_cuda.py

import logging
import os
import sys
import numpy as np

import tvm
import topi
from topi.testing import conv2d_nchw_python

from tvm import autotvm

# Note: enable logging.
import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)

# Define schedule search space.
@autotvm.template
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    data = tvm.placeholder((N, CI, H, W), name='data')
    kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype='float32')
    s = tvm.create_schedule([conv.op])

    ##### space definition begin #####
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

    # DEBUG
    # cfg.define_knob("dump_pass_ir", [True])
    ##### space definition end #####

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

    return s, [raw_data, kernel, conv]

# Search over schedule space.

conv_configs = {
  # Format: N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
  'c7': (1, 28, 28, 128, 256, 3, 3, (2, 2), (1, 1)),
  'c12': (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1))
}

feature_types = ['itervar', 'knob', 'curve', 'curve-fix']

# Get a fresh, unduplicated filename.
def fresh(name, suffix):
    count = 0
    orig_name = name
    name = orig_name + '_' + str(count)
    filename = '{}.{}'.format(name, suffix)
    while os.path.exists(filename):
        print("WARNING: Filename '{}' already exists".format(filename))
        count += 1
        name = orig_name + '_' + str(count)
        filename = '{}.{}'.format(name, suffix)
    return filename

def fresh_dir(name):
    count = 0
    orig_name = name
    dirname = orig_name + '_' + str(count)
    while os.path.exists(dirname):
        print("WARNING: Filename '{}' already exists".format(dirname))
        count += 1
        dirname = orig_name + '_' + str(count)
    # Make directory.
    os.makedirs(dirname)
    return dirname

# Main.

def main(feature_type, conv_config, n_trial):
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    for i in range(5):
        # print('MAIN ITERATION {}.'.format(i))
        main_body(feature_type, conv_config, n_trial)
        # Test with n_trial=20 to verify things work.
        # main_body(feature_type, conv_config, n_trial=20)

def main_body(feature_type, conv_config, n_trial):
    filename = 'conv2d_{}_{}_n{}'.format(conv_config, feature_type, n_trial)
    # log_file = '{}.log'.format(filename)
    # dump_file = '{}.txt'.format(filename)
    # NOTE: Important to use `fresh(filename, ...)` to prevent file overwriting.
    log_file = fresh(filename, 'log')
    dump_file = fresh(filename, 'txt')

    # NOTE: Failed results dir experiment.
    # results_dir = fresh_dir('results')
    # log_file = fresh(os.path.join(results_dir, filename), 'log')
    # dump_file = fresh(os.path.join(results_dir, filename), 'txt')
    # # log_file = os.path.join(results_dir, log_file)
    # # dump_file = os.path.join(results_dir, dump_file)

    # NOTE: Dump file will contain info from later iterations.
    # logging.getLogger('autotvm').addHandler(logging.FileHandler(dump_file))

    N, H, W, CO, CI, KH, KW, strides, padding = conv_configs[conv_config]
    task = autotvm.task.create(conv2d_no_batching,
                               args=(N, H, W, CO, CI, KH, KW, strides, padding),
                               target='cuda')
    print(task.config_space)

    tuner = autotvm.tuner.XGBTuner(task, feature_type=feature_type)

    # Run function.
    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    run_timeout = 30
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=run_timeout)
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner.tune(n_trial=n_trial,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_file)])

    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # apply history best from config file
    with autotvm.apply_history_best(log_file):
        with tvm.target.create("cuda"):
            s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
            # ir = tvm.lower(s, arg_bufs, simple_mode=True)
            # print(ir)
            func = tvm.build(s, arg_bufs)

    # check correctness
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
    repeat_number = 1000
    evaluator = func.time_evaluator(func.entry_name, ctx, number=repeat_number)
    print('Time cost of this operator: %f' % evaluator(a_tvm, w_tvm, c_tvm).mean)

# Main.
if __name__ == "__main__":
    import sys
    conv_config = sys.argv[1]
    feature_type = sys.argv[2]
    n_trial = int(sys.argv[3])
    main(feature_type, conv_config, n_trial)
