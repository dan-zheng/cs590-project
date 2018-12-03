import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_label(filename):
  filename = filename.split('.')[0]
  if filename.endswith('itervar'):
    return 'itervar'
  elif filename.endswith('knob'):
    return 'knob'
  elif filename.endswith('curve'):
    return 'curve'
  else:
    return 'itervar'

def get_data(filename):
  xs = []
  ys = []
  with open(filename, 'r') as f:
    for line in f:
      line = line.rstrip()
      if line.startswith("No:"):
        l = line.split('\t')
        # Get No. and GFLOPS.
        l = l[:2]
        x = int(l[0].split(' ')[1]) - 1
        y = float(l[1].split('/')[1])
        # data.append((x, y))
        xs.append(x)
        ys.append(y)
  return xs, ys, get_label(filename)

def plot(filenames):
  # Save dir.
  figures_dir = 'figures'
  if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

  data = []
  for f in filenames:
      data.append(get_data(f))

  # Plot.
  # for xs, ys in data:
  # assert(len(xs) == len(ys))
  fig, ax = plt.subplots()
  # ps = plt.bar(xs, ys, width = 0.55)
  # ax.set_xticks(xs)
  # ps = ax.plot(xs, ys, 'r--', xs, ys, 'bs', linewidth=2, label='Default')

  for xs, ys, label in data:
    ps = ax.plot(xs, ys, linewidth=2, label=label)

  all_xs = [d[0] for d in data]
  base_xs = max(all_xs, key=lambda x: len(x))

  all_ys = [d[1][-1] for d in data]
  max_y = max(all_ys)

  ax.legend()
  ax.set(xlabel='Number of trials', ylabel='GFLOPS',
         title='ResNet C12')

  # Optional: set x_ticks.
  # x_tick = max(100, len(base_xs) // 10)
  # print('x_ticks', base_xs[::x_tick])
  # ax.set_xticks(base_xs[::x_tick])

  # Optional: set y_ticks.
  # May want to use TFLOPS on y-axis.
  # print('max_y', max_y)
  # ax.set_yticks(np.arange(0, max_y, 100))

  plt.show()
  pylab.savefig(os.path.join(figures_dir, 'result' + '.png'))

if __name__ == "__main__":
  import sys
  filenames = sys.argv[1:]
  plot(filenames)
