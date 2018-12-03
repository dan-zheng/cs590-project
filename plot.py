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
        y = l[1].split('/')[1]
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
    # ps = ax.plot(xs, ys, linewidth=2, label='Default')
    ps = ax.plot(xs, ys, linewidth=2, label=label)
  # ps = ax.plot(xs, ys, 'r--', linewidth=2, label='Default')
  # ps = ax.plot(xs, ys, 'bs', linewidth=2, label='Curve')

  ax.legend()
  ax.set(xlabel='Number of trials', ylabel='GFLOPS',
         title='ResNet C12')

  # ps = plt.plot(ys, linewidth=2)
  x_tick = max(100, len(xs) // 10)
  # ax.set_xticks(xs[::10])
  ax.set_xticks(xs[::x_tick])
  # ax.set_yticks(xs[::x_tick])

  # start, end = ax.get_ylim()
  # ax.yaxis.set_ticks(np.arange(0, end, 2))
  # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

  plt.show()
  pylab.savefig(os.path.join(figures_dir, 'result' + '.png'))

if __name__ == "__main__":
  import sys
  filename = sys.argv[1]
  # data = get_data(filename)
  # xs, ys = data
  # print(data)
  filenames = sys.argv[1:]
  plot(filenames)

  #print(sys.argv)
  # model = sys.argv[1]
  # n_files = len(sys.argv) - 2
  # files = []
  # for i in range(n_files):
  #   files.append(sys.argv[i+2])
  # plot(files, model)
