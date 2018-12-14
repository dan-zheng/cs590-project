import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from collections import defaultdict
import operator

from datetime import datetime, timedelta

def get_label(filename):
  filename = filename.split('.')[0]
  if 'itervar' in filename:
    return 'itervar'
  elif 'knob' in filename:
    return 'knob'
  elif 'curve' in filename:
    return 'curve'
  raise ValueError('Unrecognized label')

def seconds_to_datestr(seconds):
  # s = timedelta(seconds=seconds)
  # print(seconds)
  # print(str(s))
  # return str(datetime(1,1,1) + timedelta(seconds=seconds))
  return str(timedelta(seconds=seconds))

def get_times(filename):
  times = []
  total_time = 0
  tune_count = 1
  last_time = 0
  time_counter = defaultdict(int)

  # Format: 'TIME: <time in seconds>, <note>'
  with open(filename, 'r') as f:
    for line in f:
      line = line.rstrip()
      if line.startswith("DEBUG:autotvm:"):
          line = line[len("DEBUG:autotvm:"):]
      if line.startswith("TIME:"):
        l = line.split(', ')
        note = l[1]
        # Remove numeric values from notes.
        if '=' in note:
          note = note.split('=')[0]
        time = float(l[0].split(' ')[1])
        times.append(time)

        if last_time is not None and time < last_time:
          total_time += last_time
          tune_count += 1

        print(l)
        time_counter[note] += max(0, time - last_time)
        last_time = time
        
        # print(note)
        # print(time)

  # Add the final measured time.
  total_time += time
  time_counter = sorted(time_counter.items(), key=operator.itemgetter(1))
  # time_counter = [(x[0], timedelta(seconds=x[1])) for x in time_counter]
  print(time_counter)
  time_counter = [(x[0], seconds_to_datestr(x[1])) for x in time_counter]
  print(time_counter)
  print('Total time:', total_time)
  print('Tune count:', tune_count)
  return total_time, tune_count

def parse(filename):
  # Create figure directory.
  figures_dir = 'figures'
  if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

  # Get data.
  times = get_times(filename)
  # data = []
  # for f in filenames:
  #     data.append(get_data(f))

  # Plot.
  fig, ax = plt.subplots()
  # ps = plt.bar(xs, ys, width = 0.55)
  # ax.set_xticks(xs)
  # ps = ax.plot(xs, ys, 'r--', xs, ys, 'bs', linewidth=2, label='Default')

  for xs, ys, label in data:
    print(ys)
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
  assert len(sys.argv) > 1, "Expected filename"
  filename = sys.argv[1]
  parse(filename)
