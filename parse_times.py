# A utility for parsing bottleneck times from tuning log files.
# This is designed to be used with `*_full.txt` files.
# Those files store tuning time info which is parsed by this script.

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

        # print(l)
        time_counter[note] += max(0, time - last_time)
        last_time = time

  # Add the final measured time.
  total_time += time
  time_counter = sorted(time_counter.items(), key=operator.itemgetter(1))
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

if __name__ == "__main__":
  import sys
  assert len(sys.argv) > 1, "Expected filename"
  filename = sys.argv[1]
  parse(filename)
