import sys
from tune_nnvm_cuda import *

n_trial = int(sys.argv[1])
main('curve', n_trial=n_trial)
