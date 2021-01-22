import numpy as np
cimport numpy as cnp
cimport cython
from scipy.linalg import eigh
from libcpp cimport bool

from utilities cimport idx_F_i
from system_class cimport System

#   Main function to solve a system
cpdef void solve_system(System system,
                        int max_num_iter,
                        double tol,
                        bool juction)
