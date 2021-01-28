import setuptools

import numpy as np
cimport numpy as cnp

from libcpp cimport bool
import matplotlib.pyplot as plt


cdef extern from "<complex.h>" namespace "std":
    double complex conj(double complex z) nogil
    double complex exp(double complex z ) nogil
    double abs(double complex z) nogil
cdef extern from "<math.h>" nogil:
    double pow(double r, double e)
    double exp(double r)
    double tanh(double r)
    double abs(double r)
    double sin(double)
    double cos(double)

import pyximport
pyximport.install()

from solve_hamiltonian cimport solve_system
from system_class cimport System

#from cython.parallel import prange

"""
This is the file which contains the functions to calcualte the phase-current relations for a S/HM/S junction.
"""

# This function calcualte the current_phase relation for a given alpha-representation. (and a input set of the system dimensions)
cpdef solve_for_shms_system_phase(cnp.ndarray[cnp.float64_t, ndim=1] alpha_array,
                                         int max_num_iter,
                                         double tol,
                                         int L_y,
                                         int L_z,
                                         int L_sc_0,
                                         int L_nc,
                                         int L_f,
                                         int L_soc,
                                         int L_sc,
                                         double mu_sc,
                                         double mu_nc,
                                         double mu_soc,
                                         double u_sc,
                                         double beta)

# This function search over the strength of the spin-orbit coupling, given a static vector orientation of alpha.
# For each search, the function will only save the critical current and the belonging phase.
cpdef solve_for_shms_system_strength_individual_phase(int max_num_iter,
                                                      double tol,
                                                      bool xz,
                                                      bool yz,
                                                      double [:] alpha_max,
                                                      double theta)


# This function seach over the relative orientation of the spin-orbit coupling, given a static magnitude of alpha.
# For each search, the function will only save the critical current and the belonging phase.
cpdef solve_for_shms_system_orientation_individual_phase(int max_num_iter,
                                                         double tol,
                                                         bool xz,
                                                         bool yz,
                                                         double alpha_max)