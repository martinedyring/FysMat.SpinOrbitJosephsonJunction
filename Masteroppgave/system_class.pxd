import numpy as np
cimport numpy as cnp
import matplotlib.pyplot as plt


from libcpp cimport bool

from utilities cimport idx_F_i, idx_F_ij_x_pluss, idx_F_ij_x_minus, idx_F_ij_y_pluss, idx_F_ij_y_minus, idx_F_ij_s, num_idx_F_i
from numpy import conj, tanh, exp, sqrt, cos, sin, log#, sqrt #as conj, tanh, exp, cos, sin, sqrt

"""
This script define the class System which contains all necessary information to construct one system.
"""

cimport cython
@cython.embedsignature(True)
cdef class System:
    cdef int L_sc_0
    cdef int L_sc
    cdef int L_soc
    cdef int L_nc
    cdef int L_f
    cdef int L_x
    cdef int L_y
    cdef int L_z
    cdef double t_x
    cdef double t_y
    cdef double t_sc
    cdef double t_0
    cdef double t_nc
    cdef double t

    cdef double [:] h

    cdef double u_sc
    cdef double u_soc
    cdef double u_nc
    cdef double u_f

    cdef double mu_sc
    cdef double mu_soc
    cdef double mu_nc
    cdef double mu_f
    cdef double [:] mu_array
    cdef double [:] alpha_R_initial

    cdef double beta
    cdef double wd
    cdef double phase

    cdef double complex [:,:] F_sc_0_initial
    cdef double complex [:,:] F_soc_initial
    cdef double complex [:,:] F_nc_initial
    cdef double complex [:,:] F_f_initial
    cdef double complex [:,:] F_sc_initial

    cdef double complex [:,:] phase_array

    cdef double [:] ky_array
    cdef double [:] kz_array

    cdef double complex [:,:] F_matrix
    cdef double [:] U_array
    cdef double [:,:] t_x_array
    cdef double [:,:] t_y_array
    cdef double [:,:] h_array

    cdef double [:,:] alpha_R_x_array
    cdef double [:,:] alpha_R_y_array

    cdef double complex [:,:,:,:] eigenvectors
    cdef double [:,:,:] eigenvalues
    cdef double complex [:,:] hamiltonian

    cpdef epsilon_ijk(self, int i, int j, double ky, double kz)
    cpdef set_epsilon(self, double complex [:,:] arr, int i, int j, double ky, double kz)
    cpdef delta_gap(self, int i)
    cpdef set_delta(self, double complex [:,:] arr, int i, int j)
    cpdef set_rashba_ky(self, double complex [:,:] arr, int i, int j, double ky, double kz)
    cpdef set_h(self, double complex [:,:] arr, int i, int j)
    cpdef zero_init_hamiltonian(self)
    cpdef set_hamiltonian(self, double ky, double kz)
    cpdef calculate_F_matrix(self)
    cpdef plot_components_of_hamiltonian(self, fig)
    cpdef test_valid(self)
    cpdef get_eigenvectors(self)
    cpdef get_eigenvalues(self)
    cpdef energy_vec(self, double min_E, double max_E, double resolution)
    cpdef local_density_of_states(self, double resolution, double sigma, double min_e, double max_e)
    cpdef ldos_from_problem(self, double resolution, double kernel_size, double min_E, double max_E)
    cpdef compute_energy(self, bool N)
    cpdef forcePhaseDifference(self)
    cpdef current_along_lattice(self)


