import numpy as np
cimport numpy as cnp

import matplotlib.pyplot as plt


from libcpp cimport bool

cdef extern from "complex.h":
    double complex cexp(double complex) nogil
    double complex conj(double complex) nogil;
cdef extern from "math.h":
    double sin(double x) nogil
    double cos(double x) nogil
    double tanh(double x) nogil


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

    cdef cnp.ndarray h

    cdef double u_sc
    cdef double u_soc
    cdef double u_nc
    cdef double u_f

    cdef double mu_sc
    cdef double mu_soc
    cdef double mu_nc
    cdef double mu_f
    cdef cnp.ndarray mu_array
    cdef cnp.ndarray alpha_R_initial

    cdef double beta
    cdef double wd
    cdef double phase

    cdef cnp.ndarray F_sc_0_initial #complex
    cdef cnp.ndarray F_soc_initial
    cdef cnp.ndarray F_nc_initial
    cdef cnp.ndarray F_f_initial
    cdef cnp.ndarray F_sc_initial

    cdef cnp.ndarray phase_array

    cdef cnp.ndarray ky_array
    cdef cnp.ndarray kz_array

    cdef cnp.ndarray F_matrix #complex
    cdef cnp.ndarray U_array
    cdef cnp.ndarray t_x_array
    cdef cnp.ndarray t_y_array
    cdef cnp.ndarray h_array # : :

    cdef cnp.ndarray alpha_R_x_array # : :
    cdef cnp.ndarray alpha_R_y_array # : :

    cdef cnp.ndarray eigenvectors
    cdef cnp.ndarray eigenvalues
    cdef double complex [:,:] hamiltonian # : : complex

    cdef cnp.ndarray energies
    cdef cnp.ndarray ldos # : :

    cdef void set_epsilon(self, int i, int j, double ky, double kz, double[:] mu_array, double [:] t_array) nogil
    cdef void set_delta(self, int i, int j, double [:] U_array, double complex [:] F_matrix) nogil
    cdef void set_rashba_ky(self, int i, int j, double ky, double kz, double [:,:] alpha_R_x_array, double [:,:] alpha_R_y_array) nogil
    cdef void set_h(self, int i, int j, double[:,:] h_array) nogil
    cdef void zero_init_hamiltonian(self)
    cdef void set_hamiltonian(self,
                              double ky,
                              double kz,
                              double[:] mu_array,
                              double [:] t_array,
                              int L_x,
                              double complex[:] F_matrix,
                              double [:] U_array,
                              double [:,:] h_array,
                              double [:,:] alpha_R_x_array,
                              double [:,:] alpha_R_y_array) nogil
    cdef void calculate_F_matrix(self, double complex [:] F_matrix, double[:,:,:,:] eigenvectors)
    cpdef test_valid(self)
    cpdef get_eigenvectors(self)
    cpdef get_eigenvalues(self)
    cpdef get_F_matrix(self)
    cdef double [:] energy_vec(self, double min_E, double max_E, double resolution)
    cdef double [:,:] local_density_of_states(self, double resolution, double sigma, double min_e, double max_e, double complex [:,:,:,:] eigenvectors, double [:,:,:] eigenvalues) nogil
    cpdef ldos_from_problem(self, double resolution, double kernel_size, double min_E, double max_E)
    cpdef double compute_energy(self, bool N)
    cpdef forcePhaseDifference(self)
    cpdef current_along_lattice(self)


