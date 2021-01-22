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
                        bool juction):
    #   Start by checking that input system is valid.
    system.test_valid()
    cdef int tmp_num_iter = 0
    cdef int delta_diff = 1
    cdef int num_delta_over_tol = system.L_x
    cdef double complex [:,:] delta_store = np.ones((system.L_x, 2), dtype=np.complex128) # 1.column NEW, 2.column OLD

    cdef int ky_idx, kz_idx
    cdef double [:,:,:] evalues
    cdef double complex [:,:,:,:] evectors
    while num_delta_over_tol > 0 and tmp_num_iter <= max_num_iter:
        # We need to solve the system for all k-s
        for ky_idx in range(1, len(system.ky_array)): # form k=-pi to k=pi
            for kz_idx in range(1, len(system.kz_array)):
                system.set_hamiltonian(ky=system.ky_array[ky_idx], kz=system.kz_array[kz_idx])

                # Calculates the eigenvalues from hamiltonian.
                evalues, evectors = eigh(system.hamiltonian)
                system.eigenvalues[:, ky_idx, kz_idx], system.eigenvectors[:, :, ky_idx, kz_idx] = evalues, evectors
            # Calculate and update the new pairing amplitude functions.
        #print("before calc: ", system.F_matrix[:,idx_F_i])
        system.calculate_F_matrix()

        ###--------
        if juction==True:
            system.forcePhaseDifference() #ONLY for supercurrent i sxs systems
        ###--------

        delta_store[:, 0] = system.F_matrix[:, idx_F_i]  # F_ii
        delta_diff = abs(np.subtract(delta_store[:, 0], delta_store[:, 1]) / delta_store[:, 1])
        delta_store[:, 1] = system.F_matrix[:, idx_F_i]  # F_ii
        tmp_num_iter += 1

        num_delta_over_tol = len(np.where(delta_diff > tol)[0])

