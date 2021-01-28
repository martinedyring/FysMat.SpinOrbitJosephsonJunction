import numpy as np
cimport numpy as cnp
cimport cython
from scipy.linalg import eigh
from libcpp cimport bool
cimport cython

from utilities cimport idx_F_i

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

from system_class cimport System

#   Main function to solve a system
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void solve_system(System system,
                        int max_num_iter,
                        double tol,
                        bool juction):
    #   Start by checking that input system is valid.
    system.test_valid()
    cdef int tmp_num_iter = 0
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] delta_diff = np.ones(system.L_x, dtype=np.complex128)
    cdef int num_delta_over_tol = system.L_x
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] delta_store = np.ones((system.L_x, 2), dtype=np.complex128) # 1.column NEW, 2.column OLD

    cdef int ky_idx, kz_idx, ii
    cdef cnp.ndarray[cnp.float64_t, ndim=1] evalues
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] evectors
    cdef int ky_len = len(system.ky_array)
    cdef int kz_len = len(system.kz_array)
    print("ky_len = ", ky_len)



    while num_delta_over_tol > 0 and tmp_num_iter <= max_num_iter:
        # We need to solve the system for all k-s
        for ky_idx in range(ky_len-1): # form k=-pi to k=pi
            for kz_idx in range(kz_len-1):
                system.set_hamiltonian(ky=system.ky_array[ky_idx+1],
                                       kz=system.kz_array[kz_idx+1],
                                        mu_array=system.mu_array,
                                        t_array=system.t_array,
                                        L_x=system.L_x,
                                        F_matrix=system.F_matrix,
                                        U_array=system.U_array,
                                        h_array=system.h_array,
                                        alpha_R_x_array=system.alpha_R_x_array,
                                        alpha_R_y_array=system.alpha_R_y_array)
                #print("test1")

                # Calculates the eigenvalues from hamiltonian.
                evalues, evectors = eigh(system.hamiltonian)
                #print("test2")
                system.eigenvalues[:, ky_idx, kz_idx] = evalues[:]
                system.eigenvectors[:, :, ky_idx, kz_idx] = evectors[:,:]
                #print("test3")
        # Calculate and update the new pairing amplitude functions.
        #print("before calc: ", system.F_matrix[:,idx_F_i])
        #print("forloop done")
        system.F_matrix = system.calculate_F_matrix(system.F_matrix, system.eigenvectors)

        #______________________________________
        #testing what the eigenvectors looks like
        #   Eigenvectors
        #self.eigenvectors = np.zeros(shape=(4 * self.L_x, 4 * self.L_x, self.L_y, self.L_z),dtype=np.complex128)

        #   Eigenvalues
        #self.eigenvalues = np.zeros(shape=(4 * self.L_x, self.L_y, self.L_z), dtype=np.float128)

        #   Hamiltonian
        #self.hamiltonian = np.zeros(shape=(self.L_x * 4, self.L_x * 4), dtype=np.complex128)

        #for i in range(system.L_x*4):
        #    print(system.eigenvectors[i])
        #______________________________________


        #print("test4")
        ###--------
        if juction==True:
            system.forcePhaseDifference() #ONLY for supercurrent i sxs systems
        ###--------
        #print("test5")
        for ii in range(system.L_x):
            delta_store[ii, 0] = system.F_matrix[ii]  # F_ii
            if delta_store[ii, 1] == 0.0:
                delta_diff[ii] = 0.0
            else:
                delta_diff[ii] = (delta_store[ii, 0] - delta_store[ii, 1]) / delta_store[ii, 1]

            delta_store[ii, 1] = system.F_matrix[ii]  # F_ii

        #print("delta_store = ", delta_store)
        #print("delta_diff = ", delta_diff)
        tmp_num_iter += 1
        print("tmp_num_iter = ", tmp_num_iter)

        num_delta_over_tol = 0
        for ii in range(system.L_x):
            if abs(delta_diff[ii]) > tol:
                num_delta_over_tol += 1
        print("num_delta_over_tol = ", num_delta_over_tol)

