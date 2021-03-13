import numpy as np
#from scipy.linalg import eigh
from numpy.linalg import eigh
from numba import njit
from dask import delayed
#from utilities import set_hamiltonian, calculate_F_matrix
from utilities_t import idx_F_i

from system_helper import set_hamiltonian, calculate_F_matrix, forcePhaseDifference, solve_system_numba

#idx_F_i = 0


#   Main function to solve a system
def solve_system(system, max_num_iter = 100, tol=1e-5, juction=True):
    #for ik in range((system.L_y + 2) // 2):
    #    system.set_hamiltonian(system.k_array[ik])

    #   Start by checking that input system is valid.
    # system.test_valid()
    ##print("Hermition: ", np.allclose(np.zeros_like(system.hamiltonian), system.hamiltonian - system.hamiltonian.T.conjugate(), rtol=0.000, atol=1e-4))

    #print(system.F_matrix)
    tmp_num_iter = 0
    delta_diff = 1
    num_delta_over_tol = system.L_x
    delta_store = np.ones((system.L_x, 2), dtype=np.complex128) # 1.column NEW, 2.column OLD
    #while (delta_diff / system.L_x) > tol and tmp_num_iter <= max_num_iter:
    while num_delta_over_tol > 0 and tmp_num_iter <= max_num_iter:
        print("Iteration nr. %i" % (tmp_num_iter + 1))

        #if tmp > 0:
        #    system.update_hamiltonian()
        #system.set_hamiltonian()
        # We need to solve the system for all k-s
        for ky_idx in range(len(system.ky_array)): # form k=-pi to k=pi
            #if tmp > 0:
            #    system.update_hamiltonian()
            # mulig å ta bort set_hamiltionian, og heller ha en if test i update_hamiltonian for å opdater u og initialisere hamiltionian til 0
            for kz_idx in range(len(system.kz_array)):

                system.hamiltonian = set_hamiltonian(ky=system.ky_array[ky_idx],
                                                     kz=system.kz_array[kz_idx],
                                                     hamiltonian=system.hamiltonian,
                                                     L_x=system.L_x,
                                                     L_sc=system.L_sc,
                                                     L_soc=system.L_soc,
                                                     mu_array=system.mu_array,
                                                     t_array=system.t_x_array,
                                                     U_array=system.U_array,
                                                     F_matrix=system.F_matrix,
                                                     h_array=system.h_array,
                                                     alpha_R_x_array=system.alpha_R_x_array,
                                                     alpha_R_y_array=system.alpha_R_y_array)

                #------- old:
                ##  system.set_hamiltonian(ky=system.ky_array[ky_idx], kz=system.kz_array[kz_idx])
                #________


                #ham = system.hamiltonian
                # Calculates the eigenvalues from hamiltonian.
                evalues, evectors = eigh(system.hamiltonian)
                system.eigenvalues[:, ky_idx, kz_idx], system.eigenvectors[:, :, ky_idx, kz_idx] = evalues, evectors
            #print("Hermition: ",np.allclose(np.zeros_like(system.hamiltonian), system.hamiltonian - system.hamiltonian.T.conjugate(),rtol=0.000, atol=1e-4))
            #print(system.eigenvalues[:,ik])
            #values, vectors = eigh(system.hamiltonian)
            #print("dimentions:")
            #print("Ham: ", system.hamiltonian.shape)
            #print("val: ", values.shape)
            #print("vec: ", vectors.shape)
            #system.eigenvalues[:, ik] = values[:]
            #system.eigenvectors[:,:,ik] = vectors[:,:]

        #print(system.hamiltonian)
        # Calculate and update the new pairing amplitude functions.
        #print("before calc: ", system.F_matrix[:,idx_F_i])
        print("før f")
        system.F_matrix = calculate_F_matrix(F_matrix=system.F_matrix,
                                             L_x=system.L_x,
                                             L_y=system.L_y,
                                             L_z=system.L_z,
                                             eigenvalues=system.eigenvalues,
                                             eigenvectors=system.eigenvectors,
                                             beta=system.beta)
        print("etter f")
        #________old:
        ##  system.calculate_F_matrix()
        #_____________

        ###--------
        if juction==True:
            system.F_matrix = forcePhaseDifference(F_matrix=system.F_matrix,
                                                   phase=system.phase)
            #_____old:
            ##  system.forcePhaseDifference() #ONLY for supercurrent i sxs systems
            #________
        ###--------

        """
        #print("after calc: ", system.F_matrix[:, idx_F_i])
        delta_store[:,0] = system.F_matrix[:,idx_F_i] # F_ii
        #print(system.F_matrix[:,idx_F_i])
        #print("del 0 før: \n", delta_store[:,0])
        #print("del 1 før: \n", delta_store[:,1])
        delta_diff = abs(sum((delta_store[:,0]-delta_store[:,1]) / delta_store[:,1]))
        #delta_diff = (delta_store[:, 0] - delta_store[:, 1]) / delta_store[:, 0]
        #print(delta_diff)
        #delta_diff_real = np.abs(np.real(delta_diff))
        #print(delta_diff_real)
        #delta_diff_imag = np.abs(np.imag(delta_diff))
        #print("real: ",len(np.where((delta_diff_real > tol)[0])))
        #print("imag: ", len(np.where((delta_diff_imag > tol)[0])))
        #delta_diff_latticesite = len(np.where((delta_diff_real > tol)[0])) +  len(np.where((delta_diff_imag > tol)[0]))
        delta_store[:,1] = system.F_matrix[:,idx_F_i] # F_ii
        tmp_num_iter += 1
        print("delta_diff = ", delta_diff, " -- normalized: delta_diff/L_x = ", delta_diff / system.L_x)
        #print("delta_diff_lattice_site = ", delta_diff_latticesite)
        #print("del 0 etter: \n", delta_store[:,0])
        #print("del 1 etter: \n", delta_store[:,1])
        """

        # print("after calc: ", system.F_matrix[:, idx_F_i])
        delta_store[:, 0] = system.F_matrix[:, idx_F_i]  # F_ii
        delta_diff = abs((delta_store[:, 0] - delta_store[:, 1]) / delta_store[:, 1])
        delta_store[:, 1] = system.F_matrix[:, idx_F_i]  # F_ii
        tmp_num_iter += 1

        num_delta_over_tol = len(np.where(delta_diff > tol)[0])
        ##print("num_delta_over_tol = ", num_delta_over_tol, " -- where total num lattice sites is ",system.L_x)


        # print("delta_diff_lattice_site = ", delta_diff_latticesite)
        # print("del 0 etter: \n", delta_store[:,0])
        # print("del 1 etter: \n", delta_store[:,1])

    #F_matrix = system.F_matrix[:, :]
    """
    for r in range(num_iter):
        print("Iteration nr. %i" % (r + 1))

        # We need to solve the system for all k-s
        for ik in range((system.L_y + 2) //2):
            set_hamiltonian(hamiltonian, system.L_x, system.k_array[ik], system.mu, system.F_matrix, system.U_array, system.t_array, system.hz_array)

            # Calculates the eigenvalues from hamiltonian.
            values, vectors = eigh(hamiltonian)
            eigenvalues[:, ik] = values[:]
            eigenvectors[:,:,ik] = vectors[:,:]

        # Calculate and update the new pairing amplitude functions.
        calculate_F_matrix(eigenvectors, eigenvalues, system.beta, system.L_y, system.F_matrix, system.k_array, system.orbital_indicator)

        F_matrix[r + 1, :,:] = system.F_matrix[:,:]
    """
    return #F_matrix

def solve_system_selfconsistent(system, max_num_iter = 1000, tol=1e-3, junction=True):
    F_matrix, eigenvalues, eigenvectors, hamiltonian, num_iter = solve_system_numba(max_num_iter=max_num_iter,
                                                                                                       tol=tol,
                                                                                                       junction=junction,
                                                                                                       L_x=system.L_x,
                                                                                                       L_y=system.L_y,
                                                                                                       L_z=system.L_z,
                                                                                                       L_sc=system.L_sc,
                                                                                                       L_soc=system.L_soc,
                                                                                                       mu_array=system.mu_array,
                                                                                                       h_array=system.h_array,
                                                                                                       U_array=system.U_array,
                                                                                                       F_matrix=system.F_matrix,
                                                                                                       t_x_array=system.t_x_array,
                                                                                                       ky_array=system.ky_array,
                                                                                                       kz_array=system.kz_array,
                                                                                                       alpha_R_x_array=system.alpha_R_x_array,
                                                                                                       alpha_R_y_array=system.alpha_R_y_array,
                                                                                                       beta=system.beta,
                                                                                                       phase=system.phase,
                                                                                                       eigenvalues=system.eigenvalues,
                                                                                                       eigenvectors=system.eigenvectors,
                                                                                                       hamiltonian=system.hamiltonian)

    system.F_matrix, system.eigenvalues, system.eigenvectors, system.hamiltonian = F_matrix, eigenvalues, eigenvectors, hamiltonian
    return num_iter



