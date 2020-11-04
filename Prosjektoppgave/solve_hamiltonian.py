import numpy as np
from scipy.linalg import eigh

#from utilities import set_hamiltonian, calculate_F_matrix
from utilities import idx_F_i

#   Main function to solve a system
def solve_system(system, max_num_iter = 100, tol=1e-5, juction=True):
    #for ik in range((system.L_y + 2) // 2):
    #    system.set_hamiltonian(system.k_array[ik])

    #   Start by checking that input system is valid.
    system.test_valid()
    ##print("Hermition: ", np.allclose(np.zeros_like(system.hamiltonian), system.hamiltonian - system.hamiltonian.T.conjugate(), rtol=0.000, atol=1e-4))

    #print(system.F_matrix)
    tmp_num_iter = 0
    delta_diff = 1
    num_delta_over_tol = system.L_x
    delta_store = np.ones((system.L_x, 2), dtype=np.complex128) # 1.column NEW, 2.column OLD
    #while (delta_diff / system.L_x) > tol and tmp_num_iter <= max_num_iter:
    while num_delta_over_tol > 0 and tmp_num_iter <= max_num_iter:
        ##print("Iteration nr. %i" % (tmp_num_iter + 1))

        #if tmp > 0:
        #    system.update_hamiltonian()
        #system.set_hamiltonian()
        # We need to solve the system for all k-s
        for ky_idx in range(1, len(system.ky_array)): # form k=-pi to k=pi
            #if tmp > 0:
            #    system.update_hamiltonian()
            # mulig å ta bort set_hamiltionian, og heller ha en if test i update_hamiltonian for å opdater u og initialisere hamiltionian til 0
            for kz_idx in range(1, len(system.kz_array)):
                system.set_hamiltonian(ky=system.ky_array[ky_idx], kz=system.kz_array[kz_idx])

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
        system.calculate_F_matrix()

        ###--------
        if juction==True:
            system.forcePhaseDifference() #ONLY for supercurrent i sxs systems
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

