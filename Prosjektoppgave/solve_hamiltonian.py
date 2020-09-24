import numpy as np
from scipy.linalg import eigh

#from utilities import set_hamiltonian, calculate_F_matrix
from utilities import idx_F_i

#   Main function to solve a system
def solve_system(system, num_iter = 15, tol=1e-4):
    for ik in range((system.L_y + 2) // 2):
        system.set_hamiltonian(ik)

    #   Start by checking that input system is valid.
    system.test_valid()
    print("Hermitioan : ", np.allclose(np.zeros_like(system.hamiltonian), system.hamiltonian - system.hamiltonian.T.conjugate(), rtol=0.000, atol=1e-4))

    tmp = 0
    delta_diff = 1
    delta_store = np.ones((system.L_x, 2), dtype=np.complex128) # 1.column NEW, 2.column OLD
    while delta_diff > tol:
        print("Iteration nr. %i" % (tmp + 1))
        if tmp > 0:
            system.update_hamiltonian()

        # We need to solve the system for all k-s
        for ik in range((system.L_y + 2) //2):
            # Calculates the eigenvalues from hamiltonian.
            #system.eigenvalues[:, ik], system.eigenvectors[:,:,ik] = eigh(system.hamiltonian)
            values, vectors = eigh(system.hamiltonian)
            system.eigenvalues[:, ik] = values[:]
            system.eigenvectors[:,:,ik] = vectors[:,:]

        # Calculate and update the new pairing amplitude functions.
        system.calculate_F_matrix()

        delta_store[:,0] = system.F_matrix[:,idx_F_i] # F_ii
        #print("del 0 før: \n", delta_store[:,0])
        #print("del 1 før: \n", delta_store[:,1])
        delta_diff = abs(sum((delta_store[:,0]-delta_store[:,1]) / delta_store[:,1]))
        delta_store[:,1] = system.F_matrix[:,idx_F_i] # F_ii
        tmp += 1
        print("delta_diff = ", delta_diff)
        #print("del 0 etter: \n", delta_store[:,0])
        #print("del 1 etter: \n", delta_store[:,1])

    F_matrix = system.F_matrix[:, :]
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
    return F_matrix

