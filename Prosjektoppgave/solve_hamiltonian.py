import numpy as np
from scipy.linalg import eigh

from utilities import set_hamiltonian, calculate_F_matrix


#   Main function to solve a system
def solve_system(system, num_iter = 15):
    F_matrix = np.empty(shape=(num_iter+1, system.F_matrix.shape[0], system.F_matrix.shape[1]), dtype=np.complex128)
    F_matrix[0, :, :] = system.F_matrix[:,:]

    #   Start by checking that input system is valid.
    system.test_valid()


    #   Eigenvectors
    eigenvectors = np.zeros(shape=(4*system.L_x, 4*system.L_x, (system.L_y+2)//2), dtype=np.complex128)

    #   Eigenvalues
    eigenvalues = np.zeros(shape=(4*system.L_x, (system.L_y+2)//2), dtype=np.float64)

    #   Hamiltonian
    hamiltonian = np.zeros(shape=(system.L_x*4, system.L_x*4), dtype=np.complex128)

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

    system.E_matrix = eigenvalues[:,:]
    system.eigenvectors = eigenvectors[:,:,:]
    return F_matrix

