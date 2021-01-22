import numpy as np
from scipy.linalg import eigh
from utilities import idx_F_i

#   Main function to solve a system
def solve_system(system, max_num_iter = 100, tol=1e-5, juction=True):
    #   Start by checking that input system is valid.
    system.test_valid()
    tmp_num_iter = 0
    delta_diff = 1
    num_delta_over_tol = system.L_x
    delta_store = np.ones((system.L_x, 2), dtype=np.complex128) # 1.column NEW, 2.column OLD
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
        delta_diff = abs((delta_store[:, 0] - delta_store[:, 1]) / delta_store[:, 1])
        delta_store[:, 1] = system.F_matrix[:, idx_F_i]  # F_ii
        tmp_num_iter += 1

        num_delta_over_tol = len(np.where(delta_diff > tol)[0])

    return #F_matrix

