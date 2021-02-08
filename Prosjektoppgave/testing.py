from numba import njit, prange
import numpy as np
from numba.extending import overload
from dask import delayed
from system_helper import set_hamiltonian

@delayed
@njit(fastmath=True)
def testing(max_num_iter,
                       tol,
                       juction,
                       L_x,
                       L_y,
                       L_z,
                       L_sc,
                       L_soc,
                       mu_array,
                       h_array,
                       U_array,
                       F_matrix,
                       t_x_array,
                       ky_array,
                       kz_array,
                       alpha_R_x_array,
                       alpha_R_y_array,
                       beta,
                       phase,
                       eigenvalues,
                       eigenvectors,
                       hamiltonian):
    hamiltonian = set_hamiltonian(ky=ky_array[5],
                                  kz=kz_array[10],
                                  hamiltonian=hamiltonian,
                                  L_x=L_x,
                                  L_sc=L_sc,
                                  L_soc=L_soc,
                                  mu_array=mu_array,
                                  t_array=t_x_array,
                                  U_array=U_array,
                                  F_matrix=F_matrix,
                                  h_array=h_array,
                                  alpha_R_x_array=alpha_R_x_array,
                                  alpha_R_y_array=alpha_R_y_array)

    for ky_idx in range(1, len(ky_array)): # form k=-pi to k=pi
        for kz_idx in range(1, len(kz_array)):

            # Calculates the eigenvalues from hamiltonian.
            evalues, evectors = np.linalg.eigh(hamiltonian)
