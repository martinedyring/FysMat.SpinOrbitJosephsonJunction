from system_helper import *

@njit(fastmath=True, parallel=True)
def solve_system_nonconsistent(junction,
                               L_x,
                               L_y,
                               L_z,
                               L_sc,
                               L_soc,
                               mu_array,
                               h_array,
                               U_array,
                               F_matrix,
                               t_y_array,
                               ky_array,
                               kz_array,
                               alpha_R_x_array,
                               alpha_R_y_array,
                               beta,
                               phase,
                               eigenvalues,
                               eigenvectors,
                               hamiltonian):


    #one iteration to sovle systen, and one iteration on compute the non-consitent solution after locked complete sc regime


    #for ky_idx in prange(1, len(ky_array))
    for ky_idx in prange(len(ky_array)): # form k=-pi to k=pi  #prange, set 1/2-2020
        for kz_idx in range(len(kz_array)):
            hamiltonian[:,:, ky_idx, kz_idx] = set_hamiltonian(ky=ky_array[ky_idx],
                                                                 kz=kz_array[kz_idx],
                                                                 hamiltonian=np.zeros(shape=(4*L_x, 4*L_x), dtype=np.complex128),
                                                                 L_x=L_x,
                                                                 L_sc=L_sc,
                                                                 L_soc=L_soc,
                                                                 mu_array=mu_array,
                                                                 t_array=t_y_array,
                                                                 U_array=U_array,
                                                                 F_matrix=F_matrix,
                                                                 h_array=h_array,
                                                                 alpha_R_x_array=alpha_R_x_array,
                                                                 alpha_R_y_array=alpha_R_y_array)

            # Calculates the eigenvalues from hamiltonian.
            """
            ham = hamiltonian[:,:, ky_idx, kz_idx]
            is_hermitian = np.allclose(np.zeros_like(ham), ham - ham.T.conjugate(), rtol=0.000, atol=1e-4)
            if is_hermitian == False:
                print("First: Hamiltonian is not Hermitian for ", ky_idx, kz_idx)
            evalues, evectors = np.linalg.eigh(ham)
            #if np.amax(evalues) > 1e10:
            #    evalues, evectors = np.linalg.eigh(hamiltonian[:, :, ky_idx, kz_idx])
            """
            eigenvalues[:, ky_idx, kz_idx], eigenvectors[:, :, ky_idx, kz_idx] = np.linalg.eigh(hamiltonian[:,:, ky_idx, kz_idx]) #evalues.astype(np.float64), evectors.astype(np.complex128) # #evalues, evectors

    #duration = time.time() - start
    #print(duration)
    """
    F_matrix = calculate_F_matrix(F_matrix=F_matrix,
                                 L_x=L_x,
                                 L_y=L_y,
                                 L_z=L_z,
                                 eigenvalues=eigenvalues,
                                 eigenvectors=eigenvectors,
                                 beta=beta)

    if junction==True:
        #F_matrix = forcePhaseDifference(F_matrix=F_matrix,
                                        #phase=phase)
        F_matrix[:L_sc, 0] = abs(F_matrix[L_sc-1, 0]) * np.exp(1.0j * phase)  # phase_plus
        F_matrix[-L_sc:, 0] = abs(F_matrix[-L_sc, 0])

    #for ky_idx in prange(1, len(ky_array)):
    for ky_idx in range(1, len(ky_array)): # form k=-pi to k=pi  #prange, set 1/2-2020
        for kz_idx in range(1, len(kz_array)):
            hamiltonian[:, :, ky_idx, kz_idx] = set_hamiltonian(ky=ky_array[ky_idx],
                                                                kz=kz_array[kz_idx],
                                                                hamiltonian=hamiltonian[:, :, ky_idx, kz_idx],
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
            #hamiltonian[:,:, ky_idx, kz_idx] = update_hamiltonian(hamiltonian=hamiltonian[:,:, ky_idx, kz_idx],
            #                                                          U_array=U_array,
            #                                                          F_matrix=F_matrix,
            #                                                          L_x=L_x)
            # Calculates the eigenvalues from hamiltonian.
            ham = hamiltonian[:, :, ky_idx, kz_idx]
            is_hermitian = np.allclose(np.zeros_like(ham), ham - ham.T.conjugate(), rtol=0.000, atol=1e-4)
            if is_hermitian == False:
                print("Second: Hamiltonian is not Hermitian for ", ky_idx, kz_idx)
            evalues, evectors = np.linalg.eigh(ham)
            #if np.amax(evalues) > 1e10:
            #    evalues, evectors = np.linalg.eigh(hamiltonian[:, :, ky_idx, kz_idx])
            eigenvalues[:, ky_idx, kz_idx], eigenvectors[:, :, ky_idx, kz_idx] = evalues.astype(np.float64), evectors.astype(np.complex128)  # np.linalg.eigh(hamiltonian[:,:, ky_idx, kz_idx]) #evalues, evectors
    """
    return F_matrix, eigenvalues, eigenvectors, hamiltonian
