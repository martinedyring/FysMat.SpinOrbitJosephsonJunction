import numpy as np
from numpy import conj, real, imag, cos, sin, tanh, exp, sqrt as conj, real, imag, cos, sin, tanh, exp, sqrt

"""
This script contains all extra function solve_hamiltonian.py needs
"""

#   Define index for the pairing amplitude functions:
idx_F_i = 0
idx_F_i_x_pluss = 1
idx_F_i_x_minus = 2
idx_F_i_y_pluss = 3
idx_F_i_y_minus = 4

num_idx_F_i = 5

name_map = [
    r'$F_{s,i}$',
    r'$F_i^{x+}$',
    r'$F_i^{x-}$',
    r'$F_i^{y+}$',
    r'$F_i^{y-}$'
]


def U_term(i, U, F):
    return U*F[i, idx_F_i]

def calculate_F_matrix(eigenvectors, eigenvalues, beta, L_y, F_matrix, k_array):

    #   Initialize the old F_matrix to 0+0j, so that we can start to add new values
    F_matrix[:, :] = 0.0 + 0.0j

    idx_endpoint = F_matrix.shape[0]-1

    # Calculation loops
    for k in range(eigenvalues.shape[1]):
        s_k = 1.0
        for n in range(eigenvalues.shape[0]):
            if (k_array[k]==0) or (k_array[k] == np.pi):
                s_k == 0.0
            coeff = 1 / (2* L_y) * tanh(beta*eigenvalues[n, k]/2)
            for i in range(F_matrix.shape[0]-1):
                # F X+ S
                F_matrix[i, idx_F_i_x_pluss] += coeff * (eigenvectors[4*i, n, k] * conj(eigenvectors[(4*i) + 3, n, k])  -  s_k * eigenvectors[4*i + 1, n, k] * conj(eigenvectors[(4*i) + 2, n, k]))
                # F X- S
                F_matrix[i, idx_F_i_x_minus] += coeff * (eigenvectors[4*i, n, k] * conj(eigenvectors[(4*i) + 3, n, k])  -  s_k * eigenvectors[4*i + 1, n, k] * conj(eigenvectors[(4*i) + 2, n, k]))

                # F Y+ S
                F_matrix[i, idx_F_i_y_pluss] += coeff * (eigenvectors[4*i, n, k] * conj(eigenvectors[(4*i) + 3, n, k]) * np.exp(-1.0j*k_array[k])  -  s_k * eigenvectors[4*i + 1, n, k] * conj(eigenvectors[(4*i) +2, n, k]) * np.exp(1.0j*k_array[k]))

                # F Y- S
                F_matrix[i, idx_F_i_y_minus] += coeff * (eigenvectors[4*i, n, k] * conj(eigenvectors[(4*i) + 3, n, k]) * np.exp(+1.0j*k_array[k])  -  s_k * eigenvectors[4*i + 1, n, k] * conj(eigenvectors[(4*i) +2, n, k]) * np.exp(-1.0j*k_array[k]))

                # s_i
                F_matrix[i, idx_F_i] += 1 / 8 * (F_matrix[i, idx_F_i_x_pluss] + conj(F_matrix[i, idx_F_i_x_pluss]) + F_matrix[i, idx_F_i_x_minus] + conj(F_matrix[i, idx_F_i_x_minus]) + F_matrix[i, idx_F_i_y_pluss] + conj(F_matrix[i, idx_F_i_y_pluss]) + F_matrix[i, idx_F_i_y_minus] + conj(F_matrix[i, idx_F_i_y_minus]))

            #   At the endpoint we can not calculate the correlation in x-direction
            # UP DOWN SAME POINT
            #F_matrix[idx_endpoint, idx_F_i] += (eigenvectors[4*idx_endpoint, n, k] * conj(eigenvectors[(4*idx_endpoint) + 1, n, k])) * (1-f)

            # F Y-
            F_matrix[idx_endpoint, idx_F_i_y_minus] += coeff * (eigenvectors[4*idx_endpoint, n, k]*conj(eigenvectors[4*idx_endpoint+1, n, k])*exp(-1.0j*k_array[k]))

            # F Y+
            F_matrix[idx_endpoint, idx_F_i_y_pluss] += coeff * (eigenvectors[4*idx_endpoint, n, k]*conj(eigenvectors[4*idx_endpoint+1, n, k])*exp(1.0j*k_array[k]))

            # s_i
            F_matrix[i, idx_F_i] += 1 / 4 * (F_matrix[i, idx_F_i_x_pluss] + F_matrix[i, idx_F_i_x_minus] + F_matrix[i, idx_F_i_y_pluss] + F_matrix[i, idx_F_i_y_minus])


# hamiltonian_helper
def hamilton_epsilon(i, j, k, spin, t_array, hz_array):  #spin can take two values: 1 = up, 2 = down
    h = 0.0
    t_0 = 0.0
    t_1 = 0.0

    if i == j:
        h = hz_array[i]
        t = t_array[i]
    elif i == j + 1:
        t = t_array[j]
    elif i == j - 1:
        t = t_array[i]
    if spin == 2:   #   spin down
        h = -h

    e = np.complex128(-(t_0 + 2*t_1 * np.cos(k)) - h)
    return e


def set_diagonal(arr, i, j, k, mu, t_array, hz_array):
    arr[0][0] = hamilton_epsilon(i, j, k, 1, t_array, hz_array) - mu
    arr[1][1] = hamilton_epsilon(i, j, k, 2, t_array, hz_array) - mu
    arr[2][2] = -hamilton_epsilon(i, j, k, 1, t_array, hz_array) + mu
    arr[3][3] = -hamilton_epsilon(i, j, k, 2, t_array, hz_array) + mu

    return arr


def set_u(arr, i, j, k, U, F):
    if i == j:
        arr[0][3] = -U_term(i, U[i], F) / 2
        arr[1][2] = U_term(i, U[i], F) / 2
        arr[2][1] = conj(U_term(i, U[i], F)) / 2
        arr[3][0] = -conj(U_term(i, U[i], F)) / 2
    return arr


def zero_init(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            m[i, j] = 0.0 + 0.0j


def set_hamiltonian(ham, N_x, k, mu, F, U, t_array, hz_array):
    zero_init(ham)
    for i in range(N_x):
        for j in range(N_x):
            # fix reference, send and return array
            """
            if i == N_x-1 and j == N_x-1:
                print("0: i, j = ", i, j)
                ham[4*i:, 4*j:] = set_diagonal(ham[4*i:, 4*j:], 4*N_x, i, j, k, mu, t_ijs_x, t_is_y, h_i)
                ham[4*i:, 4*j:] = set_hubbard_u(ham[4*i:, 4*j:], 4*N_x, i, j, k, U, F)
                ham[4*i:, 4*j:] = set_hxhy(ham[4*i:, 4*j:], 4*N_x, i, j, h_i)
                #ham[4*i:4, 4*j:4] = set_rashba(ham[4*i:4, 4*j:4], 4*N_x, i, j, alpha_R)
            elif i == N_x-1:
                print("1: i, j = ", i, j)
                ham[4*i:, 4*j:4] = set_diagonal(ham[4*i:, 4*j:4], 4*N_x, i, j, k, mu, t_ijs_x, t_is_y, h_i)
                ham[4*i:, 4*j:4] = set_hubbard_u(ham[4*i:, 4*j:4], 4*N_x, i, j, k, U, F)
                ham[4*i:, 4*j:4] = set_hxhy(ham[4*i:, 4*j:4], 4*N_x, i, j, h_i)
                #ham[4*i:4, 4*j:4] = set_rashba(ham[4*i:4, 4*j:4], 4*N_x, i, j, alpha_R)
            elif j == N_x-1:
                print("2: i, j = ", i, j)
                ham[4*i:4, 4*j:] = set_diagonal(ham[4*i:4, 4*j:], 4*N_x, i, j, k, mu, t_ijs_x, t_is_y, h_i)
                ham[4*i:4, 4*j:] = set_hubbard_u(ham[4*i:4, 4*j:], 4*N_x, i, j, k, U, F)
                ham[4*i:4, 4*j:] = set_hxhy(ham[4*i:4, 4*j:], 4*N_x, i, j, h_i)
                #ham[4*i:4, 4*j:4] = set_rashba(ham[4*i:4, 4*j:4], 4*N_x, i, j, alpha_R)
            """
            # print("3: i, j = ", i, j)
            # print(ham[4*i:4, 4*j:4])
            ham[4 * i:4 * i + 4, 4 * j:4 * j + 4] = set_diagonal(ham[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, k, mu, t_array, hz_array)
            ham[4 * i:4 * i + 4, 4 * j:4 * j + 4] = set_u(ham[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, k, U, F)
    return ham


def ccreate_hamiltonian(L_x, k, mu, F, U, t_array, hz_array):
    res = np.zeros((4 * L_x, 4 * L_x), dtype=np.complex128)
    res = set_hamiltonian(res, L_x, k, mu, F, U, t_array, hz_array)
    return res


def ccreate_hamiltonian_from_system(s, k):
    return ccreate_hamiltonian(s.L_x, k, s.mu, s.F_matrix, s.U_array, s.t_array, s.hz_array)


def create_hamiltonian(L_x, k_array, mu, F_matrix, U_array, t_array, hz_array):

    assert F_matrix.shape == (L_x, num_idx_F_i), "F has not the shape of (%i, %i). Got %s" % (L_x, num_idx_F_i, str(F_matrix.shape))
    assert t_array.shape == (L_x - 1, 2), "t_ijs_x has not the shape of (%i, %i). Got %s" % (L_x - 1, num_idx_F_i, str(t_array.shape))
    assert t_array.shape == (L_x, 2), "t_is_y has not the shape of (%i, %i). Got %s" % (L_x, 2, str(t_array.shape))
    assert hz_array.shape == (L_x, 3), "h_i has not the shape of (%i, %i). Got %s" % (L_x, 3, str(hz_array.shape))
    assert U_array.shape == (L_x,), "U_i has not the shape of (%i,). Got %s" % (L_x, str(U_array.shape))
    return np.asarray(ccreate_hamiltonian(L_x, k_array, mu, F_matrix, U_array, t_array, hz_array))


def create_hamiltonian_from_problem(s, k=0.0):  # cpdef create_hamiltonian_from_problem(p, k=0.0):
    return np.asarray(ccreate_hamiltonian_from_system(s, k))


