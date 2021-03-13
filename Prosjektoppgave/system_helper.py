from numba import njit
import numpy as np
import matplotlib.pyplot as plt

from numba import prange

import time

from math import exp
#from scipy.linalg import eigh
#import numpy.linalg as eigh

idx_F_i = 0

def forcePhaseDifference(F_matrix, phase):
    #phase_plus = np.exp(1.0j * phase)        #   SC_0

    F_matrix[0, 0] = np.abs(F_matrix[0, 0]) * np.exp(1.0j * phase)#phase_plus
    F_matrix[-1, 0] = np.abs(F_matrix[-1, 0])
    return F_matrix

@njit(fastmath=True, parallel=True)
def calculate_F_matrix(F_matrix, L_x, L_y, L_z, eigenvalues, eigenvectors, beta):
    F_matrix[:, idx_F_i] = 0.0 + 0.0j
    for i in prange(F_matrix.shape[0]):
        F_matrix[i, idx_F_i] += np.sum(1 / (2 * L_y * L_z) * (1 + np.tanh(beta * eigenvalues[:, 1:, 1:] / 2)) * (eigenvectors[4 * i, :, 1:, 1:] * np.conj(eigenvectors[(4 * i) + 3, :, 1:, 1:])))
    """


    for kz in range(eigenvectors.shape[3]):
        for ky in range(eigenvectors.shape[2]):
            for j in range(eigenvectors.shape[1]):
                for i in prange(F_matrix.shape[0]):
                    F_matrix[i, idx_F_i] += np.sum(1 / (2 * L_y * L_z) * (1 + np.tanh(beta * eigenvalues[j, ky, kz] / 2)) * (eigenvectors[4 * i, j, ky, kz] * np.conj(eigenvectors[(4 * i) + 3, j, ky, kz])))

    """
    return F_matrix

@njit(fastmath=True)
def epsilon_ijk(i, j, ky, kz, mu_array, t_array):  # spin can take two values: 1 = up, 2 = down
    e = 0.0
    if i == j:
        e = np.complex128(- 2 * t_array[i] * (np.cos(ky) + np.cos(kz)) - mu_array[i])  # spini in (1, 2) => (0, 1) index => (spinup, spindown)
    elif i == j + 1:
        e = np.complex128(-t_array[j])  # x #-
    elif i == j - 1:
        e = np.complex128(-t_array[i])  # x #-
    return e

@njit(fastmath=True)
def set_epsilon(arr, i, j, ky, kz, mu_array, t_array):
    arr[0][0] += epsilon_ijk(i, j, ky, kz, mu_array, t_array)
    arr[1][1] += epsilon_ijk(i, j, ky, kz, mu_array, t_array)
    arr[2][2] += -epsilon_ijk(i, j, ky, kz, mu_array, t_array)
    arr[3][3] += -epsilon_ijk(i, j, ky, kz, mu_array, t_array)
    return arr

@njit(fastmath=True)
def delta_gap(i, U_array, F_matrix):
    return U_array[i] * F_matrix[i, idx_F_i]

# har endre fra += til = grunnet update_hamiltonian. Sjekk at dette fungere fremdeles
@njit(fastmath=True)
def set_delta(arr, i, j, U_array, F_matrix):
    # Comment out +=, and remove the other comp from update_hamil to increase runtime and check if there is any diff. Shouldnt be diff in output.
    if i==j:
        #   Skjekk om du må bytte om index
        arr[0][3] = -delta_gap(i, U_array, F_matrix)#/2
        arr[1][2] = delta_gap(i, U_array, F_matrix)#/2
        arr[2][1] = np.conj(delta_gap(i, U_array, F_matrix))#/2
        arr[3][0] = -np.conj(delta_gap(i, U_array, F_matrix))#/2

    return arr

@njit(fastmath = True)
def set_rashba_ky(arr, i, j, ky, kz, alpha_R_x_array, alpha_R_y_array, L_soc, L_sc):
    #I = 1.0j
    sinky = np.sin(ky)
    sinkz = np.sin(kz)

    # barr = arr[2:][2:]
    if i == j:
        # (n_z*sigma_x - n_x*sigma_z)
        y00 = -alpha_R_y_array[i, 0]
        y01 = alpha_R_y_array[i, 2]
        y10 = alpha_R_y_array[i, 2]
        y11 = alpha_R_y_array[i, 0]

        z01_up = -alpha_R_y_array[i, 1] - 1.0j * alpha_R_y_array[i, 0]
        z10_up = -alpha_R_y_array[i, 1] + 1.0j * alpha_R_y_array[i, 0]
        z01_down = -alpha_R_y_array[i, 1] + 1.0j * alpha_R_y_array[i, 0]
        z10_down = -alpha_R_y_array[i, 1] - 1.0j * alpha_R_y_array[i, 0]


        # Upper left
        arr[0][0] += sinky * y00
        arr[0][1] += sinky * y01 + sinkz * z01_up
        arr[1][0] += sinky * y10 + sinkz * z10_up
        arr[1][1] += sinky * y11

        # Bottom right. Minus and conjugate
        arr[2][2] += sinky * y00
        arr[2][3] += sinky * y01 + sinkz * z01_down
        arr[3][2] += sinky * y10 + sinkz * z10_down
        arr[3][3] += sinky * y11

    # Backward jump X-
    elif ((i == (j - 1)) or (i == (j + 1))):
        # elif j == i + 1 or j == i - 1:
        #if j == i + 1:  # Backward jump X-
        l = j ### j
        xi = 0
        if (i == (j - 1)):  # Backward jump X-
            if (L_sc <= i < (L_sc + L_soc)):
                l = i  # i
            coeff = -1.0 / 4.0

            #if ((L_sc <= i < (L_sc + L_soc)) and (L_sc <= j-1 < (L_sc + L_soc))):  # check if both i and j are inside soc material
            #    xi = 1
        else:  # Forward jump X+
            if (L_sc <= i < (L_sc + L_soc)):
                l = i
            #l = i  # j
            coeff = 1.0 / 4.0

            #if ((L_sc <= i < (L_sc + L_soc)) and (L_sc <= j+1 < (L_sc + L_soc))):  # check if both i and j are inside soc material
            #    xi = 1

        if (L_sc <= i < (L_sc + L_soc)) and (L_sc <= j < (L_sc + L_soc)):# and (L_sc <= j-1 < (L_sc + L_soc)):  # check if both i and j are inside soc material
            xi = 1

        s00_up = 1.0j * alpha_R_x_array[int(l), 1]
        s01_up = -alpha_R_x_array[int(l), 2]  # maybe change sign on s01 and s10??
        s10_up = alpha_R_x_array[int(l), 2]
        s11_up = - 1.0j * alpha_R_x_array[int(l), 1]

        s00_down = 1.0j * alpha_R_x_array[int(l), 1]
        s01_down = alpha_R_x_array[int(l), 2]  # maybe change sign on s01 and s10??
        s10_down = -alpha_R_x_array[int(l), 2]
        s11_down = - 1.0j * alpha_R_x_array[int(l), 1]

        arr[0][0] += coeff * s00_up * (1 + xi)
        arr[0][1] += coeff * s01_up * (1 + xi)
        arr[1][0] += coeff * s10_up * (1 + xi)
        arr[1][1] += coeff * s11_up * (1 + xi)

        # arr[2][2] += conj(coeff * s00)
        # arr[2][3] += conj(coeff * s01)
        # arr[3][2] += conj(coeff * s10)
        # arr[3][3] += conj(coeff * s11)

        arr[2][2] += coeff * s00_down * (1 + xi)
        arr[2][3] += coeff * s01_down * (1 + xi)
        arr[3][2] += coeff * s10_down * (1 + xi)
        arr[3][3] += coeff * s11_down * (1 + xi)

    return arr

@njit(fastmath=True)
def set_h(arr, i, j, h_array):
# cdef double complex [:,:] n_dot_sigma = n_dot_sigma(h_i[i,:])
    if i == j:
        arr[0][0] += h_array[i, 2]
        arr[0][1] += h_array[i, 0] - 1.0j * h_array[i, 1]
        arr[1][0] += h_array[i, 0] + 1.0j * h_array[i, 1]
        arr[1][1] += -h_array[i, 2]

        arr[2][2] += -h_array[i, 2]
        arr[2][3] += -h_array[i, 0] - 1.0j * h_array[i, 1]
        arr[3][2] += -h_array[i, 0] + 1.0j * h_array[i, 1]
        arr[3][3] += h_array[i, 2]
    return arr

@njit(fastmath=True)
def zero_init_hamiltonian(hamiltonian):
    hamiltonian[:,:] = 0.0 + 0.0j
    return hamiltonian

@njit(fastmath=True, parallel=True)
def set_hamiltonian(ky, kz, hamiltonian, L_x, L_sc, L_soc, mu_array, t_array, U_array, F_matrix, h_array, alpha_R_x_array, alpha_R_y_array):
    #hamiltonian = zero_init_hamiltonian(hamiltonian)
    for i in prange(L_x):
    #for i in prange(L_x):
        for j in range(L_x):
            hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = set_epsilon(hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, ky, kz, mu_array, t_array)
            hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = set_delta(hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, U_array, F_matrix)
            hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = set_rashba_ky(hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, ky, kz, alpha_R_x_array, alpha_R_y_array, L_soc, L_sc)
            hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = set_h(hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, h_array)
    return hamiltonian

@njit(fastmath=True, parallel=True)
def update_hamiltonian(hamiltonian, U_array, F_matrix, L_x):
    for i in prange(L_x):
        for j in range(L_x):
            hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = set_delta(hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, U_array, F_matrix)
    return hamiltonian

#   Plot delta, U-term and F for the resulting hamiltonian
def plot_components_of_hamiltonian(F_matrix, U_array, alpha_R_x_array, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(10,10))

    ax = fig.subplots(nrows=1, ncols=2).flatten()

    #   Delta-term
    line = ax[0].plot(U_array, label='U')
    ax[0].plot(np.multiply(U_array, np.abs(F_matrix[:,idx_F_i])), ls=':', label=r'$|\Delta|$')
    ax[0].plot(np.real(F_matrix[:, idx_F_i]), ls='--', label=r'$F_{i}^{\uparrow\downarrow}$')
    ax[0].set_title('Delta')
    ax[0].legend()

    # rashba coupling
    line = ax[1].plot(alpha_R_x_array[:, 0], label=r'$\alpha_R^x$')
    ax[1].plot(alpha_R_x_array[:, 1], ls='--', label=r'$\alpha_R^y$')
    ax[1].plot(alpha_R_x_array[:, 2], ls=':', label=r'$\alpha_R^z$')
    ax[1].legend()
    ax[1].set_title('Rashba SOC coupling')

    #fig.savefig('Hamilton components, mu_s=0.9, mu_soc=0.85, u=-4.2.png', bbox_inches='tight')


@njit(fastmath=True)
def energy_vec(min_E, max_E, resolution):
    Ne = int((max_E - min_E) / resolution)
    Es = np.linspace(min_E, max_E, Ne, dtype=np.float64)
    return Es


@njit(fastmath=True, parallel=True)
def local_density_of_states(resolution, sigma, min_e, max_e, L_y, L_z, eigenvalues, eigenvectors):
    # sigma is the size of gaussian function
    num_energies = int((max_e - min_e) / resolution) #number energi
    num_latticesites = eigenvectors.shape[0] // 4 #number latticesties
    coeff = 1.0 / (sigma * np.sqrt(2*np.pi)) / (L_y*L_z)

    Es = energy_vec(min_e, max_e, resolution)
    ldos = np.zeros((num_latticesites, num_energies), dtype=np.float64)


    pos_e_diff = eigenvalues[:, 1:, 1:] /2
    for ii in prange(num_latticesites):
        us = pow(abs(eigenvectors[4 * ii, :, 1:, 1:]), 2) + pow(abs(eigenvectors[4 * ii + 1, :, 1:, 1:]), 2)
        for ei in range(num_energies):
            eng = Es[ei]
            pos_ldos = coeff * np.exp(-pow((pos_e_diff - eng) / sigma, 2))

            ldos[ii, ei] += np.sum(np.multiply(us, pos_ldos))

    return ldos, Es

@njit(fastmath=True)
def compute_energy(L_x, U_array, F_matrix, mu_array, ky_array, kz_array, beta, t, eigenvalues, N=False):
    # Compute the Free energy as in Linas Master Thisis

    delta_array = np.multiply(U_array, np.real(F_matrix[:,idx_F_i]))


    # u-term. Cant do if U = 0 in the region
    U_index = np.where(U_array != 0)
    U_energy = 0.0
    for u in U_index[0]:
        U_energy += np.abs(delta_array[u])**2 / U_array[u]


    hopping_energy = np.sum(2 * t * (np.cos(ky_array[1:] + np.cos(kz_array[1:]))))
    epsilon_energy = np.sum(hopping_energy + mu_array[:])
    #hopping_energy = self.L_x*np.sum(2 * self.t_0 * cos(self.k_array[:]) + np.sum(self.mu_array[:]))

    H_0 = L_x * U_energy - epsilon_energy
    F = H_0 - (1 / beta) * np.sum(np.log(1 + np.exp(-beta * eigenvalues[:, 1:, 1:] / 2)))

    return F

@njit(fastmath=True, parallel=True)
def current_along_lattice(L_x, L_y, L_z, L_sc_0, L_soc, beta, t, alpha_R_x_array, eigenvalues, eigenvectors):
    I = 1.0j
    current = np.zeros(L_x - 1, dtype=np.complex128)
    tanh_coeff = 1 / (np.exp(beta * eigenvalues) + 1)
    tanh_coeff /= (L_y * L_z)  # 1/(system.L_y*system.L_z) *(1-np.tanh(system.beta * system.eigenvalues / 2)) #-


    for ix in range(1, len(current)):  # -1 because it doesnt give sense to check last point for I+
        xi_ii = 0
        xi_minus = 0
        xi_pluss = 0

        # """
        if (L_sc_0 <= ix < (L_sc_0 + L_soc)):  # check if both i and i are inside soc material
            xi_ii = 1
        if (L_sc_0 <= ix < (L_sc_0 + L_soc)):  # and (system.L_sc_0 <= ix+1 < (system.L_sc_0 + system.L_soc)):# and (system.L_sc_0 <= ix-1 < (system.L_sc_0 + system.L_soc)): #check if both i and i+1 are inside soc material
            xi_pluss = 1
        if (L_sc_0 <= ix < (L_sc_0 + L_soc)):  # and (system.L_sc_0 <= ix-1 < (system.L_sc_0 + system.L_soc)):# and (system.L_sc_0 <= ix+1 < (system.L_sc_0 + system.L_soc)): #check if both i and i-1 are inside soc material
            xi_minus = 1

        B_opp_opp_psite = 1.0j / 4 * alpha_R_x_array[ix, 1] * (1 + xi_ii)
        B_opp_opp_pluss = 1.0j / 4 * alpha_R_x_array[ix + 1, 1] * (1 + xi_pluss)
        B_opp_opp_minus = 1.0j / 4 * alpha_R_x_array[ix - 1, 1] * (1 + xi_minus)
        B_opp_opp_msite = 1.0j / 4 * alpha_R_x_array[ix, 1] * (1 + xi_ii)

        B_ned_ned_psite = - 1.0j / 4 * alpha_R_x_array[ix, 1] * (1 + xi_ii)
        B_ned_ned_pluss = - 1.0j / 4 * alpha_R_x_array[ix + 1, 1] * (1 + xi_pluss)
        B_ned_ned_minus = - 1.0j / 4 * alpha_R_x_array[ix - 1, 1] * (1 + xi_minus)
        B_ned_ned_msite = - 1.0j / 4 * alpha_R_x_array[ix, 1] * (1 + xi_ii)

        B_opp_ned_psite = - 1.0 / 4 * alpha_R_x_array[ix, 2] * (1 + xi_ii)
        B_opp_ned_pluss = - 1.0 / 4 * alpha_R_x_array[ix + 1, 2] * (1 + xi_pluss)
        B_opp_ned_minus = - 1.0 / 4 * alpha_R_x_array[ix - 1, 2] * (1 + xi_minus)
        B_opp_ned_msite = - 1.0 / 4 * alpha_R_x_array[ix, 2] * (1 + xi_ii)

        B_ned_opp_psite = + 1.0 / 4 * alpha_R_x_array[ix, 2] * (1 + xi_ii)
        B_ned_opp_pluss = + 1.0 / 4 * alpha_R_x_array[ix + 1, 2] * (1 + xi_pluss)
        B_ned_opp_minus = + 1.0 / 4 * alpha_R_x_array[ix - 1, 2] * (1 + xi_minus)
        B_ned_opp_msite = + 1.0 / 4 * alpha_R_x_array[ix, 2] * (1 + xi_ii)


        # ---- Hopping x+ (imag)----#
        #:
        current[ix] += np.imag(2 * np.sum(t * tanh_coeff[:, 1:, 1:] * (np.conj(eigenvectors[4 * ix, :, 1:, 1:]) * eigenvectors[4 * (ix + 1), :, 1:, 1:])))  # opp opp # * (np.exp(1.0j * system.ky_array[1:]) * np.exp(1.0j * system.kz_array[1:])))) #sigma = opp
        current[ix] += np.imag(2 * np.sum(t * tanh_coeff[:, 1:, 1:] * (np.conj(eigenvectors[4 * ix + 1, :, 1:, 1:]) * eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:])))  # ned ned # * (np.exp(1.0j * system.ky_array[1:]) * np.exp(1.0j * system.kz_array[1:])))) #sigma = opp

        current[ix] -= np.imag(2 * np.sum(t * tanh_coeff[:, 1:, 1:] * (np.conj(eigenvectors[4 * ix, :, 1:, 1:]) * eigenvectors[4 * (ix - 1), :, 1:, 1:])))  # opp opp # # * (np.exp(-1.0j * system.ky_array[1:]) * np.exp(-1.0j * system.kz_array[1:])))) #sigma = opp
        current[ix] -= np.imag(2 * np.sum(t * tanh_coeff[:, 1:, 1:] * (np.conj(eigenvectors[4 * ix + 1, :, 1:, 1:]) * eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:])))  # ned ned # # * (np.exp(-1.0j * system.ky_array[1:]) * np.exp(-1.0j * system.kz_array[1:])))) #sigma = opp

        # --- Rashba x+ (real)----#
        #:
        # """
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_opp_psite * (np.conj(eigenvectors[4 * ix, :, 1:, 1:]) * eigenvectors[4 * (ix + 1), :, 1:, 1:])))  # opp opp
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_opp_pluss * (np.conj(eigenvectors[4 * (ix + 1), :, 1:, 1:]) * eigenvectors[4 * ix, :, 1:, 1:])))  # opp opp
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_opp_minus * (np.conj(eigenvectors[4 * (ix - 1), :, 1:, 1:]) * eigenvectors[4 * ix, :, 1:, 1:])))  # opp opp
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_opp_msite * (np.conj(eigenvectors[4 * ix, :, 1:, 1:]) * eigenvectors[4 * (ix - 1), :, 1:, 1:])))  # opp opp

        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_ned_psite * (np.conj(eigenvectors[4 * ix + 1, :, 1:, 1:]) * eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:])))  # ned ned
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_ned_pluss * (np.conj(eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:]) * eigenvectors[4 * ix + 1, :, 1:, 1:])))  # ned ned
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_ned_minus * (np.conj(eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:]) * eigenvectors[4 * ix + 1, :, 1:, 1:])))  # ned ned
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_ned_msite * (np.conj(eigenvectors[4 * ix + 1, :, 1:, 1:]) * eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:])))  # ned ned

        #:
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_ned_psite * (np.conj(eigenvectors[4 * ix, :, 1:, 1:]) * eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:])))  # opp ned
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_ned_pluss * (np.conj(eigenvectors[4 * (ix + 1), :, 1:, 1:]) * eigenvectors[4 * ix + 1, :, 1:, 1:])))  # opp ned
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_ned_minus * (np.conj(eigenvectors[4 * (ix - 1), :, 1:, 1:]) * eigenvectors[4 * ix + 1, :, 1:, 1:])))  # opp ned
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_ned_msite * (np.conj(eigenvectors[4 * ix, :, 1:, 1:]) * eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:])))  # opp ned

        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_opp_psite * (np.conj(eigenvectors[4 * ix + 1, :, 1:, 1:]) * eigenvectors[4 * (ix + 1), :, 1:, 1:])))  # ned opp
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_opp_pluss * (np.conj(eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:]) * eigenvectors[4 * ix, :, 1:, 1:])))  # ned opp
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_opp_minus * (np.conj(eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:]) * eigenvectors[4 * ix, :, 1:, 1:])))  # ned opp
        current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_opp_msite * (np.conj(eigenvectors[4 * ix + 1, :, 1:, 1:]) * eigenvectors[4 * (ix - 1), :, 1:, 1:])))  # ned opp
        # """

    return current

@njit(fastmath=True, parallel=True)
def solve_system_numba(max_num_iter,
                       tol,
                       junction,
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

    tmp_num_iter = 0
    #delta_diff = np.ones
    num_delta_over_tol = L_x
    delta_store = np.ones((L_x, 2), dtype=np.complex128) # 1.column NEW, 2.column OLD

    while num_delta_over_tol > 0 and tmp_num_iter <= max_num_iter:
        #print("Iteration nr. %i" % (tmp_num_iter + 1))
        #start = time.time()
        for ky_idx in prange(1, len(ky_array)): # form k=-pi to k=pi  #prange, set 1/2-2020
            for kz_idx in range(1, len(kz_array)):
                if tmp_num_iter==0:
                    ham = set_hamiltonian(ky=ky_array[ky_idx],
                                         kz=kz_array[kz_idx],
                                         hamiltonian=np.zeros(shape=(4*L_x, 4*L_x), dtype=np.complex128),
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
                else:
                    ham = update_hamiltonian(hamiltonian=np.copy(hamiltonian[:,:, ky_idx, kz_idx]),
                                          U_array=U_array,
                                          F_matrix=F_matrix,
                                          L_x=L_x)

                hamiltonian[:, :, ky_idx, kz_idx] = ham
                # Calculates the eigenvalues from hamiltonian.
                evalues, evectors = np.linalg.eigh(hamiltonian[:,:, ky_idx, kz_idx])
                eigenvalues[:, ky_idx, kz_idx], eigenvectors[:, :, ky_idx, kz_idx] = evalues, evectors
        #duration = time.time() - start
        #print(duration)
        fmatrix = calculate_F_matrix(F_matrix=F_matrix,
                                     L_x=L_x,
                                     L_y=L_y,
                                     L_z=L_z,
                                     eigenvalues=eigenvalues,
                                     eigenvectors=eigenvectors,
                                     beta=beta)
        F_matrix = fmatrix
        if junction==True:
            #F_matrix = forcePhaseDifference(F_matrix=F_matrix,
                                            #phase=phase)
            F_matrix[0, 0] = np.abs(F_matrix[0, 0]) * np.exp(1.0j * phase)  # phase_plus
            F_matrix[-1, 0] = np.abs(F_matrix[-1, 0])

        delta_store[:, 0] = F_matrix[:, idx_F_i]  # F_ii
        #delta_diff = abs((delta_store[:, 0] - delta_store[:, 1]) / delta_store[:, 1])
        delta_diff_tmp = (delta_store[:, 0] - delta_store[:, 1])
        delta_diff_tmp /= delta_store[:, 1]
        delta_diff = np.abs(delta_diff_tmp)
        delta_store[:, 1] = F_matrix[:, idx_F_i]  # F_ii
        tmp_num_iter += 1

        num_delta_over_tol = len(np.where(delta_diff > tol)[0])
    return F_matrix, eigenvalues, eigenvectors, hamiltonian, tmp_num_iter