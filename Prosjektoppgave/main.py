import numpy as np
import matplotlib.pyplot as plt

from solve_hamiltonian import solve_system
from system_class import System
from plots import plot_complex_function
from utilities import label_F_matrix

"""
This is the main function to run all programs
"""


def pairing_amplitude_all_orbitals():
    F_sc_initial_s = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # s-orbital
    F_sc_initial_d = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    F_sc_initial_px = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    F_sc_initial_py = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    F_sc_initial_pxpy = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0j, 1.0j, -1.0j, -1.0j],
    F_sc_initial_spy = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0+1.0j, 1.0+1.0j, 1.0-1.0j, 1.0-1.0j],

    mu_s = -3.5,
    mu_d = -0.5,
    mu_px = -1.5,
    mu_py = -1.5,

    system_s = System(mu_orbital=mu_s, orbital_indicator='s', F_sc_initial_orbital=F_sc_initial_s)
    system_d = System(mu_orbital=mu_d, orbital_indicator='d', F_sc_initial_orbital = F_sc_initial_d)
    system_px = System(mu_orbital=mu_px, orbital_indicator='px', F_sc_initial_orbital = F_sc_initial_px)
    system_py = System(mu_orbital=mu_py, orbital_indicator='py', F_sc_initial_orbital = F_sc_initial_py)

    F_matrix_s = np.asarray(solve_system(system_s, num_iter=2))
    F_matrix_d = np.asarray(solve_system(system_d, num_iter=2))
    F_matrix_px = np.asarray(solve_system(system_px, num_iter=2))
    F_matrix_py = np.asarray(solve_system(system_py, num_iter=2))

    tol = 1e-13
    fig = plt.figure(figsize=(20, 20))
    ax = fig.subplots(ncols=3, nrows=(F_matrix_s.shape[2] + 2) // 3, sharex=True, sharey=False).flatten()
    for i in range(F_matrix_s.shape[2]):
        ys = F_matrix_s[-1, :, i]
        # ys[np.abs(ys)< tol] = 0.0 + 0.0j
        plot_complex_function(y=ys, ax=ax[i], labels=['Real part', 'Imaginary part'])
        ax[i].grid()
        ax[i].legend()
        ax[i].set_title(label_F_matrix[i])
    fig.subplots_adjust(wspace=0.0)

    fig = plt.figure(figsize=(20, 6))
    fig.subplots_adjust(wspace=0.0)
    system_s.plot_components_of_hamiltonian(fig)

def solve_and_return_system_and_F_matrix(mu_orbital, orbital_indicator,F_sc_initial_orbital, tol=1e-4):

    system = System(mu_orbital=mu_orbital, orbital_indicator=orbital_indicator,F_sc_initial_orbital=F_sc_initial_orbital)

    F_matrix = np.asarray(solve_system(system, 3, tol))
    return system, F_matrix




#if __name__ == "__main__":
#    pairing_amplitude()