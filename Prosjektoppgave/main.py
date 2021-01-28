# !/usr/bin/python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from solve_hamiltonian import solve_system, solve_system_new
from system_class import System
from plots import plot_complex_function, plot_pairing_amplitude
from utilities import label_F_matrix

"""
This is the main function to run all programs
"""

# debye freq ???

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

def solve_and_return_system_and_F_matrix(max_num_iter=100, tol=1e-4):

    system = System(L_y = 102, mu_sc = 0.9, mu_nc = 0.9, mu_soc = 0.85, u_sc = -4.2)

    #F_matrix = np.asarray(solve_system(system, 3, tol))

    solve_system(system, max_num_iter, tol)
    F_matrix = system.F_matrix
    return system, F_matrix


def solve_and_test_small_system(max_num_iter=100, tol=1e-4):

    system = System(phase=0, L_y = 50,L_z=50, L_sc_0 = 0, L_nc=25, L_sc=25, L_soc=0, mu_sc = 0.9, mu_nc = 0.9, mu_soc = 0.85, u_sc = -4.2, beta=100)
    #system = System(phase=np.pi / 4, L_y=20, L_z=20, L_sc_0=7, L_h=5, L_sc=7, L_soc=0, mu_sc=0.9, mu_nc=0.9,mu_soc=0.85, u_sc=-4.2)


    #F_matrix = np.asarray(solve_system(system, 3, tol))

    #solve_system(system, max_num_iter, tol, juction=False)
    solve_system_new(system, max_num_iter, tol, juction=False)
    F_matrix = system.F_matrix
    return system, F_matrix


def solve_for_sns_system(max_num_iter=2000, tol=1e-4):
    phase_array = np.linspace(0, 2*np.pi, 20)
    current_midle = np.zeros(len(phase_array))

    # define first utside, so that we can change phase and solve the same system (do not need to start from scratch to solve)
    print("#_________ Phase = ", phase_array[0], "_________#")
    system = System(phase=phase_array[0], L_y=15, L_z=15, L_sc_0=5, L_nc=5, L_sc=5, L_soc=0, mu_sc=0.9, mu_nc=0.9,mu_soc=0.85, u_sc=-4.2, beta=100)
    solve_system(system, max_num_iter, tol)
    current_arr = system.current_along_lattice()
    current_midle[0] = np.imag(current_arr)[system.L_sc_0 + system.L_nc // 2]

    site_x = np.linspace(0, system.L_x - 1, system.L_x - 1)
    plt.plot(site_x[1:], np.imag(current_arr)[1:], label="imag")
    plt.legend()
    plt.xlabel("lattice site [SC-NC-SC]/[5-5-5]")
    plt.ylabel("current I_x")
    plt.grid()
    plt.show()

    #plot_pairing_amplitude(system, system.F_matrix)


    for i in range(1, len(phase_array)):
        print("#_________ Phase = ",phase_array[i],"_________#")
        #system.phase = phase_array[i]
        system = System(old_solution = True, old_F_matrix_guess=abs(system.F_matrix), phase=phase_array[i], L_y=15, L_z=15, L_sc_0=5, L_nc=5, L_sc=5, L_soc=0, mu_sc=0.9, mu_nc=0.9,mu_soc=0.85, u_sc=-4.2, beta=100)


        solve_system(system, max_num_iter, tol)
        current_arr = system.current_along_lattice()
        current_midle[i] = np.imag(current_arr)[system.L_sc_0 + system.L_nc // 2]
        site_x = np.linspace(0, system.L_x - 1, system.L_x - 1)
        plt.plot(site_x[1:], np.imag(current_arr)[1:], label="imag")
        plt.legend()
        plt.xlabel("lattice site [SC-NC-SC]/[5-5-5]")
        plt.ylabel("current I_x")
        plt.grid()
        plt.show()
        #plot_pairing_amplitude(system, system.F_matrix)

    return current_midle, phase_array

def solve_for_sfs_system(max_num_iter=200, tol=1e-5):
    L_f_array = np.linspace(5, 30, 26, dtype=int)
    current_midle = np.zeros(len(L_f_array))

    # define first utside, so that we can change phase and solve the same system (do not need to start from scratch to solve)
    print("#_________ L_f = ", L_f_array[0], "_________#")
    system = System(phase=np.pi/2, L_y=60, L_z=60, L_sc_0=15, L_nc=0, L_f = L_f_array[0], L_sc=15, L_soc=0, mu_sc=0.9, mu_nc=0.9,mu_soc=0.85, u_sc=-2.2, beta=100)
    solve_system(system, max_num_iter, tol)
    current_arr = system.current_along_lattice()
    current_midle[0] = np.imag(current_arr)[system.L_sc_0 + system.L_f // 2]

    site_x = np.linspace(1, system.L_x - 1, system.L_x - 1)
    plt.plot(site_x, np.imag(current_arr), label="imag")
    plt.legend()
    plt.xlabel("lattice site [SC-NC-SC]/[15-L_f-15]")
    plt.ylabel("current I_x")
    plt.grid()
    plt.show()


    for i in range(1, len(L_f_array)):
        print("#_________ L_f = ",L_f_array[i],"_________#")
        #system.phase = phase_array[i]
        system = System(phase=np.pi/2, L_y=60, L_z=60, L_sc_0=15, L_nc=0, L_f=L_f_array[i],  L_sc=15, L_soc=0, mu_sc=0.9, mu_nc=0.9,mu_soc=0.85, u_sc=-2.2, beta=100)

        solve_system(system, max_num_iter, tol)
        current_arr = system.current_along_lattice()
        current_midle[i] = np.imag(current_arr)[system.L_sc_0 + system.L_f // 2]
        site_x = np.linspace(1, system.L_x - 1, system.L_x - 1)
        plt.plot(site_x, np.imag(current_arr), label="imag")
        plt.legend()
        plt.xlabel("lattice site [SC-NC-SC]/[15-L_f-15]")
        plt.ylabel("current I_x")
        plt.grid()
        plt.show()
    return current_midle, L_f_array

def define_system(beta=np.inf, alpha_R_initial=[0,0,2], L_nc=50, L_soc=2, L_sc=50, L_y=102, L_z=102):
    #print("in pycharm: ", alpha_R_initial)
    system = System(beta=beta, alpha_R_initial=alpha_R_initial, L_nc=L_nc, L_soc=L_soc, L_sc=L_sc, L_y=L_y, L_z=L_z)
    return system

def solve_for_shms_system(max_num_iter=100, tol=1e-5, xz=False, yz=False, alpha_max=np.linspace(0,3,20), theta = 0):
    alpha_array = np.ones((len(alpha_max), 3), dtype=np.float64)
    for i in range(len(alpha_max)):
        alpha_array[i] = alpha_max[i] * alpha_array[i, :]

    if (xz == True):
        alpha_array[:] = alpha_array[:] * np.array([np.sin(theta), 0 * theta, np.cos(theta)])  # xz plane, sin(phi)=0
    if (yz == True):
        alpha_array[:] = alpha_array[:] * np.array([0 * theta, np.sin(theta), np.cos(theta)])  # yz plane, cos(phi)=0
    if ((yz == False) and (xz == False)):
        print("You have to choose orientation of alpha, xz or yz!")
        return

    current_midle = np.zeros(alpha_array.shape[0])


    # define first utside, so that we can change phase and solve the same system (do not need to start from scratch to solve)
    print("#_________ Phase = ", alpha_array[0], "_________#")
    system = System(alpha_R_initial = alpha_array[0], L_y=50, L_z=50, L_sc_0=15, L_nc=0, L_sc=15, L_soc=2, mu_sc=0.9, mu_nc=0.9,
                    mu_soc=0.85, u_sc=-2.2, beta=100)
    solve_system(system, max_num_iter, tol)
    current_arr = system.current_along_lattice()
    current_midle[0] = np.imag(current_arr)[system.L_sc_0 + system.L_nc // 2]

    site_x = np.linspace(0, system.L_x - 1, system.L_x - 1)
    plt.plot(site_x[1:], np.imag(current_arr)[1:], label="imag")
    plt.legend()
    plt.xlabel("lattice site [SC-HM-SC]/[15-2-15]")
    plt.ylabel("current I_x")
    plt.grid()
    plt.show()

    # plot_pairing_amplitude(system, system.F_matrix)


    for i in range(1, alpha_array.shape[0]):
        print("#_________ Phase = ", alpha_array[i], "_________#")
        # system.phase = phase_array[i]
        system = System(alpha_R_intial=alpha_array[i], L_y=50, L_z=50, L_sc_0=15, L_nc=0, L_sc=15, L_soc=2, mu_sc=0.9,
                        mu_nc=0.9, mu_soc=0.85, u_sc=-2.2, beta=100)

        solve_system(system, max_num_iter, tol)
        current_arr = system.current_along_lattice()
        current_midle[i] = np.imag(current_arr)[system.L_sc_0 + system.L_nc // 2]
        site_x = np.linspace(0, system.L_x - 1, system.L_x - 1)
        plt.plot(site_x[1:], np.imag(current_arr)[1:], label="imag")
        plt.legend()
        plt.xlabel("lattice site [SC-HM-SC]/[15-2-15]")
        plt.ylabel("current I_x")
        plt.grid()
        plt.show()
        # plot_pairing_amplitude(system, system.F_matrix)

    return current_midle, alpha_array


#if __name__ == "__main__":
#    pairing_amplitude()