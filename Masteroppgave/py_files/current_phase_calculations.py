import numpy as np
import matplotlib.pyplot as plt

from solve_hamiltonian import solve_system
from system_class import System

"""
This is the file which contains the functions to calcualte the phase-current relations for a S/HM/S junction.
"""

# This function calcualte the current_phase relation for a given alpha-representation. (and a input set of the system dimensions)
def solve_for_shms_system_phase(alpha_array=[0,0,0.5], max_num_iter=800, tol=1e-4, L_y=30, L_z=30, L_sc_0=3, L_nc=0, L_f=0, L_soc=5,L_sc=3, mu_sc=0.9, mu_nc=0.9,mu_soc=0.85, u_sc=-4.2, beta=100):
    phase_array = np.linspace(0, 2*np.pi, 20)
    current_midle = np.zeros(len(phase_array))

    # define first utside, so that we can change phase and solve the same system (do not need to start from scratch to solve)
    print("#_________ Phase = ", phase_array[0], "_________#")
    system = System(alpha_R_initial = alpha_array, phase=phase_array[0], L_y=L_y, L_z=L_z, L_sc_0=L_sc_0, L_nc=L_nc, L_f=L_f, L_sc=L_sc, L_soc=L_soc, mu_sc=mu_sc, mu_nc=mu_nc, mu_soc=mu_soc, u_sc=u_sc, beta=beta)
    solve_system(system, max_num_iter, tol)
    current_arr = system.current_along_lattice()
    current_midle[0] = np.imag(current_arr)[system.L_sc_0 + system.L_nc // 2]

    site_x = np.linspace(0, system.L_x - 1, system.L_x - 1)
    plt.plot(site_x[1:], np.imag(current_arr)[1:], label="imag")
    plt.legend()
    plt.xlabel('lattice site [SC-HM-SC]/[%d-%d-%d], k_y = %d, k_z = %d' %(L_sc_0, L_soc, L_sc, L_y, L_z))
    plt.ylabel("current I_x")
    plt.grid()
    plt.show()

    for i in range(1, len(phase_array)):
        print("#_________ Phase = ",phase_array[i],"_________#")
        system = System(alpha_R_initial = alpha_array, old_solution = True, old_F_matrix_guess=abs(system.F_matrix), phase=phase_array[i], L_y=L_y, L_z=L_z, L_sc_0=L_sc_0, L_nc=L_nc, L_f=L_f, L_sc=L_sc, L_soc=L_soc, mu_sc=mu_sc, mu_nc=mu_nc, mu_soc=mu_soc, u_sc=u_sc, beta=beta)

        solve_system(system, max_num_iter, tol)
        current_arr = system.current_along_lattice()
        current_midle[i] = np.imag(current_arr)[system.L_sc_0 + system.L_nc // 2]
        site_x = np.linspace(0, system.L_x - 1, system.L_x - 1)
        plt.plot(site_x[1:], np.imag(current_arr)[1:], label="imag")
        plt.legend()
        plt.xlabel('lattice site [SC-HM-SC]/[%d-%d-%d], k_y = %d, k_z = %d' %(L_sc_0, L_soc, L_sc, L_y, L_z))
        plt.ylabel("current I_x")
        plt.grid()
        plt.show()

    return current_midle, phase_array


# This function search over the strength of the spin-orbit coupling, given a static vector orientation of alpha.
# For each search, the function will only save the critical current and the belonging phase.
def solve_for_shms_system_strength_individual_phase(max_num_iter=500, tol=1e-4, xz=False, yz=False, alpha_max=np.linspace(0,3,20), theta = 0):
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
    used_phase = np.zeros(alpha_array.shape[0])


    for i in range(0, alpha_array.shape[0]):
        print("#_________ alpha = ", alpha_array[i], "_________#")
        current_array_tmp, phase_array_tmp = solve_for_shms_system_phase(alpha_array=alpha_array[i], max_num_iter=max_num_iter, tol=tol)

        index_max = np.argmax(current_array_tmp)
        current_midle[i] = current_array_tmp[index_max]
        used_phase[i] = phase_array_tmp[index_max]
    return current_midle, alpha_array, used_phase


# This function seach over the relative orientation of the spin-orbit coupling, given a static magnitude of alpha.
# For each search, the function will only save the critical current and the belonging phase.
def solve_for_shms_system_orientation_individual_phase(max_num_iter=500, tol=1e-4, xz=False, yz=False, alpha_max=2):
    theta_array = np.linspace(0, np.pi / 2, 20)
    if (xz == True):
        alpha_array = alpha_max * np.array([np.sin(theta_array[:]), 0 * theta_array[:], np.cos(theta_array[:])])  # xz plane, sin(phi)=0
    if (yz == True):
        alpha_array = alpha_max * np.array([0 * theta_array[:], np.sin(theta_array[:]), np.cos(theta_array[:])])  # yz plane, cos(phi)=0
    if ((yz == False) and (xz == False)):
        print("You have to choose orientation of alpha, xz or yz!")
        return

    current_midle = np.zeros(alpha_array.shape[1])
    used_phase = np.zeros(alpha_array.shape[1])

    for i in range(0, alpha_array.shape[1]):
        print("#_________ alpha = ", alpha_array[:,i], "_________#")
        current_array_tmp, phase_array_tmp = solve_for_shms_system_phase(alpha_array=alpha_array[:,i],
                                                                         max_num_iter=max_num_iter, tol=tol)

        index_max = np.argmax(current_array_tmp)
        current_midle[i] = current_array_tmp[index_max]
        used_phase[i] = phase_array_tmp[index_max]
    return current_midle, theta_array, used_phase