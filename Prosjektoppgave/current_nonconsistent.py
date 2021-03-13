from nonconsistent_solution import *
from system_class import System
import numpy as np
import pickle

def current_for_phase_array_nonconsistent(L_y,
                                          L_z,
                                          L_sc_0,
                                          L_soc,
                                          L_sc,
                                          L_nc,
                                          mu_sc,
                                          mu_soc,
                                          t,
                                          u_sc,
                                          beta,
                                          alpha_array):

    phase_array = np.linspace(0,2*np.pi, 15)
    #current_midle = np.zeros(alpha_array.shape[0])
    current_midle = np.zeros(phase_array.shape[0])

    """
    # define first utside, so that we can change phase and solve the same system (do not need to start from scratch to solve)
    print("#_________ Phase = ", phase_array[0], "_________#")
    system = System(alpha_R_initial = alpha_array,
                    phase=phase_array[0],
                    L_y=L_y,
                    L_z=L_z,
                    L_sc_0=L_sc_0,
                    L_nc=L_nc,
                    L_sc=L_sc,
                    L_soc=L_soc,
                    mu_sc=mu_sc,
                    mu_soc=mu_soc,
                    t=t,
                    u_sc=u_sc,
                    beta=beta)

    fmatrix, evals, evecs, ham = solve_system_nonconsistent(junction=True,
                                                                                                       L_x=system.L_x,
                                                                                                       L_y=system.L_y,
                                                                                                       L_z=system.L_z,
                                                                                                       L_sc=system.L_sc,
                                                                                                       L_soc=system.L_soc,
                                                                                                       mu_array=system.mu_array,
                                                                                                       h_array=system.h_array,
                                                                                                       U_array=system.U_array,
                                                                                                       F_matrix=system.F_matrix,
                                                                                                       t_x_array=system.t_x_array,
                                                                                                       ky_array=system.ky_array,
                                                                                                       kz_array=system.kz_array,
                                                                                                       alpha_R_x_array=system.alpha_R_x_array,
                                                                                                       alpha_R_y_array=system.alpha_R_y_array,
                                                                                                       beta=system.beta,
                                                                                                       phase=system.phase,
                                                                                                       eigenvalues=system.eigenvalues,
                                                                                                       eigenvectors=system.eigenvectors,
                                                                                                       hamiltonian=system.hamiltonian)

    system.F_matrix, system.eigenvalues, system.eigenvectors, system.hamiltonian = fmatrix, evals, evecs, ham
    current_arr = system.current_along_lattice()
    current_midle[0] = np.real(current_arr)[system.L_sc_0 + system.L_soc // 2]

    site_x = np.linspace(0, system.L_x - 1, system.L_x - 1)
    plt.plot(site_x[1:], np.real(current_arr)[1:], label="real")
    plt.legend()
    plt.xlabel("lattice site [SC-HM-SC]/[25-5-25]")
    plt.ylabel("current I_x")
    plt.grid()
    plt.show()

    # plot_pairing_amplitude(system, system.F_matrix)

    # delete old system class object
    del system.L_sc_0, system.L_sc, system.L_soc, system.L_nc, system.L_f, system.L_x, system.L_y, system.L_z
    del system.t, system.h, system.u_sc, system.u_soc, system.u_nc, system.u_f, system.mu_sc, system.mu_soc, system.mu_nc
    del system.mu_array, system.alpha_R_initial, system.beta, system.debye_freq, system.phase
    del system.F_sc_0_initial, system.F_soc_initial, system.F_nc_initial, system.F_f_initial, system.F_sc_initial
    del system.phase_array, system.orbital_indicator, system.ky_array, system.kz_array
    del system.F_matrix, system.U_array, system.t_x_array, system.t_y_array, system.h_array, system.alpha_R_x_array, system.alpha_R_y_array
    del system.eigenvectors, system.eigenvalues, system.hamiltonian
    del system, fmatrix, evals, evecs, ham
    """

    for i in range(0, phase_array.shape[0]):
        print("#_________ Phase = ", phase_array[i], "_________#")
        # system.phase = phase_array[i]

        system = System(alpha_R_initial = alpha_array,
                        phase=phase_array[i],
                        L_y=L_y,
                        L_z=L_z,
                        L_sc_0=L_sc_0,
                        L_nc=L_nc,
                        L_sc=L_sc,
                        L_soc=L_soc,
                        mu_sc=mu_sc,
                        mu_soc=mu_soc,
                        t=t,
                        u_sc=u_sc,
                        beta=beta)

        fmatrix, evals, evecs, ham = solve_system_nonconsistent(junction=True,
                                                                                                       L_x=system.L_x,
                                                                                                       L_y=system.L_y,
                                                                                                       L_z=system.L_z,
                                                                                                       L_sc=system.L_sc,
                                                                                                       L_soc=system.L_soc,
                                                                                                       mu_array=system.mu_array,
                                                                                                       h_array=system.h_array,
                                                                                                       U_array=system.U_array,
                                                                                                       F_matrix=system.F_matrix,
                                                                                                       t_x_array=system.t_x_array,
                                                                                                       ky_array=system.ky_array,
                                                                                                       kz_array=system.kz_array,
                                                                                                       alpha_R_x_array=system.alpha_R_x_array,
                                                                                                       alpha_R_y_array=system.alpha_R_y_array,
                                                                                                       beta=system.beta,
                                                                                                       phase=system.phase,
                                                                                                       eigenvalues=system.eigenvalues,
                                                                                                       eigenvectors=system.eigenvectors,
                                                                                                       hamiltonian=system.hamiltonian)
        system.F_matrix, system.eigenvalues, system.eigenvectors, system.hamiltonian = fmatrix, evals, evecs, ham
        current_arr = system.current_along_lattice()
        current_midle[i] = np.real(current_arr)[system.L_sc_0 + system.L_soc // 2]

        """
        site_x = np.linspace(0, system.L_x - 1, system.L_x - 1)
        plt.plot(site_x[1:], np.real(current_arr)[1:], label="real")
        plt.legend()
        plt.xlabel("lattice site [SC-HM-SC]/[25-5-25]")
        plt.ylabel("current I_x")
        plt.grid()
        plt.show()
        """
        # plot_pairing_amplitude(system, system.F_matrix)

        # delete old system class object


        del system.L_sc_0, system.L_sc, system.L_soc, system.L_nc, system.L_f, system.L_x, system.L_y, system.L_z
        del system.t, system.h, system.u_sc, system.u_soc, system.u_nc, system.u_f, system.mu_sc, system.mu_soc, system.mu_nc
        del system.mu_array, system.alpha_R_initial, system.beta, system.debye_freq, system.phase
        del system.F_sc_0_initial, system.F_soc_initial, system.F_nc_initial, system.F_f_initial, system.F_sc_initial
        del system.phase_array, system.orbital_indicator, system.ky_array, system.kz_array
        del system.F_matrix, system.U_array, system.t_x_array, system.t_y_array, system.h_array, system.alpha_R_x_array, system.alpha_R_y_array
        del system.eigenvectors, system.eigenvalues, system.hamiltonian
        del system, fmatrix, evals, evecs, ham

    return current_midle, phase_array

def calculate_current_for_phase_of_all_alpha_orientations(phase_array=np.linspace(0,2*np.pi,15),
                                                          alpha_max=np.linspace(0,1,10),
                                                          theta=0,
                                                          xz=False,
                                                          yz=False,
                                                          ):

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

    current_all = np.zeros(shape=(len(phase_array),len(alpha_max)))

    for i in range(len(alpha_array)):
        print('-----------------------Alpha value #', i+1, ' of ', len(alpha_array), '-----------------------')
        current_all[:,i], phase_array = current_for_phase_array_nonconsistent(L_y=30,
                                                                      L_z=30,
                                                                      L_sc_0=25,
                                                                      L_soc=5,
                                                                      L_sc=25,
                                                                      L_nc=0,
                                                                      mu_sc=0.9,
                                                                      mu_soc=0.9,
                                                                      t=1,
                                                                      u_sc=-2,
                                                                      beta=33.3,
                                                                      alpha_array=alpha_array[i,:])

    return current_all, alpha_array, phase_array

def calculate_current_for_phase_of_all_alpha_given_theta_and_phi(phase_array=np.linspace(0, 2 * np.pi, 15),
                                                                  alpha_max=np.linspace(0, 1, 10),
                                                                  theta=0,
                                                                  phi=0
                                                                  ):

    alpha_array = np.ones((len(alpha_max), 3), dtype=np.float64)
    for i in range(len(alpha_max)):
        alpha_array[i] = alpha_max[i] * alpha_array[i, :]

    alpha_array[:] = alpha_array[:] * np.array([np.cos(phi)*np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])


    current_all = np.zeros(shape=(len(phase_array), len(alpha_max)))

    for i in range(len(alpha_array)):
        print('-----Alpha value #', i + 1, ' of ', len(alpha_array), ' ----- theta=',theta,' phi=',phi,' -----------------------')
        current_all[:, i], phase_array = current_for_phase_array_nonconsistent(L_y=30,
                                                                               L_z=30,
                                                                               L_sc_0=25,
                                                                               L_soc=5,
                                                                               L_sc=25,
                                                                               L_nc=0,
                                                                               mu_sc=0.0,
                                                                               mu_soc=0.0,
                                                                               t=1,
                                                                               u_sc=-1,
                                                                               beta=33.3,
                                                                               alpha_array=alpha_array[i, :])

    return current_all, alpha_array, phase_array

#curr, al, ph = calculate_current_for_phase_of_all_alpha_orientations(xz=True)

#with open('test_current_idun.pkl', 'wb') as output:
#    pickle.dump([curr, al, ph], output, pickle.HIGHEST_PROTOCOL)
#np.savez('test_current_idun.npz', curr, al, ph)