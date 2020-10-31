from main import *

def calculate_systems(max_iter=100,
                      beta=30,
                      alpha_R_initial=[0, 0, 2],
                      tol_iter=1e-3):
    """
    Calculate the energy for SC and SOC systems at a beta.
    """
    L_y = 85
    L_z = 85
    # NC
    nc_system = define_system(beta=beta, alpha_R_initial=alpha_R_initial, L_y=L_y, L_z=L_z, L_nc=5, L_sc=0, L_soc=0)
    solve_system(nc_system, max_iter, tol_iter)

    # SOC
    soc_system = define_system(beta=beta, alpha_R_initial=alpha_R_initial, L_y=L_y, L_z=L_z, L_nc=0, L_sc=0, L_soc=5)
    solve_system(soc_system, max_iter, tol_iter)

    # SC
    sc_system = define_system(beta=beta, alpha_R_initial=alpha_R_initial, L_y=L_y, L_z=L_z, L_nc=0, L_sc=5, L_soc=0)
    solve_system(sc_system, max_iter, tol_iter)

    energies_nc_soc_sc = np.array([nc_system.compute_energy(), soc_system.compute_energy(), sc_system.compute_energy()])
    return energies_nc_soc_sc

# Try to compute pd for beta - not working
def pd_search_along_beta(min_beta = 200, max_beta=400, num_beta_step=5, tol=1e-3):
    es = np.zeros(shape=(num_beta_step, 3), dtype=np.float128)
    tps = ['P', 'SOC', 'SC']
    beta_array = np.linspace(min_beta, max_beta, num_beta_step)
    for i in range(num_beta_step):
        print("---- beta = ", beta_array[i], "-----")
        e = calculate_systems(beta=beta_array[i])
        es[i,:] = e[:] # es:soc energy, sc energy
    return es, beta_array

# PD for the absolute value of alpha. 1D
def pd_search_along_alpha_strength(alpha_max, L_nc=50, L_soc=2, L_sc=50, theta=0, xz=False, yz=False):
    alpha = np.ones((len(alpha_max), 3), dtype=np.float64)
    for i in range(len(alpha_max)):
        alpha[i] = alpha_max[i] * alpha[i, :]

    if (xz == True):
        alpha[:] = alpha[:] * np.array([np.sin(theta), 0 * theta, np.cos(theta)])  # xz plane, sin(phi)=0
    if (yz == True):
        alpha[:] = alpha[:] * np.array([0 * theta, np.sin(theta), np.cos(theta)])  # yz plane, cos(phi)=0
    if ((yz == False) and (xz == False)):
        print("You have to choose orientation of alpha, xz or yz!")
        return

    free_energy = np.zeros(shape=(alpha.shape[0], 3), dtype=np.float128)
    tps = ['NC', 'SOC', 'SC']

    for i in range(alpha.shape[0]):
        print("---- alpha = ", alpha[i, :], "-----")
        e = calculate_systems(alpha_R_initial=alpha[i, :])
        free_energy[i, :] = e[:]  # es:soc energy, sc energy
    return free_energy, alpha_max


# PD for the ortentation og alpha. 1D
def pd_search_along_alpha_angle(alpha_max=2, L_nc=7, L_soc=3, L_sc=7, xz=False, yz=False):
    theta = np.linspace(0, np.pi / 2, 30)
    if (xz == True):
        alpha = alpha_max * np.array([np.sin(theta[:]), 0 * theta[:], np.cos(theta[:])])  # xz plane, sin(phi)=0
    if (yz == True):
        alpha = alpha_max * np.array([0 * theta[:], np.sin(theta[:]), np.cos(theta[:])])  # yz plane, cos(phi)=0
    if ((yz == False) and (xz == False)):
        print("You have to choose orientation of alpha, xz or yz!")
        return

    free_energy = np.zeros(shape=(alpha.shape[1], 3), dtype=np.float128)
    tps = ['NC', 'SOC']

    for i in range(alpha.shape[1]):
        print("---- alpha = ", alpha[:, i], "-----")
        e = calculate_systems(alpha_R_initial=alpha[:, i])
        free_energy[i, :] = e[:]  # es:soc energy, sc energy
    return free_energy, theta


# Free energy for one system
def calculate_energy_of_system(max_iter=100,
                               beta=33.3,
                               alpha_R_initial=[0, 0, 2],
                               tol_iter=1e-3,
                               L_nc=50, L_soc=2, L_sc=50):
    """
    Calculate the energy for nc - soc - sc system at a given alpha.
    """
    L_y = 85  # L_nc+L_soc+L_sc
    L_z = 85  # L_nc+L_soc+L_sc
    # nc - soc - sc
    nc_soc_sc_system = define_system(beta=beta, alpha_R_initial=alpha_R_initial, L_y=L_y, L_z=L_z, L_nc=L_nc, L_sc=L_sc,
                                     L_soc=L_soc)
    solve_system(nc_soc_sc_system, max_iter, tol_iter)

    energies_nc_soc_sc = np.array(
        [nc_soc_sc_system.compute_energy(N=False)])  # N=True gives only free energy in NC layer
    return energies_nc_soc_sc

# used to compute for HM-NC system. Reproduce free energy plot in Linas thisis
def search_along_alpha(alpha_max=2, L_nc=50, L_soc=2, L_sc=50, xz=False, yz=False):
    # phi = np.linspace(0,np.pi,30)
    theta = np.linspace(0, np.pi / 2, 20)
    # alpha = alpha_max*np.array([np.cos(phi[:]), np.sin(phi[:]), 0*phi[:]]) # xy plane, Theta=0
    if (xz == True):
        alpha = alpha_max * np.array([np.sin(theta[:]), 0 * theta[:], np.cos(theta[:])])  # xz plane, sin(phi)=0
    if (yz == True):
        alpha = alpha_max * np.array([0 * theta[:], np.sin(theta[:]), np.cos(theta[:])])  # yz plane, cos(phi)=0
    if ((yz == False) and (xz == False)):
        print("You have to choose orientation of alpha, xz or yz!")
        return

    free_energy = np.zeros(shape=(alpha.shape[1], 1), dtype=np.float128)
    tps = ['NC', 'SOC']

    for i in range(alpha.shape[1]):
        print("---- alpha = ", alpha[:, i], "-----")
        e = calculate_energy_of_system(alpha_R_initial=alpha[:, i], L_nc=L_nc, L_soc=L_soc, L_sc=L_sc)
        free_energy[i, :] = e[:]
    return free_energy, theta

# 2D PD for alpha angle and strength
def calculate_systems_2d(max_iter=10,
                         beta=30,
                         alpha_R_initial=[0, 0, 2],
                         tol_iter=1e-3):
    """
    Calculate the energy for SC and SOC systems at a beta.
    """
    L_y = 20
    L_z = 20
    # NC
    nc_system = define_system(beta=beta, alpha_R_initial=alpha_R_initial, L_y=L_y, L_z=L_z, L_nc=5, L_sc=0, L_soc=0)
    solve_system(nc_system, max_iter, tol_iter)

    # SOC
    soc_system = define_system(beta=beta, alpha_R_initial=alpha_R_initial, L_y=L_y, L_z=L_z, L_nc=0, L_sc=0, L_soc=5)
    solve_system(soc_system, max_iter, tol_iter)

    # SC
    sc_system = define_system(beta=beta, alpha_R_initial=alpha_R_initial, L_y=L_y, L_z=L_z, L_nc=0, L_sc=5, L_soc=0)
    solve_system(sc_system, max_iter, tol_iter)

    energies_nc_soc_sc = np.array([nc_system.compute_energy(), soc_system.compute_energy(), sc_system.compute_energy()])
    return energies_nc_soc_sc


def pd_search_along_alpha_2d(alpha_max, xz=False, yz=False):
    theta = np.linspace(0, np.pi / 2, 15)

    free_energies_2d = np.zeros(((len(theta), len(alpha_max), 3)))
    for j in range(len(theta)):
        # for each theta value, we do a PD over alpha-strength
        alpha = np.ones((len(alpha_max), 3), dtype=np.float64)
        for i in range(len(alpha_max)):
            alpha[i] = alpha_max[i] * alpha[i, :]

        if (xz == True):
            alpha = alpha[:] * np.array([np.sin(theta[j]), 0 * theta[j], np.cos(theta[j])])  # xz plane, sin(phi)=0
        if (yz == True):
            alpha = alpha[:] * np.array([0 * theta[j], np.sin(theta[j]), np.cos(theta[j])])  # yz plane, cos(phi)=0
        if ((yz == False) and (xz == False)):
            print("You have to choose orientation of alpha, xz or yz!")
            return
        # print(alpha)
        free_energy = np.zeros(shape=(alpha.shape[0], 3), dtype=np.float128)
        tps = ['SC', 'NC', 'SOC']

        for i in range(alpha.shape[0]):
            print("---- alpha = ", alpha[i, :], "-----")
            e = calculate_systems_2d(alpha_R_initial=alpha[i, :])
            free_energy[i, :] = e[:]  # es:soc energy, sc energy

        # add new PD for theis theta value to the total PD arrray
        free_energies_2d[j, :, :] = free_energy
    return free_energies_2d, theta, alpha_max




