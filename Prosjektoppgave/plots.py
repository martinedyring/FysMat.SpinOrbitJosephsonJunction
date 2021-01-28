import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/martinedh/Documents/NTNU/GitHub/FysMat/Prosjektoppgave")

#from utilities import label_F_matrix
label_F_matrix = [
    r'$F_{ii}$',
    r'$F_{i}^{x+}$',
    r'$F_{i}^{x-}$',
    r'$F_{i}^{y+}$',
    r'$F_{i}^{y-}$',
    r'$F_{s_i}$'
]

"""
This script contains ass plotting functions
"""

def plot_pairing_amplitude(system, F_matrix):
    """
    F_sc_initial_s = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #s-orbital
    F_sc_initial_d = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    F_sc_initial_px = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    F_sc_initial_py = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    F_sc_initial_pxpy = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0j, 1.0j, -1.0j, -1.0j],
    F_sc_initial_spy = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0+1.0j, 1.0+1.0j, 1.0-1.0j, 1.0-1.0j],

    mu_s = -3.5,
    mu_d = -0.5,
    mu_pxpy = -1.5,
    """

    fig = plt.figure(figsize=(20, 15))
    ax = fig.subplots(ncols=3, nrows=(F_matrix.shape[-1] + 2) // 3, sharex=True, sharey=False).flatten()
    for i in range(F_matrix.shape[-1]):
        ys = F_matrix[:, i]
        # ys[np.abs(ys)< tol] = 0.0 + 0.0j
        plot_complex_function(y=ys, ax=ax[i], labels=['Real part', 'Imaginary part'])
        ax[i].grid()
        ax[i].legend()
        ax[i].set_title(label_F_matrix[i])
        ax[i].set_xlabel("Lattice i")
    fig.suptitle("Correlation function: NC - SOC -  SC")
    fig.subplots_adjust(wspace=0.0)
    plt.show()
    #fig.savefig('correlation function, mu_s=0.9, mu_soc=0.85, u=-4.2.png', bbox_inches='tight')

    fig = plt.figure(figsize=(20, 6))
    fig.subplots_adjust(wspace=0.0)

    system.plot_components_of_hamiltonian(fig)


def plot_complex_function(x=None, y=None, ax=None, labels=None, **kwargs):
    #   Plot the real- and imaginary part of a function individually
    if y is None:
        raise ValueError('Y must not be None.')
    if  x is None:
        x = np.arange(y.shape[0])
    if ax is None:
        fig, ax = plt.subplots()
    if labels is None:
        labels = [None, None]

    lr = ax.plot(x, np.real(y), label=labels[0], **kwargs)
    ax.plot(np.imag(y), ls=':', c=lr[0].get_color(), label=labels[1], **kwargs)

def plot_density_of_states(es, ldos, L_sc_0=50, L_nc=50, L_soc=2, L_sc=50):
    print(es.shape)
    print(ldos.shape)
    #plt.figure(figsize=(20, 10))
    #fig, ax = plt.figure(figsize=(20,15))
    plt.plot(es, np.sum(ldos[:L_sc_0], axis=0) / L_sc_0, label='LDOS in SC')
    plt.xlabel("Energy E")
    plt.legend()
    # plt.savefig('DOS nc, mu_s=0.9, mu_soc=0.85, u=-4.2.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(es, np.sum(ldos[L_sc_0:L_sc_0 + L_nc], axis=0)/L_nc, label='LDOS in NC')
    plt.xlabel("Energy E")
    plt.legend()
    #plt.savefig('DOS nc, mu_s=0.9, mu_soc=0.85, u=-4.2.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(es, np.sum(ldos[L_sc_0+L_nc:L_sc_0+L_nc+L_soc], axis=0)/L_soc, label='LDOS in S0C')
    plt.xlabel("Energy E")
    plt.legend()
    #plt.savefig('DOS soc, mu_s=0.9, mu_soc=0.85, u=-4.2.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(es, np.sum(ldos[L_sc_0+L_soc+L_nc:], axis=0)/L_sc, label='LDOS in SC')
    plt.xlabel("Energy E")
    plt.legend()
    #plt.savefig('DOS sc, mu_s=0.9, mu_soc=0.85, u=-4.2.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(es, np.sum(ldos, axis=0)/(L_sc_0+L_nc+L_soc+L_sc), label='Total DOS')
    #plt.grid()
    plt.xlabel("Energy E")
    plt.legend()
    #plt.savefig('DOS all, mu_s=0.9, mu_soc=0.85, u=-4.2.png', dpi=300, bbox_inches='tight')
    plt.show()
    #ax.set_xlabel("Energy E")
    #ax.set_title("Density of states : NC - SOC - SC")
