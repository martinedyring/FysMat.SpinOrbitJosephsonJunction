import numpy as np
import matplotlib.pyplot as plt

from utilities import label_F_matrix

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

    fig = plt.figure(figsize=(20, 30))
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

def plot_density_of_states(es, ldos):
    #fig = plt.figure(figsize=(18,6))
    plt.plot(es, np.sum(ldos[0:50], axis=0), label='LDOS in NC')
    plt.legend()
    plt.show()
    plt.plot(es, np.sum(ldos[50:52], axis=0), label='LDOS in S0C')
    plt.legend()
    plt.show()
    plt.plot(es, np.sum(ldos[52:], axis=0), label='LDOS in SC')
    plt.legend()
    plt.show()
    plt.plot(es, np.sum(ldos, axis=0), label='Total DOS')
    plt.grid()
    plt.legend()
    plt.xlabel("Energy E")
    plt.title("Density of states : NC - SOC - SC")