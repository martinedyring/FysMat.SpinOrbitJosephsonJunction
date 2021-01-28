import numpy as np
import matplotlib.pyplot as plt


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

    plt.show()

def plot_pairing_amplitude(system, F_matrix):
    plt.figure(figsize=(20, 15))
    ys = F_matrix[:]
    print(F_matrix)
    # ys[np.abs(ys)< tol] = 0.0 + 0.0j
    plot_complex_function(y=ys, ax=None, labels=['Real part', 'Imaginary part'])
    plt.grid()
    plt.legend()
    plt.xlabel("Lattice i")
    #fig.subplots_adjust(wspace=0.0)
    plt.show()
    #fig.savefig('correlation function, mu_s=0.9, mu_soc=0.85, u=-4.2.png', bbox_inches='tight')

    #fig = plt.figure(figsize=(20, 6))
    #fig.subplots_adjust(wspace=0.0)

    #system.plot_components_of_hamiltonian()