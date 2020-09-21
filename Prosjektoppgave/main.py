import numpy as np
import matplotlib.pyplot as plt

#from Prosjektoppgave.solve_hamiltonian import solve_system
#from Prosjektoppgave.system_class import System
#from Prosjektoppgave.plots import plot_complex_function

from solve_hamiltonian import solve_system
from system_class import System
from plots import plot_complex_function
from utilities import label_F_matrix

"""
This is the main function to run all programs
"""

def pairing_amplitude():
    s = System()
    F_matrix = np.asarray(solve_system(s, num_iter=1))

    tol = 1e-13
    fig = plt.figure(figsize=(20, 20))
    ax = fig.subplots(ncols=3, nrows=(F_matrix.shape[2] + 2) // 3, sharex=True, sharey=False).flatten()
    for i in range(F_matrix.shape[2]):
        ys = F_matrix[-1, :, i]
        # ys[np.abs(ys)< tol] = 0.0 + 0.0j
        plot_complex_function(y=ys, ax=ax[i], labels=['Real part', 'Imaginary part'])
        ax[i].grid()
        ax[i].legend()
        ax[i].set_title(label_F_matrix[i])
    fig.subplots_adjust(wspace=0.0)

    fig = plt.figure(figsize=(20, 6))
    fig.subplots_adjust(wspace=0.0)
    s.plot_components_of_hamiltonian(fig)

    """
    def __init__(self,
                 L_y = 100,
                 L_sc = 100,
                 L_nc = 100,

                 t_sc = 1.0,
                 t_0 = 1.0,
                 t_nc = 1.0,

                 u_sc = 0.0,
                 u_nc = 0.0,

                 mu_s = -3.5,
                 mu_d = -0.5,
                 mu_pxpy = -1.5,

                 h_sc = 0.0,
                 h_nc = 0.0,

                 beta=np.inf,

                 F_sc_initial = 1.0,
                 F_nc_initial = 0.0,
                 self_consistent_h = 0
                 ):
    """


if __name__ == "__main__":
    pairing_amplitude()