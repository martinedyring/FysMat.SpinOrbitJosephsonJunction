import numpy as np
import matplotlib.pyplot as plt

#from Prosjektoppgave.utilities import idx_F_i, idx_F_i_x_pluss, idx_F_i_x_minus, idx_F_i_y_pluss, idx_F_i_y_minus, num_idx_F_i
from utilities import idx_F_i, idx_F_i_x_pluss, idx_F_i_x_minus, idx_F_i_y_pluss, idx_F_i_y_minus, num_idx_F_i

"""
This script define the class System which contains all necessary information to construct one system.
"""


class System:
    def __init__(self,
                 L_y = 100,
                 L_sc = 100,
                 L_nc = 100,

                 t_sc = 1.0,
                 t_0 = 1.0,
                 t_nc = 1.0,

                 u_sc = -2.5, # V_ij in superconductor
                 u_nc = 0.0,

                 mu_s = -3.5,
                 mu_d = -0.5,
                 mu_pxpy = -1.5,

                 h_sc = 0.0,
                 h_nc = 0.0,

                 beta=np.inf,

                 F_sc_initial_s = 1.0,
                 F_sc_initial_d = 1.0,
                 F_sc_initial_px = 1.0,
                 F_sc_initial_py = 1.0,
                 F_sc_initial_pxpy = 1.0,
                 F_sc_initial_spy = 1.0,

                 F_nc_initial = 0.0
                 ):


        self.L_x = L_nc + L_sc
        self.L_sc = L_sc
        self.L_nc = L_nc

        self.t_sc = t_sc
        self.t_0 = t_0
        self.t_nc = t_nc

        self.u_sc = u_sc
        self.u_nc = u_nc

        self.mu = mu_s

        self.L_y = L_y
        self.h_sc = h_sc
        self.h_nc = h_nc

        self.beta = beta

        self.F_sc_initial_s = F_sc_initial_s
        self.F_sc_initial_d = F_sc_initial_d
        self.F_sc_initial_px = F_sc_initial_px
        self.F_sc_initial_py = F_sc_initial_py
        self.F_sc_initial_pxpy = F_sc_initial_pxpy
        self.F_sc_initial_spy = F_sc_initial_spy
        self.F_nc_initial = F_nc_initial

        #   To define k, I have to take case of even/odd number of lattice sites.
        #   Choose k from 0 to pi, so we only sum over k >= 0 in later calculations
        if L_y % 2 == 0:
            self.k_array = np.linspace(0.0, np.pi, num = 1+ self.L_y//2, endpoint=True, dtype=np.float64)
        else:
            self.k_array = np.linspace(0.0, np.pi, num = 1 + self.L_y//2, endpoint=False, dtype=np.float64)

        self.F_matrix = np.zeros((self.L_x, num_idx_F_i), dtype=np.complex128)  #   2D, one row for each F-comp
        self.U_array = np.zeros(self.L_x, dtype=np.float64)                     #   1D
        self.t_array = np.zeros(self.L_x, dtype=np.float64)                     #   1D
        self.hz_array = np.zeros(self.L_x, dtype=np.float64)                    #   1D

        #   Fill inn values in matrix
        # L_x = L_nc + L_sc
        for i in range(self.L_x):
            if i < L_nc:    #   NC
                self.t_array[i] = t_nc
                self.hz_array[i] = h_nc
                self.F_matrix[i, :] = F_nc_initial           #   Set all F values to inital condition = 1, s-orbital
                self.U_array[i] = u_nc
            else:           #   SC
                self.t_array[i] = t_sc
                self.hz_array[i] = h_sc
                self.F_matrix[i, :] = F_sc_initial_s           # Set all F values to inital condition = 1, s-orbital
                self.U_array[i] = u_sc

    def plot_components_of_hamiltonian(self, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(10,10))

        ax = fig.subplots(nrows=1, ncols=2).flatten()

        #   Delta-term
        line = ax[0].plot(self.U_array, label='U')
        ax[0].plot(np.multiply(self.U_array, np.real(self.F_matrix[:,idx_F_i])), ls=':', label=r'$\Delta$')
        ax[0].plot(np.real(self.F_matrix[:, idx_F_i]), ls='--', label=r'$F_{i}^{\uparrow\downarrow}$')
        ax[0].set_title('Delta')
        ax[0].legend()

    def test_valid(self):
        # dimensions
        assert self.L_x > 0, "L_x must be larger than 0."
        assert self.L_y > 0, "L_x must be larger than 0."

        # U term
        assert self.U_array.shape[0] == self.L_x

        # magnetic field
        assert self.hz_array.shape[0] == self.L_x

        # F
        assert self.F_matrix.shape[0] == self.L_x
        assert self.F_matrix.shape[1] == num_idx_F_i

        # t_ij
        assert self.t_array.shape[0] == self.L_x

    def get_energies(self):
        return self.energies

    def get_phi(self):
        return self.energies
