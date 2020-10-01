import numpy as np
import matplotlib.pyplot as plt

from utilities import idx_F_i, idx_F_ij_x_pluss, idx_F_ji_x_pluss,idx_F_ij_x_minus, idx_F_ji_x_minus, idx_F_ij_y_pluss, idx_F_ji_y_pluss, idx_F_ij_y_minus, idx_F_ji_y_minus, idx_F_ij_s, num_idx_F_i
from numpy import conj, tanh, exp, sqrt, cos, sin#, sqrt #as conj, tanh, exp, cos, sin, sqrt

"""
This script define the class System which contains all necessary information to construct one system.
"""

#(beta=np.inf, u_sc=-1.0, F_init=0.3, alpha_R_z=2, NY=50, N_SOC=5, N_F=100, N_SC=100, h_f_z=0.3, h_f_x=0.0)

class System:
    def __init__(self,
                 L_y = 102,
                 L_sc = 50,
                 L_nc = 50,
                 L_soc = 2,

                 t_x = 1.0,
                 t_y = 1.0,

                 t_sc = 1.0,
                 t_0 = 1.0,
                 t_nc = 1.0,

                 u_sc = 1.0,#-4.2, # V_ij in superconductor
                 u_nc = 1.0,#-4.2,
                 u_soc = 1.0,#-4.2,

                 mu_s = -3.5, #s
                 mu_d = -0.5,
                 mu_pxpy = -1.5,
                 mu_nc = -3.5,
                 mu_sc = -3.5,
                 mu_soc = -3.5,
                 alpha_r = np.array([0.1, 0.0, 0.0], dtype=np.float64),
                 t = 0.5,
                 U = -4.2,
                 #beta = 200,
                 wd = 0.6,
                 F = 0.3,

                 alpha_R_x = 0.0,
                 alpha_R_y = 0.0,
                 alpha_R_z = 0.0, # 1

                 beta = np.inf,

                 #F_sc_initial = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 F_sc_initial = np.pad(np.array([1.0], dtype=np.complex128),  (0,num_idx_F_i-1), mode='constant', constant_values=0.0),

                 F_sc_initial_s = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #s-orbital
                 F_sc_initial_d = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                 F_sc_initial_px = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                 F_sc_initial_py = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                 F_sc_initial_pxpy = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0j, 1.0j, -1.0j, -1.0j],
                 F_sc_initial_spy = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0+1.0j, 1.0+1.0j, 1.0-1.0j, 1.0-1.0j],

                 F_nc_initial = np.pad(np.array([1.0], dtype=np.complex128),  (0,num_idx_F_i-1), mode='constant', constant_values=0.0),#0.3,
                 F_soc_initial = np.pad(np.array([1.0], dtype=np.complex128),  (0,num_idx_F_i-1), mode='constant', constant_values=0.0),#0.3,

                 orbital_indicator = "s"
                 ):

        self.L_y = L_y
        self.L_sc = L_sc
        self.L_soc = L_soc
        self.L_nc = L_nc
        self.L_x = self.L_nc + self.L_soc + self.L_sc

        self.t_x = t_x
        self.t_y = t_y
        self.t_sc = t_sc
        self.t_0 = t_0
        self.t_nc = t_nc

        self.u_sc = u_sc
        self.u_soc = u_soc
        self.u_nc = u_nc

        self.mu_array = np.zeros(shape=(self.L_x,), dtype=np.float64)

        self.alpha_R_x = alpha_R_x
        self.alpha_R_y = alpha_R_y
        self.alpha_R_z = alpha_R_z

        self.beta = beta

        self.debye_freq = wd

        self.F_sc_initial_orbital = F_sc_initial_s
        self.F_nc_initial = F_nc_initial

        self.orbital_indicator = str(orbital_indicator)

        #   To define k, I have to take case of even/odd number of lattice sites.
        #   Choose k from 0 to pi, so we only sum over k >= 0 in later calculations
        if L_y % 2 == 0:
            self.k_array = np.linspace(0.0, np.pi, num = (self.L_y + 2)//2, endpoint=True, dtype=np.float64)
        else:
            self.k_array = np.linspace(0.0, np.pi, num = (self.L_y + 2)//2, endpoint=False, dtype=np.float64)

        # F_matrix: F_ii, F_orbital, F_x+, F_x-, F_y+, F_y-
        self.F_matrix = np.zeros((self.L_x, num_idx_F_i), dtype=np.complex128)  #   2D, one row for each F-comp
        self.U_array = np.zeros(self.L_x, dtype=np.float64)                     #   1D
        self.t_x = np.zeros((self.L_x - 1), dtype=np.float64)
        self.t_y = np.zeros((self.L_x), dtype=np.float64)

        self.alpha_r_x = np.zeros((self.L_x, 3), dtype=np.float64)
        self.alpha_r_y = np.zeros((self.L_x, 3), dtype=np.float64)
        self.alpha_R_array = np.zeros((self.L_x, 3), dtype=np.float64)


        #   Eigenvectors
        self.eigenvectors = np.zeros(shape=(4 * self.L_x, 4 * self.L_x, (self.L_y + 2) // 2), dtype=np.complex128)

        #   Eigenvalues
        self.eigenvalues = np.zeros(shape=(4 * self.L_x, (self.L_y + 2) // 2), dtype=np.float64)

        #   Hamiltonian
        self.hamiltonian = np.zeros(shape=(self.L_x * 4, self.L_x * 4), dtype=np.complex128)

        #   Fill inn values in matrix
        # L_x = L_nc + L_soc + L_sc
        for i in range(self.L_x):
            self.t_y[i] = t_y

            if i < L_nc:    #   NC
                self.F_matrix[i, :] = F_nc_initial                   #   Set all F values to inital condition for NC material

                self.U_array[i] = u_nc
                self.mu_array[i] = mu_nc

            elif i < L_nc + L_soc :  # SOC
                self.F_matrix[i, :] = F_soc_initial
                self.U_array[i] = u_soc
                self.mu_array[i] = mu_soc

                self.alpha_R_array[i, 0] = self.alpha_R_x
                self.alpha_R_array[i, 1] = self.alpha_R_y
                self.alpha_R_array[i, 2] = self.alpha_R_z

            else:           #   SC
                self.F_matrix[i, :] = F_sc_initial         # Set all F values to inital condition for SC material (+1 s-orbital)
                self.U_array[i] = u_sc
                self.mu_array[i] = mu_sc
        """
        stp = L_start_soc + L_soc
        for i in range(L_start_soc, stp):
            self.alpha_r_y[i, 0] = alpha[0]
            self.alpha_r_y[i, 1] = alpha[1]
            self.alpha_r_y[i, 2] = alpha[2]

            if i != stp - 1:  # only add x-coupling of both layers are in SOC
                self.alpha_r_x[i, 0] = alpha[0]
                self.alpha_r_x[i, 1] = alpha[1]
                self.alpha_r_x[i, 2] = alpha[2]
        """
        # Some parameters only rely on neighbors in x-direction, and thus has only NX-1 links
        for i in range(self.L_x - 1):
            self.t_x[i] = t_x

    def epsilon_ijk(self, i, j, k, spin):  # spin can take two values: 1 = up, 2 = down
        #t_0 = 0.0
        #t_1 = 0.0
        e = 0.0
        #print(i, j)
        if i == j:
            #t_1 = self.t_array[i]
            e = np.complex128(- 2 * self.t_y[i] * cos(k) - self.mu_array[i]) # spini in (1, 2) => (0, 1) index => (spinup, spindown)
        elif i == j + 1:
            #t_0 = self.t_array[j]
            e = np.complex128(self.t_x[j])
        elif i == j - 1:
            #t_0 = self.t_array[i]
            e = np.complex128(self.t_x[i])

        #e = np.complex128(-(t_0 + 2 * t_1 * np.cos(k)) - h - self.mu)
        #e = np.complex128(-2 * t_1 * np.cos(k) - t_0)
        return e


    def set_epsilon(self, arr, i, j, k):
        arr[0][0] = self.epsilon_ijk(i, j, k, 1)
        arr[1][1] = self.epsilon_ijk(i, j, k, 2)
        arr[2][2] = -self.epsilon_ijk(i, j, k, 1)
        arr[3][3] = -self.epsilon_ijk(i, j, k, 2)
        return arr

    def delta_gap(self, i):
        return -self.U_array[i] * self.F_matrix[i, idx_F_i]


    def set_delta(self, arr, i, j):
        # Comment out +=, and remove the other comp from update_hamil to increase runtime and check if there is any diff. Shouldnt be diff in output.
        if i==j:
            #print("arr", arr)
            arr[0][3] += self.delta_gap(i)/2
            arr[1][2] += -self.delta_gap(i)/2
            arr[2][1] += -conj(self.delta_gap(i))/2
            arr[3][0] += conj(self.delta_gap(i))/2

        """"
        def set_delta(self, i, j):
            if i==j:
                self.hamiltonian[4 * i + 3][4 * j + 0] += self.delta_gap(i) / 2
                self.hamiltonian[4 * i + 2][4 * j + 1] += self.delta_gap(i) / 2
                self.hamiltonian[4 * i + 1][4 * j + 2] += self.delta_gap(i) / 2
                self.hamiltonian[4 * i + 0][4 * j + 3] += self.delta_gap(i) / 2
        """
        return arr

    def set_rashba(self, arr, i, j, k):
        I = 1.0j
        sinka = sin(k)

        # barr = arr[2:][2:]
        if i == j:
            # (n_z*sigma_x - n_x*sigma_z)
            s00 = -self.alpha_r_y[i, 0]
            s01 = self.alpha_r_y[i, 2]
            s10 = self.alpha_r_y[i, 2]
            s11 = self.alpha_r_y[i, 0]

            # Upper left
            arr[0][0] += 2 * sinka * s00
            arr[0][1] += 2 * sinka * s01
            arr[1][0] += 2 * sinka * s10
            arr[1][1] += 2 * sinka * s11

            # Bottom right. Minus and conjugate
            arr[2][2] += 2 * sinka * s00
            arr[2][3] += 2 * sinka * s01
            arr[3][2] += 2 * sinka * s10
            arr[3][3] += 2 * sinka * s11

        # Backward jump X-
        elif j == i + 1 or j == i - 1:
            if j == i + 1:  # Backward jump X-
                l = i
                coeff = -1.0
            else:  # Forward jump X+
                l = j
                coeff = 1.0
            s00 = self.alpha_r_x[l, 1]
            s11 = -self.alpha_r_x[l, 1]
            s01 = I * self.alpha_r_x[l, 2]
            s10 = -I * self.alpha_r_x[l, 2]

            arr[0][0] += coeff * I * s00
            arr[0][1] += coeff * I * s01
            arr[1][0] += coeff * I * s10
            arr[1][1] += coeff * I * s11

            arr[2][2] += conj(coeff * I * s00)
            arr[2][3] += conj(coeff * I * s01)
            arr[3][2] += conj(coeff * I * s10)
            arr[3][3] += conj(coeff * I * s11)

        return arr

    def zero_init_hamiltonian(self):
        #for i in range(matrix.shape[0]):
        #    for j in range(matrix.shape[1]):
        #        matrix[i, j] = 0.0 + 0.0j
        self.hamiltonian[:, :] = 0.0 + 0.0j
        return self

    def set_hamiltonian(self):

        """
        self.zero_init(self.hamiltonian)
        for i in range(self.L_x):
            for j in range(self.L_x):
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_epsilon(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, k)
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_delta(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j)
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_rashba(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, k)
        """
        self.zero_init_hamiltonian()
        for i in range(self.L_x):
            for j in range(self.L_x):
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_delta(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j)
        return self

    def update_hamiltonian(self, k):
        #self.calc_new_mu()
        for i in range(self.L_x):
            for j in range(self.L_x):
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_epsilon(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, k)
                #self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_rashba(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, k)
        return self

    def long_calculate_F_matrix(self):
        #   Initialize the old F_matrix to 0+0j, so that we can start to add new values
        for a in range(num_idx_F_i):
            for b in range(self.F_matrix.shape[0]):
                self.F_matrix[b, a] = 0.0 + 0.0j
        #self.F_matrix[:, :] = 0.0 + 0.0j

        #print("dim F: ", self.F_matrix.shape)
        #print("dim eigenvalues: ", self.eigenvalues.shape)
        #print("dim eigenvectors: ", self.eigenvectors.shape)



        # Calculation loops
        #for k in range(self.eigenvalues.shape[1]):
        s_k = 1.0
        #    for n in range(self.eigenvalues.shape[0]):
        #        if abs(self.eigenvalues[n, k]) >= np.inf:
        #            print("debye")
        #            continue
        #        if (self.k_array[k] == 0) or (self.k_array[k] == np.pi):
        #            #print("k")
        #            s_k = 0.0
        #        coeff = 1 / (2* self.L_y) * tanh(self.beta*self.eigenvalues[n, k])

        # Not done, fix so that all k values are summarized under each i value

        #debye freq, stemp function. = if energy>debye freq
        idx1, idx2 = np.where(abs(self.eigenvalues) >= self.debye_freq)
        step_func = np.ones(shape=(4 * self.L_x, (self.L_y + 2) // 2), dtype=int)
        for i in range(len(idx1)):
            step_func[idx1[i]][idx2[i]] = 0

        for i in range(self.F_matrix.shape[0]-1):
            # F_ii - same point
            self.F_matrix[i, idx_F_i] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4 * i, :, 0] * conj(self.eigenvectors[(4 * i) + 3, :, 0])))# - s_k * self.eigenvectors[(4 * i) + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0])))
            self.F_matrix[i, idx_F_i] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[(4 * i) + 3, :, -1])))# - s_k * self.eigenvectors[(4 * i) + 1, :,-1] * conj(self.eigenvectors[(4 * i) + 2, :, -1])))
            self.F_matrix[i, idx_F_i] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 3, :, 0:-2]) - s_k * self.eigenvectors[(4 * i) + 1, :,0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2])))

            # F ij X+ S, i_x, j_x
            self.F_matrix[i, idx_F_ij_x_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*i, :, 0] * conj(self.eigenvectors[(4*(i+1)) + 3, :, 0])))#  -  s_k * self.eigenvectors[(4*(i+1)) + 1, :, 0] * conj(self.eigenvectors[(4*i) + 2, :, 0])))
            self.F_matrix[i, idx_F_ij_x_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[(4*(i+1)) + 3, :, -1])))# - s_k * self.eigenvectors[(4*(i+1)) + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1])))
            self.F_matrix[i, idx_F_ij_x_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[(4*(i+1)) + 3, :, 0:-2]) - s_k * self.eigenvectors[(4*(i+1)) + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2])))

            # F ji X+ S, i_x, j_x
            self.F_matrix[i, idx_F_ji_x_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*(i+1), :, 0] * conj(self.eigenvectors[4 *i + 3, :, 0])))# - s_k * self.eigenvectors[4*i + 1, :, 0] * conj(self.eigenvectors[4*(i+1) + 2, :, 0])))
            self.F_matrix[i, idx_F_ji_x_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * (i + 1), :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[4 * (i + 1) + 2, :, -1])))
            self.F_matrix[i, idx_F_ji_x_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * (i + 1), :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[4 * (i + 1) + 2, :, 0:-2])))

            # F ij X- S, i_x, j_x
            self.F_matrix[i, idx_F_ij_x_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*i, :, 0] * conj(self.eigenvectors[4*(i+1) + 3, :, 0])))#  -  s_k * self.eigenvectors[4*(i+1) + 1, :, 0] * conj(self.eigenvectors[(4*i) + 2, :, 0])))
            self.F_matrix[i, idx_F_ij_x_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * (i + 1) + 3, :, -1])))# - s_k * self.eigenvectors[4 * (i + 1) + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1])))
            self.F_matrix[i, idx_F_ij_x_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * (i + 1) + 3, :, 0:-2]) - s_k * self.eigenvectors[4 * (i + 1) + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2])))

            # F ji X- S, i_x, j_x
            self.F_matrix[i, idx_F_ji_x_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4 * (i + 1), :, 0] * conj(self.eigenvectors[4 * i + 3, :, 0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * (i + 1)) + 2, :, 0])))
            self.F_matrix[i, idx_F_ji_x_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * (i + 1), :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * (i + 1)) + 2, :, -1])))
            self.F_matrix[i, idx_F_ji_x_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * (i + 1), :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * (i + 1)) + 2, :, 0:-2])))

            # F ij Y+ S, i_y, j_y = i_y,i_y
            self.F_matrix[i, idx_F_ij_y_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*i, :, 0] * conj(self.eigenvectors[4*i + 3, :, 0]) * np.exp(-1.0j*self.k_array[0])))#  -  s_k * self.eigenvectors[4*i + 1, :, 0] * conj(self.eigenvectors[(4*i) + 2, :, 0]) * np.exp(1.0j*self.k_array[0])))
            self.F_matrix[i, idx_F_ij_y_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1]) * np.exp(-1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(1.0j * self.k_array[-1])))
            self.F_matrix[i, idx_F_ij_y_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2]) * np.exp(1.0j * self.k_array[0:-2])))

            # F ji Y+ S, i_y, j_y = i_y,i_y
            self.F_matrix[i, idx_F_ji_y_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4 * i, :, 0] * conj(self.eigenvectors[4 * i + 3, :, 0]) * np.exp(-1.0j * self.k_array[0])))# - s_k *self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(1.0j * self.k_array[0])))
            self.F_matrix[i, idx_F_ji_y_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1]) * np.exp(-1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(1.0j * self.k_array[-1])))
            self.F_matrix[i, idx_F_ji_y_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2]) * np.exp(1.0j * self.k_array[0:-2])))

            # F ij Y- S, i_y, j_y = i_y,i_y
            self.F_matrix[i, idx_F_ij_y_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*i, :, 0] * conj(self.eigenvectors[4*i + 3, :, 0]) * np.exp(+1.0j*self.k_array[0])))#  -  s_k * self.eigenvectors[4*i + 1, :, 0] * conj(self.eigenvectors[(4*i) + 2, :, 0]) * np.exp(-1.0j*self.k_array[0])))
            self.F_matrix[i, idx_F_ij_y_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1]) * np.exp(+1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(-1.0j * self.k_array[-1])))
            self.F_matrix[i, idx_F_ij_y_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) * np.exp(+1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2])))

            # F ji Y- S, i_y, j_y = i_y,i_y
            self.F_matrix[i, idx_F_ji_y_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4 * i, :, 0] * conj(self.eigenvectors[4 * i + 3, :, 0]) * np.exp(+1.0j * self.k_array[0])))# - s_k *self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(-1.0j * self.k_array[0])))
            self.F_matrix[i, idx_F_ji_y_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1]) * np.exp(+1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(-1.0j * self.k_array[-1])))
            self.F_matrix[i, idx_F_ji_y_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) * np.exp(+1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2])))

            #print("obital_indicator = ", self.orbital_indicator)
            # orbital_i
            if (self.orbital_indicator == 's'):
                #print("orbital")
                self.F_matrix[i, idx_F_ij_s] += (1 / 8 * (self.F_matrix[i, idx_F_ij_x_pluss] + self.F_matrix[i, idx_F_ji_x_pluss] + self.F_matrix[i, idx_F_ij_x_minus] + self.F_matrix[i, idx_F_ji_x_minus] + self.F_matrix[i, idx_F_ij_y_pluss] + self.F_matrix[i, idx_F_ji_y_pluss] + self.F_matrix[i, idx_F_ij_y_minus] + self.F_matrix[i, idx_F_ji_y_minus]))

            elif (self.orbital_indicator == 'd'):
                self.F_matrix[i, idx_F_ij_s] += (1 / 8 * (self.F_matrix[i, idx_F_ij_x_pluss] + self.F_matrix[i, idx_F_ji_x_pluss] + self.F_matrix[i, idx_F_ij_x_minus] + self.F_matrix[i, idx_F_ji_x_minus] - self.F_matrix[i, idx_F_ij_y_pluss] - self.F_matrix[i, idx_F_ji_y_pluss] - self.F_matrix[i, idx_F_ij_y_minus] - self.F_matrix[i, idx_F_ji_y_minus]))

            elif (self.orbital_indicator == 'px'):
                self.F_matrix[i, idx_F_ij_s] += (1 / 4 * (self.F_matrix[i, idx_F_ij_x_pluss] - self.F_matrix[i, idx_F_ji_x_pluss] - self.F_matrix[i, idx_F_ij_x_minus] + self.F_matrix[i, idx_F_ji_x_minus]))

            elif (self.orbital_indicator == 'py'):
                self.F_matrix[i, idx_F_ij_s] += 1 / 4 * (self.F_matrix[i, idx_F_ij_y_pluss] - self.F_matrix[i, idx_F_ji_y_pluss] - self.F_matrix[i, idx_F_ij_y_minus] + self.F_matrix[i, idx_F_ji_y_minus])

        #Fix this part
        #   At the endpoint we can not calculate the correlation in x-direction - k = 0
        idx_endpoint = self.F_matrix.shape[0] - 1
        #idx_endpoint =
        # F_ii - same point
        self.F_matrix[idx_endpoint, idx_F_i] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[(4 * idx_endpoint) + 3, :, 0])))# - s_k * self.eigenvectors[(4 * i) + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0])))
        self.F_matrix[idx_endpoint, idx_F_i] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[(4 * idx_endpoint) + 3, :, -1])))# - s_k * self.eigenvectors[(4 * i) + 1, :,-1] * conj(self.eigenvectors[(4 * i) + 2, :, -1])))
        self.F_matrix[idx_endpoint, idx_F_i] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 3, :, 0:-2]) - s_k * self.eigenvectors[(4 * i) + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2])))

        # F ij Y+ S, i_y, j_y = i_y,i_y
        self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0]) * np.exp(-1.0j * self.k_array[0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(1.0j * self.k_array[0])))
        self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, -1]) * np.exp(-1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(1.0j * self.k_array[-1])))
        self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * idx_endpoint + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2]) * np.exp(1.0j * self.k_array[0:-2])))

        # F ji Y+ S, i_y, j_y = i_y,i_y
        self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0]) * np.exp(-1.0j * self.k_array[0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(1.0j * self.k_array[0])))
        self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, -1]) * np.exp(-1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(1.0j * self.k_array[-1])))
        self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * idx_endpoint + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2]) * np.exp(1.0j * self.k_array[0:-2])))

        # F ij Y- S, i_y, j_y = i_y,i_y
        self.F_matrix[idx_endpoint, idx_F_ij_y_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0]) * np.exp(+1.0j * self.k_array[0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(-1.0j * self.k_array[0])))
        self.F_matrix[idx_endpoint, idx_F_ij_y_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, -1]) * np.exp(+1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(-1.0j * self.k_array[-1])))
        self.F_matrix[idx_endpoint, idx_F_ij_y_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0:-2]) * np.exp(+1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * idx_endpoint + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2])))

        # F ji Y- S, i_y, j_y = i_y,i_y
        self.F_matrix[idx_endpoint, idx_F_ji_y_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0]) * np.exp(+1.0j * self.k_array[0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(-1.0j * self.k_array[0])))
        self.F_matrix[idx_endpoint, idx_F_ji_y_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, -1]) * np.exp(+1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(-1.0j * self.k_array[-1])))
        self.F_matrix[idx_endpoint, idx_F_ji_y_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0:-2]) * np.exp(+1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * idx_endpoint + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2])))

        # orbital_i
        if (self.orbital_indicator == 's'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 8 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] + self.F_matrix[idx_endpoint, idx_F_ji_x_pluss] + self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + self.F_matrix[idx_endpoint, idx_F_ji_x_minus] + self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] + self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] + self.F_matrix[idx_endpoint, idx_F_ij_y_minus] + self.F_matrix[idx_endpoint, idx_F_ji_y_minus]))

        elif (self.orbital_indicator == 'd'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 8 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] + self.F_matrix[idx_endpoint, idx_F_ji_x_pluss] + self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + self.F_matrix[idx_endpoint, idx_F_ji_x_minus] - self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] -self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] - self.F_matrix[idx_endpoint, idx_F_ij_y_minus] - self.F_matrix[idx_endpoint, idx_F_ji_y_minus]))

        elif (self.orbital_indicator == 'px'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 4 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] - self.F_matrix[idx_endpoint, idx_F_ji_x_pluss] - self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + self.F_matrix[idx_endpoint, idx_F_ji_x_minus]))

        elif (self.orbital_indicator == 'py'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += 1 / 4 * (self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] - self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] - self.F_matrix[idx_endpoint, idx_F_ij_y_minus] + self.F_matrix[idx_endpoint, idx_F_ji_y_minus])
        return self

    def calculate_F_matrix(self):

        #   Initialize the old F_matrix to 0+0j, so that we can start to add new values
        self.F_matrix[:, :] = 0.0 + 0.0j
        #print(self.F_matrix)

        #print("dim F: ", self.F_matrix.shape)
        #print("dim eigenvalues: ", self.eigenvalues.shape)
        #print("dim eigenvectors: ", self.eigenvectors.shape)

        #idx_endpoint = self.F_matrix.shape[0] - 1

        # Calculation loops
        s_k = 1.0
        #if abs(self.eigenvalues[n, k]) >= np.inf:
        #    print("debye")
        #    continue
        #if (self.k_array[k] == 0) or (self.k_array[k] == np.pi):
            # print("k")
        #    s_k = 0.0

        #coeff = 1 / (2 * self.L_y) * tanh(self.beta * self.eigenvalues[n, k])  # /2)

        # debye freq, stemp function. = if energy>debye freq
        idx1, idx2 = np.where(abs(self.eigenvalues) >= self.debye_freq)
        step_func = np.ones(shape=(4 * self.L_x, (self.L_y + 2) // 2), dtype=int) #equal dimension as eigenvalues
        if len(idx1) == (4 * self.L_x * len(self.eigenvalues[1])-1):
            for i in range(len(idx1)):
                step_func[idx1[i]][idx2[i]] = 0

        for i in range(self.F_matrix.shape[0]):
            # F_ii - same point
            self.F_matrix[i, idx_F_i] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0])  * (self.eigenvectors[4 * i, :, 0] * conj(self.eigenvectors[(4 * i) + 3, :, 0]))) #k = 0
            self.F_matrix[i, idx_F_i] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1])  * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[(4 * i) + 3, :, -1]))) # k = pi
            self.F_matrix[i, idx_F_i] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2])  * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 3, :, 0:-2]) - s_k * self.eigenvectors[(4 * i) + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2])))
        """
        for i in range(self.F_matrix.shape[0]):
            # F_ii - same point
            tmp0 = np.sum(1 / (2 * self.L_y) * tanh(self.beta * self.eigenvalues[:, 0] )* np.matmul(self.eigenvectors[4 * i, :, 0], conj(self.eigenvectors[(4 * i) + 3, :, 0]))) #k = 0
            self.F_matrix[i, idx_F_i] += tmp0
            tmp1 = np.sum(1 / (2 * self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * np.matmul(self.eigenvectors[4 * i, :, -1], conj(self.eigenvectors[(4 * i) + 3, :, -1]))) # k = pi
            self.F_matrix[i, idx_F_i] += tmp1
            tmp2 = np.sum(1 / (2 * self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * np.matmul(np.transpose(self.eigenvectors[4 * i, :, 0:-2]), conj(self.eigenvectors[(4 * i) + 3, :, 0:-2])) - s_k * np.matmul(np.transpose(self.eigenvectors[(4 * i) + 1, :, 0:-2]), conj(self.eigenvectors[(4 * i) + 2, :, 0:-2])))
            self.F_matrix[i, idx_F_i] += tmp2
            print("--------")
            print(tmp0)
            print(tmp1)
            print(tmp2)
        """
            # Calculation loops
        #for k in range(self.eigenvalues.shape[1]):
        #    sk = 1.0
        #    if k == 0 or (k == (self.eigenvalues.shape[1] - 1) and self.L_y % 2 == 0):
        #        sk = 0.0
        #    for n in range(self.eigenvalues.shape[0]):
        #        f = np.tanh(self.beta * self.eigenvalues[n, k])
        #        f /= 2*  self.L_y
        #        for i in range(self.F_matrix.shape[0]):
                    # UP DOWN SAME POINT
        #            self.F_matrix[i, idx_F_i] += (self.eigenvectors[4 * i, n, k] * conj(self.eigenvectors[(4 * i) + 3, n, k]) - sk * self.eigenvectors[(4 * i) + 1, n, k] * conj(self.eigenvectors[(4 * i) + 2, n, k])) * f
        #print(self.F_matrix)

        #Sjekket output fra trippel for-løkken og bruk av sum og *, samme output
        return self

    def calc_new_mu(self):  # cdef double calc_new_mu(double complex [:,:,:] psi, double [:, :] energies, int Ny, double beta) nogil:

        # shape [Nx, Nk]
        num = 0.0
        f = 0.0
        for i in range(self.eigenvectors.shape[0] // 4):
            for k in range(self.eigenvectors.shape[2]):
                sk = 1.0
                if k == 0 or (k == (self.eigenvectors.shape[2] - 1) and self.L_y % 2 == 0):
                    sk = 0.0
                for n in range(self.eigenvectors.shape[1]):
                    # sum_ikn (|u_ink|^2 + |v_ink|^2)f(E_nk) + (|w_ink|^2+|x_ink|^2)(1-f(E_nk)
                    f = 1 / (1 + np.exp(self.beta * self.eigenvalues[n, k]))
                    num += (pow(abs(self.eigenvectors[4 * i, n, k]), 2) + pow(abs(self.eigenvectors[4 * i + 1, n, k]), 2)) * f
                    num += sk * (pow(abs(self.eigenvectors[4 * i + 2, n, k]), 2) + pow(abs(self.eigenvectors[4 * i + 3, n, k]), 2)) * (1 - f)
        print("n_e:%4.3g" % num)
        return self

    #   Plot delta, U-term and F for the resulting hamiltonian
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

        # rashba coupling
        line = ax[1].plot(self.alpha_R_array[:, 0], label=r'$\alpha_R^x$')
        ax[1].plot(self.alpha_R_array[:, 1], ls='--', label=r'$\alpha_R^y$')
        ax[1].plot(self.alpha_R_array[:, 2], ls=':', label=r'$\alpha_R^z$')
        ax[1].legend()
        ax[1].set_title('Rashba SOC coupling')

    #   Created a small test of dimensjon for each matrix/variable
    #   This test is done before we start solving the system to avoid trivial error due to runtime
    def test_valid(self):
        # dimensions
        assert self.L_x > 0, "L_x must be larger than 0."
        assert self.L_y > 0, "L_x must be larger than 0."

        # U term - e-e interaction
        assert self.U_array.shape[0] == self.L_x

        # F_matrix - correlation function
        assert self.F_matrix.shape[0] == self.L_x
        assert self.F_matrix.shape[1] == num_idx_F_i

        # t_ij - hopping term
        #assert self.t_array.shape[0] == self.L_x
        # t_ij
        assert self.t_x.shape[0] == self.L_x - 1
        assert self.t_y.shape[0] == self.L_x

    #   Get-functions
    def get_eigenvectors(self):
        return self.eigenvectors

    def get_eigenvalues(self):
        return self.eigenvalues

    def energy_vec(self, min_E, max_E, resolution):
        Ne = int((max_E - min_E) / resolution)
        Es = np.linspace(min_E, max_E, Ne, dtype=np.float64)
        return Es


    def local_density_of_states(self, resolution, sigma, min_e, max_e):
        # sigma is the size of gaussian function
        num_energies = int((max_e - min_e) / resolution) #number energu
        num_latticesites = self.eigenvectors.shape[0] // 4 #number latticesties
        coeff = 1.0 / (sigma * sqrt(np.pi))

        Es = self.energy_vec(min_e, max_e, resolution)
        ldos = np.zeros((num_latticesites, num_energies), dtype=np.float64)

        sk = 1.0
        #if ki == 0 or (ki == (self.eigenvectors.shape[2] - 1) and self.L_y % 2 == 0):
        #    sk = 0.0
        for ei in range(num_energies):
            pos_e_diff = self.eigenvalues[:, :] - Es[ei]
            neg_e_diff = self.eigenvalues[:, :] + Es[ei]
            pos_ldos = coeff * np.exp(-pow(pos_e_diff / sigma, 2))
            neg_ldos = coeff * np.exp(-pow(neg_e_diff / sigma, 2))

            for ii in range(num_latticesites):
                ldos[ii, ei] += np.sum(pow(abs(self.eigenvectors[4 * ii, :, :]), 2) + pow(abs(self.eigenvectors[4 * ii + 1, :, :]),2) * pos_ldos)
                ldos[ii, ei] += np.sum(sk * (pow(abs(self.eigenvectors[4 * ii + 2, :, 0:-2]), 2) + pow(abs(self.eigenvectors[4 * ii + 3, :, 0:-2]), 2)) * neg_ldos[:,0:-2]) #k != 0 or k!= pi
        return ldos


    def ldos_from_problem(self, resolution, kernel_size, min_E, max_E):
        ldos = self.local_density_of_states(resolution, kernel_size, min_E, max_E)
        energies = self.energy_vec(min_E, max_E, resolution)

        return np.asarray(ldos), np.asarray(energies)