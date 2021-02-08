import numpy as np
import matplotlib.pyplot as plt
from numba import jit


from utilities_t import idx_F_i, idx_F_ij_x_pluss, idx_F_ij_x_minus, idx_F_ij_y_pluss, idx_F_ij_y_minus, idx_F_ij_s, num_idx_F_i
from numpy import conj, tanh, exp, sqrt, cos, sin, log#, sqrt #as conj, tanh, exp, cos, sin, sqrt

"""
This script define the class System which contains all necessary information to construct one system.
"""

#(beta=np.inf, u_sc=-1.0, F_init=0.3, alpha_R_z=2, NY=50, N_SOC=5, N_F=100, N_SC=100, h_f_z=0.3, h_f_x=0.0)

class System:
    def __init__(self,
                 L_y = 100,
                 L_z = 102,
                 L_sc = 50,
                 L_nc = 50,
                 L_soc = 2,
                 L_sc_0 = 0,
                 L_f = 0,

                 t_x = 1,  #  0.5,
                 t_y = 1,  # 0.5,
                 t = 0.5,  # t used in compute energy
                 t_sc = 0.5,
                 t_0 = 1,  #0.5,
                 t_nc = 0.5,

                 h = [0.0,0.0,1.0],  #hx, hy, hz

                 u_sc = -2.1,  #-4.2, # V_ij in superconductor
                 u_nc = 0.0,  #-4.2,
                 u_soc = 0.0,  #-4.2,
                 u_f = 0.0,

                 mu_s = -3.5,  #s
                 mu_d = -0.5,
                 mu_pxpy = -1.5,
                 mu_nc = 1.9,  #0.9,
                 mu_sc = 1.9,  #0.9,
                 mu_soc = 1.7,  #0.85,
                 alpha_r = np.array([0.1, 0.0, 0.0], dtype=np.float64),
                 U = -4.2,
                 wd = 0.6,
                 F = 0.3,

                 alpha_R_initial = [0.0, 0.0, 0.5],  #0.1

                 beta = 200,  #np.inf,

                 phase=np.pi/4,
                 old_solution = False, #True if there is sendt in an initial phase from last system
                 old_F_matrix_guess = np.array([0.0],dtype=np.complex128),
                 old_phase_array=np.array([0.0], dtype=np.complex128),

                 #F_sc_initial = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 #F_sc_initial = np.pad(np.array([0.3], dtype=np.complex128),  (0,num_idx_F_i-1), mode='constant', constant_values=0.0),

                 F_sc_initial_s = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  #s-orbital
                 F_sc_initial_d = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                 F_sc_initial_px = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                 F_sc_initial_py = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                 F_sc_initial_pxpy = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0j, 1.0j, -1.0j, -1.0j],
                 F_sc_initial_spy = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0+1.0j, 1.0+1.0j, 1.0-1.0j, 1.0-1.0j],

                 #F_nc_initial = np.pad(np.array([0.3], dtype=np.complex128),  (0,num_idx_F_i-1), mode='constant', constant_values=0.0),  #0.3,
                 #F_soc_initial = np.pad(np.array([0.0], dtype=np.complex128),  (0,num_idx_F_i-1), mode='constant', constant_values=0.0),  #0.3,
                 #F_f_initial=np.pad(np.array([0.0], dtype=np.complex128), (0, num_idx_F_i - 1), mode='constant',constant_values=0.0),  # 0.3,

                 orbital_indicator = "s"
                 ):

        self.L_sc_0 = L_sc_0
        self.L_sc = L_sc
        self.L_soc = L_soc
        self.L_nc = L_nc
        self.L_f = L_f
        self.L_x = self.L_sc_0 + self.L_nc + self.L_f + self.L_soc + self.L_sc
        self.L_y = L_y
        self.L_z = L_z
        self.t_x = t_x
        self.t_y = t_y
        self.t_sc = t_sc
        self.t_0 = t_0
        self.t_nc = t_nc
        self.t = t


        self.h = h

        self.u_sc = u_sc
        self.u_soc = u_soc
        self.u_nc = u_nc
        self.u_f = u_f

        self.mu_sc = mu_sc
        self.mu_soc = mu_soc
        self.mu_nc = mu_nc
        self.mu_array = np.zeros(shape=(self.L_x,), dtype=np.float64)

        self.alpha_R_initial = alpha_R_initial

        self.beta = beta

        self.debye_freq = wd

        self.phase = phase
        if old_solution == True:
            #self.phase_array = np.hstack((np.ones(self.L_sc_0) * self.phase, np.zeros(L_nc + L_sc))).ravel()
            self.F_sc_0_initial = old_F_matrix_guess[:self.L_sc_0, :]
            self.F_soc_initial = old_F_matrix_guess[self.L_sc_0:(self.L_sc_0+self.L_soc), :]
            self.F_nc_initial = old_F_matrix_guess[self.L_sc_0:(self.L_sc_0+self.L_nc), :]
            self.F_f_initial = old_F_matrix_guess[self.L_sc_0:(self.L_sc_0+self.L_f), :]
            self.F_sc_initial = old_F_matrix_guess[-self.L_sc:]

            self.phase_array = np.hstack((np.ones(self.L_sc_0)*self.phase, np.linspace(self.phase, 0, self.L_x-self.L_sc_0-self.L_sc),np.zeros(L_sc))).ravel()
            #lattice = np.linspace(0, self.L_x, self.L_x)
            #plt.plot(lattice, self.phase_array)
            #plt.show()
        else:
            self.phase_array = np.hstack((np.ones(self.L_sc_0)*self.phase, np.linspace(self.phase, 0, self.L_x-self.L_sc_0-self.L_sc),np.zeros(L_sc))).ravel()
            #lattice = np.linspace(0, self.L_x, self.L_x)
            #plt.plot(lattice, self.phase_array)
            #plt.show()

            #self.phase_array = np.hstack((np.linspace(self.phase,self.phase * 4 / 4, self.L_sc_0) , np.linspace(self.phase * 4 / 5, self.phase * 1/ 5, (self.L_x-self.L_sc_0-self.L_sc)) , np.linspace(self.phase * 1/ 5, 0, self.L_sc))).ravel()

            self.F_sc_0_initial = np.zeros((self.L_sc_0, num_idx_F_i), dtype=np.complex128)
            self.F_sc_0_initial[:, 0] = 0.3

            self.F_soc_initial = np.zeros((self.L_soc, num_idx_F_i), dtype=np.complex128)
            self.F_soc_initial[:, 0] = 0.3

            self.F_nc_initial = np.zeros((self.L_nc, num_idx_F_i), dtype=np.complex128)
            self.F_nc_initial[:, 0] = 0.3

            self.F_f_initial = np.zeros((self.L_f, num_idx_F_i), dtype=np.complex128)
            self.F_f_initial[:, 0] = 0.3

            self.F_sc_initial = np.zeros((self.L_sc, num_idx_F_i), dtype=np.complex128)
            self.F_sc_initial[:, 0] = 0.3

        #print("phase_array, ", self.phase_array.shape)
        #print("sc_0, ",self.F_sc_0_initial.shape)
        #print("nc, ", self.F_nc_initial.shape)
        #print("sc, ", self.F_sc_initial.shape)

        #self.F_sc_0_initial = np.pad(np.array([0.3 * np.exp(1.0j * self.phase)], dtype=np.complex128), (0, num_idx_F_i - 1), mode='constant',constant_values=0.0)

        self.orbital_indicator = str(orbital_indicator)

        #   To define k, I have to take case of even/odd number of lattice sites.
        #   Choose k from 0 to pi, so we only sum over k >= 0 in later calculations
        #if L_y % 2 == 0:
            #self.ky_array = np.linspace(0.0, np.pi, num =(self.L_y + 2)//2, endpoint=True, dtype=np.float64)
            #self.kz_array = np.linspace(0.0, np.pi, num =(self.L_z + 2)// 2, endpoint=True, dtype=np.float64)

        #    self.ky_array = np.linspace(-np.pi, np.pi, num=(self.L_y), endpoint=True, dtype=np.float64)
        #    self.kz_array = np.linspace(-np.pi, np.pi, num =(self.L_z), endpoint=True, dtype=np.float64)
        #else:
            #self.ky_array = np.linspace(0.0, np.pi, num =(self.L_y + 2)//2, endpoint=False, dtype=np.float64)
            #self.kz_array = np.linspace(0.0, np.pi, num =(self.L_z + 2)// 2, endpoint=False, dtype=np.float64)
        #    self.ky_array = np.linspace(-np.pi, np.pi, num=(self.L_y), endpoint=False, dtype=np.float64)
        #    self.kz_array = np.linspace(-np.pi, np.pi, num =(self.L_z), endpoint=False, dtype=np.float64)

        self.ky_array = np.linspace(-np.pi, np.pi, num=(self.L_y), endpoint=True, dtype=np.float64)
        self.kz_array = np.linspace(-np.pi, np.pi, num =(self.L_z), endpoint=True, dtype=np.float64)

        # F_matrix: F_ii, F_orbital, F_x+, F_x-, F_y+, F_y-
        self.F_matrix = np.zeros((self.L_x, num_idx_F_i), dtype=np.complex128)  #   2D, one row for each F-comp
        self.U_array = np.zeros(self.L_x, dtype=np.float64)                     #   1D
        self.t_x_array = np.zeros((self.L_x - 1), dtype=np.float64)
        self.t_y_array = np.zeros((self.L_x), dtype=np.float64)
        self.h_array = np.zeros((self.L_x, 3), dtype=np.float64)

        self.alpha_R_x_array = np.zeros(shape=(self.L_x, 3), dtype=np.float64)
        self.alpha_R_y_array = np.zeros(shape=(self.L_x, 3), dtype=np.float64)
        #self.alpha_R_array = np.zeros((self.L_x, 3), dtype=np.float64)


        #   Eigenvectors
        #self.eigenvectors = np.zeros(shape=(4 * self.L_x, 4 * self.L_x, (self.L_y + 2) // 2, (self.L_z + 2) // 2), dtype=np.complex128)
        self.eigenvectors = np.zeros(shape=(4 * self.L_x, 4 * self.L_x, (self.L_y), (self.L_z)),dtype=np.complex128)


        #   Eigenvalues
        #self.eigenvalues = np.zeros(shape=(4 * self.L_x, (self.L_y + 2) // 2, (self.L_z + 2) // 2), dtype=np.float128)
        self.eigenvalues = np.zeros(shape=(4 * self.L_x, (self.L_y), (self.L_z)), dtype=np.float64)

        #   Hamiltonian
        #self.hamiltonian = np.zeros(shape=(self.L_x * 4, self.L_x * 4), dtype=np.complex128)
        self.hamiltonian = np.zeros(shape=(self.L_x * 4, self.L_x * 4, self.L_y, self.L_z), dtype=np.complex128)

        #   Fill inn values in matrix
        # L_x = L_nc + L_soc + L_sc
        #self.set_phase_initial = np.linspace(1.0j * self.phase, 0, self.L_x, dtype=np.complex128)
        for i in range(self.L_x):
            self.t_y_array[i] = t_y
            if i < self.L_sc_0:           #   SC
                self.F_matrix[i, :] = self.F_sc_0_initial[i, :] *  np.exp(1.0j * self.phase_array[i])     # Set all F values to inital condition for SC material (+1 s-orbital)
                self.U_array[i] = self.u_sc #* np.exp(1.0j * np.pi/2)
                self.mu_array[i] = self.mu_sc

            elif i < (self.L_sc_0 + self.L_nc):    #   NC
                self.F_matrix[i, :] = self.F_nc_initial[i-self.L_sc_0, :] *  np.exp(1.0j * self.phase_array[i])                 #   Set all F values to inital condition for NC material

                self.U_array[i] = self.u_nc
                self.mu_array[i] = self.mu_nc

            elif i < (self.L_sc_0 + self.L_nc + self.L_f):
                self.F_matrix[i, :] = self.F_f_initial[i-(self.L_sc_0 + self.L_nc), :] *  np.exp(1.0j * self.phase_array[i])
                self.h_array[i, 0] = self.h[0]
                self.h_array[i, 1] = self.h[1]
                self.h_array[i, 2] = self.h[2]
                self.U_array[i] = self.u_f

            elif i < (self.L_sc_0 + self.L_nc + self.L_f + self.L_soc):  # SOC
                self.F_matrix[i, :] = self.F_soc_initial[i-(self.L_sc_0 + self.L_nc + self.L_f), :] *  np.exp(1.0j * self.phase_array[i])
                self.U_array[i] = self.u_soc
                self.mu_array[i] = self.mu_soc

                #self.alpha_R_array[i, 0] = self.alpha_R_x
                #self.alpha_R_array[i, 1] = self.alpha_R_y
                #self.alpha_R_array[i, 2] = self.alpha_R_z
                self.alpha_R_y_array[i, 0] = self.alpha_R_initial[0]
                self.alpha_R_y_array[i, 1] = self.alpha_R_initial[1]
                self.alpha_R_y_array[i, 2] = self.alpha_R_initial[2]

                self.alpha_R_x_array[i, 0] = self.alpha_R_initial[0]
                self.alpha_R_x_array[i, 1] = self.alpha_R_initial[1]
                self.alpha_R_x_array[i, 2] = self.alpha_R_initial[2]

                #if i < L_sc + self.L_nc + self.L_f + L_soc - 1:  # only add x-coupling of both layers are in SOC
                #    self.alpha_R_x_array[i, 0] = alpha_R_initial[0]
                #    self.alpha_R_x_array[i, 1] = alpha_R_initial[1]
                #    self.alpha_R_x_array[i, 2] = alpha_R_initial[2]


            else:           #   SC
                self.F_matrix[i, :] = self.F_sc_initial[i-(self.L_sc_0 + self.L_nc + self.L_f + self.L_soc), :] *  np.exp(1.0j * self.phase_array[i])     # Set all F values to inital condition for SC material (+1 s-orbital)
                self.U_array[i] = self.u_sc
                self.mu_array[i] = self.mu_sc
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
        for i in range(self.L_x -1):
            self.t_x_array[i] = self.t_x


    def epsilon_ijk(self, i, j, ky, kz, spin):  # spin can take two values: 1 = up, 2 = down
        #t_0 = 0.0
        #t_1 = 0.0
        e = 0.0
        #print(i, j)
        if i == j:
            #t_1 = self.t_array[i]
            e = np.complex128(- 2 * self.t_y_array[i] * (cos(ky) + cos(kz)) - self.mu_array[i]) # spini in (1, 2) => (0, 1) index => (spinup, spindown)
        elif i == j + 1:
            #t_0 = self.t_array[j]
            e = np.complex128(-self.t_y_array[j]) #x #-
        elif i == j - 1:
            #t_0 = self.t_array[i]
            e = np.complex128(-self.t_y_array[i]) #x #-

        #e = np.complex128(-(t_0 + 2 * t_1 * np.cos(k)) - h - self.mu)
        #e = np.complex128(-2 * t_1 * np.cos(k) - t_0)
        return e

    def set_epsilon(self, arr, i, j, ky, kz):
        arr[0][0] += self.epsilon_ijk(i, j, ky, kz, 1)
        arr[1][1] += self.epsilon_ijk(i, j, ky, kz, 2)
        arr[2][2] += -self.epsilon_ijk(i, j, ky, kz, 1)
        arr[3][3] += -self.epsilon_ijk(i, j, ky, kz, 2)
        return arr

    def delta_gap(self, i):
        return self.U_array[i] * self.F_matrix[i, idx_F_i]

    def set_delta(self, arr, i, j):
        # Comment out +=, and remove the other comp from update_hamil to increase runtime and check if there is any diff. Shouldnt be diff in output.
        if i==j:
            #   Skjekk om du må bytte om index
            arr[0][3] = -self.delta_gap(i)#/2
            arr[1][2] = self.delta_gap(i)#/2
            arr[2][1] = conj(self.delta_gap(i))#/2
            arr[3][0] = -conj(self.delta_gap(i))#/2

        """"
        def set_delta(self, i, j):
            if i==j:
                self.hamiltonian[4 * i + 3][4 * j + 0] += self.delta_gap(i) / 2
                self.hamiltonian[4 * i + 2][4 * j + 1] += self.delta_gap(i) / 2
                self.hamiltonian[4 * i + 1][4 * j + 2] += self.delta_gap(i) / 2
                self.hamiltonian[4 * i + 0][4 * j + 3] += self.delta_gap(i) / 2
        """
        return arr

    def set_rashba_ky(self, arr, i, j, ky, kz):
        I = 1.0j
        sinky = sin(ky)
        sinkz = sin(kz)

        # barr = arr[2:][2:]
        if i == j:
            # (n_z*sigma_x - n_x*sigma_z)
            y00 = -self.alpha_R_y_array[i, 0]
            y01 = self.alpha_R_y_array[i, 2]
            y10 = self.alpha_R_y_array[i, 2]
            y11 = self.alpha_R_y_array[i, 0]

            z01_up = -self.alpha_R_y_array[i, 1] - I * self.alpha_R_y_array[i, 0]
            z10_up = -self.alpha_R_y_array[i, 1] + I * self.alpha_R_y_array[i, 0]
            z01_down = -self.alpha_R_y_array[i, 1] + I * self.alpha_R_y_array[i, 0]
            z10_down = -self.alpha_R_y_array[i, 1] - I * self.alpha_R_y_array[i, 0]


            # Upper left
            arr[0][0] += sinky * y00
            arr[0][1] += sinky * y01 + sinkz * z01_up
            arr[1][0] += sinky * y10 + sinkz * z10_up
            arr[1][1] += sinky * y11

            # Bottom right. Minus and conjugate
            arr[2][2] += sinky * y00
            arr[2][3] += sinky * y01 + sinkz * z01_down
            arr[3][2] += sinky * y10 + sinkz * z10_down
            arr[3][3] += sinky * y11

        # Backward jump X-
        elif (i == (j - 1)) or (i == (j + 1)):
        #elif j == i + 1 or j == i - 1:
            #if j == i + 1:  # Backward jump X-
            xi = 0
            if i == (j - 1):  # Backward jump X-
                l = i #i
                coeff = -1.0/4.0
            else:  # Forward jump X+
                l = j #j
                coeff = 1.0/4.0

            if (self.L_sc_0 <= i < (self.L_sc_0 + self.L_soc)) and (self.L_sc_0 <= j < (self.L_sc_0 + self.L_soc)): #check if both i and j are inside soc material
                xi = 1

            s00_up = I *self.alpha_R_x_array[l, 1]
            s01_up = -self.alpha_R_x_array[l, 2] #maybe change sign on s01 and s10??
            s10_up = self.alpha_R_x_array[l, 2]
            s11_up = - I * self.alpha_R_x_array[l, 1]

            s00_down = I * self.alpha_R_x_array[l, 1]
            s01_down = self.alpha_R_x_array[l, 2]  # maybe change sign on s01 and s10??
            s10_down = -self.alpha_R_x_array[l, 2]
            s11_down = - I * self.alpha_R_x_array[l, 1]

            arr[0][0] += coeff * s00_up * (1 + xi)
            arr[0][1] += coeff * s01_up * (1 + xi)
            arr[1][0] += coeff * s10_up * (1 + xi)
            arr[1][1] += coeff * s11_up * (1 + xi)

            #arr[2][2] += conj(coeff * s00)
            #arr[2][3] += conj(coeff * s01)
            #arr[3][2] += conj(coeff * s10)
            #arr[3][3] += conj(coeff * s11)

            arr[2][2] += coeff * s00_down * (1 + xi)
            arr[2][3] += coeff * s01_down * (1 + xi)
            arr[3][2] += coeff * s10_down * (1 + xi)
            arr[3][3] += coeff * s11_down * (1 + xi)

        return arr

    def set_h(self, arr, i, j):
    # cdef double complex [:,:] n_dot_sigma = n_dot_sigma(h_i[i,:])
        if i == j:
            arr[0][0] += self.h_array[i, 2]
            arr[0][1] += self.h_array[i, 0] - 1.0j * self.h_array[i, 1]
            arr[1][0] += self.h_array[i, 0] + 1.0j * self.h_array[i, 1]
            arr[1][1] += -self.h_array[i, 2]

            arr[2][2] += -self.h_array[i, 2]
            arr[2][3] += -self.h_array[i, 0] - 1.0j * self.h_array[i, 1]
            arr[3][2] += -self.h_array[i, 0] + 1.0j * self.h_array[i, 1]
            arr[3][3] += self.h_array[i, 2]
        return arr

    def zero_init_hamiltonian(self):
        #for i in range(matrix.shape[0]):
        #    for j in range(matrix.shape[1]):
        #        matrix[i, j] = 0.0 + 0.0j
        self.hamiltonian[:, :] = 0.0 + 0.0j
        return self

    def set_hamiltonian(self, ky, kz):
        self.zero_init_hamiltonian()
        for i in range(self.L_x):
            for j in range(self.L_x):
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_epsilon(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, ky, kz)
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_delta(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j)
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_rashba_ky(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, ky, kz)
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_h(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j)
        return self

    def calculate_F_matrix(self):
        #   Initialize the old F_matrix to 0+0j, so that we can start to add new values
        #for a in range(num_idx_F_i):
        #    for b in range(self.F_matrix.shape[0]):
        #        self.F_matrix[b, a] = 0.0 + 0.0j
        self.F_matrix[:, :] = 0.0 + 0.0j

        #print("dim F: ", self.F_matrix.shape)
        #print("dim eigenvalues: ", self.eigenvalues.shape)
        #print("dim eigenvectors: ", self.eigenvectors.shape)



        # Calculation loops
        #for k in range(self.eigenvalues.shape[1]):
        #s_k = 1.0
        s_k = 0.0
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
        #idx1, idx2, idx3 = np.where(abs(self.eigenvalues) >= self.debye_freq)
        #step_func = np.ones(shape=(4 * self.L_x, (self.L_y + 2) // 2, (self.L_z + 2) // 2), dtype=int)
        step_func = np.ones(shape=(4 * self.L_x, (self.L_y), (self.L_z)), dtype=int)

        #if len(idx1) == (len(self.eigenvalues[0]) * len(self.eigenvalues[1]) * len(self.eigenvalues[2])-1):
        #    for i in range(len(idx1)):
         #       step_func[idx1[i]][idx2[i]][idx3[i]] = 0

        for i in range(self.F_matrix.shape[0]-1):
            # F_ii - same point
            #self.F_matrix[i, idx_F_i] += np.sum(step_func[:, 0, 0] * 1 / (self.L_y * self.L_z) * tanh(self.beta*self.eigenvalues[:, 0, 0]/2)*(self.eigenvectors[4 * i, :, 0, 0] * conj(self.eigenvectors[(4 * i) + 3, :, 0, 0])))# - s_k * self.eigenvectors[(4 * i) + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0])))
            #self.F_matrix[i, idx_F_i] += np.sum(step_func[:, -1, -1] * 1 / (self.L_y * self.L_z) * tanh(self.beta * self.eigenvalues[:, -1, -1]/2) * (self.eigenvectors[4 * i, :, -1, -1] * conj(self.eigenvectors[(4 * i) + 3, :, -1, -1])))# - s_k * self.eigenvectors[(4 * i) + 1, :,-1] * conj(self.eigenvectors[(4 * i) + 2, :, -1])))
            #self.F_matrix[i, idx_F_i] += np.sum(step_func[:, 0:-2, 0:-2] * 1 / (self.L_y * self.L_z) * tanh(self.beta * self.eigenvalues[:, 0:-2, 0:-2]/2) * (self.eigenvectors[4 * i, :, 0:-2, 0:-2] * conj(self.eigenvectors[(4 * i) + 3, :, 0:-2, 0:-2]) - s_k * self.eigenvectors[(4 * i) + 1, :,0:-2, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2, 0:-2])))
            self.F_matrix[i, idx_F_i] += np.sum(step_func[:, 1:, 1:] * 1 / (self.L_y * self.L_z) * (1+tanh(self.beta * self.eigenvalues[:, 1:, 1:] / 2)) * (self.eigenvectors[4 * i, :, 1:, 1:] * conj(self.eigenvectors[(4 * i) + 3, :, 1:, 1:])))


            # F ij X+ S, i_x, j_x
            self.F_matrix[i, idx_F_ij_x_pluss] += np.sum(step_func[:, 1:, 1:] * 1 / (self.L_y*self.L_z) * (1+tanh(self.beta * self.eigenvalues[:, 1:, 1:]/2)) * (self.eigenvectors[4 * i, :, 1:, 1:] * conj(self.eigenvectors[(4*(i+1)) + 3, :, 1:, 1:])))

            # F ij X- S, i_x, j_x
            self.F_matrix[i, idx_F_ij_x_minus] += np.sum(step_func[:, 1:, 1:] * 1 / (self.L_y*self.L_z) * (1+tanh(self.beta * self.eigenvalues[:, 1:, 1:]/2)) * (self.eigenvectors[4 * i, :, 1:, 1:] * conj(self.eigenvectors[4 * (i + 1) + 3, :, 1:, 1:])))

            # F ij Y+ S, i_y, j_y = i_y,i_y
            self.F_matrix[i, idx_F_ij_y_pluss] += np.sum(step_func[:, 1:, 1:] * 1 / (self.L_y*self.L_z) * (1+tanh(self.beta * self.eigenvalues[:, 1:, 1:]/2)) * (self.eigenvectors[4 * i, :, 1:, 1:] * conj(self.eigenvectors[4 * i + 3, :, 1:, 1:]) * np.exp(-1.0j * self.ky_array[1:]) * np.exp(-1.0j * self.kz_array[1:])))

            # F ij Y- S, i_y, j_y = i_y,i_y
            self.F_matrix[i, idx_F_ij_y_minus] += np.sum(step_func[:, 1:, 1:] * 1 / (self.L_y*self.L_z) * (1+tanh(self.beta * self.eigenvalues[:, 1:, 1:]/2)) * (self.eigenvectors[4 * i, :, 1:, 1:] * conj(self.eigenvectors[4 * i + 3, :, 1:, 1:]) * np.exp(+1.0j * self.ky_array[1:]) * np.exp(+1.0j * self.kz_array[1:])))

            #print("obital_indicator = ", self.orbital_indicator)
            # orbital_i
            if (self.orbital_indicator == 's'):
                #print("orbital")
                self.F_matrix[i, idx_F_ij_s] += (1 / 8 * (self.F_matrix[i, idx_F_ij_x_pluss] + conj(self.F_matrix[i, idx_F_ij_x_pluss]) + self.F_matrix[i, idx_F_ij_x_minus] + conj(self.F_matrix[i, idx_F_ij_x_minus]) + self.F_matrix[i, idx_F_ij_y_pluss] + conj(self.F_matrix[i, idx_F_ij_y_pluss]) + self.F_matrix[i, idx_F_ij_y_minus] + conj(self.F_matrix[i, idx_F_ij_y_minus])))

            elif (self.orbital_indicator == 'd'):
                self.F_matrix[i, idx_F_ij_s] += (1 / 8 * (self.F_matrix[i, idx_F_ij_x_pluss] + conj(self.F_matrix[i, idx_F_ij_x_pluss]) + self.F_matrix[i, idx_F_ij_x_minus] + conj(self.F_matrix[i, idx_F_ij_x_minus]) - self.F_matrix[i, idx_F_ij_y_pluss] - conj(self.F_matrix[i, idx_F_ij_y_pluss]) - self.F_matrix[i, idx_F_ij_y_minus] - conj(self.F_matrix[i, idx_F_ij_y_minus])))

            elif (self.orbital_indicator == 'px'):
                self.F_matrix[i, idx_F_ij_s] += (1 / 4 * (self.F_matrix[i, idx_F_ij_x_pluss] - conj(self.F_matrix[i, idx_F_ij_x_pluss]) - self.F_matrix[i, idx_F_ij_x_minus] + conj(self.F_matrix[i, idx_F_ij_x_minus])))

            elif (self.orbital_indicator == 'py'):
                self.F_matrix[i, idx_F_ij_s] += 1 / 4 * (self.F_matrix[i, idx_F_ij_y_pluss] - conj(self.F_matrix[i, idx_F_ij_y_pluss]) - self.F_matrix[i, idx_F_ij_y_minus] + conj(self.F_matrix[i, idx_F_ij_y_minus]))



            """
            # F ij X+ S, i_x, j_x
            self.F_matrix[i, idx_F_ij_x_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*i, :, 0] * conj(self.eigenvectors[(4*(i+1)) + 3, :, 0])))#  -  s_k * self.eigenvectors[(4*(i+1)) + 1, :, 0] * conj(self.eigenvectors[(4*i) + 2, :, 0])))
            self.F_matrix[i, idx_F_ij_x_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[(4*(i+1)) + 3, :, -1])))# - s_k * self.eigenvectors[(4*(i+1)) + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1])))
            self.F_matrix[i, idx_F_ij_x_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[(4*(i+1)) + 3, :, 0:-2]) - s_k * self.eigenvectors[(4*(i+1)) + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2])))

            # F ji X+ S, i_x, j_x
            #self.F_matrix[i, idx_F_ji_x_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*(i+1), :, 0] * conj(self.eigenvectors[4 *i + 3, :, 0])))# - s_k * self.eigenvectors[4*i + 1, :, 0] * conj(self.eigenvectors[4*(i+1) + 2, :, 0])))
            #self.F_matrix[i, idx_F_ji_x_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * (i + 1), :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[4 * (i + 1) + 2, :, -1])))
            #self.F_matrix[i, idx_F_ji_x_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * (i + 1), :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[4 * (i + 1) + 2, :, 0:-2])))

            # F ij X- S, i_x, j_x
            self.F_matrix[i, idx_F_ij_x_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*i, :, 0] * conj(self.eigenvectors[4*(i+1) + 3, :, 0])))#  -  s_k * self.eigenvectors[4*(i+1) + 1, :, 0] * conj(self.eigenvectors[(4*i) + 2, :, 0])))
            self.F_matrix[i, idx_F_ij_x_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * (i + 1) + 3, :, -1])))# - s_k * self.eigenvectors[4 * (i + 1) + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1])))
            self.F_matrix[i, idx_F_ij_x_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * (i + 1) + 3, :, 0:-2]) - s_k * self.eigenvectors[4 * (i + 1) + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2])))

            # F ji X- S, i_x, j_x
            #self.F_matrix[i, idx_F_ji_x_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4 * (i + 1), :, 0] * conj(self.eigenvectors[4 * i + 3, :, 0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * (i + 1)) + 2, :, 0])))
            #self.F_matrix[i, idx_F_ji_x_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * (i + 1), :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * (i + 1)) + 2, :, -1])))
            #self.F_matrix[i, idx_F_ji_x_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * (i + 1), :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * (i + 1)) + 2, :, 0:-2])))

            # F ij Y+ S, i_y, j_y = i_y,i_y
            self.F_matrix[i, idx_F_ij_y_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*i, :, 0] * conj(self.eigenvectors[4*i + 3, :, 0]) * np.exp(-1.0j*self.k_array[0])))#  -  s_k * self.eigenvectors[4*i + 1, :, 0] * conj(self.eigenvectors[(4*i) + 2, :, 0]) * np.exp(1.0j*self.k_array[0])))
            self.F_matrix[i, idx_F_ij_y_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1]) * np.exp(-1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(1.0j * self.k_array[-1])))
            self.F_matrix[i, idx_F_ij_y_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2]) * np.exp(1.0j * self.k_array[0:-2])))

            # F ji Y+ S, i_y, j_y = i_y,i_y
            #self.F_matrix[i, idx_F_ji_y_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4 * i, :, 0] * conj(self.eigenvectors[4 * i + 3, :, 0]) * np.exp(-1.0j * self.k_array[0])))# - s_k *self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(1.0j * self.k_array[0])))
            #self.F_matrix[i, idx_F_ji_y_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1]) * np.exp(-1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(1.0j * self.k_array[-1])))
            #self.F_matrix[i, idx_F_ji_y_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2]) * np.exp(1.0j * self.k_array[0:-2])))

            # F ij Y- S, i_y, j_y = i_y,i_y
            self.F_matrix[i, idx_F_ij_y_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4*i, :, 0] * conj(self.eigenvectors[4*i + 3, :, 0]) * np.exp(+1.0j*self.k_array[0])))#  -  s_k * self.eigenvectors[4*i + 1, :, 0] * conj(self.eigenvectors[(4*i) + 2, :, 0]) * np.exp(-1.0j*self.k_array[0])))
            self.F_matrix[i, idx_F_ij_y_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1]) * np.exp(+1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(-1.0j * self.k_array[-1])))
            self.F_matrix[i, idx_F_ij_y_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) * np.exp(+1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2])))

            # F ji Y- S, i_y, j_y = i_y,i_y
            #self.F_matrix[i, idx_F_ji_y_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta*self.eigenvalues[:, 0])*(self.eigenvectors[4 * i, :, 0] * conj(self.eigenvectors[4 * i + 3, :, 0]) * np.exp(+1.0j * self.k_array[0])))# - s_k *self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(-1.0j * self.k_array[0])))
            #self.F_matrix[i, idx_F_ji_y_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * i, :, -1] * conj(self.eigenvectors[4 * i + 3, :, -1]) * np.exp(+1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(-1.0j * self.k_array[-1])))
            #self.F_matrix[i, idx_F_ji_y_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * i, :, 0:-2] * conj(self.eigenvectors[4 * i + 3, :, 0:-2]) * np.exp(+1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * i + 1, :, 0:-2] * conj(self.eigenvectors[(4 * i) + 2, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2])))

            #print("obital_indicator = ", self.orbital_indicator)
            # orbital_i
            if (self.orbital_indicator == 's'):
                #print("orbital")
                self.F_matrix[i, idx_F_ij_s] += (1 / 8 * (self.F_matrix[i, idx_F_ij_x_pluss] + conj(self.F_matrix[i, idx_F_ij_x_pluss]) + self.F_matrix[i, idx_F_ij_x_minus] + conj(self.F_matrix[i, idx_F_ij_x_minus]) + self.F_matrix[i, idx_F_ij_y_pluss] + conj(self.F_matrix[i, idx_F_ij_y_pluss]) + self.F_matrix[i, idx_F_ij_y_minus] + conj(self.F_matrix[i, idx_F_ij_y_minus])))

            elif (self.orbital_indicator == 'd'):
                self.F_matrix[i, idx_F_ij_s] += (1 / 8 * (self.F_matrix[i, idx_F_ij_x_pluss] + conj(self.F_matrix[i, idx_F_ij_x_pluss]) + self.F_matrix[i, idx_F_ij_x_minus] + conj(self.F_matrix[i, idx_F_ij_x_minus]) - self.F_matrix[i, idx_F_ij_y_pluss] - conj(self.F_matrix[i, idx_F_ij_y_pluss]) - self.F_matrix[i, idx_F_ij_y_minus] - conj(self.F_matrix[i, idx_F_ij_y_minus])))

            elif (self.orbital_indicator == 'px'):
                self.F_matrix[i, idx_F_ij_s] += (1 / 4 * (self.F_matrix[i, idx_F_ij_x_pluss] - conj(self.F_matrix[i, idx_F_ij_x_pluss]) - self.F_matrix[i, idx_F_ij_x_minus] + conj(self.F_matrix[i, idx_F_ij_x_minus])))

            elif (self.orbital_indicator == 'py'):
                self.F_matrix[i, idx_F_ij_s] += 1 / 4 * (self.F_matrix[i, idx_F_ij_y_pluss] - conj(self.F_matrix[i, idx_F_ij_y_pluss]) - self.F_matrix[i, idx_F_ij_y_minus] + conj(self.F_matrix[i, idx_F_ij_y_minus]))
            """
        #Fix this part
        #   At the endpoint we can not calculate the correlation in x-direction - k = 0
        idx_endpoint = self.F_matrix.shape[0] - 1
        #idx_endpoint =
        # F_ii - same point
        self.F_matrix[idx_endpoint, idx_F_i] += np.sum(step_func[:, 1:, 1:] * 1 / (self.L_y * self.L_z) * (1+tanh(self.beta * self.eigenvalues[:, 1:, 1:]/2)) * (self.eigenvectors[4 * idx_endpoint, :, 1:, 1:] * conj(self.eigenvectors[(4 * idx_endpoint) + 3, :, 1:, 1:]) ))

        # F ij Y+ S, i_y, j_y = i_y,i_y
        self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] += np.sum(step_func[:, 1:, 1:] * 1 / (self.L_y * self.L_z) * (1+tanh(self.beta * self.eigenvalues[:, 1:, 1:]/2)) * (self.eigenvectors[4 * idx_endpoint, :, 1:, 1:] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 1:, 1:]) * np.exp(-1.0j * self.ky_array[1:]) * np.exp(-1.0j * self.kz_array[1:])))

        # F ij Y- S, i_y, j_y = i_y,i_y
        self.F_matrix[idx_endpoint, idx_F_ij_y_minus] += np.sum(step_func[:, 1:, 1:] * 1 / (self.L_y * self.L_z) * (1+tanh(self.beta * self.eigenvalues[:, 1:, 1:]/2)) * (self.eigenvectors[4 * idx_endpoint, :, 1:, 1:] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 1:, 1:]) * np.exp(+1.0j * self.ky_array[1:]) * np.exp(+1.0j * self.kz_array[1:])))

        # orbital_i
        if (self.orbital_indicator == 's'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 8 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_pluss]) + self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_minus]) + self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] + conj(self.F_matrix[idx_endpoint, idx_F_ij_y_pluss]) + self.F_matrix[idx_endpoint, idx_F_ij_y_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_y_minus])))

        elif (self.orbital_indicator == 'd'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 8 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_pluss]) + self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_minus]) - self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] - conj(self.F_matrix[idx_endpoint, idx_F_ij_y_pluss]) - self.F_matrix[idx_endpoint, idx_F_ij_y_minus] - conj(self.F_matrix[idx_endpoint, idx_F_ij_y_minus])))

        elif (self.orbital_indicator == 'px'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 4 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] - conj(self.F_matrix[idx_endpoint, idx_F_ij_x_pluss]) - self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_minus])))

        elif (self.orbital_indicator == 'py'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += 1 / 4 * (self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] - conj(self.F_matrix[idx_endpoint, idx_F_ij_y_pluss]) - self.F_matrix[idx_endpoint, idx_F_ij_y_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_y_minus]))



        """
        # F ij Y+ S, i_y, j_y = i_y,i_y
        self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0]) * np.exp(-1.0j * self.k_array[0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(1.0j * self.k_array[0])))
        self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, -1]) * np.exp(-1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(1.0j * self.k_array[-1])))
        self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * idx_endpoint + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2]) * np.exp(1.0j * self.k_array[0:-2])))

        # F ji Y+ S, i_y, j_y = i_y,i_y
        #self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0]) * np.exp(-1.0j * self.k_array[0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(1.0j * self.k_array[0])))
        #self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, -1]) * np.exp(-1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(1.0j * self.k_array[-1])))
        #self.F_matrix[idx_endpoint, idx_F_ji_y_pluss] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * idx_endpoint + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2]) * np.exp(1.0j * self.k_array[0:-2])))

        # F ij Y- S, i_y, j_y = i_y,i_y
        self.F_matrix[idx_endpoint, idx_F_ij_y_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0]) * np.exp(+1.0j * self.k_array[0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(-1.0j * self.k_array[0])))
        self.F_matrix[idx_endpoint, idx_F_ij_y_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, -1]) * np.exp(+1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(-1.0j * self.k_array[-1])))
        self.F_matrix[idx_endpoint, idx_F_ij_y_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0:-2]) * np.exp(+1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * idx_endpoint + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2])))

        # F ji Y- S, i_y, j_y = i_y,i_y
        #self.F_matrix[idx_endpoint, idx_F_ji_y_minus] += np.sum(step_func[:, 0] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0]) * (self.eigenvectors[4 * idx_endpoint, :, 0] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0]) * np.exp(+1.0j * self.k_array[0])))# - s_k * self.eigenvectors[4 * i + 1, :, 0] * conj(self.eigenvectors[(4 * i) + 2, :, 0]) * np.exp(-1.0j * self.k_array[0])))
        #self.F_matrix[idx_endpoint, idx_F_ji_y_minus] += np.sum(step_func[:, -1] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, -1]) * (self.eigenvectors[4 * idx_endpoint, :, -1] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, -1]) * np.exp(+1.0j * self.k_array[-1])))# - s_k * self.eigenvectors[4 * i + 1, :, -1] * conj(self.eigenvectors[(4 * i) + 2, :, -1]) * np.exp(-1.0j * self.k_array[-1])))
        #self.F_matrix[idx_endpoint, idx_F_ji_y_minus] += np.sum(step_func[:, 0:-2] * 1 / (self.L_y) * tanh(self.beta * self.eigenvalues[:, 0:-2]) * (self.eigenvectors[4 * idx_endpoint, :, 0:-2] * conj(self.eigenvectors[4 * idx_endpoint + 3, :, 0:-2]) * np.exp(+1.0j * self.k_array[0:-2]) - s_k * self.eigenvectors[4 * idx_endpoint + 1, :, 0:-2] * conj(self.eigenvectors[(4 * idx_endpoint) + 2, :, 0:-2]) * np.exp(-1.0j * self.k_array[0:-2])))

        # orbital_i
        if (self.orbital_indicator == 's'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 8 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_pluss]) + self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_minus]) + self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] + conj(self.F_matrix[idx_endpoint, idx_F_ij_y_pluss]) + self.F_matrix[idx_endpoint, idx_F_ij_y_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_y_minus])))

        elif (self.orbital_indicator == 'd'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 8 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_pluss]) + self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_minus]) - self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] - conj(self.F_matrix[idx_endpoint, idx_F_ij_y_pluss]) - self.F_matrix[idx_endpoint, idx_F_ij_y_minus] - conj(self.F_matrix[idx_endpoint, idx_F_ij_y_minus])))

        elif (self.orbital_indicator == 'px'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += (1 / 4 * (self.F_matrix[idx_endpoint, idx_F_ij_x_pluss] - conj(self.F_matrix[idx_endpoint, idx_F_ij_x_pluss]) - self.F_matrix[idx_endpoint, idx_F_ij_x_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_x_minus])))

        elif (self.orbital_indicator == 'py'):
            self.F_matrix[idx_endpoint, idx_F_ij_s] += 1 / 4 * (self.F_matrix[idx_endpoint, idx_F_ij_y_pluss] - conj(self.F_matrix[idx_endpoint, idx_F_ij_y_pluss]) - self.F_matrix[idx_endpoint, idx_F_ij_y_minus] + conj(self.F_matrix[idx_endpoint, idx_F_ij_y_minus]))
        """
        return self

    def short_calculate_F_matrix(self):

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
        ax[0].plot(np.multiply(self.U_array, np.abs(self.F_matrix[:,idx_F_i])), ls=':', label=r'$|\Delta|$')
        ax[0].plot(np.real(self.F_matrix[:, idx_F_i]), ls='--', label=r'$F_{i}^{\uparrow\downarrow}$')
        ax[0].set_title('Delta')
        ax[0].legend()

        # rashba coupling
        line = ax[1].plot(self.alpha_R_x_array[:, 0], label=r'$\alpha_R^x$')
        ax[1].plot(self.alpha_R_x_array[:, 1], ls='--', label=r'$\alpha_R^y$')
        ax[1].plot(self.alpha_R_x_array[:, 2], ls=':', label=r'$\alpha_R^z$')
        ax[1].legend()
        ax[1].set_title('Rashba SOC coupling')

        #fig.savefig('Hamilton components, mu_s=0.9, mu_soc=0.85, u=-4.2.png', bbox_inches='tight')
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
        assert self.t_x_array.shape[0] == self.L_x - 1
        assert self.t_y_array.shape[0] == self.L_x

        # magnetic field
        assert self.h_array.shape[0] == self.L_x
        assert self.h_array.shape[1] == 3

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
        num_energies = int((max_e - min_e) / resolution) #number energi
        num_latticesites = self.eigenvectors.shape[0] // 4 #number latticesties
        coeff = 1.0 / (sigma * sqrt(2*np.pi)) / (self.L_y*self.L_z)

        Es = self.energy_vec(min_e, max_e, resolution)
        ldos = np.zeros((num_latticesites, num_energies), dtype=np.float64)

        #sk = 1.0
        #if ki == 0 or (ki == (self.eigenvectors.shape[2] - 1) and self.L_y % 2 == 0):
        #    sk = 0.0
        """
        for ei in range(num_energies):
            pos_e_diff = self.eigenvalues[:, :] - Es[ei]
            neg_e_diff = self.eigenvalues[:, :] + Es[ei]
            pos_ldos = coeff * np.exp(-pow(pos_e_diff / sigma, 2))
            neg_ldos = coeff * np.exp(-pow(neg_e_diff / sigma, 2))

            for ii in range(num_latticesites):
                ldos[ii, ei] += np.sum(pow(abs(self.eigenvectors[4 * ii, :, :]), 2) + pow(abs(self.eigenvectors[4 * ii + 1, :, :]),2) * pos_ldos)
                ldos[ii, ei] += np.sum(sk * (pow(abs(self.eigenvectors[4 * ii + 2, :, 0:-2]), 2) + pow(abs(self.eigenvectors[4 * ii + 3, :, 0:-2]), 2)) * neg_ldos[:,0:-2]) #k != 0 or k!= pi
        """
        #pos_e_diff = self.eigenvalues[:, :, :]
        pos_e_diff = self.eigenvalues[:, 1:, 1:] /2
        #print(np.any(np.iscomplex(pos_e_diff)))
        #print(conj(self.eigenvectors[4 * 4, :, 1:, 1:])*self.eigenvectors[4 * 4, :, 1:, 1:])
        #neg_e_diff = self.eigenvalues[:, 1:, 1:]/2
        for ii in range(num_latticesites):
            us = pow(abs(self.eigenvectors[4 * ii, :, 1:, 1:]), 2) + pow(abs(self.eigenvectors[4 * ii + 1, :, 1:, 1:]), 2)
            #us = conj(self.eigenvectors[4 * ii, :, 1:, 1:])*self.eigenvectors[4 * ii, :, 1:, 1:] + conj(self.eigenvectors[4 * ii + 1, :, 1:, 1:])*self.eigenvectors[4 * ii + 1, :, 1:, 1:] #spin opp - spin ned

            #vs = (pow(abs(self.eigenvectors[4 * ii + 2, :, 1:, 1:]), 2) + pow(abs(self.eigenvectors[4 * ii + 3, :, 1:, 1:]), 2)) #sk

            for ei in range(num_energies):
                eng = Es[ei]
                pos_ldos = coeff * exp(-pow((pos_e_diff - eng) / sigma, 2))
                #neg_ldos = coeff * exp(-pow((neg_e_diff + eng) / sigma, 2))

                ldos[ii, ei] += np.sum(np.multiply(us, pos_ldos))
                #ldos[ii, ei] += np.sum(vs * neg_ldos)

        return ldos

    def long_local_density_of_states(self, res, kernel_size, min_e, max_e):
        Ne = int((max_e - min_e) / res)
        nx = self.eigenvectors.shape[0] // 4
        NI = self.eigenvectors.shape[1]
        NK = self.eigenvectors.shape[2]

        prefac = 1.0 / (kernel_size * sqrt(3.141592))
        Es = self.energy_vec(min_e, max_e, res)
        ldos = np.zeros((nx, Ne), dtype=np.float64)

        for ni in range(NI):
            for ki in range(NK):
                sk = 1.0
                if ki == 0 or (ki == (self.eigenvectors.shape[2] - 1) and self.L_y % 2 == 0):
                    sk = 0.0

                pos_e_diff = self.eigenvalues[ni, ki]
                neg_e_diff = self.eigenvalues[ni, ki]
                for ii in range(nx):
                    us = pow(abs(self.eigenvectors[4 * ii, ni, ki]), 2) + pow(abs(self.eigenvectors[4 * ii + 1, ni, ki]), 2)
                    vs = sk * (pow(abs(self.eigenvectors[4 * ii + 2, ni, ki]), 2) + pow(abs(self.eigenvectors[4 * ii + 3, ni, ki]), 2))
                    for ei in range(Ne):
                        eng = Es[ei]
                        pos_ldos = prefac * exp(-pow((pos_e_diff - eng) / kernel_size, 2))
                        neg_ldos = prefac * exp(-pow((neg_e_diff + eng) / kernel_size, 2))

                        ldos[ii, ei] += us * pos_ldos
                        ldos[ii, ei] += vs * neg_ldos
        return ldos

    def ldos_from_problem(self, resolution, kernel_size, min_E, max_E):
        ldos = self.local_density_of_states(resolution, kernel_size, min_E, max_E)
        energies = self.energy_vec(min_E, max_E, resolution)

        return np.asarray(ldos), np.asarray(energies)

    def compute_energy(self, N=False):
        # Compute the Free energy as in Linas Master Thisis

        delta_array = np.multiply(self.U_array, np.real(self.F_matrix[:,idx_F_i]))
        """
        if N==True:
            # u-term. Cant do if U = 0 in the region
            U_index = np.where(self.U_array[:self.L_nc] != 0)
            U_energy_nc = 0.0
            for u in U_index[0]:
                U_energy_nc += np.abs(delta_array[u]) ** 2 / self.U_array[u]
            #hopping_energy_nc = np.sum(np.sum(2 * self.t_0 * cos(self.k_array[:])) + self.mu_array[:self.L_nc])
            hopping_energy_nc = np.sum(np.sum(2 * self.t_0 * cos(self.k_array[:])) + self.mu_array[:self.L_nc])
            H_0_nc = self.L_nc * U_energy_nc - hopping_energy_nc
            # H_0 = - hopping_energy

            #print("U_energy_nc: ", U_energy_nc)
            #print("hopping_energy_nc: ", hopping_energy_nc)
            #print("H_0_nc: ", H_0_nc)
            eigenvalues_nc = self.eigenvalues[:self.L_nc]
            # F = H_0 - (1/self.beta) * np.sum(log(1 + exp(-self.beta * self.eigenvalues[:,:] / 2)))
            F_nc = H_0_nc + 1 / 2 * np.sum(eigenvalues_nc[eigenvalues_nc < 0])  # t->0, sum over only negative eigenvalues
            #print("Free energy: ", F_nc)
            return F_nc
        """

        # u-term. Cant do if U = 0 in the region
        U_index = np.where(self.U_array != 0)
        U_energy = 0.0
        for u in U_index[0]:
            U_energy += np.abs(delta_array[u])**2 / self.U_array[u]


        #hopping_energy = np.sum(np.sum(2 * self.t_0* cos(self.k_array[:])) + self.mu_array[:])
        #hopping_energy = np.sum(2 * self.t_0 * (cos(self.ky_array[:] + cos(self.kz_array[:]))))
        hopping_energy = np.sum(2 * self.t_0 * (cos(self.ky_array[1:] + cos(self.kz_array[1:]))))
        epsilon_energy = np.sum(hopping_energy + self.mu_array[:])
        #hopping_energy = self.L_x*np.sum(2 * self.t_0 * cos(self.k_array[:]) + np.sum(self.mu_array[:]))

        H_0 = self.L_x * U_energy - epsilon_energy
        #H_0 = - hopping_energy

        #print("U_energy: ", U_energy)
        #print("hopping_energy: ", hopping_energy)
        #print("H_0: ",H_0)
        #F = H_0 - (1/self.beta) * np.sum(log(1 + exp(-self.beta * self.eigenvalues[:,:,:] / 2)))
        F = H_0 - (1 / self.beta) * np.sum(log(1 + exp(-self.beta * self.eigenvalues[:, 1:, 1:] / 2)))
        #F = H_0 + 1/2 * np.sum(self.eigenvalues[self.eigenvalues < 0]) #t->0, sum over only negative eigenvalues
        #print("Free energy: ", F)

        return F

    def forcePhaseDifference(self):
        """
        Har testet min låsing, og vemunds tilnærming. Det viser seg min min har en svakhet der det ikke er symetrisk slik at strømmen
        ikke blir symetrisk. Correlation function ser nogen lunde lik ut for begge.

        Derfor velger jeg å bruke vemunds låsing:
        I = 1.0j
        phase_plus = np.exp(I * self.phase, dtype=np.complex128)
        self.F_matrix[0, 0] = np.abs(self.F_matrix[0, 0])* phase_plus
        self.F_matrix[-1, 0] = np.abs(self.F_matrix[-1, 0])
        """
        I = 1.0j
        phaseDiff = np.exp(I * self.phase, dtype=np.complex128)
        phase_plus = np.exp(I * self.phase, dtype=np.complex128)         #   SC_0
        ##self.F_matrix[0, :] = self.F_matrix[0,:] / (self.F_matrix[0,:]/self.F_matrix[-1, :]) * phaseDiff
        #self.F_matrix[1, :] = self.F_matrix[1,:] / (self.F_matrix[1,:]/self.F_matrix[-2, :]) * phaseDiff
        #self.F_matrix[2, :] = self.F_matrix[self.L_x - 3, :] * phaseDiff

        self.F_matrix[0, 0] = np.abs(self.F_matrix[0, 0])* phase_plus
        #self.F_matrix[1, 0] = np.abs(self.F_matrix[1, 0]) * phase_plus

        self.F_matrix[-1, 0] = np.abs(self.F_matrix[-1, 0])
        #self.F_matrix[-2, 0] = np.abs(self.F_matrix[-2, 0])
        return self

    def current_along_lattice(self):

        current = np.zeros(self.L_x - 1, dtype=np.float64)
        tanh_coeff = 1 / (np.exp(self.beta * self.eigenvalues) + 1)
        tanh_coeff /= (self.L_y * self.L_z)  # 1/(system.L_y*system.L_z) *(1-np.tanh(system.beta * system.eigenvalues / 2)) #-

        t = self.t_0

        for ix in range(1, len(current)):  # -1 because it doesnt give sense to check last point for I+
            xi_ii = 0
            xi_minus = 0
            xi_pluss = 0


            if (self.L_sc_0 <= ix < (self.L_sc_0 + self.L_soc)):  # check if both i and i are inside soc material
                xi_ii = 1
            if (self.L_sc_0 <= ix < (self.L_sc_0 + self.L_soc)):  # and (system.L_sc_0 <= ix+1 < (system.L_sc_0 + system.L_soc)):# and (system.L_sc_0 <= ix-1 < (system.L_sc_0 + system.L_soc)): #check if both i and i+1 are inside soc material
                xi_pluss = 1
            if (self.L_sc_0 <= ix < (self.L_sc_0 + self.L_soc)):  # and (system.L_sc_0 <= ix-1 < (system.L_sc_0 + system.L_soc)):# and (system.L_sc_0 <= ix+1 < (system.L_sc_0 + system.L_soc)): #check if both i and i-1 are inside soc material
                xi_minus = 1

            B_opp_opp_psite = 1.0j / 4 * self.alpha_R_x_array[ix, 1] * (1 + xi_ii)
            B_opp_opp_pluss = 1.0j / 4 * self.alpha_R_x_array[ix + 1, 1] * (1 + xi_pluss)
            B_opp_opp_minus = 1.0j / 4 * self.alpha_R_x_array[ix - 1, 1] * (1 + xi_minus)
            B_opp_opp_msite = 1.0j / 4 * self.alpha_R_x_array[ix, 1] * (1 + xi_ii)

            B_ned_ned_psite = - 1.0j / 4 * self.alpha_R_x_array[ix, 1] * (1 + xi_ii)
            B_ned_ned_pluss = - 1.0j / 4 * self.alpha_R_x_array[ix + 1, 1] * (1 + xi_pluss)
            B_ned_ned_minus = - 1.0j / 4 * self.alpha_R_x_array[ix - 1, 1] * (1 + xi_minus)
            B_ned_ned_msite = - 1.0j / 4 * self.alpha_R_x_array[ix, 1] * (1 + xi_ii)

            B_opp_ned_psite = - 1.0 / 4 * self.alpha_R_x_array[ix, 2] * (1 + xi_ii)
            B_opp_ned_pluss = - 1.0 / 4 * self.alpha_R_x_array[ix + 1, 2] * (1 + xi_pluss)
            B_opp_ned_minus = - 1.0 / 4 * self.alpha_R_x_array[ix - 1, 2] * (1 + xi_minus)
            B_opp_ned_msite = - 1.0 / 4 * self.alpha_R_x_array[ix, 2] * (1 + xi_ii)

            B_ned_opp_psite = + 1.0 / 4 * self.alpha_R_x_array[ix, 2] * (1 + xi_ii)
            B_ned_opp_pluss = + 1.0 / 4 * self.alpha_R_x_array[ix + 1, 2] * (1 + xi_pluss)
            B_ned_opp_minus = + 1.0 / 4 * self.alpha_R_x_array[ix - 1, 2] * (1 + xi_minus)
            B_ned_opp_msite = + 1.0 / 4 * self.alpha_R_x_array[ix, 2] * (1 + xi_ii)

            # ---- Hopping x+ (imag)----#
            #:
            current[ix] += np.imag(2 * np.sum(t * tanh_coeff[:, 1:, 1:] * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1), :, 1:, 1:])))  # opp opp # * (np.exp(1.0j * system.ky_array[1:]) * np.exp(1.0j * system.kz_array[1:])))) #sigma = opp
            current[ix] += np.imag(2 * np.sum(t * tanh_coeff[:, 1:, 1:] * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:])))  # ned ned # * (np.exp(1.0j * system.ky_array[1:]) * np.exp(1.0j * system.kz_array[1:])))) #sigma = opp

            current[ix] -= np.imag(2 * np.sum(t * tanh_coeff[:, 1:, 1:] * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1), :, 1:, 1:])))  # opp opp # # * (np.exp(-1.0j * system.ky_array[1:]) * np.exp(-1.0j * system.kz_array[1:])))) #sigma = opp
            current[ix] -= np.imag(2 * np.sum(t * tanh_coeff[:, 1:, 1:] * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:])))  # ned ned # # * (np.exp(-1.0j * system.ky_array[1:]) * np.exp(-1.0j * system.kz_array[1:])))) #sigma = opp

            # --- Rashba x+ (real)----#
            #:
            # """
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_opp_psite * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1), :, 1:, 1:])))  # opp opp
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_opp_pluss * (np.conj(self.eigenvectors[4 * (ix + 1), :, 1:, 1:]) * self.eigenvectors[4 * ix, :, 1:, 1:])))  # opp opp
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_opp_minus * (np.conj(self.eigenvectors[4 * (ix - 1), :, 1:, 1:]) * self.eigenvectors[4 * ix, :, 1:, 1:])))  # opp opp
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_opp_msite * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1), :, 1:, 1:])))  # opp opp

            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_ned_psite * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:])))  # ned ned
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_ned_pluss * (np.conj(self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:]) * self.eigenvectors[4 * ix + 1, :, 1:, 1:])))  # ned ned
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_ned_minus * (np.conj(self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:]) * self.eigenvectors[4 * ix + 1, :, 1:, 1:])))  # ned ned
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_ned_msite * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:])))  # ned ned

            #:
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_ned_psite * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:])))  # opp ned
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_ned_pluss * (np.conj(self.eigenvectors[4 * (ix + 1), :, 1:, 1:]) * self.eigenvectors[4 * ix + 1, :, 1:, 1:])))  # opp ned
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_ned_minus * (np.conj(self.eigenvectors[4 * (ix - 1), :, 1:, 1:]) * self.eigenvectors[4 * ix + 1, :, 1:, 1:])))  # opp ned
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_opp_ned_msite * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:])))  # opp ned

            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_opp_psite * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1), :, 1:, 1:])))  # ned opp
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_opp_pluss * (np.conj(self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:]) * self.eigenvectors[4 * ix, :, 1:, 1:])))  # ned opp
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_opp_minus * (np.conj(self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:]) * self.eigenvectors[4 * ix, :, 1:, 1:])))  # ned opp
            current[ix] -= np.real(1.0j * np.sum(tanh_coeff[:, 1:, 1:] * B_ned_opp_msite * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1), :, 1:, 1:])))  # ned opp
            # """

        return current




