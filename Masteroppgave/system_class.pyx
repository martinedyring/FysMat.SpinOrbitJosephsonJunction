import numpy as np
cimport numpy as cnp
import matplotlib.pyplot as plt


from libcpp cimport bool

from utilities cimport idx_F_i, idx_F_ij_x_pluss, idx_F_ij_x_minus, idx_F_ij_y_pluss, idx_F_ij_y_minus, idx_F_ij_s, num_idx_F_i
from numpy import conj, tanh, exp, sqrt, cos, sin, log#, sqrt #as conj, tanh, exp, cos, sin, sqrt

"""
This script define the class System which contains all necessary information to construct one system.
"""

cimport cython
@cython.embedsignature(True)
cdef class System:
    def __init__(self,
                 int L_y = 0,
                 int L_z = 0,
                 int L_sc = 0,
                 int L_nc = 0,
                 int L_soc = 0,
                 int L_sc_0 = 0,
                 int L_f = 0,

                 double t_x = 1,  #  0.5,
                 double t_y = 1,  # 0.5,
                 double t = 1,  # t used in compute energy
                 double t_sc = 1,
                 double t_0 = 1,  #0.5,
                 double t_nc = 1,

                 double [:] h = np.array([0.0, 0.0, 0.1], dtype=np.float64),  #hx, hy, hz

                 double u_sc = 0.0,  #-4.2, # V_ij in superconductor
                 double u_nc = 0.0,  #-4.2,
                 double u_soc = 0.0,  #-4.2,
                 double u_f = 0.0,

                 double mu_s = -3.5,  #s
                 double mu_d = -0.5,
                 double mu_pxpy = -1.5,
                 double mu_nc = 1.9,  #0.9,
                 double mu_sc = 1.9,  #0.9,
                 double mu_soc = 1.7,  #0.85,
                 double mu_f = 1.9,
                 double [:] alpha_r = np.array([0.1, 0.0, 0.0], dtype=np.float64),
                 double U = -4.2,
                 double wd = 0.6,
                 double F = 0.3,

                 double [:] alpha_R_initial = np.array([0.0, 0.0, 0.2], dtype=np.float64),  #0.1

                 double beta = 200,  #np.inf,

                 double phase=0.0,#np.pi/4,
                 bool old_solution = False, #True if there is sendt in an initial phase from last system
                 double complex [:,:] old_F_matrix_guess = np.zeros((0,0), dtype=np.complex128),
                 double complex [:,:] old_phase_array = np.zeros((0,0), dtype=np.complex128),
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
        self.mu_f = mu_f
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

            self.phase_array = np.hstack((np.ones(self.L_sc_0)*(-self.phase), np.linspace(-self.phase, 0, self.L_x-self.L_sc_0-self.L_sc),np.zeros(L_sc))).ravel()

        else:
            self.phase_array = np.hstack((np.ones(self.L_sc_0)*(-self.phase), np.linspace(-self.phase, 0, self.L_x-self.L_sc_0-self.L_sc),np.zeros(L_sc))).ravel()

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

        self.ky_array = np.linspace(-np.pi, np.pi, num=(self.L_y), endpoint=True, dtype=np.float64)
        self.kz_array = np.linspace(-np.pi, np.pi, num=(self.L_z), endpoint=True, dtype=np.float64)

        # F_matrix: F_ii, F_orbital, F_x+, F_x-, F_y+, F_y-
        self.F_matrix = np.zeros((self.L_x, num_idx_F_i), dtype=np.complex128)  #   2D, one row for each F-comp
        self.U_array = np.zeros(self.L_x, dtype=np.float64)                     #   1D
        self.t_x_array = np.zeros((self.L_x - 1), dtype=np.float64)
        self.t_y_array = np.zeros((self.L_x), dtype=np.float64)
        self.h_array = np.zeros((self.L_x, 3), dtype=np.float64)

        self.alpha_R_x_array = np.zeros((self.L_x, 3), dtype=np.float64)
        self.alpha_R_y_array = np.zeros((self.L_x, 3), dtype=np.float64)

        #   Eigenvectors
        self.eigenvectors = np.zeros(shape=(4 * self.L_x, 4 * self.L_x, (self.L_y), (self.L_z)),dtype=np.complex128)

        #   Eigenvalues
        self.eigenvalues = np.zeros(shape=(4 * self.L_x, (self.L_y), (self.L_z)), dtype=np.float128)

        #   Hamiltonian
        self.hamiltonian = np.zeros(shape=(self.L_x * 4, self.L_x * 4), dtype=np.complex128)

        #   Fill inn values in matrix
        # L_x = L_nc + L_soc + L_sc
        cdef int i
        for i in range(self.L_x):
            self.t_y_array[i] = t_y
            if i < self.L_sc_0:           #   SC
                self.F_matrix[i, :] = self.F_sc_0_initial[i, :] *  np.exp(np.multiply(1.0j, self.phase_array[i]))     # Set all F values to inital condition for SC material (+1 s-orbital)
                self.U_array[i] = self.u_sc #* np.exp(1.0j * np.pi/2)
                self.mu_array[i] = self.mu_sc

            elif i < (self.L_sc_0 + self.L_nc):    #   NC
                self.F_matrix[i, :] = self.F_nc_initial[i-self.L_sc_0, :] *  np.exp(np.multiply(1.0j, self.phase_array[i]))                 #   Set all F values to inital condition for NC material

                #self.U_array[i] = self.u_nc
                self.mu_array[i] = self.mu_nc

            elif i < (self.L_sc_0 + self.L_nc + self.L_f):  #   F
                self.F_matrix[i, :] = self.F_f_initial[i-(self.L_sc_0 + self.L_nc), :] *  np.exp(np.multiply(1.0j, self.phase_array[i]))
                self.h_array[i, 0] = self.h[0]
                self.h_array[i, 1] = self.h[1]
                self.h_array[i, 2] = self.h[2]
                #self.U_array[i] = self.u_f
                self.mu_array[i] = self.mu_f

            elif i < (self.L_sc_0 + self.L_nc + self.L_f + self.L_soc):  # SOC
                self.F_matrix[i, :] = self.F_soc_initial[i-(self.L_sc_0 + self.L_nc + self.L_f), :] *  np.exp(np.multiply(1.0j, self.phase_array[i]))
                #self.U_array[i] = self.u_soc
                self.mu_array[i] = self.mu_soc

                self.alpha_R_y_array[i, 0] = self.alpha_R_initial[0]
                self.alpha_R_y_array[i, 1] = self.alpha_R_initial[1]
                self.alpha_R_y_array[i, 2] = self.alpha_R_initial[2]

                self.alpha_R_x_array[i, 0] = self.alpha_R_initial[0]
                self.alpha_R_x_array[i, 1] = self.alpha_R_initial[1]
                self.alpha_R_x_array[i, 2] = self.alpha_R_initial[2]

            else:           #   SC
                self.F_matrix[i, :] = self.F_sc_initial[i-(self.L_sc_0 + self.L_nc + self.L_f + self.L_soc), :] *  np.exp(np.multiply(1.0j, self.phase_array[i]))     # Set all F values to inital condition for SC material (+1 s-orbital)
                self.U_array[i] = self.u_sc
                self.mu_array[i] = self.mu_sc

        # Some parameters only rely on neighbors in x-direction, and thus has only NX-1 links
        for i in range(self.L_x -1):
            self.t_x_array[i] = self.t_x

    cpdef epsilon_ijk(self, int i, int j, double ky, double kz):  # spin can take two values: 1 = up, 2 = down
        cdef double complex e = 0.0
        if i == j:
            e = np.complex128(- 2 * np.multiply(self.t_y_array[i], (cos(ky) + cos(kz))) - self.mu_array[i]) # spini in (1, 2) => (0, 1) index => (spinup, spindown)
        elif i == (j + 1):
            e = -1 * np.complex128(self.t_y_array[j]) #x #-
        elif i == (j - 1):
            e = -1 * np.complex128(self.t_y_array[i]) #x #-
        return e

    cpdef set_epsilon(self, double complex [:,:] arr, int i, int j, double ky, double kz):
        arr[0][0] += self.epsilon_ijk(i, j, ky, kz)
        arr[1][1] += self.epsilon_ijk(i, j, ky, kz)
        arr[2][2] += -self.epsilon_ijk(i, j, ky, kz)
        arr[3][3] += -self.epsilon_ijk(i, j, ky, kz)
        return arr

    cpdef delta_gap(self, int i):
        return self.U_array[i] * self.F_matrix[i, idx_F_i]

    cpdef set_delta(self, double complex [:,:] arr, int i, int j):
        if i==j:
            arr[0][3] += -self.delta_gap(i)#/2
            arr[1][2] += self.delta_gap(i)#/2
            arr[2][1] += conj(self.delta_gap(i))#/2
            arr[3][0] += -conj(self.delta_gap(i))#/2
        return arr

    cpdef set_rashba_ky(self, double complex [:,:] arr, int i, int j, double ky, double kz):
        I = 1.0j
        cdef double sinky = sin(ky)
        cdef double sinkz = sin(kz)

        # if statement
        cdef double y00
        cdef double y01
        cdef double y10
        cdef double y11

        cdef double complex z01_up
        cdef double complex z10_up
        cdef double complex z01_down
        cdef double complex z10_down

        # elif statement
        cdef int l
        cdef double coeff

        cdef double complex s00
        cdef double complex s01
        cdef double complex s10
        cdef double complex s11

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
            if i == (j - 1):  # Backward jump X-
                l = i #i
                coeff = -1.0/4.0
            else:  # Forward jump X+
                l = j#j
                coeff = 1.0/4.0

            if (self.L_sc_0 + self.L_nc <= i < (self.L_sc_0 + self.L_nc + self.L_soc)) and (self.L_sc_0 + self.L_nc <= j < (self.L_sc_0 + self.L_nc + self.L_soc)): #check if both i and j are inside soc material
                coeff = coeff * 2

            s00 = I *self.alpha_R_x_array[l, 1]
            s01 = -self.alpha_R_x_array[l, 2] #maybe change sign on s01 and s10??
            s10 = self.alpha_R_x_array[l, 2]
            s11 = - I * self.alpha_R_x_array[l, 1]

            arr[0][0] += coeff * s00
            arr[0][1] += coeff * s01
            arr[1][0] += coeff * s10
            arr[1][1] += coeff * s11

            arr[2][2] += coeff * s00
            arr[2][3] -= coeff * s01
            arr[3][2] -= coeff * s10
            arr[3][3] += coeff * s11

        return arr

    cpdef set_h(self, double complex [:,:] arr, int i, int j):
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

    cpdef zero_init_hamiltonian(self):
        self.hamiltonian[:, :] = 0.0 + 0.0j
        return self

    cpdef set_hamiltonian(self, double ky, double kz):
        self.zero_init_hamiltonian()
        cdef int i,j
        for i in range(self.L_x):
            for j in range(self.L_x):
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_epsilon(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, ky, kz)
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_delta(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j)
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_rashba_ky(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, ky, kz)
                self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_h(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j)
        return self

    cpdef calculate_F_matrix(self):
        self.F_matrix[:, :] = 0.0 + 0.0j
        cdef int i
        for i in range(self.L_x):
            self.F_matrix[i, idx_F_i] += np.sum(1 / (2 * self.L_y * self.L_z) * (1+tanh(np.multiply(self.beta, self.eigenvalues[:, 1:, 1:]) / 2)) * (self.eigenvectors[4 * i, :, 1:, 1:] * conj(self.eigenvectors[(4 * i) + 3, :, 1:, 1:])))
        return self


    #   Plot delta, U-term and F for the resulting hamiltonian
    cpdef plot_components_of_hamiltonian(self, fig):
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
    cpdef test_valid(self):
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
    cpdef get_eigenvectors(self):
        return self.eigenvectors

    cpdef get_eigenvalues(self):
        return self.eigenvalues

    cpdef energy_vec(self, double min_E, double max_E, double resolution):
        cdef double Ne = int((max_E - min_E) / resolution)
        cdef double [:] Es = np.linspace(min_E, max_E, Ne, dtype=np.float64)
        return Es


    cpdef local_density_of_states(self, double resolution, double sigma, double min_e, double max_e):
        cdef int num_latticesites = self.L_x #number latticesties
        cdef double coeff = 1.0 / (sigma * sqrt(2*np.pi) * self.L_y * self.L_z)

        cdef double [:] Es = self.energy_vec(min_e, max_e, resolution)
        cdef int num_energies = len(Es)
        cdef double [:,:,:,:] ldos_lattice = np.zeros(shape=(num_latticesites, self.eigenvalues.shape[0], self.eigenvalues.shape[1]-1, self.eigenvalues.shape[2]-1), dtype=np.float64)
        cdef double [:,:,:,:] ldos_energies = np.zeros(shape=(num_energies, self.eigenvalues.shape[0], self.eigenvalues.shape[1]-1, self.eigenvalues.shape[2]-1), dtype=np.float64)
        cdef double [:,:] ldos = np.zeros(shape=(num_latticesites, num_energies), dtype=np.float64)

        cdef double [:,:,:] pos_e_diff = self.eigenvalues[:, 1:, 1:] #/ 2

        cdef double [:,:,:,:] us
        cdef double eng
        cdef double [:,:,:] pos_ldos

        cdef int ii, ei
        for ii in range(num_latticesites):
            us = conj(self.eigenvectors[4 * ii, :, 1:, 1:])*self.eigenvectors[4 * ii, :, 1:, 1:] + conj(self.eigenvectors[4 * ii + 1, :, 1:, 1:])*self.eigenvectors[4 * ii + 1, :, 1:, 1:] #spin opp + spin ned

            for ei in range(num_energies):
                eng = Es[ei]
                pos_ldos = coeff * exp(-pow(np.subtrace(pos_e_diff, eng), 2) / pow(sigma*np.sqrt(2), 2))
                ldos[ii, ei] += np.sum(np.multiply(us[:,:,:], pos_ldos[:,:,:]))

        return ldos, Es


    cpdef ldos_from_problem(self, double resolution, double kernel_size, double min_E, double max_E):
        self.ldos, self.energies = self.local_density_of_states(resolution, kernel_size, min_E, max_E)
        return np.asarray(self.ldos), np.asarray(self.energies)

    cpdef compute_energy(self, bool N):
        # Compute the Free energy as in Linas Master Thisis

        cdef double [:] delta_array = np.multiply(self.U_array, self.F_matrix[:,idx_F_i])

        # u-term. Cant do if U = 0 in the region
        cdef double[:,:] U_index = np.where(np.array(self.U_array) != 0.0)
        cdef double U_energy = 0.0

        cdef int u
        for u in U_index[0]:
            U_energy += np.abs(delta_array[u])**2 / self.U_array[u]

        cdef double H_0 = self.L_y*self.L_z * U_energy #- epsilon_energy
        cdef double F = H_0 - (1 / self.beta) * np.sum(log(1 + exp(-np.multiply(self.beta, self.eigenvalues[:, 1:, 1:]) / 2)))
        return F

    cpdef forcePhaseDifference(self):
        cdef double complex I = 1.0j
        #cdef double complex [:] phaseDiff = np.exp(I * (-self.phase), dtype=np.complex128)
        cdef double complex phase_plus = np.exp(I * (-self.phase), dtype=np.complex128)         #   SC_0
        self.F_matrix[0, 0] = np.abs(self.F_matrix[0, 0])* phase_plus
        self.F_matrix[-1, 0] = np.abs(self.F_matrix[-1, 0])
        return self


    cpdef current_along_lattice(self):
        cdef double complex I = 1.0j
        cdef double complex [:] current = np.zeros(self.L_x - 1, dtype=np.complex128)
        cdef double [:,:,:] tanh_coeff = 1 / (np.exp(np.multiply(self.beta, self.eigenvalues)) + 1) / (self.L_y * self.L_z) # 1/(system.L_y*system.L_z) *(1-np.tanh(system.beta * system.eigenvalues / 2)) #-

        cdef double t = self.t_0

        cdef int xi_ii
        cdef int xi_minus
        cdef int xi_pluss

        cdef double complex B_pluss
        cdef double complex B_minus
        cdef double complex C_pluss
        cdef double complex C_minus

        cdef int ix
        for ix in range(1, len(current)):  # -1 because it doesnt give sense to check last point for I+
            xi_ii = 0
            xi_minus = 0
            xi_pluss = 0
            if (self.L_sc_0 <= ix < (self.L_sc_0 + self.L_soc)): #check if both i and i are inside soc material
                xi_ii = 1
            if (self.L_sc_0 <= ix < (self.L_sc_0 + self.L_soc)) and (self.L_sc_0 <= ix+1 < (self.L_sc_0 + self.L_soc)): #check if both i and i+1 are inside soc material
                xi_pluss = 1
            if (self.L_sc_0 <= ix < (self.L_sc_0 + self.L_soc)) and (self.L_sc_0 <= ix-1 < (self.L_sc_0 + self.L_soc)): #check if both i and i-1 are inside soc material
                xi_minus = 1


            B_pluss = 0.0 + I / 4 * (self.alpha_R_x_array[ix, 1] - self.alpha_R_x_array[ix, 2]) * (1+xi_ii)
            B_minus = 0.0 + I / 4 * (self.alpha_R_x_array[ix-1, 1] - self.alpha_R_x_array[ix-1, 2]) * (1+xi_minus)
            C_pluss = 0.0 - I / 4 * (self.alpha_R_x_array[ix+1, 1] - self.alpha_R_x_array[ix+1, 2]) * (1+xi_pluss)
            C_minus = 0.0 - I / 4 * (self.alpha_R_x_array[ix, 1] - self.alpha_R_x_array[ix, 2]) * (1+xi_ii)

            # ---- Hopping x+ (imag)----#
            #:
            current[ix] += 2 * np.sum(np.multiply(t, tanh_coeff[:, 1:, 1:]) * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1), :, 1:, 1:]))  # * (np.exp(1.0j * system.ky_array[1:]) * np.exp(1.0j * system.kz_array[1:])))) #sigma = opp
            current[ix] -= 2 * np.sum(np.multiply(t, tanh_coeff[:, 1:, 1:]) * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1), :, 1:, 1:]))  # * (np.exp(-1.0j * system.ky_array[1:]) * np.exp(-1.0j * system.kz_array[1:])))) #sigma = opp

            # --- Rashba x+ (real)----#
            #:
            current[ix] += 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], C_minus) * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1), :, 1:, 1:])) # opp opp
            current[ix] += 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], C_pluss) * (np.conj(self.eigenvectors[4 * (ix + 1), :, 1:, 1:]) * self.eigenvectors[4 * ix, :, 1:, 1:])) # opp opp
            current[ix] -= 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], B_minus) * (np.conj(self.eigenvectors[4 * (ix - 1), :, 1:, 1:]) * self.eigenvectors[4 * ix, :, 1:, 1:]))  # opp opp
            current[ix] -= 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], B_pluss) * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1), :, 1:, 1:]))  # opp opp


            current[ix] += 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], C_minus) * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:])) #ned ned
            current[ix] += 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], C_pluss) * (np.conj(self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:]) * self.eigenvectors[4 * ix + 1, :, 1:, 1:])) #ned ned
            current[ix] -= 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], B_minus) * (np.conj(self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:]) * self.eigenvectors[4 * ix + 1, :, 1:, 1:]))  # ned ned
            current[ix] -= 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], B_pluss) * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:]))  # ned ned

            #:
            current[ix] += 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], C_minus) * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:])) # opp ned
            current[ix] += 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], C_pluss) * (np.conj(self.eigenvectors[4 * (ix + 1), :, 1:, 1:]) * self.eigenvectors[4 * ix + 1, :, 1:, 1:])) # opp ned
            current[ix] -= 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], B_minus) * (np.conj(self.eigenvectors[4 * (ix - 1), :, 1:, 1:]) * self.eigenvectors[4 * ix + 1, :, 1:, 1:]))  # opp ned
            current[ix] -= 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], B_pluss) * (np.conj(self.eigenvectors[4 * ix, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:]))  # opp ned


            current[ix] += 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], C_minus) * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix + 1), :, 1:, 1:])) # ned opp
            current[ix] += 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], C_pluss) * (np.conj(self.eigenvectors[4 * (ix + 1) + 1, :, 1:, 1:]) * self.eigenvectors[4 * ix, :, 1:, 1:])) # ned opp
            current[ix] -= 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], B_minus) * (np.conj(self.eigenvectors[4 * (ix - 1) + 1, :, 1:, 1:]) * self.eigenvectors[4 * ix, :, 1:, 1:]))  # ned opp
            current[ix] -= 1.0j * np.sum(np.multiply(tanh_coeff[:, 1:, 1:], B_pluss) * (np.conj(self.eigenvectors[4 * ix + 1, :, 1:, 1:]) * self.eigenvectors[4 * (ix - 1), :, 1:, 1:]))  # ned opp

        return current


