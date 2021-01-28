import numpy as np
cimport numpy as cnp
import matplotlib.pyplot as plt

cimport cython
from cython.parallel cimport prange

from libcpp cimport bool

cdef extern from "complex.h":
    double complex cexp(double complex) nogil
    double complex conj(double complex) nogil
    double imag(double complex z) nogil
    double real(double complex z) nogil
    double abs(double complex z) nogil


cdef extern from "math.h":
    double sin(double x) nogil
    double cos(double x) nogil
    double tanh(double x) nogil
    double sqrt(double x) nogil
    double exp(double x) nogil
    double pow(double r, double e) nogil


"""
This script define the class System which contains all necessary information to construct one system.
"""

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
## FLOAT_64_TYPE = cnp.float64
## FLOAT_128_TYPE = cnp.float128
## COMPLEX_128_TYPE = cnp.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
## ctypedef np.float64 FLOAT_64_TYPE_t
## ctypedef np.float128 FLOAT_128_TYPE_t
## ctypedef np.complex128 COMPLEX_128_TYPE_t


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

                 cnp.ndarray[cnp.float64_t, ndim=1] h = np.zeros(3, dtype=np.float64),  #hx, hy, hz

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
                 double U = -4.2,
                 double wd = 0.6,
                 double F = 0.3,

                 cnp.ndarray[cnp.float64_t, ndim=1] alpha_R_initial = np.zeros(3, dtype=np.float64),  #0.1

                 double beta = 200,  #np.inf,

                 double phase=0.0,#np.pi/4,
                 bool old_solution = False, #True if there is sendt in an initial phase from last system
                 cnp.ndarray[cnp.complex128_t, ndim=1] old_F_matrix_guess = np.zeros(1, dtype=np.complex128),
                 cnp.ndarray[cnp.complex128_t, ndim=1] old_phase_array = np.zeros(1, dtype=np.complex128),
                 ):

        self.L_sc_0 = L_sc_0
        self.L_sc = L_sc
        self.L_soc = L_soc
        self.L_nc = L_nc
        self.L_f = L_f
        self.L_x = L_sc_0 + L_nc + L_f + L_soc + L_sc
        self.L_y = L_y
        self.L_z = L_z
        self.t_x = t_x
        self.t_y = t_y
        self.t_sc = t_sc
        self.t_0 = t_0
        self.t_nc = t_nc
        self.t = t

        #self.h = h[:]

        #self.u_sc = u_sc
        #self.u_soc = u_soc
        #self.u_nc = u_nc
        #self.u_f = u_f

        #self.mu_sc = mu_sc
        #self.mu_soc = mu_soc
        #self.mu_nc = mu_nc
        #self.mu_f = mu_f
        self.mu_array = np.zeros(shape=(self.L_x), dtype=np.float64)

        #self.alpha_R_initial = alpha_R_initial[:]

        self.beta = beta

        #self.phase = phase
        if old_solution == True:
            #self.phase_array = np.hstack((np.ones(self.L_sc_0) * self.phase, np.zeros(L_nc + L_sc))).ravel()
            self.F_sc_0_initial = old_F_matrix_guess[:L_sc_0]
            self.F_soc_initial = old_F_matrix_guess[L_sc_0:(L_sc_0 + L_soc)]
            self.F_nc_initial = old_F_matrix_guess[L_sc_0:(L_sc_0 + L_nc)]
            self.F_f_initial = old_F_matrix_guess[L_sc_0:(L_sc_0, + L_f)]
            self.F_sc_initial = old_F_matrix_guess[(self.L_x - L_sc):]

            self.phase_array = np.hstack((np.ones(L_sc_0)*(np.multiply(-1, phase)), np.linspace(np.multiply(-1, phase), 0, self.L_x-L_sc_0-L_sc), np.zeros(L_sc)))

        else:
            self.phase_array = np.hstack((np.ones(L_sc_0)*(np.multiply(-1, phase)), np.linspace(np.multiply(-1, phase), 0, self.L_x-L_sc_0-L_sc), np.zeros(L_sc)))

            self.F_sc_0_initial = np.zeros(L_sc_0, dtype=np.complex128)
            self.F_sc_0_initial[:] = 0.3

            self.F_soc_initial = np.zeros(L_soc, dtype=np.complex128)
            self.F_soc_initial[:] = 0.3

            self.F_nc_initial = np.zeros(L_nc, dtype=np.complex128)
            self.F_nc_initial[:] = 0.3

            self.F_f_initial = np.zeros(L_f, dtype=np.complex128)
            self.F_f_initial[:] = 0.3

            self.F_sc_initial = np.zeros(L_sc, dtype=np.complex128)
            self.F_sc_initial[:] = 0.3

        self.ky_array = np.linspace(-np.pi, np.pi, num=(L_y)) #, dtype=np.float64)
        self.kz_array = np.linspace(-np.pi, np.pi, num=(L_z)) #, dtype=np.float64)

        # F_matrix: F_ii, F_orbital, F_x+, F_x-, F_y+, F_y-
        self.F_matrix = np.zeros(self.L_x, dtype=np.complex128)  #   2D, one row for each F-comp
        self.U_array = np.zeros(self.L_x, dtype=np.float64)                     #   1D
        self.t_x_array = np.zeros((self.L_x - 1), dtype=np.float64)
        self.t_y_array = np.zeros((self.L_x), dtype=np.float64)
        self.h_array = np.zeros((self.L_x, 3), dtype=np.float64)

        self.alpha_R_x_array = np.zeros((self.L_x, 3), dtype=np.float64)
        self.alpha_R_y_array = np.zeros((self.L_x, 3), dtype=np.float64)

        #   Eigenvectors
        self.eigenvectors = np.zeros(shape=(4 * self.L_x, 4 * self.L_x, L_y-1, L_z-1),dtype=np.complex128)

        #   Eigenvalues
        self.eigenvalues = np.zeros(shape=(4 * self.L_x, L_y-1, L_z-1), dtype=np.float128)

        #   Hamiltonian
        self.hamiltonian = np.zeros(shape=(self.L_x * 4, self.L_x * 4), dtype=np.complex128)

        #   Fill inn values in matrix
        # L_x = L_nc + L_soc + L_sc
        cdef int i
        for i in range(self.L_x):
            self.t_y_array[i] = t_y
            if i < self.L_sc_0:           #   SC
                self.F_matrix[i] = np.multiply(self.F_sc_0_initial[i],  np.exp(np.multiply(1.0j, self.phase_array[i])))    # Set all F values to inital condition for SC material (+1 s-orbital)
                self.U_array[i] = u_sc #* np.exp(1.0j * np.pi/2)
                self.mu_array[i] = mu_sc

            elif i < (L_sc_0 + L_nc):    #   NC
                self.F_matrix[i] = np.multiply(self.F_nc_initial[i-L_sc_0],  np.exp(np.multiply(1.0j, self.phase_array[i])))                #   Set all F values to inital condition for NC material

                #self.U_array[i] = self.u_nc
                self.mu_array[i] = mu_nc

            elif i < (L_sc_0 + L_nc + L_f):  #   F
                self.F_matrix[i] = np.multiply(self.F_f_initial[i-(L_sc_0 + L_nc)],  np.exp(np.multiply(1.0j, self.phase_array[i])))
                self.h_array[i, 0] = h[0]
                self.h_array[i, 1] = h[1]
                self.h_array[i, 2] = h[2]
                #self.U_array[i] = u_f
                self.mu_array[i] = mu_f

            elif i < (np.sum([L_sc_0, L_nc, L_f, L_soc])):  # SOC
                self.F_matrix[i] = np.multiply(self.F_soc_initial[i-(np.sum([L_sc_0, L_nc, L_f]))],  np.exp(np.multiply(1.0j, self.phase_array[i])))
                #self.U_array[i] = self.u_soc
                self.mu_array[i] = mu_soc

                self.alpha_R_y_array[i, 0] = alpha_R_initial[0]
                self.alpha_R_y_array[i, 1] = alpha_R_initial[1]
                self.alpha_R_y_array[i, 2] = alpha_R_initial[2]

                self.alpha_R_x_array[i, 0] = alpha_R_initial[0]
                self.alpha_R_x_array[i, 1] = alpha_R_initial[1]
                self.alpha_R_x_array[i, 2] = alpha_R_initial[2]

            else:           #   SC
                self.F_matrix[i] = np.multiply(self.F_sc_initial[i-(L_sc_0 + L_nc + L_f + L_soc)],  np.exp(np.multiply(1.0j, self.phase_array[i])))     # Set all F values to inital condition for SC material (+1 s-orbital)
                self.U_array[i] = u_sc
                self.mu_array[i] = mu_sc

        # Some parameters only rely on neighbors in x-direction, and thus has only NX-1 links
        for i in range(self.L_x -1):
            self.t_x_array[i] = t_x


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_epsilon(self, int i, int j, double ky, double kz, double[:] mu_array, double [:] t_array) nogil:
        cdef double complex epsilon = 0.0 #= self.epsilon_ijk(i, j, ky, kz)

        cdef double cos_ky = <double>(cos(ky))
        cdef double cos_kz = <double>(cos(kz))

        if i == j:
            epsilon = <double complex>(- 2 * (t_array[i] * (cos_ky + cos_kz)) - mu_array[i]) # spini in (1, 2) => (0, 1) index => (spinup, spindown)
        elif i == (j + 1):
            epsilon = <double complex>(-t_array[j]) #x #-
        elif i == (j - 1):
            epsilon = <double complex>(-t_array[i]) #x #-

        self.hamiltonian[4 * i + 0, 4 * j + 0] += epsilon #self.epsilon_ijk(i, j, ky, kz)
        self.hamiltonian[4 * i + 1, 4 * j + 1] += epsilon #self.epsilon_ijk(i, j, ky, kz)
        self.hamiltonian[4 * i + 2, 4 * j + 2] += -epsilon #-self.epsilon_ijk(i, j, ky, kz)
        self.hamiltonian[4 * i + 3, 4 * j + 3] += -epsilon #-self.epsilon_ijk(i, j, ky, kz)

        #arr[0][0] += self.epsilon_ijk(i, j, ky, kz)
        #arr[1][1] += self.epsilon_ijk(i, j, ky, kz)
        #arr[2][2] += -self.epsilon_ijk(i, j, ky, kz)
        #arr[3][3] += -self.epsilon_ijk(i, j, ky, kz)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_delta(self, int i, int j, double [:] U_array, double complex [:] F_matrix) nogil:
        cdef double complex u
        if i==j:
            u = <double complex>(F_matrix[i] * U_array[i]) #=self.delta_gap(i)

            self.hamiltonian[4 * i + 0, 4 * j + 3] += -u #-self.delta_gap(i)#/2
            self.hamiltonian[4 * i + 1, 4 * j + 2] += u #self.delta_gap(i)#/2
            self.hamiltonian[4 * i + 2, 4 * j + 1] += conj(u) #conj(self.delta_gap(i))#/2
            self.hamiltonian[4 * i + 3, 4 * j + 0] += -conj(u) #-conj(self.delta_gap(i))#/2

            #arr[0][3] += -self.delta_gap(i)#/2
            #arr[1][2] += self.delta_gap(i)#/2
            #arr[2][1] += conj(self.delta_gap(i))#/2
            #arr[3][0] += -conj(self.delta_gap(i))#/2


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_rashba_ky(self, int i, int j, double ky, double kz, double [:,:] alpha_R_x_array, double [:,:] alpha_R_y_array) nogil:
        cdef cnp.complex128_t I = 1.0j
        cdef cnp.float64_t sinky = sin(ky)
        cdef cnp.float64_t sinkz = sin(kz)

        # if statement
        cdef cnp.float64_t y00
        cdef cnp.float64_t y01
        cdef cnp.float64_t y10
        cdef cnp.float64_t y11

        cdef cnp.complex128_t z01_up
        cdef cnp.complex128_t z10_up
        cdef cnp.complex128_t z01_down
        cdef cnp.complex128_t z10_down

        # elif statement
        cdef int l
        cdef cnp.float64_t coeff

        cdef cnp.complex128_t s00
        cdef cnp.complex128_t s01
        cdef cnp.complex128_t s10
        cdef cnp.complex128_t s11

        cdef int lower_i = self.L_sc_0 + self.L_nc
        cdef int upper_i = self.L_sc_0 + self.L_nc + self.L_soc
        cdef int lower_j = self.L_sc_0 + self.L_nc
        cdef int upper_j = self.L_sc_0 + self.L_nc + self.L_soc


        if i == j:
            # (n_z*sigma_x - n_x*sigma_z)
            y00 = -alpha_R_y_array[i, 0]
            y01 = alpha_R_y_array[i, 2]
            y10 = alpha_R_y_array[i, 2]
            y11 = alpha_R_y_array[i, 0]

            z01_up = -alpha_R_y_array[i, 1] - I * alpha_R_y_array[i, 0]
            z10_up = -alpha_R_y_array[i, 1] + I * alpha_R_y_array[i, 0]
            z01_down = -alpha_R_y_array[i, 1] + I * alpha_R_y_array[i, 0]
            z10_down = -alpha_R_y_array[i, 1] - I * alpha_R_y_array[i, 0]


            # Upper left
            self.hamiltonian[4 * i + 0, 4 * j + 0] += sinky * y00
            self.hamiltonian[4 * i + 0, 4 * j + 1] += sinky * y01 + sinkz * z01_up
            self.hamiltonian[4 * i + 1, 4 * j + 0] += sinky * y10 + sinkz * z10_up
            self.hamiltonian[4 * i + 1, 4 * j + 1] += sinky * y11

            #arr[0][0] += sinky * y00
            #arr[0][1] += sinky * y01 + sinkz * z01_up
            #arr[1][0] += sinky * y10 + sinkz * z10_up
            #arr[1][1] += sinky * y11

            # Bottom right. Minus and conjugate
            self.hamiltonian[4 * i + 2, 4 * j + 2] += sinky * y00
            self.hamiltonian[4 * i + 2, 4 * j + 3] += sinky * y01 + sinkz * z01_down
            self.hamiltonian[4 * i + 3, 4 * j + 2] += sinky * y10 + sinkz * z10_down
            self.hamiltonian[4 * i + 3, 4 * j + 3] += sinky * y11

            #arr[2][2] += sinky * y00
            #arr[2][3] += sinky * y01 + sinkz * z01_down
            #arr[3][2] += sinky * y10 + sinkz * z10_down
            #arr[3][3] += sinky * y11

        # Backward jump X-
        elif (i == (j - 1)) or (i == (j + 1)):
            if i == (j - 1):  # Backward jump X-
                l = i #i
                coeff = -1.0/4.0
            else:  # Forward jump X+
                l = j#j
                coeff = 1.0/4.0

            if ((lower_i <= i < upper_i) and (lower_j <= j < upper_j)): #check if both i and j are inside soc material
                coeff = coeff * 2

            s00 = I * alpha_R_x_array[l, 1]
            s01 = -alpha_R_x_array[l, 2] #maybe change sign on s01 and s10??
            s10 = alpha_R_x_array[l, 2]
            s11 = - I * alpha_R_x_array[l, 1]

            self.hamiltonian[4 * i + 0, 4 * j + 0] += coeff * s00
            self.hamiltonian[4 * i + 0, 4 * j + 1] += coeff * s01
            self.hamiltonian[4 * i + 1, 4 * j + 0] += coeff * s10
            self.hamiltonian[4 * i + 1, 4 * j + 1] += coeff * s11

            #arr[0][0] += coeff * s00
            #arr[0][1] += coeff * s01
            #arr[1][0] += coeff * s10
            #arr[1][1] += coeff * s11

            self.hamiltonian[4 * i + 2, 4 * j + 2] += coeff * s00
            self.hamiltonian[4 * i + 2, 4 * j + 3] -= coeff * s01
            self.hamiltonian[4 * i + 3, 4 * j + 2] -= coeff * s10
            self.hamiltonian[4 * i + 3, 4 * j + 3] += coeff * s11

            #arr[2][2] += coeff * s00
            #arr[2][3] -= coeff * s01
            #arr[3][2] -= coeff * s10
            #arr[3][3] += coeff * s11

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_h(self, int i, int j, double [:,:] h_array) nogil:
        if i == j:
            self.hamiltonian[4 * i + 0, 4 * j + 0] += h_array[i, 2]
            self.hamiltonian[4 * i + 0, 4 * j + 1] += h_array[i, 0] - 1.0j * h_array[i, 1]
            self.hamiltonian[4 * i + 1, 4 * j + 0] += h_array[i, 0] + 1.0j * h_array[i, 1]
            self.hamiltonian[4 * i + 1, 4 * j + 1] += -h_array[i, 2]

            self.hamiltonian[4 * i + 2, 4 * j + 2] += -h_array[i, 2]
            self.hamiltonian[4 * i + 2, 4 * j + 3] += -h_array[i, 0] - 1.0j * h_array[i, 1]
            self.hamiltonian[4 * i + 3, 4 * j + 2] += -h_array[i, 0] + 1.0j * h_array[i, 1]
            self.hamiltonian[4 * i + 3, 4 * j + 3] += h_array[i, 2]

            #arr[0][0] += self.h_array[i, 2]
            #arr[0][1] += self.h_array[i, 0] - 1.0j * self.h_array[i, 1]
            #arr[1][0] += self.h_array[i, 0] + 1.0j * self.h_array[i, 1]
            #arr[1][1] += -self.h_array[i, 2]

            #arr[2][2] += -self.h_array[i, 2]
            #arr[2][3] += -self.h_array[i, 0] - 1.0j * self.h_array[i, 1]
            #arr[3][2] += -self.h_array[i, 0] + 1.0j * self.h_array[i, 1]
            #arr[3][3] += self.h_array[i, 2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void zero_init_hamiltonian(self):
        cdef int i,j
        for i in range(self.hamiltonian.shape[0]):
            for j in range(self.hamiltonian.shape[1]):
                self.hamiltonian[i,j] = 0.0 + 0.0j

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_hamiltonian(self,
                              double ky,
                              double kz,
                              double[:] mu_array,
                              double [:] t_array,
                              int L_x,
                              double complex[:] F_matrix,
                              double [:] U_array,
                              double [:,:] h_array,
                              double [:,:] alpha_R_x_array,
                              double [:,:] alpha_R_y_array) nogil:
        with gil:
            self.zero_init_hamiltonian()
        #cdef double complex [:,:] arr = np.ones((2,2), dtype=np.complex128)

        cdef int i,j
        for i in prange(L_x):
            for j in range(L_x):
                self.set_epsilon(i, j, ky, kz, mu_array, t_array)
                self.set_delta(i, j, U_array, F_matrix)
                self.set_rashba_ky(i, j, ky, kz, alpha_R_x_array, alpha_R_y_array)
                self.set_h(i, j, h_array)

                #self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_epsilon(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, ky, kz)
                #self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_delta(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j)
                #self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_rashba_ky(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j, ky, kz)
                #self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4] = self.set_h(self.hamiltonian[4 * i:4 * i + 4, 4 * j:4 * j + 4], i, j)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.overflowcheck(False)
    cdef void calculate_F_matrix(self, double complex [:] F_matrix, double[:,:,:,:] eigenvectors):
        cdef int i,j,k,l
        cdef double f = 0.0

        for i in prange(self.L_x, nogil=True):
            F_matrix[i] = 0.0 + 0.0j

        #print("calculateFmatrix 1")
        for l in range(self.L_z - 1): #4*L_x is the entire dimention
            for k in range(self.L_y - 1):
                for j in range(4 * self.L_x):
                    f = <double>(tanh(self.eigenvalues[j, k, l] * self.beta)/2.0 + 1.0)
                    f /= 2.0 * self.L_y * self.L_z
                    for i in prange(self.L_x, nogil=True):
                        #print("calculateFmatrix 2")
                        #self.F_matrix[i] += np.multiply(1 / (2 * self.L_y * self.L_z),np.sum(np.multiply(1+np.tanh(np.multiply(self.beta, self.eigenvalues[:, 1:, 1:]) / 2),np.multiply(self.eigenvectors[4 * i, :, 1:, 1:],conj(self.eigenvectors[(4 * i) + 3, :, 1:, 1:])))))
                        #self.F_matrix[i] += 1 / (2 * self.L_y * self.L_z) * (1+np.tanh(self.beta * self.eigenvalues[j, k, l]) / 2) * (self.eigenvectors[4 * i, j, k, l] * conj(self.eigenvectors[(4 * i) + 3, j, k, l]))
                        F_matrix[i] += f * (eigenvectors[4 * i, j, k, l] * conj(eigenvectors[(4 * i) + 3, j, k, l]))
        #print("f_matrix : ", self.F_matrix[:])


    #   Plot delta, U-term and F for the resulting hamiltonian
    def plot_components_of_hamiltonian(self):
        fig = plt.figure(figsize=(10,10))

        ax = fig.subplots(nrows=1, ncols=2).flatten()

        #   Delta-term
        line = ax[0].plot(self.U_array, label='U')
        ax[0].plot(np.multiply(self.U_array, np.abs(self.F_matrix[:])), ls=':', label=r'$|\Delta|$')
        ax[0].plot(np.real(self.F_matrix[:]), ls='--', label=r'$F_{i}^{\uparrow\downarrow}$')
        ax[0].set_title('Delta')
        ax[0].legend()

        # rashba coupling
        line = ax[1].plot(self.alpha_R_x_array[:, 0], label=r'$\alpha_R^x$')
        ax[1].plot(self.alpha_R_x_array[:, 1], ls='--', label=r'$\alpha_R^y$')
        ax[1].plot(self.alpha_R_x_array[:, 2], ls=':', label=r'$\alpha_R^z$')
        ax[1].legend()
        ax[1].set_title('Rashba SOC coupling')

        plt.show()
        #fig.savefig('Hamilton components, mu_s=0.9, mu_soc=0.85, u=-4.2.png', bbox_inches='tight')

    #   Created a small test of dimensjon for each matrix/variable
    #   This test is done before we start solving the system to avoid trivial error due to runtime
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef test_valid(self):
        # dimensions
        assert self.L_x > 0, "L_x must be larger than 0."
        assert self.L_y > 0, "L_x must be larger than 0."

        # U term - e-e interaction
        assert self.U_array.shape[0] == self.L_x

        # F_matrix - correlation function
        assert self.F_matrix.shape[0] == self.L_x

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

    cpdef get_F_matrix(self):
        return self.F_matrix

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double [:] energy_vec(self, double min_E, double max_E, double resolution):
        cdef int Ne = <int>((max_E - min_E) / resolution)
        cdef double [:] Es = np.linspace(min_E, max_E, Ne) #, dtype=np.float64)
        return Es

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef double [:,:] local_density_of_states(self, double resolution, double sigma, double min_e, double max_e, double complex [:,:,:,:] eigenvectors, double [:,:,:] eigenvalues) nogil:

        cdef int num_latticesites = self.L_x #number latticesties
        cdef double coeff = 1.0 / (sigma * sqrt(2*3.1415) * self.L_y * self.L_z)
        cdef double [:] Es #= self.energy_vec(min_e, max_e, resolution)
        cdef int num_energies = <int>((max_e - min_e) / resolution)
        #cdef double [:,:,:,:] ldos_lattice = np.zeros(shape=(num_latticesites, self.eigenvalues.shape[0], self.eigenvalues.shape[1]-1, self.eigenvalues.shape[2]-1), dtype=np.float64)
        #cdef double [:,:,:,:] ldos_energies = np.zeros(shape=(num_energies, self.eigenvalues.shape[0], self.eigenvalues.shape[1]-1, self.eigenvalues.shape[2]-1), dtype=np.float64)
        cdef double [:,:] ldos #= np.zeros(shape=(num_latticesites, num_energies), dtype=np.float64) #shape=(num_latticesites, num_energies)

        with gil:
            print('Calculating with four loops of size (%i, %i, %i, %i). Totaling %3g ops.' % \
                  (eigenvectors.shape[1], eigenvectors.shape[2], num_energies, num_latticesites, eigenvectors.shape[1]*eigenvectors.shape[2]*num_energies*num_latticesites))
            Es = self.energy_vec(min_e, max_e, resolution)
            ldos = np.zeros(shape=(self.L_x, num_energies), dtype=np.float64) #shape=(num_latticesites, num_energies)


        #cdef long double [:,:,:] pos_e_diff = self.eigenvalues[:, 1:, 1:] #/ 2

        cdef double us
        cdef double eng
        cdef double pos_ldos

        cdef int ii, ei, l, k, j, i
        #for ii in range(num_latticesites):
        #    us = conj(self.eigenvectors[4 * ii, :, 1:, 1:])*self.eigenvectors[4 * ii, :, 1:, 1:] + conj(self.eigenvectors[4 * ii + 1, :, 1:, 1:])*self.eigenvectors[4 * ii + 1, :, 1:, 1:] #spin opp + spin ned

        for l in range(self.L_z-1): #4*L_x is the entire dimention
            for k in range(self.L_y-1):
                for j in range(4 * self.L_x):
                    for i in prange(self.L_x, nogil=True):
                        us = pow(abs(eigenvectors[4 * i, j, k, l]),2) + pow(abs(eigenvectors[4 * i + 1, j, k, l]),2) #spin opp + spin ned

                        for ei in range(num_energies):
                            #eng = Es[ei]
                            pos_ldos = coeff * exp(-pow((eigenvalues[j,k,l] - Es[ei]), 2) / pow(sigma*sqrt(2), 2))
                            ldos[i, ei] += us * pos_ldos
        # need Es as well
        return ldos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ldos_from_problem(self, double resolution, double kernel_size, double min_E, double max_E):
        self.ldos, self.energies = self.local_density_of_states(resolution, kernel_size, min_E, max_E, self.eigenvectors, self.eigenvalues)
        return np.asarray(self.ldos), np.asarray(self.energies)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double compute_energy(self, bool N):
        # Compute the Free energy as in Linas Master Thisis

        cdef cnp.ndarray[cnp.complex128_t, ndim=1] delta_array = np.multiply(self.U_array, self.F_matrix[:])

        # u-term. Cant do if U = 0 in the region
        cdef cnp.ndarray[cnp.float64_t, ndim=2] U_index = np.where(np.array(self.U_array) != 0.0)
        cdef double U_energy = 0.0

        cdef int u
        for u in U_index[0]:
            U_energy += np.abs(delta_array[u])**2 / self.U_array[u]

        cdef double H_0 = U_energy * self.L_y * self.L_z  #- epsilon_energy
        cdef double F = H_0 - (1 / self.beta) * np.sum(np.log(1 + np.exp(-np.multiply(self.beta, self.eigenvalues[:, 1:, 1:]) / 2)))
        return F

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef forcePhaseDifference(self):
        cdef cnp.complex128_t I = 1.0j
        #cdef double complex [:] phaseDiff = np.exp(I * (-self.phase), dtype=np.complex128)
        cdef cnp.complex128_t phase_plus = np.exp(I * (-1 * self.phase), dtype=np.complex128)         #   SC_0

        #print("forcePhaseDiff 1")
        self.F_matrix[0] = np.abs(self.F_matrix[0])* phase_plus
        #print("forcePhaseDiff 2")
        self.F_matrix[self.L_x - 1] = np.abs(self.F_matrix[self.L_x - 1])
        #print("forcePhaseDiff 3")
        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef current_along_lattice(self):
        cdef double complex I = 1.0j
        cdef cnp.ndarray current = np.zeros(self.L_x - 1, dtype=np.complex128)
        #cdef long double [:,:,:] tanh_coeff = 1 / (np.exp(np.multiply(self.beta, self.eigenvalues[:,:,:])) + 1) / (self.L_y * self.L_z) # 1/(system.L_y*system.L_z) *(1-np.tanh(system.beta * system.eigenvalues / 2)) #-
        cdef cnp.float64_t tanh_coeff

        cdef double t = self.t_x

        cdef int xi_ii
        cdef int xi_minus
        cdef int xi_pluss

        cdef cnp.complex128_t B_pluss
        cdef cnp.complex128_t B_minus
        cdef cnp.complex128_t C_pluss
        cdef cnp.complex128_t C_minus

        cdef int ix, j, k, l
        cdef int lower = self.L_sc_0
        cdef int upper = self.L_sc_0 + self.L_soc

        for l in range(self.L_z-1): #4*L_x is the entire dimention
            for k in range(self.L_y-1):
                for j in range(4 * self.L_x):
                    #for i in range(self.L_x):
                    tanh_coeff = 1.0 / (np.exp(self.eigenvalues[j,k,l] * self.beta, dtype=np.float64) + 1)
                    tanh_coeff /= self.L_y * self.L_z # 1/(system.L_y*system.L_z) *(1-np.tanh(system.beta * system.eigenvalues / 2)) #-


                    for ix in range(1, len(current)):  # -1 because it doesnt give sense to check last point for I+
                        xi_ii = 0
                        xi_minus = 0
                        xi_pluss = 0
                        if (lower <= ix < upper): #check if both i and i are inside soc material
                            xi_ii = 1
                        if (lower <= ix < upper) and (lower <= ix+1 < upper): #check if both i and i+1 are inside soc material
                            xi_pluss = 1
                        if (lower <= ix < upper) and (lower <= ix-1 < upper): #check if both i and i-1 are inside soc material
                            xi_minus = 1


                        B_pluss = I / 4 * (self.alpha_R_x_array[ix, 1] - self.alpha_R_x_array[ix, 2]) * (1+xi_ii)
                        B_minus = I / 4 * (self.alpha_R_x_array[ix-1, 1] - self.alpha_R_x_array[ix-1, 2]) * (1+xi_minus)
                        C_pluss = - I / 4 * (self.alpha_R_x_array[ix+1, 1] - self.alpha_R_x_array[ix+1, 2]) * (1+xi_pluss)
                        C_minus = - I / 4 * (self.alpha_R_x_array[ix, 1] - self.alpha_R_x_array[ix, 2]) * (1+xi_ii)

                        # ---- Hopping x+ (imag)----#
                        #:
                        current[ix] += 2 * t * tanh_coeff * (np.conj(self.eigenvectors[4 * ix, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix + 1), j, k, l])  # * (np.exp(1.0j * system.ky_array[1:]) * np.exp(1.0j * system.kz_array[1:])))) #sigma = opp
                        current[ix] -= 2 * t * tanh_coeff * (np.conj(self.eigenvectors[4 * ix, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix - 1), j, k, l])  # * (np.exp(-1.0j * system.ky_array[1:]) * np.exp(-1.0j * system.kz_array[1:])))) #sigma = opp

                        # --- Rashba x+ (real)----#
                        #:
                        current[ix] += 1.0j * tanh_coeff * C_minus * (np.conj(self.eigenvectors[4 * ix, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix + 1), j, k, l]) # opp opp
                        current[ix] += 1.0j * tanh_coeff * C_pluss * (np.conj(self.eigenvectors[4 * (ix + 1), j, k, l], dtype=np.complex128) * self.eigenvectors[4 * ix, j, k, l]) # opp opp
                        current[ix] -= 1.0j * tanh_coeff * B_minus * (np.conj(self.eigenvectors[4 * (ix - 1), j, k, l], dtype=np.complex128) * self.eigenvectors[4 * ix, j, k, l])  # opp opp
                        current[ix] -= 1.0j * tanh_coeff * B_pluss * (np.conj(self.eigenvectors[4 * ix, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix - 1), j, k, l])  # opp opp


                        current[ix] += 1.0j * tanh_coeff * C_minus * (np.conj(self.eigenvectors[4 * ix + 1, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix + 1) + 1, j, k, l]) #ned ned
                        current[ix] += 1.0j * tanh_coeff * C_pluss * (np.conj(self.eigenvectors[4 * (ix + 1) + 1, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * ix + 1, j, k, l]) #ned ned
                        current[ix] -= 1.0j * tanh_coeff * B_minus * (np.conj(self.eigenvectors[4 * (ix - 1) + 1, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * ix + 1, j, k, l])  # ned ned
                        current[ix] -= 1.0j * tanh_coeff * B_pluss * (np.conj(self.eigenvectors[4 * ix + 1, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix - 1) + 1, j, k, l])  # ned ned

                        #:
                        current[ix] += 1.0j * tanh_coeff * C_minus * (np.conj(self.eigenvectors[4 * ix, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix + 1) + 1, j, k, l]) # opp ned
                        current[ix] += 1.0j * tanh_coeff * C_pluss * (np.conj(self.eigenvectors[4 * (ix + 1), j, k, l], dtype=np.complex128) * self.eigenvectors[4 * ix + 1, j, k, l]) # opp ned
                        current[ix] -= 1.0j * tanh_coeff * B_minus * (np.conj(self.eigenvectors[4 * (ix - 1), j, k, l], dtype=np.complex128) * self.eigenvectors[4 * ix + 1, j, k, l])  # opp ned
                        current[ix] -= 1.0j * tanh_coeff * B_pluss * (np.conj(self.eigenvectors[4 * ix, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix - 1) + 1, j, k, l])  # opp ned


                        current[ix] += 1.0j * tanh_coeff * C_minus * (np.conj(self.eigenvectors[4 * ix + 1, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix + 1), j, k, l]) # ned opp
                        current[ix] += 1.0j * tanh_coeff * C_pluss * (np.conj(self.eigenvectors[4 * (ix + 1) + 1, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * ix, j, k, l]) # ned opp
                        current[ix] -= 1.0j * tanh_coeff * B_minus * (np.conj(self.eigenvectors[4 * (ix - 1) + 1, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * ix, j, k, l])  # ned opp
                        current[ix] -= 1.0j * tanh_coeff * B_pluss * (np.conj(self.eigenvectors[4 * ix + 1, j, k, l], dtype=np.complex128) * self.eigenvectors[4 * (ix - 1), j, k, l])  # ned opp

        return current[:]


