import current_phase_calculations
#import solve_for_shms_system_phase

import numpy as np
import matplotlib.pyplot as plt
from system_class import System
import solve_hamiltonian

"""
current_for_phase, phase_arr = current_phase_calculations.solve_for_shms_system_phase(alpha_array=np.array([0,0,0],dtype=np.float64),
                                                                                      max_num_iter=100,
                                                                                      tol=1e-2,
                                                                                      L_y=10,
                                                                                      L_z=10,
                                                                                      L_sc_0=5,
                                                                                      L_soc=0,
                                                                                      L_sc=5,
                                                                                      L_nc=5,
                                                                                      L_f=0,
                                                                                      mu_sc=0.9,
                                                                                      mu_nc=0.9,
                                                                                      mu_soc=0.9,
                                                                                      u_sc=-4.2,
                                                                                      beta=33.3)

print("current_for_phase = ", current_for_phase)
for i in range(len(current_for_phase)):
    print("current - phase : ", current_for_phase[i], phase_arr[i])

np.savez('shms_5_3_5_test.npz', phase_arr, current_for_phase)
#plt.plot(phase_arr, np.real(current_for_phase))
#"""

#"""
s = System(alpha_R_initial = np.array([0,0,0.5],dtype=np.float64),
           phase=0,
           L_y=30,
           L_z=30,
           L_sc_0=0,
           L_nc=25,
           L_f=0,
           L_sc=25,
           L_soc=0,
           mu_sc=0.9,
           mu_nc=0.9,
           mu_soc=0.9,
           u_sc=-4.2,
           beta=np.inf)

solve_hamiltonian.solve_system(system=s, max_num_iter=100, tol=1e-4, juction=False)

#ldos, energy_state = s.ldos_from_problem(0.1, 0.6, -6, 6) # resolution, sigma, min e, max e #0.01, 0.03, -6, 6


f_matrix = s.get_F_matrix()

eigen = s.get_eigenvalues()

# save the object after converge, such that we can look at it in jupyter lab
#np.savez('sc_dos_test.npz', ldos, energy_state)
np.savez('sc_fmatrix_test.npz', f_matrix)
np.savez('sc_energies_test.npz', eigen)
#plt.plot(phase_arr, np.real(current_for_phase))

#"""

## pickle the person
 # d = pickle.dumps(dave)


# unpickle the person
 # dave = pickle.loads(d)