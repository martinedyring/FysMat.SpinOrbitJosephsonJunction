import numpy as np


L_x = 100
L_y = 100
int_tol = 1*10^(-5)
KbT = 0.0
mu_s = -3.5
t_sc = 1.0
t_0 = 1.0
V = -2.5

def crate_lattice(L_x, L_y):
    # (2019) have decribed the lattice on pg. 23. Here there is a lattice of L_y  in y-dir, and 2*L_x in x-dir. (One L_x in each medium)

    lattice = np.array(())