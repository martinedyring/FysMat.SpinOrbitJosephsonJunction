"""
This script contains all extra function solve_hamiltonian.py needs
"""

#   Define index for the pairing amplitude functions: d: down, u: up
#   Expecting x_pluss to be equal to x_minus, so I am only considering x_pluss
#   Expecting ud_x to be equal to du_x, so I am only considering ud_x. Similar for y.

cdef int idx_F_i
cdef int idx_F_ij_x_pluss
cdef int idx_F_ij_x_minus
cdef int idx_F_ij_y_pluss
cdef int idx_F_ij_y_minus
cdef int idx_F_ij_s

cdef int num_idx_F_i

