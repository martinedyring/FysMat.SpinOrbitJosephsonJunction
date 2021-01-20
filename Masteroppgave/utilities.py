"""
This script contains all extra function solve_hamiltonian.py needs
"""

#   Define index for the pairing amplitude functions: d: down, u: up
#   Expecting x_pluss to be equal to x_minus, so I am only considering x_pluss
#   Expecting ud_x to be equal to du_x, so I am only considering ud_x. Similar for y.

idx_F_i = 0
idx_F_ij_x_pluss = 1
idx_F_ij_x_minus = 2
idx_F_ij_y_pluss = 3
idx_F_ij_y_minus = 4
idx_F_ij_s = 5

num_idx_F_i = 6

#   Label name for each component in F-matrix
label_F_matrix = [
    r'$F_{ii}$',
    r'$F_{i}^{x+}$',
    r'$F_{i}^{x-}$',
    r'$F_{i}^{y+}$',
    r'$F_{i}^{y-}$',
    r'$F_{s_i}$'
]

