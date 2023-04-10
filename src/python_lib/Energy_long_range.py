# in this script we will calculate the long range Energy term
# the calculatioin should be divided in to three part
# 1. k = 0 term, call it Energy_long_kzero
# 2. k > 0, split term, Energy_long_split
# 3. k > 0, non-split term, Energy_long_nonsplit

import green_function as g
from Gauss_quadrature import Gauss_Ledendra_int
import numpy as np
from scipy.special import iv, jv
from numpy import sqrt, pi, exp
from neighbor_check import neighbor_check, rho_cal
from k_space_summation import k_space_summation
from Sigma_Gamma_func import Sigma_Gamma_func_s, Sigma_Gamma_func_ns

def Energy_long_kzero(POSCAR, INCAR, CELL_LIST):
    Z_Cell = CELL_LIST['Z_Cell']
    z_list = []
    for room in Z_Cell:
        for member in room:
            z_list.append(member)
    eps_0 = INCAR['eps_0']
    eps_2 = INCAR['eps_2']
    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    NUM_particle = INCAR['NUM_particle']

    Q_a = np.zeros(NUM_particle)
    Q_b = np.zeros(NUM_particle)

    Q_a[0] = 0
    Q_b[0] = - POSCAR[z_list[0]][0]

    for i in range(0, NUM_particle - 1):
        Q_a[i + 1] = Q_a[i] + POSCAR[z_list[i]][0]
        Q_b[i + 1] = Q_b[i] - POSCAR[z_list[i + 1]][0]

    #print(Q_a, Q_b)

    Energy_long_kzero_val = 0
    for i in range(0, NUM_particle):
        Energy_long_kzero_val += 2 * POSCAR[z_list[i]][0] * POSCAR[z_list[i]][3] * (Q_a[i] - Q_b[i])

    Energy_long_kzero_val = - Energy_long_kzero_val / (4 * eps_2 * eps_0 * L_x * L_y)

    return Energy_long_kzero_val


def Energy_long_k_nonzero(POSCAR, INCAR, CELL_LIST):

    eps_0 = INCAR['eps_0']
    eps_2 = INCAR['eps_2']
    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    NUM_particle = INCAR['NUM_particle']

    func_para = [POSCAR, INCAR, CELL_LIST]
    Energy_long_k_nonzero_split_val = - k_space_summation(Sigma_Gamma_func_s, func_para, INCAR) / (2 * eps_0 * eps_2 * L_x * L_y)
    Energy_long_k_nonzero_nonsplit_val = - k_space_summation(Sigma_Gamma_func_ns, func_para, INCAR) / (2 * eps_0 * eps_2 * L_x * L_y)

    Energy_long_k_nonzero_val = Energy_long_k_nonzero_split_val + Energy_long_k_nonzero_nonsplit_val

    return Energy_long_k_nonzero_val


def Energy_long(POSCAR, INCAR, CELL_LIST):
    Energy_long_kzero_val = Energy_long_kzero(POSCAR, INCAR, CELL_LIST)
    Energy_long_k_nonzero_val = Energy_long_k_nonzero(POSCAR, INCAR, CELL_LIST)

    Energy_long_val = Energy_long_kzero_val + Energy_long_k_nonzero_val
    
    return Energy_long_val