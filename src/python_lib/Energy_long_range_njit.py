import green_function_njit as g
import numpy as np
from numpy import exp, cos, sin
import Sigma_Gamma_func_njit
from math import floor
from random import random
import numba as nb

@nb.njit()
def Energy_long_k_nonzero(POSCAR, RBM_p, K_set_shape, K_set, L_x, L_y, L_z, eps_0, eps_1, eps_2, eps_3, S, z_list, equal_cell, equal_size, equal_cell_room_num):
    NUM_particle = np.shape(POSCAR)[0]
    Energy_long_k_nonzero_split_val = 0
    Energy_long_k_nonzero_nonsplit_val = 0
    for l in range(RBM_p):
        k_p_num = floor(K_set_shape * random())
        k_p = K_set[k_p_num]
        Energy_long_k_nonzero_split_val +=  - (S/RBM_p) * Sigma_Gamma_func_njit.Sigma_Gamma_func_s(k_p, eps_1, eps_2, eps_3, POSCAR, L_z) / (2 * eps_0 * eps_2 * L_x * L_y)
        Energy_long_k_nonzero_nonsplit_val +=  - (S/RBM_p) * Sigma_Gamma_func_njit.Sigma_Gamma_func_ns(k_p, POSCAR, z_list) / (2 * eps_0 * eps_2 * L_x * L_y)
    
    Energy_long_k_nonzero = Energy_long_k_nonzero_split_val + Energy_long_k_nonzero_nonsplit_val

    return Energy_long_k_nonzero

@nb.njit()
def Energy_long_kzero(POSCAR, RBM_p, K_set_shape, K_set, L_x, L_y, L_z, eps_0, eps_1, eps_2, eps_3, S, z_list, equal_cell, equal_size, equal_cell_room_num):
    NUM_particle = np.shape(POSCAR)[0]
    
    Q_a = np.zeros(NUM_particle)
    Q_b = np.zeros(NUM_particle)

    Q_a[0] = 0
    Q_b[0] = - POSCAR[z_list[0]][0]

    for i in range(0, NUM_particle - 1):
        Q_a[i + 1] = Q_a[i] + POSCAR[z_list[i]][0]
        Q_b[i + 1] = Q_b[i] - POSCAR[z_list[i + 1]][0]

    Energy_long_kzero_val = 0
    for i in range(0, NUM_particle):
        Energy_long_kzero_val += 2 * POSCAR[z_list[i]][0] * POSCAR[z_list[i]][3] * (Q_a[i] - Q_b[i])

    Energy_long_kzero_val = - Energy_long_kzero_val / (4 * eps_2 * eps_0 * L_x * L_y)

    return Energy_long_kzero_val


def Energy_long(POSCAR, INCAR, CELL_LIST):
    RBM_p = INCAR['RBM_p']
    K_set = INCAR['K_set']
    K_set_shape = np.shape(K_set)[0]
    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    L_z = INCAR['L_z']
    eps_0 = INCAR['eps_0']
    eps_1 = INCAR['eps_1']
    eps_2 = INCAR['eps_2']
    eps_3 = INCAR['eps_3']
    NUM_particle = np.shape(POSCAR)[0]
    S = INCAR['Total_P']

    Z_Cell = CELL_LIST['Z_Cell']
    z_list = np.full(NUM_particle, 0)
    mark = 0
    for room in Z_Cell:
        for member in room:
            z_list[mark] = member
            mark += 1
            
    z_0 = POSCAR[int(z_list[0])][3]
    equal_cell = np.full([NUM_particle, NUM_particle], 9999)
    equal_size = np.full(NUM_particle, 0)
    equal_cell[0][0] = int(z_list[0])
    equal_cell_num = 0
    z_pre = z_0

    equal_cell_num = 0
    equal_room_num = 1
    room_flag = 1
    for i in range(1, NUM_particle):
        z_mem = int(z_list[i])
        z_mem_pos = POSCAR[z_mem][3]
        if z_mem_pos > z_pre:
            equal_size[equal_cell_num] = equal_room_num
            equal_cell_num += 1
            equal_cell[equal_cell_num][0] = z_mem
            equal_room_num += 1
            room_flag = 1
            z_pre = z_mem_pos
        else:
            equal_cell[equal_cell_num][room_flag] = z_mem
            equal_room_num += 1
            room_flag += 1
        equal_size[equal_cell_num] = equal_room_num

    equal_cell_room_num = equal_cell_num + 1

    Energy_long_k_nonzero_val = Energy_long_k_nonzero(POSCAR, RBM_p, K_set_shape, K_set, L_x, L_y, L_z, eps_0, eps_1, eps_2, eps_3, S, z_list, equal_cell, equal_size, equal_cell_room_num)
    Energy_long_kzero_val = Energy_long_kzero(POSCAR, RBM_p, K_set_shape, K_set, L_x, L_y, L_z, eps_0, eps_1, eps_2, eps_3, S, z_list, equal_cell, equal_size, equal_cell_room_num)

    Energy_long_val = Energy_long_kzero_val + Energy_long_k_nonzero_val
    
    return Energy_long_val
