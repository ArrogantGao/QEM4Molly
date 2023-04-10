import green_function_njit as g
import numpy as np
from numpy import exp, cos, sin
import Sigma_Gamma_func_njit
from math import floor
from random import random
import numba as nb

@nb.njit()
def Force_long_cal(POSCAR, RBM_p, K_set_shape, K_set, L_x, L_y, L_z, eps_0, eps_1, eps_2, eps_3, S, z_list, equal_cell, equal_size, equal_cell_room_num):
    NUM_particle = np.shape(POSCAR)[0]
    Force_long_x_k_sum_val = np.zeros(NUM_particle)
    Force_long_y_k_sum_val = np.zeros(NUM_particle)
    Force_long_z_k_sum_val = np.zeros(NUM_particle)

    for l in range(RBM_p):
        k_p_num = floor(K_set_shape * random())
        k_p = K_set[k_p_num]
        F_x_sum1_val, F_y_sum1_val, F_z_sum1_val = Sigma_Gamma_func_njit.F_long_sum_1(k_p, eps_1, eps_2, eps_3, POSCAR, L_z)
        F_x_sum2_val, F_y_sum2_val = Sigma_Gamma_func_njit.F_long_xy_sum_2(k_p, POSCAR, z_list)
        F_z_sum0_val, F_z_sum2_val = Sigma_Gamma_func_njit.F_long_z_sum_2(k_p, POSCAR, z_list, equal_cell, equal_size, equal_cell_room_num)
        F_z_self_val = Sigma_Gamma_func_njit.F_long_z_self_sum(k_p, POSCAR, eps_1, eps_2, eps_3, L_z)
        
        for j in range(NUM_particle):
            Force_long_x_k_sum_val[j] += + (1/(L_x * L_y * eps_0 * eps_2)) * (F_x_sum1_val[j] + F_x_sum2_val[j]) * (S/RBM_p)
            Force_long_y_k_sum_val[j] += + (1/(L_x * L_y * eps_0 * eps_2)) * (F_y_sum1_val[j] + F_y_sum2_val[j]) * (S/RBM_p)
            Force_long_z_k_sum_val[j] += + (1/(L_x * L_y * eps_0 * eps_2)) * (F_z_sum1_val[j] - F_z_sum2_val[j] + F_z_self_val[j]) * (S/RBM_p)

    Force_long_z_sum0_val = F_z_sum0_val 
    
    Force_long_x = Force_long_x_k_sum_val
    Force_long_y = Force_long_y_k_sum_val

    Force_long_z = np.zeros(NUM_particle)
    for i in range(NUM_particle):
        Force_long_z[i] = Force_long_z_k_sum_val[i] + (1/(2 * L_x * L_y * eps_0 * eps_2)) * Force_long_z_sum0_val[i]

    return Force_long_x, Force_long_y, Force_long_z


def Force_long(POSCAR, INCAR, CELL_LIST):
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

    Force_long_x, Force_long_y, Force_long_z = Force_long_cal(POSCAR, RBM_p, K_set_shape, K_set, L_x, L_y, L_z, eps_0, eps_1, eps_2, eps_3, S, z_list, equal_cell, equal_size, equal_cell_room_num)

    return Force_long_x, Force_long_y, Force_long_z