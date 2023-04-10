import green_function as g
import numpy as np
from numpy import exp, cos, sin
import Sigma_Gamma_func
from math import floor
from random import random


def Force_long(POSCAR, INCAR, CELL_LIST):
    RBM_p = INCAR['RBM_p']
    K_set = INCAR['K_set']
    K_set_shape = np.shape(K_set)[0]
    func_para = [POSCAR, INCAR, CELL_LIST]
    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    eps_0 = INCAR['eps_0']
    eps_2 = INCAR['eps_2']
    NUM_particle = np.shape(POSCAR)[0]
    S = INCAR['Total_P']

    Z_Cell = CELL_LIST['Z_Cell']
    z_list = np.zeros(NUM_particle)
    mark = 0
    for room in Z_Cell:
        for member in room:
            z_list[mark] = member
            mark += 1

    Force_long_x_k_sum_val = np.zeros(NUM_particle)
    Force_long_y_k_sum_val = np.zeros(NUM_particle)
    Force_long_z_k_sum_val = np.zeros(NUM_particle)


    for l in range(RBM_p):
        k_p_num = floor(K_set_shape * random())
        k_p = K_set[k_p_num]
        F_x_sum1_val, F_y_sum1_val, F_z_sum1_val = Sigma_Gamma_func.F_long_sum_1(k_p, func_para)
        F_x_sum2_val, F_y_sum2_val = Sigma_Gamma_func.F_long_xy_sum_2(k_p, func_para)
        F_z_sum2_val = Sigma_Gamma_func.F_long_z_sum_2(k_p, func_para)
        #Force_long_z_self_val = Sigma_Gamma_func.Force_long_z_self(k_p, func_para)
        
        for j in range(NUM_particle):
            Force_long_x_k_sum_val[j] += + (1/(L_x * L_y * eps_0 * eps_2)) * (F_x_sum1_val[j] + F_x_sum2_val[j]) * (S/RBM_p)
            Force_long_y_k_sum_val[j] += + (1/(L_x * L_y * eps_0 * eps_2)) * (F_y_sum1_val[j] + F_y_sum2_val[j]) * (S/RBM_p)
            Force_long_z_k_sum_val[j] += - (1/(L_x * L_y * eps_0 * eps_2)) * (F_z_sum1_val[j] + F_z_sum2_val[j]) * (S/RBM_p)

    Force_long_z_sum0_val = Sigma_Gamma_func.F_long_z_sum_0(func_para)    
    
    Force_long_x = Force_long_x_k_sum_val
    Force_long_y = Force_long_y_k_sum_val

    Force_long_z = np.zeros(NUM_particle)
    for i in range(NUM_particle):
        Force_long_z[i] = Force_long_z_k_sum_val[i] + (1/(2 * L_x * L_y * eps_0 * eps_2)) * Force_long_z_sum0_val[i]

    return Force_long_x, Force_long_y, Force_long_z