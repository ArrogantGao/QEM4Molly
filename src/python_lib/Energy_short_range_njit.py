import green_function_njit as g
import numpy as np
from Bessel import Bessel0, Bessel1
from numpy import sqrt, pi, exp
from neighbor_check import neighbor_check
import numba as nb

@nb.njit()
def Gauss_Ledendra_int(func, Gauss_para, k_f, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha):
    Gauss_order = np.shape(Gauss_para)[0]
    SUM = 0
    for i in range(Gauss_order):
        s_i = Gauss_para[i][0] * k_f/2 + k_f/2
        w_i = Gauss_para[i][1]
        t = w_i * (k_f/2) * func(s_i, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)
        SUM += t
    return SUM

@nb.njit()
def gauss_charge_intcore(k, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha): #func_para = [INCAR, POSCAR, j, i]
    gauss_chare_intcore_val = g.Gamma_func(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- k**2/ (4 * alpha)) * Bessel0(k * rho_ji)
    return gauss_chare_intcore_val

@nb.njit()
def point_charge_intcore(k, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha): #func_para = [INCAR, POSCAR, j, i]
    gauss_chare_intcore_val =  g.Gamma_funca(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)  * Bessel0(k * rho_ji)
    return gauss_chare_intcore_val

@nb.njit()
def rho_cal(x_1, y_1, x_2, y_2):
    return sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)

@nb.njit()
def Energy_short_other_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha):

    Energy_short_other_val_j = 0
    
    i = 0
    while np_neighbor_list_j[i][0] != 9999:
        q_i, x_i, y_i, z_i ,rho_ji = np_neighbor_list_j[i]

        if rho_ji == 0:
            Energy_short_other_val_j = 0
        else:
            Energy_short_other_ji_gauss = Gauss_Ledendra_int(gauss_charge_intcore, Gauss_para_1, k_f1, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

            Energy_short_other_ji_point_a = Gauss_Ledendra_int(point_charge_intcore, Gauss_para_2, k_f2, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

            Energy_short_other_ji_point_b = 0
            Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
            z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
            gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
            for l in [0, 1, 2, 3]:
                Energy_short_other_ji_point_b += Gamma_para_val[l] * gamma_2_val / sqrt(z_para_val[l]**2 + rho_ji**2)

            Energy_short_other_ji_val = (q_i * q_j / (4 * pi * eps_2 * eps_0)) * (Energy_short_other_ji_gauss - Energy_short_other_ji_point_a + Energy_short_other_ji_point_b)

            Energy_short_other_val_j += Energy_short_other_ji_val
        i += 1

    return Energy_short_other_val_j

@nb.njit()
def Energy_short_self_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha):

    rho_ji = 0

    Energy_short_self_j_gauss = Gauss_Ledendra_int(gauss_charge_intcore, Gauss_para_1, k_f1, eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha)

    Energy_short_self_j_point_a = Gauss_Ledendra_int(point_charge_intcore, Gauss_para_2, k_f2, eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha)

    Energy_short_self_j_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    for l in [0, 2, 3]:
        Energy_short_self_j_point_b += Gamma_para_val[l] * gamma_2_val / (-z_para_val[l])

    Energy_short_self_val_j = (q_j**2/(4 * pi * eps_2 * eps_0)) * (Energy_short_self_j_gauss - Energy_short_self_j_point_a + Energy_short_self_j_point_b)

    return Energy_short_self_val_j

@nb.njit()
def Energy_short_cal(np_neighbor_list, eps_0, eps_1, eps_2, eps_3, L_z, alpha, POSCAR, Gauss_para_1, Gauss_para_2, k_f1, k_f2):

    NUM_particle = np.shape(POSCAR)[0]
    Energy_short_val = 0

    for j in range(NUM_particle):
        q_j, x_j, y_j, z_j = POSCAR[j]
        np_neighbor_list_j = np_neighbor_list[j]

        Energy_short_self_j_val = Energy_short_self_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha)

        Energy_short_other_j_val = Energy_short_other_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha)

        Energy_short_val += Energy_short_other_j_val + Energy_short_self_j_val

    return Energy_short_val

def Energy_short(POSCAR, INCAR, CELL_LIST):
    NUM_particle = INCAR['NUM_particle']
    eps_0 = INCAR['eps_0']
    eps_1 = INCAR['eps_1']
    eps_2 = INCAR['eps_2']
    eps_3 = INCAR['eps_3']
    L_z = INCAR['L_z']
    alpha = INCAR['alpha']

    Gauss_para_1 = INCAR['Gauss_para1']
    Gauss_para_2 = INCAR['Gauss_para2']
    k_f1 = INCAR['k_f1']
    k_f2 = INCAR['k_f2']

    np_neighbor_list = np.full([NUM_particle, NUM_particle, 5], 9999.0)
    for j in range(NUM_particle):
        q_j, x_j, y_j, z_j = POSCAR[j]
        neighbor_list = neighbor_check(j, POSCAR, INCAR, CELL_LIST)
        l = 0
        for neighbor_member in neighbor_list:
            q_i, x_i, y_i, z_i ,i = neighbor_member
            rho_ji = rho_cal(x_j, y_j, x_i, y_i)
            np_neighbor_list[j][l] = np.array([q_i, x_i, y_i, z_i, rho_ji], dtype = 'float_')
            l += 1

    Energy_short_val = Energy_short_cal(np_neighbor_list, eps_0, eps_1, eps_2, eps_3, L_z, alpha, POSCAR, Gauss_para_1, Gauss_para_2, k_f1, k_f2)

    return Energy_short_val