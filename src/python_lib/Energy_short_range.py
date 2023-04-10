import green_function as g
from Gauss_quadrature import Gauss_Ledendra_int
import numpy as np
from scipy.special import iv, jv
from numpy import sqrt, pi, exp
from neighbor_check import neighbor_check

def rho_cal(pos_1, pos_2):
    q_1, x_1, y_1, z_1 = pos_1
    q_2, x_2, y_2, z_2 = pos_2
    delta_x = x_1 - x_2
    delta_y = y_1 - y_2
    rho = sqrt(delta_x**2 + delta_y**2)
    return rho


def gauss_charge_intcore(k, func_para): #func_para = [INCAR, POSCAR, j, i]
    INCAR, POSCAR, j, i, rho_ji = func_para
    gauss_chare_intcore_val = g.Gamma_func(k, INCAR, POSCAR, j, i) * exp(- k**2/ (4 * INCAR['alpha'])) * jv(0, k * rho_ji)
    return gauss_chare_intcore_val

def point_charge_intcore(k, func_para):
    INCAR, POSCAR, j, i, rho_ji = func_para
    gauss_chare_intcore_val = g.Gamma_funca(k, INCAR, POSCAR, j, i) * jv(0, k * rho_ji)
    return gauss_chare_intcore_val


def Energy_short_other(POSCAR, INCAR, CELL_LIST):

    Energy_short_other_val = 0

    for j in range(INCAR['NUM_particle']):
        q_j, x_j, y_j, z_j = POSCAR[j]

        neighbor_list = neighbor_check(j, POSCAR, INCAR, CELL_LIST)
        Energy_short_other_val_j = 0
        for neighbor_member in neighbor_list:
            q_i, x_i, y_i, z_i ,i = neighbor_member
            rho_ji = rho_cal([q_j, x_j, y_j, z_j], [q_i, x_i, y_i, z_i])

            #gauss charge part
            func = gauss_charge_intcore
            Gauss_para = INCAR['Gauss_para1']
            k_f = INCAR['k_f1']
            func_para = [INCAR, POSCAR, j, i, rho_ji]
            Energy_short_other_ji_gauss = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)
            
            #point charge part a
            func = point_charge_intcore
            Gauss_para = INCAR['Gauss_para2']
            k_f = INCAR['k_f2']
            func_para = [INCAR, POSCAR, j, i, rho_ji]
            Energy_short_other_ji_point_a = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

            #point charge part b
            Energy_short_other_ji_point_b = 0
            Gamma_para_val = g.Gamma_para(0, INCAR, POSCAR, j, i)
            z_para_val = g.z_para(0, INCAR, POSCAR, j, i)
            gamma_2_val = g.gamma_2(0, INCAR, POSCAR, j, i)
            for l in [0, 1, 2, 3]:
                Energy_short_other_ji_point_b += Gamma_para_val[l] * gamma_2_val / sqrt(z_para_val[l]**2 + rho_ji**2)
            
            Energy_short_other_ji_point_val = (q_i * q_j / (4 * pi * INCAR['eps_2'] * INCAR['eps_0'])) * (Energy_short_other_ji_gauss - Energy_short_other_ji_point_a + Energy_short_other_ji_point_b)

            Energy_short_other_val_j += Energy_short_other_ji_point_val

        Energy_short_other_val += Energy_short_other_val_j

    return Energy_short_other_val
            





def Energy_short_self(POSCAR, INCAR, CELL_LIST):
    Energy_short_self_val = 0
    # calculate the short range self ineteraction energy of particle j
    for j in range(INCAR['NUM_particle']):
        q_j, x_j, y_j, z_j = POSCAR[j]
        
        #gauss charge part
        func = gauss_charge_intcore
        Gauss_para = INCAR['Gauss_para1']
        k_f = INCAR['k_f1']
        func_para = [INCAR, POSCAR, j, j, 0]
        Energy_short_self_gauss = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

        #point charge part a
        func = point_charge_intcore
        Gauss_para = INCAR['Gauss_para2']
        k_f = INCAR['k_f2']
        func_para = [INCAR, POSCAR, j, j, 0]
        Energy_short_self_point_a = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

        #point charge part b
        Energy_short_self_point_b = 0
        Gamma_para_val = g.Gamma_para(0, INCAR, POSCAR, j, j)
        z_para_val = g.z_para(0, INCAR, POSCAR, j, j)
        gamma_2_val = g.gamma_2(0, INCAR, POSCAR, j, j)
        for l in [0, 2, 3]:
            Energy_short_self_point_b += Gamma_para_val[l] * gamma_2_val / (-z_para_val[l])

        Energy_short_self_val_j = (q_j**2/(4 * pi * INCAR['eps_2'] * INCAR['eps_0'])) * (Energy_short_self_gauss - Energy_short_self_point_a + Energy_short_self_point_b)

        Energy_short_self_val += Energy_short_self_val_j
        
    return Energy_short_self_val


def Energy_short(POSCAR, INCAR, CELL_LIST):
    Energy_short_other_val = Energy_short_other(POSCAR, INCAR, CELL_LIST)
    Energy_short_self_val = Energy_short_self(POSCAR, INCAR, CELL_LIST)

    Energy_short_val = Energy_short_other_val + Energy_short_self_val

    return Energy_short_val
