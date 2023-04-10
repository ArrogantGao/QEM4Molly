import green_function as g
from Gauss_quadrature import Gauss_Ledendra_int
import numpy as np
from scipy.special import iv, jv
from numpy import sqrt, pi, exp
from neighbor_check import neighbor_check, rho_cal

def gauss_charge_Force_rho_intcore(k, func_para): #func_para = [INCAR, POSCAR, j, i]
    INCAR, POSCAR, j, i, rho_ji = func_para
    gauss_chare_Force_rho_intcore_val = k * g.Gamma_func(k, INCAR, POSCAR, j, i) * exp(- k**2/ (4 * INCAR['alpha'])) * jv(1, k * rho_ji)
    return gauss_chare_Force_rho_intcore_val

def point_charge_Force_rho_intcore(k, func_para):
    INCAR, POSCAR, j, i, rho_ji = func_para
    gauss_chare_Force_rho_intcore_val = k * g.Gamma_funca(k, INCAR, POSCAR, j, i) * jv(1, k * rho_ji)
    return gauss_chare_Force_rho_intcore_val

def gauss_charge_Force_z_intcore(k, func_para): #func_para = [INCAR, POSCAR, j, i]
    INCAR, POSCAR, j, i, rho_ji = func_para
    gauss_chare_Force_z_intcore_val = g.partial_Gamma_func(k, INCAR, POSCAR, j, i) * exp(- k**2/ (4 * INCAR['alpha'])) * jv(0, k * rho_ji)
    return gauss_chare_Force_z_intcore_val

def point_charge_Force_z_intcore(k, func_para):
    INCAR, POSCAR, j, i, rho_ji = func_para
    gauss_chare_Force_z_intcore_val = g.partial_Gamma_funca(k, INCAR, POSCAR, j, i) * jv(0, k * rho_ji)
    return gauss_chare_Force_z_intcore_val

def Force_short_rho(POSCAR, INCAR, CELL_LIST):
    NUM_particle = INCAR['NUM_particle']
    Force_short_x_val = np.zeros(NUM_particle)
    Force_short_y_val = np.zeros(NUM_particle)


    for j in range(NUM_particle):
        q_j, x_j, y_j, z_j = POSCAR[j]

        neighbor_list = neighbor_check(j, POSCAR, INCAR, CELL_LIST)
        #Here we will calculate the terms in Force_short_rho_val[j]
        for neighbor_member in neighbor_list:
            q_i, x_i, y_i, z_i ,i = neighbor_member
            rho_ji = rho_cal([q_j, x_j, y_j, z_j], [q_i, x_i, y_i, z_i])

            if rho_ji == 0:
                Force_short_x_val[j] += 0
                Force_short_y_val[j] += 0
            else:

                #gauss charge part
                func = gauss_charge_Force_rho_intcore
                Gauss_para = INCAR['Gauss_para1']
                k_f = INCAR['k_f1']
                func_para = [INCAR, POSCAR, j, i, rho_ji]
                Force_short_rho_other_ji_gauss = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

                #point charge part a
                func = point_charge_Force_rho_intcore
                Gauss_para = INCAR['Gauss_para2']
                k_f = INCAR['k_f2']
                func_para = [INCAR, POSCAR, j, i, rho_ji]
                Force_short_rho_other_ji_point_a = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

                #point charge part b
                Force_short_rho_other_ji_point_b = 0
                Gamma_para_val = g.Gamma_para(0, INCAR, POSCAR, j, i)
                z_para_val = g.z_para(0, INCAR, POSCAR, j, i)
                gamma_2_val = g.gamma_2(0, INCAR, POSCAR, j, i)
                for l in [0, 1, 2, 3]:
                    Force_short_rho_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * rho_ji / (z_para_val[l]**2 + rho_ji**2)**1.5


                Force_short_rho_other_ji_val = (q_i * q_j / (2 * pi * INCAR['eps_2'] * INCAR['eps_0'])) * (Force_short_rho_other_ji_gauss - Force_short_rho_other_ji_point_a + Force_short_rho_other_ji_point_b)

                Force_short_x_other_ji_val = ((x_j - x_i)/rho_ji) * Force_short_rho_other_ji_val
                Force_short_y_other_ji_val = ((y_j - y_i)/rho_ji) * Force_short_rho_other_ji_val
                Force_short_x_val[j] += Force_short_x_other_ji_val
                Force_short_y_val[j] += Force_short_y_other_ji_val

    return Force_short_x_val, Force_short_y_val


def Force_short_z_other(POSCAR, INCAR, CELL_LIST):
    NUM_particle = INCAR['NUM_particle']
    Force_short_z_other_val = np.zeros(NUM_particle)

    for j in range(NUM_particle):
        q_j, x_j, y_j, z_j = POSCAR[j]

        neighbor_list = neighbor_check(j, POSCAR, INCAR, CELL_LIST)
        #Here we will calculate the terms in Force_short_z_other_val[j]
        for neighbor_member in neighbor_list:
            q_i, x_i, y_i, z_i ,i = neighbor_member
            rho_ji = rho_cal([q_j, x_j, y_j, z_j], [q_i, x_i, y_i, z_i])

            #gauss charge part
            func = gauss_charge_Force_z_intcore
            Gauss_para = INCAR['Gauss_para1']
            k_f = INCAR['k_f1']
            func_para = [INCAR, POSCAR, j, i, rho_ji]
            Force_short_z_other_ji_gauss = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

            #point charge part a
            func = point_charge_Force_z_intcore
            Gauss_para = INCAR['Gauss_para2']
            k_f = INCAR['k_f2']
            func_para = [INCAR, POSCAR, j, i, rho_ji]
            Force_short_z_other_ji_point_a = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

            #point charge part b
            Force_short_z_other_ji_point_b = 0
            Gamma_para_val = g.Gamma_para(0, INCAR, POSCAR, j, i)
            z_para_val = g.z_para(0, INCAR, POSCAR, j, i)
            gamma_2_val = g.gamma_2(0, INCAR, POSCAR, j, i)
            partial_para_val = g.partial_para(0, INCAR, POSCAR, j, i)
            for l in [0, 1, 2, 3]:
                Force_short_z_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l] * z_para_val[l] / (z_para_val[l]**2 + rho_ji**2)**1.5


            Force_short_z_other_ji = (q_i * q_j / (2 * pi * INCAR['eps_2'] * INCAR['eps_0'])) * ( - Force_short_z_other_ji_gauss + Force_short_z_other_ji_point_a + Force_short_z_other_ji_point_b)
            
            Force_short_z_other_val[j] += Force_short_z_other_ji

    return Force_short_z_other_val



def Force_short_z_self(POSCAR, INCAR, CELL_LIST):
    NUM_particle = INCAR['NUM_particle']
    Force_short_z_self_val = np.zeros(NUM_particle)

    for j in range(NUM_particle):
        q_j, x_j, y_j, z_j = POSCAR[j]
        rho_ji = 0

        #gauss charge part
        func = gauss_charge_Force_z_intcore
        Gauss_para = INCAR['Gauss_para1']
        k_f = INCAR['k_f1']
        func_para = [INCAR, POSCAR, j, j, rho_ji]
        Force_short_z_self_j_gauss = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

        #point charge part a
        func = point_charge_Force_z_intcore
        Gauss_para = INCAR['Gauss_para2']
        k_f = INCAR['k_f2']
        func_para = [INCAR, POSCAR, j, j, rho_ji]
        Force_short_z_self_j_point_a = Gauss_Ledendra_int(func, Gauss_para, k_f, func_para)

        #point charge part b
        Force_short_z_self_j_point_b = 0
        Gamma_para_val = g.Gamma_para(0, INCAR, POSCAR, j, j)
        z_para_val = g.z_para(0, INCAR, POSCAR, j, j)
        gamma_2_val = g.gamma_2(0, INCAR, POSCAR, j, j)
        partial_para_val = g.partial_para(0, INCAR, POSCAR, j, j)
        for l in [0, 2]:
            Force_short_z_self_j_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l]  / (z_para_val[l]**2)

        Force_short_z_self_j = (q_j**2/(2 * pi * INCAR['eps_2'] * INCAR['eps_0'])) * (- Force_short_z_self_j_gauss + Force_short_z_self_j_point_a - Force_short_z_self_j_point_b)

        Force_short_z_self_val[j] = Force_short_z_self_j
    
    return Force_short_z_self_val


def Force_short_z(POSCAR, INCAR, CELL_LIST):
    NUM_particle = INCAR['NUM_particle']

    Force_short_z_other_val = Force_short_z_other(POSCAR, INCAR, CELL_LIST)
    Force_short_z_self_val = Force_short_z_self(POSCAR, INCAR, CELL_LIST)

    Force_short_z_val = np.zeros(NUM_particle)
    for i in range(NUM_particle):
        Force_short_z_val[i] = Force_short_z_other_val[i] + Force_short_z_self_val[i]

    return Force_short_z_val

def Force_short(POSCAR, INCAR, CELL_LIST):

    Force_short_x_val, Force_short_y_val = Force_short_rho(POSCAR, INCAR, CELL_LIST)
    Force_short_z_val = Force_short_z(POSCAR, INCAR, CELL_LIST)

    return Force_short_x_val, Force_short_y_val, Force_short_z_val