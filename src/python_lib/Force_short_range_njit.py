import green_function_njit as g
import numpy as np
from Bessel import Bessel0, Bessel1
from numpy import sqrt, pi, exp, log
from neighbor_check import neighbor_check
import numba as nb
import Force_short_norm_njit as Fs_norm

@nb.njit()
def Gauss_Ledendra_int(func, Gauss_para, k_f, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha):
    G = 0
    s = Gauss_para[0]
    w = Gauss_para[1]
    N = np.shape(s)[0]
    for i in range(N):
        s_i = (k_f/2)*s[i] + (k_f/2)
        w_i = w[i]
        t = w_i * (k_f/2) * func(s_i, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)
        G += t
    return G

@nb.njit()
def gauss_charge_Force_rho_intcore(k, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha): #func_para = [INCAR, POSCAR, j, i]
    gauss_chare_Force_rho_intcore_val = k * g.Gamma_func(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- k**2/ (4 * alpha)) * Bessel1(k * rho_ji)
    return gauss_chare_Force_rho_intcore_val

@nb.njit()
def point_charge_Force_rho_intcore(k, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha):
    gauss_chare_Force_rho_intcore_val = k * g.Gamma_funca(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel1(k * rho_ji)
    return gauss_chare_Force_rho_intcore_val

@nb.njit()
def gauss_charge_Force_z_intcore(k, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha): #func_para = [INCAR, POSCAR, j, i]
    gauss_chare_Force_z_intcore_val = g.partial_Gamma_func(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- k**2/ (4 * alpha)) * Bessel0(k * rho_ji)
    return gauss_chare_Force_z_intcore_val

@nb.njit()
def point_charge_Force_z_intcore(k, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha):
    gauss_chare_Force_z_intcore_val = g.partial_Gamma_funca(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k * rho_ji)
    return gauss_chare_Force_z_intcore_val

@nb.njit()
def rho_cal(x_1, y_1, x_2, y_2):
    return sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)

@nb.njit()
def Force_rhos_ji(pos_info, INCAR, Gauss_para_1, Gauss_para_2, eps_0):
    q_j, q_i, x_j, x_i, y_j, y_i, z_j, z_i = pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR

    k_f1 = sqrt( - 4 * alpha * log(cutoff))
    k_f2 = - log(cutoff) / (2 * L_z)

    rho_ji = sqrt((x_j - x_i)**2 + (y_j - y_i)**2)

    Force_short_rho_other_ji_gauss = Gauss_Ledendra_int(gauss_charge_Force_rho_intcore, Gauss_para_1, k_f1, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

    Force_short_rho_other_ji_point_a = Gauss_Ledendra_int(point_charge_Force_rho_intcore, Gauss_para_2, k_f2, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

    Force_short_rho_other_ji_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    for l in np.array([0, 1, 2, 3]):
        Force_short_rho_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * rho_ji / (z_para_val[l]**2 + rho_ji**2)**1.5

    Force_short_rho_other_ji_val = (q_i * q_j / (2 * pi * eps_2 * eps_0)) * (Force_short_rho_other_ji_gauss - Force_short_rho_other_ji_point_a + Force_short_rho_other_ji_point_b)

    return Force_short_rho_other_ji_val

@nb.njit()
def Force_zs_ji(pos_info, INCAR, Gauss_para_1, Gauss_para_2, eps_0):
    q_j, q_i, x_j, x_i, y_j, y_i, z_j, z_i = pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR

    k_f1 = sqrt( - 4 * alpha * log(cutoff))
    k_f2 = - log(cutoff) / (2 * L_z)

    rho_ji = sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
    
    Force_short_z_other_ji_gauss = Gauss_Ledendra_int(gauss_charge_Force_z_intcore, Gauss_para_1, k_f1, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

    Force_short_z_other_ji_point_a = Gauss_Ledendra_int(point_charge_Force_z_intcore, Gauss_para_2, k_f2, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

    Force_short_z_other_ji_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    partial_para_val = g.partial_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    for l in np.array([0, 1, 2, 3]):
        Force_short_z_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l] * z_para_val[l] / (z_para_val[l]**2 + rho_ji**2)**1.5

    Force_short_z_other_ji_val = (q_i * q_j / (2 * pi * eps_2 * eps_0)) * ( - Force_short_z_other_ji_gauss + Force_short_z_other_ji_point_a + Force_short_z_other_ji_point_b)

    return Force_short_z_other_ji_val

@nb.njit()
def Force_zs_self_ji(pos_info, INCAR, Gauss_para_1, Gauss_para_2, eps_0):
    q_j, x_j, y_j, z_j = pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR

    k_f1 = sqrt( - 4 * alpha * log(cutoff))
    k_f2 = - log(cutoff) / (2 * L_z)
    rho_ji = 0
    
    Force_short_z_self_j_gauss = Gauss_Ledendra_int(gauss_charge_Force_z_intcore, Gauss_para_1, k_f1, eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha)

    Force_short_z_self_j_point_a = Gauss_Ledendra_int(point_charge_Force_z_intcore, Gauss_para_2, k_f2, eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha)

    Force_short_z_self_j_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    partial_para_val = g.partial_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    for l in [0, 2]:
        Force_short_z_self_j_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l]  / (z_para_val[l]**2)

    Force_short_z_self_j_val = (q_j * q_j / (2 * pi * eps_2 * eps_0)) * (- Force_short_z_self_j_gauss + Force_short_z_self_j_point_a - Force_short_z_self_j_point_b)

    return Force_short_z_self_j_val
    

@nb.njit()
def Force_short_rho_cal_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha, N_t, accuracy, eps, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    Force_short_x_other_j = 0
    Force_short_y_other_j = 0

    i = 0
    while np_neighbor_list_j[i][0] != 9999:
        q_i, x_i, y_i, z_i ,rho_ji = np_neighbor_list_j[i]
        
        if rho_ji == 0:
            Force_short_x_other_ji_val = 0
            Force_short_y_other_ji_val = 0
        else:
            gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
            gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
            g12 = gamma_1 * gamma_2
            if g12  < 1:
                Force_short_rho_other_ji_gauss = Gauss_Ledendra_int(gauss_charge_Force_rho_intcore, Gauss_para_1, k_f1, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

                Force_short_rho_other_ji_point_a = Gauss_Ledendra_int(point_charge_Force_rho_intcore, Gauss_para_2, k_f2, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

                Force_short_rho_other_ji_point_b = 0
                Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
                z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
                gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
                for l in np.array([0, 1, 2, 3]):
                    Force_short_rho_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * rho_ji / (z_para_val[l]**2 + rho_ji**2)**1.5

                Force_short_rho_other_ji_val = (q_i * q_j / (2 * pi * eps_2 * eps_0)) * (Force_short_rho_other_ji_gauss - Force_short_rho_other_ji_point_a + Force_short_rho_other_ji_point_b)

            else:
                pos_info = q_j, q_i, x_j, x_i, y_j, y_i, z_j, z_i
                INCAR = 1, 1, 1, 1, L_z, eps_1, eps_2, eps_3, alpha, accuracy
                Force_short_rho_other_ji_val = Fs_norm.Force_rhos_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m) / eps_0

            Force_short_x_other_ji_val = ((x_j - x_i)/rho_ji) * Force_short_rho_other_ji_val
            Force_short_y_other_ji_val = ((y_j - y_i)/rho_ji) * Force_short_rho_other_ji_val

        Force_short_x_other_j += Force_short_x_other_ji_val
        Force_short_y_other_j += Force_short_y_other_ji_val

        i += 1

    return Force_short_x_other_j, Force_short_y_other_j

@nb.njit()
def Force_short_z_other_cal_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha, N_t, accuracy, eps, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    Force_short_z_other_j = 0
    i = 0
    while np_neighbor_list_j[i][0] != 9999:
        q_i, x_i, y_i, z_i ,rho_ji = np_neighbor_list_j[i]

        gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
        gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
        g12 = gamma_1 * gamma_2
        if g12  < 1:
            Force_short_z_other_ji_gauss = Gauss_Ledendra_int(gauss_charge_Force_z_intcore, Gauss_para_1, k_f1, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

            Force_short_z_other_ji_point_a = Gauss_Ledendra_int(point_charge_Force_z_intcore, Gauss_para_2, k_f2, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha)

            Force_short_z_other_ji_point_b = 0
            Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
            z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
            gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
            partial_para_val = g.partial_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
            for l in np.array([0, 1, 2, 3]):
                Force_short_z_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l] * z_para_val[l] / (z_para_val[l]**2 + rho_ji**2)**1.5

            Force_short_z_other_ji_val = (q_i * q_j / (2 * pi * eps_2 * eps_0)) * ( - Force_short_z_other_ji_gauss + Force_short_z_other_ji_point_a + Force_short_z_other_ji_point_b)
        else:
            pos_info = q_j, q_i, x_j, x_i, y_j, y_i, z_j, z_i
            INCAR = 1, 1, 1, 1, L_z, eps_1, eps_2, eps_3, alpha, accuracy
            Force_short_z_other_ji_val = Fs_norm.Force_zs_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m) / eps_0

        Force_short_z_other_j += Force_short_z_other_ji_val

        i += 1

    return Force_short_z_other_j

@nb.njit()
def Force_short_z_self_cal_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha, N_t, accuracy, eps, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):

    rho_ji = 0
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2

    if g12  < 1:
        Force_short_z_self_j_gauss = Gauss_Ledendra_int(gauss_charge_Force_z_intcore, Gauss_para_1, k_f1, eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha)

        Force_short_z_self_j_point_a = Gauss_Ledendra_int(point_charge_Force_z_intcore, Gauss_para_2, k_f2, eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha)

        Force_short_z_self_j_point_b = 0
        Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
        z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
        gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
        partial_para_val = g.partial_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
        for l in [0, 2]:
            Force_short_z_self_j_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l]  / (z_para_val[l]**2)

        Force_short_z_self_j_val = (q_j * q_j / (2 * pi * eps_2 * eps_0)) * (- Force_short_z_self_j_gauss + Force_short_z_self_j_point_a - Force_short_z_self_j_point_b)
    else:
        pos_info = q_j, x_j, y_j, z_j
        INCAR = 1, 1, 1, 1, L_z, eps_1, eps_2, eps_3, alpha, accuracy
        Force_short_z_self_j_val = Fs_norm.Force_zs_self_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m) / eps_0

    return Force_short_z_self_j_val

@nb.njit()
def Force_short_cal(np_neighbor_list, eps_0, eps_1, eps_2, eps_3, L_z, alpha, POSCAR, Gauss_para_1, Gauss_para_2, k_f1, k_f2, N_t, accuracy, eps, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    NUM_particle = np.shape(POSCAR)[0]
    Force_short_x_val = np.zeros(NUM_particle)
    Force_short_y_val = np.zeros(NUM_particle)
    Force_short_z_val = np.zeros(NUM_particle)

    for j in range(NUM_particle):
        q_j, x_j, y_j, z_j = POSCAR[j]
        np_neighbor_list_j = np_neighbor_list[j]

        Force_short_x_j, Force_short_y_j = Force_short_rho_cal_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha, N_t, accuracy, eps, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m)

        Force_short_z_other_j = Force_short_z_other_cal_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha, N_t, accuracy, eps, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m)

        Force_short_z_self_j = Force_short_z_self_cal_j(q_j, x_j, y_j, z_j, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha, N_t, accuracy, eps, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m)

        Force_short_z_j = Force_short_z_other_j + Force_short_z_self_j

        Force_short_x_val[j] = Force_short_x_j
        Force_short_y_val[j] = Force_short_y_j
        Force_short_z_val[j] = Force_short_z_j

    
    return Force_short_x_val, Force_short_y_val, Force_short_z_val

def Gauss_para_gen(eps_1, eps_2, eps_3, L_z, N_t, accuracy, alpha):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = np.log(g12) / (2 * L_z)

    k_f_g = max(sqrt( - 4 * alpha * np.log(accuracy)), 2.1 * k_0)
    Ng_1 = int(N_t * k_0 / k_f_g)
    Ng_m = 10
    Ng_2 = Ng_1
    Ng_3 = N_t - 2 * Ng_1

    k_f_p = max( - np.log(accuracy) / (2 * L_z), 2.1 * k_0)
    Np_1 = int(N_t * k_0 / k_f_p)
    Np_m = 10
    Np_2 = Np_1
    Np_3 = N_t - 2 * Np_1

    Gg_1 = np.polynomial.legendre.leggauss(Ng_1)
    Gg_2 = np.polynomial.legendre.leggauss(Ng_2)
    Gg_3 = np.polynomial.legendre.leggauss(Ng_3)
    Gg_m = np.polynomial.legendre.leggauss(Ng_m)

    Gp_1 = np.polynomial.legendre.leggauss(Np_1)
    Gp_2 = np.polynomial.legendre.leggauss(Np_2)
    Gp_3 = np.polynomial.legendre.leggauss(Np_3)
    Gp_m = np.polynomial.legendre.leggauss(Np_m)

    return Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m




def Force_short(POSCAR, INCAR, CELL_LIST):
    NUM_particle = INCAR['NUM_particle']
    eps_0 = INCAR['eps_0']
    eps_1 = INCAR['eps_1']
    eps_2 = INCAR['eps_2']
    eps_3 = INCAR['eps_3']
    L_z = INCAR['L_z']
    alpha = INCAR['alpha']
    accuracy = INCAR['accuracy']
    N_t = INCAR['N_t']
    eps = INCAR['eps']

    Gauss_para_1 = INCAR['Gauss_para1']
    Gauss_para_2 = INCAR['Gauss_para2']
    k_f1 = INCAR['k_f1']
    k_f2 = INCAR['k_f2']

    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2

    if g12 > 1:
        Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m = Gauss_para_gen(eps_1, eps_2, eps_3, L_z, N_t, accuracy, alpha)
    else:
        Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m = Gauss_para_gen(-1.1, 1, -1.1, L_z, 50, accuracy, alpha)

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

    Force_short_x_val, Force_short_y_val, Force_short_z_val = Force_short_cal(np_neighbor_list, eps_0, eps_1, eps_2, eps_3, L_z, alpha, POSCAR, Gauss_para_1, Gauss_para_2, k_f1, k_f2, N_t, accuracy, eps, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m)

    return Force_short_x_val, Force_short_y_val, Force_short_z_val




    

