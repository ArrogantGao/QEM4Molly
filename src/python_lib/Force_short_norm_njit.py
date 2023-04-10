import numpy as np
from numpy import exp, sign, log, sqrt, pi
# import green_function
import green_norm_njit as green_norm
from Bessel import Bessel0, Bessel1
# from scipy.special.orthogonal import p_roots,roots_jacobi
import numba as nb
import green_function_njit as g


@nb.njit()
def gauss(f, para_list, Gauss_para, a, b):
    G = 0
    s = Gauss_para[0]
    w = Gauss_para[1]
    N = np.shape(s)[0]
    for i in range(N):
        s_i = 0.5*(b-a)*s[i] + 0.5*(b+a)
        w_i = w[i]
        G += 0.5*(b-a) * (w_i * f(s_i, para_list))
    return G

@nb.njit()
def F_rs_p_up(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0 = para_list
    f = k * green_norm.Gamma_a_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)* Bessel1(k * rho_ji)
    return f

@nb.njit()
def F_rs_p_intcore(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0 = para_list

    if abs(k - k_0) <= eps:
        k_1 = k_0 - 0.01 * eps
        k_2 = k_0 + 0.01 * eps
        df = (F_rs_p_up(k_2, para_list) - F_rs_p_up(k_1, para_list)) / (k_2 - k_1)
        f = k * green_norm.Gamma_a_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel1(k * rho_ji) - df / (2 * L_z)
    elif k <= 2 * k_0:
        f = k * green_norm.Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel1(k * rho_ji) + F_rs_p_up(k_0, para_list) / (2 * L_z * (k - k_0))
    else:
        f = k * green_norm.Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel1(k * rho_ji)

    return f

@nb.njit()
def F_rs_g_up(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha, eps, k_0 = para_list
    f = k * green_norm.green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(-k**2/(4*alpha)) * Bessel1(k * rho_ji)
    return f

@nb.njit()
def F_rs_g_intcore(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha, eps, k_0 = para_list

    if abs(k - k_0) <= eps:
        k_1 = k_0 - 0.01 * eps
        k_2 = k_0 + 0.01 * eps
        df = (F_rs_g_up(k_2, para_list) - F_rs_g_up(k_1, para_list)) / (k_2 - k_1)
        f = k * green_norm.Gamma_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel1(k * rho_ji) - df / (2 * L_z)
    elif k <= 2 * k_0:
        f = k * green_norm.Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(-k**2/(4*alpha)) * Bessel1(k * rho_ji) + F_rs_g_up(k_0, para_list) / (2 * L_z * (k - k_0))
    else:
        f = k * green_norm.Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(-k**2/(4*alpha)) * Bessel1(k * rho_ji)

    return f

@nb.njit()
def Force_rhos_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    q_j, q_i, x_j, x_i, y_j, y_i, z_j, z_i = pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    rho_ji = sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
    if rho_ji == 0:
        return 0
    
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = log(g12) / (2 * L_z)

    k_f_g = max(sqrt( - 4 * alpha * log(cutoff)), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_g)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_g = (eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha, eps, k_0)
    F_g = - gauss(F_rs_g_intcore, para_list_g, Gg_1, 0, k_0 - eps) - gauss(F_rs_g_intcore, para_list_g, Gg_m, k_0 - eps, k_0 + eps) - gauss(F_rs_g_intcore, para_list_g, Gg_2, k_0 + eps, 2 * k_0) - gauss(F_rs_g_intcore, para_list_g, Gg_3, 2 * k_0, k_f_g)


    k_f_p = max( - log(cutoff) / (2 * L_z), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_p)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_p = (eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0)
    F_pa = gauss(F_rs_p_intcore, para_list_p,  Gp_1, 0, k_0 - eps) + gauss(F_rs_p_intcore, para_list_p, Gp_m, k_0 - eps, k_0 + eps) + gauss(F_rs_p_intcore, para_list_p, Gp_2, k_0 + eps, 2 * k_0) + gauss(F_rs_p_intcore, para_list_p, Gp_3, 2 * k_0, k_f_p)

    Force_short_rho_other_ji_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    for l in np.array([0, 1, 2, 3]):
        Force_short_rho_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * rho_ji / (z_para_val[l]**2 + rho_ji**2)**1.5

    F_rhos =  - (q_i*q_j/(2 * pi * eps_2)) * (F_g + F_pa - Force_short_rho_other_ji_point_b)
    return F_rhos

@nb.njit()
def Force_rhos_p_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    q_j, q_i, x_j, x_i, y_j, y_i, z_j, z_i = pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    rho_ji = sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
    if rho_ji == 0:
        return 0
    
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = log(g12) / (2 * L_z)

    F_g = 0

    k_f_p = max( - log(cutoff) / (2 * L_z), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_p)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_p = (eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0)
    F_pa = gauss(F_rs_p_intcore, para_list_p,  Gp_1, 0, k_0 - eps) + gauss(F_rs_p_intcore, para_list_p, Gp_m, k_0 - eps, k_0 + eps) + gauss(F_rs_p_intcore, para_list_p, Gp_2, k_0 + eps, 2 * k_0) + gauss(F_rs_p_intcore, para_list_p, Gp_3, 2 * k_0, k_f_p)

    Force_short_rho_other_ji_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    for l in np.array([0, 1, 2, 3]):
        Force_short_rho_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * rho_ji / (z_para_val[l]**2 + rho_ji**2)**1.5

    F_rhos =  - (q_i*q_j/(2 * pi * eps_2)) * (F_g + F_pa - Force_short_rho_other_ji_point_b)
    return F_rhos


@nb.njit()
def F_zs_p_up(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0 = para_list
    f = green_norm.dz_Gamma_a_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji)
    return f

@nb.njit()
def F_zs_p_intcore(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0 = para_list
    if abs(k - k_0) <= eps:
        k_1 = k_0 - 0.01 * eps
        k_2 = k_0 + 0.01 * eps
        df = (F_zs_p_up(k_2, para_list) - F_zs_p_up(k_1, para_list)) / (k_2 - k_1)
        f = green_norm.dz_Gamma_a_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) - df / (2 * L_z)
    elif k <= 2 * k_0:
        f = green_norm.dz_Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) + F_zs_p_up(k_0, para_list) / (2 * L_z * (k - k_0))
    else:
        f = green_norm.dz_Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji)

    return f

@nb.njit()
def F_zs_g_up(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha, eps, k_0 = para_list 
    f = green_norm.dz_green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k * rho_ji) * exp(-k ** 2/(4*alpha))
    return f

@nb.njit()
def F_zs_g_intcore(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha, eps, k_0 = para_list
    if abs(k - k_0) <= eps:
        k_1 = k_0 - 0.01 * eps
        k_2 = k_0 + 0.01 * eps
        df = (F_zs_g_up(k_2, para_list) - F_zs_g_up(k_1, para_list)) / (k_2 - k_1)
        f = green_norm.dz_Gamma_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) * exp(-k**2/(4*alpha)) - df / (2 * L_z)
    elif k <= 2 * k_0:
        f = green_norm.dz_Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) * exp(-k**2/(4*alpha)) + F_zs_g_up(k_0, para_list) / (2 * L_z * (k - k_0))
    else:
        f = green_norm.dz_Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) * exp(-k**2/(4*alpha))

    return f

@nb.njit()
def Force_zs_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    q_j, q_i, x_j, x_i, y_j, y_i, z_j, z_i = pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    rho_ji = sqrt((x_j - x_i)**2 + (y_j - y_i)**2)

    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = log(g12) / (2 * L_z)
    # print(k_0, gamma_1, gamma_2, g12)


    k_f_g = max(sqrt( - 4 * alpha * log(cutoff)), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_g)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_g = (eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha, eps, k_0)
    F_g = gauss(F_zs_g_intcore, para_list_g, Gg_1, 0, k_0 - eps) + gauss(F_zs_g_intcore, para_list_g, Gg_m, k_0 - eps, k_0 + eps) + gauss(F_zs_g_intcore, para_list_g, Gg_2, k_0 + eps, 2 * k_0) + gauss(F_zs_g_intcore, para_list_g, Gg_3, 2 * k_0, k_f_g)


    k_f_p = max( - log(cutoff) / (2 * L_z), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_p)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_p = (eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0)
    F_pa = -(gauss(F_zs_p_intcore, para_list_p,  Gp_1, 0, k_0 - eps) + gauss(F_zs_p_intcore, para_list_p, Gp_m, k_0 - eps, k_0 + eps) + gauss(F_zs_p_intcore, para_list_p, Gp_2, k_0 + eps, 2 * k_0) + gauss(F_zs_p_intcore, para_list_p, Gp_3, 2 * k_0, k_f_p))
    # print(para_list_p)
    # print(INCAR)

    Force_short_z_other_ji_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    partial_para_val = g.partial_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    for l in np.array([0, 1, 2, 3]):
        Force_short_z_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l] * z_para_val[l] / (z_para_val[l]**2 + rho_ji**2)**1.5

    # print(F_g, F_pa, F_pb)

    F_zs = - (q_i*q_j/(2 * pi * eps_2)) * (F_g + F_pa - Force_short_z_other_ji_point_b)
    return F_zs

@nb.njit()
def Force_zs_p_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    q_j, q_i, x_j, x_i, y_j, y_i, z_j, z_i = pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    rho_ji = sqrt((x_j - x_i)**2 + (y_j - y_i)**2)

    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = log(g12) / (2 * L_z)
    # print(k_0, gamma_1, gamma_2, g12)


    F_g = 0


    k_f_p = max( - log(cutoff) / (2 * L_z), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_p)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_p = (eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0)
    F_pa = -(gauss(F_zs_p_intcore, para_list_p,  Gp_1, 0, k_0 - eps) + gauss(F_zs_p_intcore, para_list_p, Gp_m, k_0 - eps, k_0 + eps) + gauss(F_zs_p_intcore, para_list_p, Gp_2, k_0 + eps, 2 * k_0) + gauss(F_zs_p_intcore, para_list_p, Gp_3, 2 * k_0, k_f_p))
    # print(para_list_p)
    # print(INCAR)

    Force_short_z_other_ji_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    partial_para_val = g.partial_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    for l in np.array([0, 1, 2, 3]):
        Force_short_z_other_ji_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l] * z_para_val[l] / (z_para_val[l]**2 + rho_ji**2)**1.5

    # print(F_g, F_pa, F_pb)

    F_zsp = - (q_i*q_j/(2 * pi * eps_2)) * (F_g + F_pa - Force_short_z_other_ji_point_b)
    return F_zsp


@nb.njit()
def F_zs_p_self_up(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0 = para_list
    f = green_norm.dz_Gamma_a_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji)
    return f

@nb.njit()
def F_zs_p_self_intcore(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, eps, k_0 = para_list
    if abs(k - k_0) <= eps:
        k_1 = k_0 - 0.01 * eps
        k_2 = k_0 + 0.01 * eps
        df = (F_zs_p_self_up(k_2, para_list) - F_zs_p_self_up(k_1, para_list)) / (k_2 - k_1)
        f = green_norm.dz_Gamma_a_self_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) - df / (2 * L_z)
    elif k <= 2 * k_0:
        f = green_norm.dz_Gamma_a_self(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) + F_zs_p_self_up(k_0, para_list) / (2 * L_z * (k - k_0))
    else:
        f = green_norm.dz_Gamma_a_self(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji)

    return f

@nb.njit()
def F_zs_g_self_up(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha, eps, k_0 = para_list 
    f = green_norm.dz_Gamma_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k * rho_ji) * exp(-k ** 2/(4*alpha))
    return f

@nb.njit()
def F_zs_g_self_intcore(k, para_list):
    eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ji, alpha, eps, k_0 = para_list
    if abs(k - k_0) <= eps:
        k_1 = k_0 - 0.01 * eps
        k_2 = k_0 + 0.01 * eps
        df = (F_zs_g_self_up(k_2, para_list) - F_zs_g_self_up(k_1, para_list)) / (k_2 - k_1)
        f = green_norm.dz_Gamma_self_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) * exp(-k**2/(4*alpha)) - df / (2 * L_z)
    elif k <= 2 * k_0:
        f = green_norm.dz_Gamma_self(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) * exp(-k**2/(4*alpha)) + F_zs_g_self_up(k_0, para_list) / (2 * L_z * (k - k_0))
    else:
        f = green_norm.dz_Gamma_self(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * Bessel0(k*rho_ji) * exp(-k**2/(4*alpha))

    return f

@nb.njit()
def Force_zs_self_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    q_j, x_j, y_j, z_j= pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    rho_ji = 0

    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = log(g12) / (2 * L_z)
    # print(k_0, gamma_1, gamma_2, g12)


    k_f_g = max(sqrt( - 4 * alpha * log(cutoff)), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_g)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_g = (eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha, eps, k_0)
    F_g = gauss(F_zs_g_self_intcore, para_list_g, Gg_1, 0, k_0 - eps) + gauss(F_zs_g_self_intcore, para_list_g, Gg_m, k_0 - eps, k_0 + eps) + gauss(F_zs_g_self_intcore, para_list_g, Gg_2, k_0 + eps, 2 * k_0) + gauss(F_zs_g_self_intcore, para_list_g, Gg_3, 2 * k_0, k_f_g)
    # F_g = 0


    k_f_p = max( - log(cutoff) / (2 * L_z), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_p)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_p = (eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, eps, k_0)
    F_pa = -(gauss(F_zs_p_self_intcore, para_list_p,  Gp_1, 0, k_0 - eps) + gauss(F_zs_p_self_intcore, para_list_p, Gp_m, k_0 - eps, k_0 + eps) + gauss(F_zs_p_self_intcore, para_list_p, Gp_2, k_0 + eps, 2 * k_0) + gauss(F_zs_p_self_intcore, para_list_p, Gp_3, 2 * k_0, k_f_p))
    # print(para_list_p)
    # print(INCAR)

    Force_short_z_self_j_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    partial_para_val = g.partial_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    for l in [0, 2]:
        Force_short_z_self_j_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l]  / (z_para_val[l]**2)

    # print(F_g, F_pa, F_pb)

    F_zs = - (q_j*q_j/(2 * pi * eps_2)) * (F_g - F_pa + Force_short_z_self_j_point_b)
    return F_zs

@nb.njit()
def Force_zs_self_p_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    q_j, x_j, y_j, z_j= pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    rho_ji = 0

    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = log(g12) / (2 * L_z)
    # print(k_0, gamma_1, gamma_2, g12)


    # k_f_g = max(sqrt( - 4 * alpha * log(cutoff)), 2.1 * k_0)
    # N_1 = int(N_t * k_0 / k_f_g)
    # N_2 = N_1
    # N_3 = N_t - 2 * N_1
    # para_list_g = (eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha, eps, k_0)
    # F_g = gauss(F_zs_g_self_intcore, para_list_g, Gg_1, 0, k_0 - eps) + gauss(F_zs_g_self_intcore, para_list_g, Gg_m, k_0 - eps, k_0 + eps) + gauss(F_zs_g_self_intcore, para_list_g, Gg_2, k_0 + eps, 2 * k_0) + gauss(F_zs_g_self_intcore, para_list_g, Gg_3, 2 * k_0, k_f_g)
    F_g = 0


    k_f_p = max( - log(cutoff) / (2 * L_z), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_p)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_p = (eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, eps, k_0)
    F_pa = -(gauss(F_zs_p_self_intcore, para_list_p,  Gp_1, 0, k_0 - eps) + gauss(F_zs_p_self_intcore, para_list_p, Gp_m, k_0 - eps, k_0 + eps) + gauss(F_zs_p_self_intcore, para_list_p, Gp_2, k_0 + eps, 2 * k_0) + gauss(F_zs_p_self_intcore, para_list_p, Gp_3, 2 * k_0, k_f_p))
    # print(para_list_p)
    # print(INCAR)

    Force_short_z_self_j_point_b = 0
    Gamma_para_val = g.Gamma_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    z_para_val = g.z_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    gamma_2_val = g.gamma_2(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    partial_para_val = g.partial_para(0, eps_1, eps_2, eps_3, L_z, z_j, z_j)
    for l in [0, 2]:
        Force_short_z_self_j_point_b += Gamma_para_val[l] * gamma_2_val * partial_para_val[l]  / (z_para_val[l]**2)

    # print(F_g, F_pa, F_pb)

    F_zs = - (q_j*q_j/(2 * pi * eps_2)) * (F_g - F_pa + Force_short_z_self_j_point_b)
    return F_zs

@nb.njit()
def Force_zs_self_g_norm(pos_info, INCAR, eps, N_t, Gg_1, Gg_2, Gg_3, Gg_m, Gp_1, Gp_2, Gp_3, Gp_m):
    q_j, x_j, y_j, z_j= pos_info
    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    rho_ji = 0

    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = log(g12) / (2 * L_z)
    # print(k_0, gamma_1, gamma_2, g12)


    k_f_g = max(sqrt( - 4 * alpha * log(cutoff)), 2.1 * k_0)
    N_1 = int(N_t * k_0 / k_f_g)
    N_2 = N_1
    N_3 = N_t - 2 * N_1
    para_list_g = (eps_1, eps_2, eps_3, L_z, z_j, z_j, rho_ji, alpha, eps, k_0)
    F_g = gauss(F_zs_g_self_intcore, para_list_g, Gg_1, 0, k_0 - eps) + gauss(F_zs_g_self_intcore, para_list_g, Gg_m, k_0 - eps, k_0 + eps) + gauss(F_zs_g_self_intcore, para_list_g, Gg_2, k_0 + eps, 2 * k_0) + gauss(F_zs_g_self_intcore, para_list_g, Gg_3, 2 * k_0, k_f_g)
    # F_g = 0


    F_zs = - (q_j*q_j/(2 * pi * eps_2)) * F_g
    return F_zs