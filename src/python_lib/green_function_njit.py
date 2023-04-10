from numpy import exp, sign
import numpy as np
import numba as nb

@nb.njit()
def gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    gamma = 1/((eps_2 - eps_1) * (eps_2 - eps_3) * exp(-2*k*L_z) - (eps_2 + eps_1) * (eps_2 + eps_3))
    return gamma

@nb.njit()
def gamma_1(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    gamma_1 = ((eps_2 - eps_1) * (eps_2 - eps_3)/((eps_2 + eps_1) * (eps_2 + eps_3))) * gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)

    return gamma_1

@nb.njit()
def gamma_2(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    gamma_2 = 1/((eps_2 + eps_1) * (eps_2 + eps_3))

    return gamma_2

@nb.njit()
def z_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    z_para = np.zeros(4)
    z_para[0] = z_i + z_j - 2 * L_z
    z_para[1] = - abs(z_i - z_j)
    z_para[2] = - (z_i + z_j)
    z_para[3] = abs(z_i - z_j) - 2 * L_z

    return z_para

@nb.njit()
def Gamma_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    
    Gamma_para = np.zeros(4)
    Gamma_para[0] = (eps_2 + eps_1) * (eps_2 - eps_3) / 2
    Gamma_para[1] = (eps_2 + eps_1) * (eps_2 + eps_3) / 2
    Gamma_para[2] = (eps_2 - eps_1) * (eps_2 + eps_3) / 2
    Gamma_para[3] = (eps_2 - eps_1) * (eps_2 - eps_3) / 2
    
    return Gamma_para

@nb.njit()
def partial_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    partial_para = np.zeros(4)
    partial_para[0] = + 1
    partial_para[1] = + sign(z_i - z_j)
    partial_para[2] = - 1
    partial_para[3] = - sign(z_i - z_j)

    return partial_para

@nb.njit()
def Gamma_func(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    gamma_val = gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    Gamma_para_val = Gamma_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = z_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)

    Gamma_func = 0
    for l in range(4):
        Gamma_func += gamma_val * Gamma_para_val[l] * exp(k * z_para_val[l])

    return Gamma_func

@nb.njit()
def Gamma_funca(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    gamma_1_val = gamma_1(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    Gamma_para_val = Gamma_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = z_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)

    Gamma_funca = 0
    for l in range(4):
        Gamma_funca += gamma_1_val * Gamma_para_val[l] * exp(- 2 * k * L_z) * exp(k * z_para_val[l])
    
    return Gamma_funca

@nb.njit()
def partial_Gamma_func(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    gamma_val = gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    Gamma_para_val = Gamma_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = z_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    partial_para_val = partial_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)

    partial_Gamma_func = 0
    for l in range(4):
        partial_Gamma_func += k * Gamma_para_val[l] * gamma_val * partial_para_val[l] * exp(k * z_para_val[l])
    
    return partial_Gamma_func

@nb.njit()
def partial_Gamma_funca(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):

    gamma_1_val = gamma_1(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    Gamma_para_val = Gamma_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    z_para_val = z_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
    partial_para_val = partial_para(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)

    partial_Gamma_funca = 0
    for l in range(4):
        partial_Gamma_funca += k * Gamma_para_val[l] * gamma_1_val * exp(-2*k*L_z) * exp(k * z_para_val[l]) * partial_para_val[l]

    return partial_Gamma_funca