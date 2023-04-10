from numpy import exp, sign
import numpy as np
import numba as nb

#this footscript provides functions below:
# green_function(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
# partial_green_function(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
# Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
# partial_Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
# Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)
# partial_Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i)

def input_translate(INCAR, POSCAR, j, i):
    eps_1 = INCAR['eps_1']
    eps_2 = INCAR['eps_2']
    eps_3 = INCAR['eps_3']
    L_z = INCAR['L_z']
    z_j = POSCAR[j][3]
    z_i = POSCAR[i][3]
    return [eps_1, eps_2, eps_3, L_z, z_j, z_i]


def gamma(k, INCAR, POSCAR, j, i):
    eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    gamma = 1/((eps_2 - eps_1) * (eps_2 - eps_3) * exp(-2*k*L_z) - (eps_2 + eps_1) * (eps_2 + eps_3))
    return gamma

def gamma_1(k, INCAR, POSCAR, j, i):
    eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    gamma_1 = ((eps_2 - eps_1) * (eps_2 - eps_3)/((eps_2 + eps_1) * (eps_2 + eps_3))) * gamma(k, INCAR, POSCAR, j, i)

    return gamma_1

def gamma_2(k, INCAR, POSCAR, j, i):
    eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    gamma_2 = 1/((eps_2 + eps_1) * (eps_2 + eps_3))

    return gamma_2

def z_para(k, INCAR, POSCAR, j, i):
    eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    z_para_0 = z_i + z_j - 2 * L_z
    z_para_1 = - abs(z_i - z_j)
    z_para_2 = - (z_i + z_j)
    z_para_3 = abs(z_i - z_j) - 2 * L_z

    return [z_para_0, z_para_1, z_para_2, z_para_3] 

def Gamma_para(k, INCAR, POSCAR, j, i):
    eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    Gamma_para_0 = (eps_2 + eps_1) * (eps_2 - eps_3) / 2
    Gamma_para_1 = (eps_2 + eps_1) * (eps_2 + eps_3) / 2
    Gamma_para_2 = (eps_2 - eps_1) * (eps_2 + eps_3) / 2
    Gamma_para_3 = (eps_2 - eps_1) * (eps_2 - eps_3) / 2
    
    return [Gamma_para_0, Gamma_para_1, Gamma_para_2, Gamma_para_3]

def partial_para(k, INCAR, POSCAR, j, i):
    eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    partial_para_0 = + 1
    partial_para_1 = + sign(z_i - z_j)
    partial_para_2 = - 1
    partial_para_3 = - sign(z_i - z_j)

    return [partial_para_0, partial_para_1, partial_para_2, partial_para_3]

def Gamma_func(k, INCAR, POSCAR, j, i):
    #eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    gamma_val = gamma(k, INCAR, POSCAR, j, i)
    Gamma_para_val = Gamma_para(k, INCAR, POSCAR, j, i)
    z_para_val = z_para(k, INCAR, POSCAR, j, i)

    Gamma_func = 0
    for l in range(4):
        Gamma_func += gamma_val * Gamma_para_val[l] * exp(k * z_para_val[l])

    return Gamma_func

def Gamma_funca(k, INCAR, POSCAR, j, i):
    eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    gamma_1_val = gamma_1(k, INCAR, POSCAR, j, i)
    Gamma_para_val = Gamma_para(k, INCAR, POSCAR, j, i)
    z_para_val = z_para(k, INCAR, POSCAR, j, i)

    Gamma_funca = 0
    for l in range(4):
        Gamma_funca += gamma_1_val * Gamma_para_val[l] * exp(- 2 * k * L_z) * exp(k * z_para_val[l])
    
    return Gamma_funca

def partial_Gamma_func(k, INCAR, POSCAR, j, i):
    #eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    gamma_val = gamma(k, INCAR, POSCAR, j, i)
    Gamma_para_val = Gamma_para(k, INCAR, POSCAR, j, i)
    z_para_val = z_para(k, INCAR, POSCAR, j, i)
    partial_para_val = partial_para(k, INCAR, POSCAR, j, i)

    partial_Gamma_func = 0
    for l in range(4):
        partial_Gamma_func += k * Gamma_para_val[l] * gamma_val * partial_para_val[l] * exp(k * z_para_val[l])
    
    return partial_Gamma_func

def partial_Gamma_funca(k, INCAR, POSCAR, j, i):
    eps_1, eps_2, eps_3, L_z, z_j, z_i = input_translate(INCAR, POSCAR, j, i)

    gamma_1_val = gamma_1(k, INCAR, POSCAR, j, i)
    Gamma_para_val = Gamma_para(k, INCAR, POSCAR, j, i)
    z_para_val = z_para(k, INCAR, POSCAR, j, i)
    partial_para_val = partial_para(k, INCAR, POSCAR, j, i)

    partial_Gamma_funca = 0
    for l in range(4):
        partial_Gamma_funca += k * Gamma_para_val[l] * gamma_1_val * exp(-2*k*L_z) * exp(k * z_para_val[l]) * partial_para_val[l]

    return partial_Gamma_funca