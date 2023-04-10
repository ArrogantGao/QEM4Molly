import numpy as np
import json
from numpy import sqrt, log
from Gauss_order_check import Gauss_order1_check, Gauss_order2_check
from random_K_gen import random_K_generator
from math import floor, ceil
from KCAR_gen import k_gen

# In the script, first we will define the POSCAR load function, LOAD_POSCAR(), this function will return a ict that contains the infromation as [q_i, x_i, y_i, z_i]
# Then using the POSCAR dict, we will load the INCAR from file 'INCAR' and add the extra terms into it.
# INCAR file contains: L_x, L_y, L_z, eps_1, eps_2, eps_3, accuracy, length_scale, RBM_p
# INCAR dict contains: L_x, L_y, L_z, eps_1, eps_2, eps_3, accuracy, length_scale, RBM_p, NUM_particle, alpha, k_f1, k_f2, Gauss_order1, Gauss_order2

def LOAD_PARCAR():
    parcar_file = open('./INPUT/PARCAR', 'r')
    js = parcar_file.read()
    PARCAR = json.loads(js)
    parcar_file.close()

    return PARCAR

def LOAD_INCAR(PARCAR):
    incar_file = open('./INPUT/INCAR', 'r')
    js = incar_file.read()
    INCAR = json.loads(js)
    incar_file.close()
    
    NUM_TYPE = len(PARCAR)
    NUM_particle = 0
    for num_type in range(1, NUM_TYPE + 1):
        mass, charge, num_particle = PARCAR[str(num_type)]
        NUM_particle += num_particle

    INCAR['NUM_particle'] = NUM_particle

    g_1 = INCAR['g_1']
    g_2 = INCAR['g_2']
    INCAR['eps_1'] = (1 - g_1) / (1 + g_1)
    INCAR['eps_2'] = 1
    INCAR['eps_3'] = (1 - g_2) / (1 + g_2)



    if INCAR['mode'] == 'rbm':
        alpha = (NUM_particle) / (INCAR['L_x'] * INCAR['L_y'])

    if INCAR['mode'] == 'exact':
        # alpha = sqrt(NUM_particle) / (INCAR['L_x'] * INCAR['L_y'])
        alpha = (NUM_particle) / (INCAR['L_x'] * INCAR['L_y'])
    
    INCAR['alpha'] = alpha


    accuracy = INCAR['accuracy']
    s = sqrt( - log(accuracy))
    r_c = s / sqrt(INCAR['alpha'])
    INCAR['r_c'] = r_c
    k_f1 = 2 * sqrt(alpha) * s
    k_f2 = (s ** 2) / (2 * INCAR['L_z'])
    INCAR['k_f1'] = k_f1
    INCAR['k_f2'] = k_f2

    Guass_para1 = Gauss_order1_check(INCAR)
    Guass_para2 = Gauss_order2_check(INCAR)
    INCAR['Gauss_para1'] = Guass_para1
    INCAR['Gauss_para2'] = Guass_para2

    K_set, Total_P = random_K_generator(INCAR)
    INCAR['K_set'] = K_set
    INCAR['Total_P'] = Total_P


    X_Cell_num = floor(INCAR['L_x']/r_c)
    Y_Cell_num = floor(INCAR['L_y']/r_c)
    Z_Cell_num = NUM_particle
    X_Cell_length = INCAR['L_x']/X_Cell_num
    Y_Cell_length = INCAR['L_y']/Y_Cell_num   
    Z_Cell_length = INCAR['L_z']/Z_Cell_num
    INCAR['Cell_info'] = [X_Cell_num, Y_Cell_num, Z_Cell_num, X_Cell_length, Y_Cell_length, Z_Cell_length]

    return INCAR

# def LOAD_INCAR_mode(PARCAR, mode):
#     incar_file = open('./INPUT/INCAR', 'r')
#     js = incar_file.read()
#     INCAR = json.loads(js)
#     incar_file.close()
    
#     NUM_TYPE = len(PARCAR)
#     NUM_particle = 0
#     for num_type in range(1, NUM_TYPE + 1):
#         mass, charge, num_particle = PARCAR[str(num_type)]
#         NUM_particle += num_particle

#     INCAR['NUM_particle'] = NUM_particle

#     if mode == 'rbm':
#         alpha = (NUM_particle) / (INCAR['L_x'] * INCAR['L_y'])
#         INCAR['alpha'] = alpha

#     if mode == 'exact':
#         alpha = sqrt(NUM_particle) / (INCAR['L_x'] * INCAR['L_y'])
#         INCAR['alpha'] = alpha

#     INCAR["s"] = int(INCAR["s"])
#     r_c = INCAR['s'] / sqrt(INCAR['alpha'])
#     INCAR['r_c'] = r_c
#     k_f1 = 2 * sqrt(alpha) * INCAR['s']
#     k_f2 = (INCAR['s'] ** 2) / (2 * INCAR['L_z'])
#     INCAR['k_f1'] = k_f1
#     INCAR['k_f2'] = k_f2

#     Guass_para1 = Gauss_order1_check(INCAR)
#     Guass_para2 = Gauss_order2_check(INCAR)
#     INCAR['Gauss_para1'] = Guass_para1
#     INCAR['Gauss_para2'] = Guass_para2

#     K_set, Total_P = random_K_generator(INCAR)
#     INCAR['K_set'] = K_set
#     INCAR['Total_P'] = Total_P


#     X_Cell_num = floor(INCAR['L_x']/r_c)
#     Y_Cell_num = floor(INCAR['L_y']/r_c)
#     Z_Cell_num = NUM_particle
#     X_Cell_length = INCAR['L_x']/X_Cell_num
#     Y_Cell_length = INCAR['L_y']/Y_Cell_num   
#     Z_Cell_length = INCAR['L_z']/Z_Cell_num
#     INCAR['Cell_info'] = [X_Cell_num, Y_Cell_num, Z_Cell_num, X_Cell_length, Y_Cell_length, Z_Cell_length]

#     return INCAR

def LOAD_KCAR(INCAR):
    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    KCAR = k_gen(10000, L_x, L_y)
    return KCAR