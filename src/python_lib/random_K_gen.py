from random import random
import numpy as np
from numpy import sqrt, pi, exp
from math import ceil, floor

# In the script, the random K_generator will be difined, which will create a random set of k parameter sampling from the discrete Gauss distrubation and return it back to INCAR dict

def random_K_generator(INCAR):
    sample_num = 10000

    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    k_f = 2 * INCAR['k_f1']
    NUM_particle = INCAR['NUM_particle']
    alpha = INCAR['alpha']

    m_x_max = ceil(k_f * L_x / (2 * pi))
    m_y_max = ceil(k_f * L_y / (2 * pi))


    S_x = 0
    S_y = 0
    P_x = []
    K_x = []
    P_y = []
    K_y = []

    for m_x in np.arange(- m_x_max, m_x_max + 1):
        k_x = m_x * 2 * pi/L_x
        S_x += exp(-k_x**2 / (4 * alpha))

    for m_x in np.arange(- m_x_max, m_x_max + 1):
        k_x = m_x * 2 * pi/L_x
        K_x.append(k_x)
        P_x.append(exp(-k_x**2 / (4 * alpha)) / S_x)
        
    for m_y in np.arange(- m_y_max, m_y_max + 1):
        k_y = m_y * 2 * pi/L_y
        S_y += exp(-k_y**2 / (4 * alpha))

    for m_y in np.arange(- m_y_max, m_y_max + 1):
        k_y = m_y * 2 * pi/L_y
        K_y.append(k_y)
        P_y.append(exp(-k_y**2 / (4 * alpha)) / S_y)
        
    sample_k_x = np.random.choice(K_x, sample_num, p=P_x)
    sample_k_y = np.random.choice(K_y, sample_num, p=P_y)


    K_xy = []
    for i in range(sample_num):
        k_x = sample_k_x[i]
        k_y = sample_k_y[i]
        K = k_x**2 + k_y**2
        k = sqrt(K)
        if k > 0:
            K_xy.append(np.array([k_x, k_y, K, k]))
    K_xy = np.array(K_xy)

    Total_P = S_x * S_y - 1

    return K_xy, Total_P