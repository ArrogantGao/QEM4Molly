import numpy as np
from scipy.special import jv
from numpy import exp, sin, cos, pi, sign, sqrt
from math import ceil
import pickle
import json
import numba as nb

@nb.njit()
def k_gen(N, L_x, L_y):
    flag = L_x > L_y
    L1 = max(L_x, L_y)
    L2 = min(L_x, L_y)
    a = L1/L2
    K_list = []
    for K in range(N+1):
        #print(K)
        #print(K)
        t = sqrt(K)
        for m1 in range(-ceil(a)*ceil(t)-1, ceil(a)*ceil(t)+2):
            for m2 in range(-ceil(t)-1, ceil(t)+2):
                R = (m1/a)**2 + (m2)**2
                if R > K and  R <= K + 1:
                    k1 = 2*pi*m1/L1
                    k2 = 2*pi*m2/L2
                    k_d = k1**2 + k2**2
                    k = sqrt(k_d)
                    if flag == 1:
                        K_list.append([k1, k2, k_d, k])
                    if flag == 0:
                        K_list.append([k2, k1, k_d, k])
    
    KCAR = np.array(K_list)
    return KCAR
    


# incar_file = open('../INPUT/INCAR', 'r')
# js = incar_file.read()
# INCAR = json.loads(js)
# incar_file.close()

# L_x = INCAR['L_x']
# L_y = INCAR['L_y']
# KCAR = k_gen(10000, L_x, L_y)
# print('success')
# # np.savetxt('KCAR', KCAR)
# np.savetxt('../INPUT/KCAR', KCAR)
# print('K calculation complete')