# in this script the function used to calculate the k space summation will be defined
# with the given random_p, that number of sampling poingts will be chosen from the K_set and used to calculate the value of the summation.

import numpy as np
from math import floor
from random import random
import numba as nb

def k_space_summation(func, func_para, INCAR): # func is the funcation to be sum up, func([k_x, k_y, K, k], [...])
    RBM_p = INCAR['RBM_p']
    K_set = INCAR['K_set']
    K_set_shape = np.shape(K_set)[0]

    SUM = 0
    for l in range(RBM_p):
        k_p_num = floor(K_set_shape * random())
        k_p = K_set[k_p_num]
        SUM += func(k_p, func_para)
    
    S = INCAR['Total_P']
    RBM_p = INCAR['RBM_p']
    SUM = (S/RBM_p) * SUM
    
    return SUM
