import numpy as np
import os


# this footscript provides the function Gauss_Ledendra_int(N, func, para_list, a, b)
# NUM is the order of the Gauss Ledendra int
# func is the intcore, which should be defined as func(k, [x_1, x_2, ...])
# para_list is the list of parameter of the intcore, [x_1, x_2, ...]
# a, b are the start and the end of the integral

def Gauss_Ledendra_int(func, Gauss_para, k_f, func_para):
    Gauss_order = np.shape(Gauss_para)[0]
    SUM = 0
    for i in range(Gauss_order):
        s_i = Gauss_para[i][0] * k_f/2 + k_f/2
        w_i = Gauss_para[i][1]
        t = w_i * (k_f/2) * func(s_i, func_para)
        SUM += t
    return SUM
