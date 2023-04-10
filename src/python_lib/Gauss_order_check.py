import numpy as np
from scipy.special import iv, jv
from numpy import sqrt, pi, exp

# In this script Gauss_order1_check(INCAR) and Gauss_order2_check(INCAR) wil be difined, they are used to check the order needed by the Gauss_quadrature
# 1. exp(-k^2/4alpha)J_0(k r_c)
# 2. exp(-2k L_z) J_0(k r_c)
# return the Gauss quadrature parameter S = [s, w]


def Gauss_order1_check(INCAR):
    # k_f = INCAR['k_f1']
    # alpha = INCAR['alpha']
    # r_c = INCAR['r_c']
    # accuracy = INCAR['accuracy']
    # Gauss_path = '/home/xzgao/Numerical_method/Integral_Method/GLint/'

    # Analysis_sloution = sqrt(alpha * pi) * exp(- alpha * r_c**2 / 2) * iv(0, alpha * r_c**2 / 2)

    # Gauss_order1 = 2
    # flag = 1
    # while flag == 1:
    #     Error = 1
    #     Guass_para1 = np.loadtxt(Gauss_path + str(Gauss_order1))

    #     Numerical_solution = 0
    #     for para in Guass_para1:
    #         s_i, w_i = para
    #         s_i = k_f * s_i/2 + k_f/2
    #         Numerical_solution += k_f/2 * w_i * exp(- s_i**2/(4*alpha)) * jv(0, s_i* r_c)
        
    #     Error = abs(Numerical_solution - Analysis_sloution)
    #     if Error > accuracy:
    #         Gauss_order1 += 1
    #     else:
    #         flag = 0

    N_t = INCAR['N_t']
    Guass_para1 = np.polynomial.legendre.leggauss(N_t)

    return Guass_para1

def Gauss_order2_check(INCAR):
    # k_f = INCAR['k_f2']
    # alpha = INCAR['alpha']
    # r_c = INCAR['r_c']
    # accuracy = INCAR['accuracy']
    # L_z = INCAR['L_z']
    # Gauss_path = '/home/xzgao/Numerical_method/Integral_Method/GLint/'

    # Analysis_sloution = 1 /(sqrt(r_c**2 + 4 * L_z**2))

    # Gauss_order2 = 2
    # flag = 1
    # while flag == 1:
    #     Error = 1
    #     Guass_para2 = np.loadtxt(Gauss_path + str(Gauss_order2))

    #     Numerical_solution = 0
    #     for para in Guass_para2:
    #         s_i, w_i = para
    #         s_i = k_f * s_i/2 + k_f/2
    #         Numerical_solution += k_f/2 * w_i * exp(- 2 * L_z * s_i) * jv(0, s_i* r_c)
        
    #     Error = abs(Numerical_solution - Analysis_sloution)
    #     if Error > accuracy:
    #         Gauss_order2 += 1
    #     else:
    #         flag = 0

    N_t = INCAR['N_t']
    Guass_para2 = np.polynomial.legendre.leggauss(N_t)
    
    return Guass_para2