from numpy import exp, sign, log
import numpy as np
import numba as nb


@nb.njit()
def gamma(k, eps_1, eps_2, eps_3, L_z):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g = 1 / (gamma_1* gamma_2 * exp(-2 * k * L_z) - 1)
    return g

@nb.njit()
def gamma_norm(k, eps_1, eps_2, eps_3, L_z):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    k_0 = log(g12) / (2 * L_z)
    x = 2 * L_z * (k - k_0)
    u = 0.5
    d = -1 + x / 2
    # for i in range(10):
    #     u += (-x)**(i + 1) / np.math.factorial(i + 3)
    #     d += x**(i + 2) * (- 1)**(i + 1) / np.math.factorial(i + 3)
    u_0 = 1 / 2
    d_0 = x / 2
    for i in range(10):
        u_0 = u_0 * (-x) / (i + 3)
        d_0 = d_0 * (-x) / (i + 3)
        u += u_0
        d += d_0
    return u / d

@nb.njit()
def green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    z_n = abs(z_i - z_j)
    z_p = z_i + z_j

    gu = 0.5 * (exp(-k * z_n) + gamma_1 * exp(- k * z_p) + gamma_2 * exp(- k *(2 * L_z - z_p)) + gamma_1 * gamma_2 * exp(- k *(2 * L_z - z_n)))

    return gu

@nb.njit()
# here dz is for dz_j
def dz_green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    z_n = abs(z_i - z_j)
    sign_z_n = sign(z_i - z_j)
    z_p = z_i + z_j

    dz_gu = 0.5 * (k * sign_z_n * exp(-k * z_n) - k * gamma_1 * exp(- k * z_p) + k * gamma_2 * exp(- k *(2 * L_z - z_p)) - k * sign_z_n * gamma_1 * gamma_2 * exp(- k *(2 * L_z - z_n)))

    return dz_gu

@nb.njit()
def Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    G = green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * gamma(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def Gamma_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    G = green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * gamma_norm(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def Gamma_a_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z)
    return G

@nb.njit()
def Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z) * gamma(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def Gamma_a_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z) * gamma_norm(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def dz_Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    G = dz_green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * gamma(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def dz_Gamma_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    G = dz_green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * gamma_norm(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def dz_Gamma_a_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * dz_green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z)
    return G

@nb.njit()
def dz_Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * dz_green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z) * gamma(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def dz_Gamma_a_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * dz_green_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z) * gamma_norm(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def dz_Gamma_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    z_p = z_i + z_j

    dz_gsu = 0.5 * (- k * gamma_1 * exp(- k * z_p) + k * gamma_2 * exp(- k *(2 * L_z - z_p)))

    return dz_gsu

@nb.njit()
def dz_Gamma_a_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * dz_Gamma_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z)
    return G

@nb.njit()
def dz_Gamma_self(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    G = dz_Gamma_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * gamma(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def dz_Gamma_self_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    G = dz_Gamma_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * gamma_norm(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def dz_Gamma_a_self(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * dz_Gamma_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z) * gamma(k, eps_1, eps_2, eps_3, L_z)
    return G

@nb.njit()
def dz_Gamma_a_self_norm(k, eps_1, eps_2, eps_3, L_z, z_j, z_i):
    gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
    gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)
    g12 = gamma_1 * gamma_2
    G = g12 * dz_Gamma_self_up(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) * exp(- 2 * k * L_z) * gamma_norm(k, eps_1, eps_2, eps_3, L_z)
    return G