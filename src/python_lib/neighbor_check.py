import numpy as np
from numpy import sqrt
from math import floor
import numba as nb

# In this script, the function to get the neighbor hood of the center particle will be difined. In that function particles in the surrounding cells will be checked and if rho<r_c, the sign of this particle will be add to the neighbor_list and returned

@nb.njit()
def rho_cal(pos_1, pos_2):
    q_1, x_1, y_1, z_1 = pos_1
    q_2, x_2, y_2, z_2 = pos_2
    delta_x = x_1 - x_2
    delta_y = y_1 - y_2
    rho = sqrt(delta_x**2 + delta_y**2)
    return rho

@nb.njit()
def r_cal(pos_1, pos_2):
    q_1, x_1, y_1, z_1 = pos_1
    q_2, x_2, y_2, z_2 = pos_2
    delta_x = x_1 - x_2
    delta_y = y_1 - y_2
    delta_z = z_1 - z_2
    r = sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    return r

@nb.njit()
def normalize(pos_2, X_neighbor_roomnum, Y_neighbor_roomnum, X_Cell_num, Y_Cell_num, L_x, L_y):
    q_j, x_j, y_j, z_j = pos_2
    if X_neighbor_roomnum == -1: # if the X_neighbor_roomnum<0, it is because X_center_roonum = 0 and mx = -1, x_j here is at the L_x boundary, we should renormalize it so that x_i< 0
        x_j -= L_x
    if X_neighbor_roomnum == X_Cell_num:
        x_j += L_x
    if Y_neighbor_roomnum == -1:
        y_j -= L_y
    if Y_neighbor_roomnum == Y_Cell_num:
        y_j += L_y
    return np.array([q_j, x_j, y_j, z_j])



def neighbor_check(center_sign, POSCAR, INCAR, CELL_LIST):
    XY_Cell = CELL_LIST['XY_Cell']
    NUM_particle = INCAR['NUM_particle']
    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    L_z = INCAR['L_z']
    alpha = INCAR['alpha']
    r_c = INCAR['r_c']

    X_Cell_num, Y_Cell_num, Z_Cell_num, X_Cell_length, Y_Cell_length, Z_Cell_length = INCAR['Cell_info']

    q_i, x_i, y_i, z_i = POSCAR[center_sign]
    X_Cell_roomnum = floor(x_i/X_Cell_length)
    Y_Cell_roomnum = floor(y_i/Y_Cell_length)

    neighbor_list = []
    for m_x in [-1, 0, 1]:
        for m_y in [-1, 0, 1]:
            X_neighbor_roomnum = (X_Cell_roomnum + m_x)
            Y_neighbor_roomnum = (Y_Cell_roomnum + m_y)
            X_neighbor_roomnum_normalized = X_neighbor_roomnum % X_Cell_num
            Y_neighbor_roomnum_normalized = Y_neighbor_roomnum % Y_Cell_num
            for neighbor_member in XY_Cell[X_neighbor_roomnum_normalized][Y_neighbor_roomnum_normalized]:
                q_j, x_j, y_j, z_j = normalize(POSCAR[neighbor_member], X_neighbor_roomnum, Y_neighbor_roomnum, X_Cell_num, Y_Cell_num, L_x, L_y)

                rho = rho_cal(np.array(POSCAR[center_sign]), np.array([q_j, x_j, y_j, z_j]))
                r = r_cal(np.array(POSCAR[center_sign]), np.array([q_j, x_j, y_j, z_j]))
                if rho < r_c and r != 0:
                    neighbor_list.append([q_j, x_j, y_j, z_j, neighbor_member])

    return neighbor_list