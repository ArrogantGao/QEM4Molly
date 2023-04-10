import numpy as np
from numpy import pi, sqrt
import numba as nb
import matplotlib.pyplot as plt


#print(scale_up, scale_down)
@nb.njit()
def rho_cal(pos_1, pos_2):
    x_1, y_1, z_1 = pos_1
    x_2, y_2, z_2 = pos_2
    rho = np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)
    return rho

@nb.njit()
def rho_s_cal(pos_1, pos_2):
    x_1, y_1, z_1 = pos_1
    x_2, y_2, z_2 = pos_2
    rho_s = (x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2
    return rho_s

@nb.njit()
def dir_cal(pos_1, pos_2):
    x_1, y_1, z_1 = pos_1
    x_2, y_2, z_2 = pos_2
    rho = rho_cal(pos_1, pos_2)
    return np.array([(x_1 - x_2)/rho, (y_1 - y_2)/rho, (z_1 - z_2)/rho])

@nb.njit()
def reflect_up(pos_info, L_z, scale_up):
    q, x, y, z = pos_info
    z_new = 2 * L_z - z
    q_new = scale_up * q
    return np.array([q_new, x, y, z_new])

@nb.njit()
def reflect_down(pos_info, L_z, scale_down):
    q, x, y, z = pos_info
    z_new = - z
    q_new = scale_down * q
    return np.array([q_new, x, y, z_new])

@nb.njit()
def U_ij(pos_i, pos_j, m_x, m_y, L_x, L_y, eps_2):
    q_i, x_i, y_i, z_i = pos_i
    q_j, x_j, y_j, z_j = pos_j
    R = (x_i - x_j + m_x * L_x)**2 + (y_i - y_j + m_y * L_y)**2 + (z_i - z_j)**2
    r = sqrt(R)
    if r == 0:
        return 0
    U_ij = 0.5 * q_i * q_j / (4 * pi * eps_2 * r)
    return U_ij


@nb.njit()
def F_ij_cal(pos_i, pos_j, m_x, m_y, L_x, L_y, eps_2):
    q_i, x_i, y_i, z_i = pos_i
    q_j, x_j, y_j, z_j = pos_j
    pos_1 = np.array([x_i + m_x * L_x, y_i + m_y * L_y, z_i])
    pos_2 = np.array([x_j, y_j, z_j])
    R = rho_s_cal(pos_1, pos_2)
    # print(R)
    if R == 0:
        return np.zeros(3)
    else:
        F = q_i * q_j / (4 * pi * eps_2 * R)
        cosx, cosy, cosz = dir_cal(pos_1, pos_2)
        return np.array([cosx * F, cosy * F, cosz * F])

@nb.njit()
def F_cal(INCAR, POSCAR, POS, N_image, N_real):

    # N_real = 100 # the grid in xy plane considered


    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    N = int(N)

    scale_up = (eps_2 - eps_3)/(eps_2 + eps_3)
    scale_down = (eps_2 - eps_1)/(eps_2 + eps_1)
    POS[N_image] = POSCAR

    for num_layer in range(N_image):
        #print(POS)
        up_source = N_image - num_layer
        up_target = N_image + num_layer + 1
        down_source = N_image + num_layer
        down_target = N_image - num_layer - 1
        for num_q in range(N):
            up_source_pos = POS[up_source][num_q]
            up_target_pos = reflect_up(up_source_pos, L_z, scale_up)
            POS[up_target][num_q] = up_target_pos

            down_source_pos = POS[down_source][num_q]
            down_target_pos = reflect_down(down_source_pos, L_z, scale_down)
            POS[down_target][num_q] = down_target_pos

    Fx_array = np.zeros(N)
    Fy_array = np.zeros(N)
    Fz_array = np.zeros(N)
    Fx_self_array = np.zeros(N)
    Fy_self_array = np.zeros(N)
    Fz_self_array = np.zeros(N)
    for num_particle_i in range(N):
        pos_i = POS[N_image][num_particle_i]
        for m_x in range(-N_real, N_real + 1):
            for m_y in range(-N_real, N_real + 1):
                for num_layer in range(2 * N_image + 1):
                    for num_particle_j in range(N):
                        pos_j = POS[num_layer][num_particle_j]
                        F_ij = np.zeros(3)
                        if num_particle_i != num_particle_j:
                            F_ij = F_ij_cal(pos_i, pos_j, m_x, m_y, L_x, L_y, eps_2)
                            Fx_array[num_particle_i] += F_ij[0]
                            Fy_array[num_particle_i] += F_ij[1]
                            Fz_array[num_particle_i] += F_ij[2]
                        else:
                            F_ij = F_ij_cal(pos_i, pos_j, m_x, m_y, L_x, L_y, eps_2)
                            Fx_self_array[num_particle_i] += F_ij[0]
                            Fy_self_array[num_particle_i] += F_ij[1]
                            Fz_self_array[num_particle_i] += F_ij[2]
    # print(Fx_array, Fy_array, Fz_array)
    return Fx_array, Fy_array, Fz_array, Fx_self_array, Fy_self_array, Fz_self_array




@nb.njit()
def U_cal(INCAR, POSCAR, POS, N_image, N_real):

     # the grid in xy plane considered


    N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
    N = int(N)

    scale_up = (eps_2 - eps_3)/(eps_2 + eps_3)
    scale_down = (eps_2 - eps_1)/(eps_2 + eps_1)
    POS[N_image] = POSCAR

    for num_layer in range(N_image):
        #print(POS)
        up_source = N_image - num_layer
        up_target = N_image + num_layer + 1
        down_source = N_image + num_layer
        down_target = N_image - num_layer - 1
        for num_q in range(N):
            up_source_pos = POS[up_source][num_q]
            up_target_pos = reflect_up(up_source_pos, L_z, scale_up)
            POS[up_target][num_q] = up_target_pos

            down_source_pos = POS[down_source][num_q]
            down_target_pos = reflect_down(down_source_pos, L_z, scale_down)
            POS[down_target][num_q] = down_target_pos

    U_total = 0
    for num_particle_i in range(N):
        pos_i = POS[N_image][num_particle_i]
        for m_x in range(-N_real, N_real + 1):
            for m_y in range(-N_real, N_real + 1):
                for num_layer in range(2 * N_image + 1):
                    for num_particle_j in range(N):
                        pos_j = POS[num_layer][num_particle_j]
                        U_total += U_ij(pos_i, pos_j, m_x, m_y, L_x, L_y, eps_2)

    print('U_total = ')
    print(U_total)
    return U_total



# q_i, x_i, y_i, z_i = 1, 15, 15, 1
# q_j, x_j, y_j, z_j = 1, 15, 15, 1
# STEP = 50

# INCAR = np.loadtxt('./INPUT/INCAR')
# POSCAR = np.loadtxt('./INPUT/POSCAR')
# N_image = 10
# N_real = 10
# N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = INCAR
# file_name = 'F_z_dir'

# gamma = [0.95, -0.95]

# F_X = []
# for g in gamma:
#     F_x_list = []
#     X = []
#     eps = (1 - g)/ (1 + g)
#     INCAR = N, scale, 30, 30, 2, eps, 1, eps, alpha, cutoff
#     for x_step in range(STEP):
#         print(x_step)
#         z_j = 1 + (x_step + 1) * 1 / (STEP)
#         pos_info_1 = np.array([[q_i, x_i, y_i, z_i], [q_j, x_j, y_j ,z_j]])
#         POS = np.zeros([2 * int(N_image) + 1, int(N), 4])
#         F_x, F_y, F_z = F_cal(INCAR, pos_info_1, POS, N_image, N_real)
#         F_x_list.append(F_z[0])
#         X.append(x_j)
#     F_X.append(F_x_list)
# np.save(file_name, F_X)



# U = move(INCAR, N_image, POS)
# np.savetxt('U', U)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x_j_array, y_j_array, z_j_array)
# plt.show()
# U_cal(INCAR, POSCAR, POS, N_image)
