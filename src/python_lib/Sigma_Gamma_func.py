import green_function
import numpy as np
from numpy import exp, cos, sin
from green_function import Gamma_func, gamma, z_para, Gamma_para, partial_Gamma_func

def Sigma_Gamma_func_s(k_set, func_para):
    k_x, k_y, K, k = k_set
    POSCAR, INCAR, CELL_LIST = func_para

    NUM_particle = INCAR['NUM_particle']
    L_z = INCAR['L_z']

    f_func = np.zeros(4)
    for i in range(NUM_particle):
        q_i, x_i, y_i, z_i = POSCAR[i]
        f_func[0] += q_i * cos(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func[1] += q_i * sin(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func[2] += q_i * cos(k_x * x_i + k_y * y_i) * exp( - k * z_i)
        f_func[3] += q_i * sin(k_x * x_i + k_y * y_i) * exp( - k * z_i)

    gamma_val = green_function.gamma(k, INCAR, POSCAR, 0, 0)
    Gamma_para_val = green_function.Gamma_para(k, INCAR, POSCAR, 0, 0)

    Sigma_Gamma_func_s_val = Gamma_para_val[0] * gamma_val * (f_func[0]**2 + f_func[1]**2) * exp(-2*k*L_z) + Gamma_para_val[2] * gamma_val * (f_func[2]**2 + f_func[3]**2) + Gamma_para_val[3] * gamma_val * (f_func[0]*f_func[2] + f_func[1]*f_func[3]) * exp(-2*k*L_z)

    Sigma_Gamma_func_s_val = Sigma_Gamma_func_s_val/k

    return Sigma_Gamma_func_s_val


def Sigma_Gamma_func_ns(k_set, func_para):
    k_x, k_y, K, k = k_set
    POSCAR, INCAR, CELL_LIST = func_para

    NUM_particle = INCAR['NUM_particle']
    L_z = INCAR['L_z']

    Z_Cell = CELL_LIST['Z_Cell']
    z_list = []
    for room in Z_Cell:
        for member in room:
            z_list.append(member)

    A = np.zeros(NUM_particle)
    B = np.zeros(NUM_particle)
    C = np.zeros(NUM_particle)
    D = np.zeros(NUM_particle)

    A[0], B[0], C[0], D[0] = [0, 0, 0, 0]
    for i in range(1, NUM_particle):
        l = z_list[i]
        q_i, x_i, y_i, z_i = POSCAR[l]
        C[0] += q_i * exp( - k * z_i) * cos(k_x * x_i + k_y * y_i)
        D[0] += q_i * exp( - k * z_i) * sin(k_x * x_i + k_y * y_i)

    for i in range(0, NUM_particle - 1):
        l = z_list[i]
        lp1 = z_list[i + 1]
        q_i, x_i, y_i, z_i = POSCAR[l]
        q_ip1, x_ip1, y_ip1, z_ip1 = POSCAR[lp1]
        A[i + 1] = A[i] + q_i * exp( + k*z_i) * cos(k_x * x_i + k_y * y_i)
        B[i + 1] = B[i] + q_i * exp( + k*z_i) * sin(k_x * x_i + k_y * y_i)
        C[i + 1] = C[i] - q_ip1 * exp( - k*z_ip1) * cos(k_x * x_ip1 + k_y * y_ip1)
        D[i + 1] = D[i] - q_ip1 * exp( - k*z_ip1) * sin(k_x * x_ip1 + k_y * y_ip1)

    Sigma_Gamma_func_ns_val = 0
    for i in range(NUM_particle):
        l = z_list[i]
        q_i, x_i, y_i, z_i = POSCAR[l]
        Sigma_Gamma_func_ns_val += - q_i**2
        Sigma_Gamma_func_ns_val += - q_i * exp( - k * z_i) * cos(k_x * x_i + k_y * y_i) * A[i]
        Sigma_Gamma_func_ns_val += - q_i * exp( - k * z_i) * sin(k_x * x_i + k_y * y_i) * B[i]
        Sigma_Gamma_func_ns_val += - q_i * exp( + k * z_i) * cos(k_x * x_i + k_y * y_i) * C[i]
        Sigma_Gamma_func_ns_val += - q_i * exp( + k * z_i) * sin(k_x * x_i + k_y * y_i) * D[i]

    Sigma_Gamma_func_ns_val = Sigma_Gamma_func_ns_val/k

    return Sigma_Gamma_func_ns_val/2


def F_long_sum_1(k_set, func_para):
    k_x, k_y, K, k = k_set
    POSCAR, INCAR, CELL_LIST = func_para

    NUM_particle = INCAR['NUM_particle']
    L_z = INCAR['L_z']

    f_func = np.zeros(4)
    for i in range(NUM_particle):
        q_i, x_i, y_i, z_i = POSCAR[i]
        f_func[0] += q_i * cos(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func[1] += q_i * sin(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func[2] += q_i * cos(k_x * x_i + k_y * y_i) * exp( - k * z_i)
        f_func[3] += q_i * sin(k_x * x_i + k_y * y_i) * exp( - k * z_i)

    gamma_val = green_function.gamma(k, INCAR, POSCAR, 0, 0)
    Gamma_para_val = green_function.Gamma_para(k, INCAR, POSCAR, 0, 0)

    F_long_x_sum1_val = np.zeros(NUM_particle)
    F_long_y_sum1_val = np.zeros(NUM_particle)
    F_long_z_sum1_val = np.zeros(NUM_particle)

    for j in range(NUM_particle):
        q_j, x_j, y_j, z_j = POSCAR[j]
        #here the force we calculate is the jth particle in z_list but not the j_number one, it is the lth number one
        sum_core_xy_j = (cos(k_x * x_j + k_y * y_j) * (Gamma_para_val[0] * gamma_val * exp(-2*k*L_z) * exp(k * z_j) * f_func[1] + Gamma_para_val[2] * gamma_val * exp( - k * z_j) * f_func[3] + Gamma_para_val[3] * gamma_val * exp(-2*k*L_z) * exp(-k * z_j) * f_func[1] + Gamma_para_val[3] * gamma_val * exp(-2*k*L_z) * exp(k * z_j) * f_func[3]) - sin(k_x * x_j + k_y * y_j) * (Gamma_para_val[0] * gamma_val * exp(-2*k*L_z) * exp(k * z_j) * f_func[0] + Gamma_para_val[2] * gamma_val * exp( - k * z_j) * f_func[2] + Gamma_para_val[3] * gamma_val * exp(-2*k*L_z) * exp(-k * z_j) * f_func[0] + Gamma_para_val[3] * gamma_val * exp(-2*k*L_z) * exp(k * z_j) * f_func[2]))

        F_long_x_sum1_val_j = (k_x * q_j/k) * sum_core_xy_j
        F_long_y_sum1_val_j = (k_y * q_j/k) * sum_core_xy_j

        sum_core_z_j = (cos(k_x * x_j + k_y * y_j) * (Gamma_para_val[0] * gamma_val * exp(-2*k*L_z) * exp(k * z_j) * f_func[0] - Gamma_para_val[2] * gamma_val * exp( - k * z_j) * f_func[2] - Gamma_para_val[3] * gamma_val * exp(-2*k*L_z) * exp(-k * z_j) * f_func[0] + Gamma_para_val[3] * gamma_val * exp(-2*k*L_z) * exp(k * z_j) * f_func[2]) + sin(k_x * x_j + k_y * y_j) * (Gamma_para_val[0] * gamma_val * exp(-2*k*L_z) * exp(k * z_j) * f_func[1] - Gamma_para_val[2] * gamma_val * exp( - k * z_j) * f_func[3] - Gamma_para_val[3] * gamma_val * exp(-2*k*L_z) * exp(-k * z_j) * f_func[1] + Gamma_para_val[3] * gamma_val * exp(-2*k*L_z) * exp(k * z_j) * f_func[3]))

        F_long_z_sum1_val_j = q_j * sum_core_z_j

        F_long_x_sum1_val[j] = F_long_x_sum1_val_j
        F_long_y_sum1_val[j] = F_long_y_sum1_val_j
        F_long_z_sum1_val[j] = F_long_z_sum1_val_j

    return F_long_x_sum1_val, F_long_y_sum1_val, F_long_z_sum1_val


def F_long_xy_sum_2(k_set, func_para):
    k_x, k_y, K, k = k_set
    POSCAR, INCAR, CELL_LIST = func_para

    NUM_particle = INCAR['NUM_particle']
    L_z = INCAR['L_z']

    Z_Cell = CELL_LIST['Z_Cell']
    z_list = []
    for room in Z_Cell:
        for member in room:
            z_list.append(member)

    A = np.zeros(NUM_particle)
    B = np.zeros(NUM_particle)
    C = np.zeros(NUM_particle)
    D = np.zeros(NUM_particle)

    A[0], B[0], C[0], D[0] = [0, 0, 0, 0]
    for i in range(1, NUM_particle):
        l = z_list[i]
        q_i, x_i, y_i, z_i = POSCAR[l]
        C[0] += q_i * exp( - k * z_i) * cos(k_x * x_i + k_y * y_i)
        D[0] += q_i * exp( - k * z_i) * sin(k_x * x_i + k_y * y_i)

    for i in range(0, NUM_particle - 1):
        l = z_list[i]
        lp1 = z_list[i + 1]
        q_i, x_i, y_i, z_i = POSCAR[l]
        q_ip1, x_ip1, y_ip1, z_ip1 = POSCAR[lp1]
        A[i + 1] = A[i] + q_i * exp( + k*z_i) * cos(k_x * x_i + k_y * y_i)
        B[i + 1] = B[i] + q_i * exp( + k*z_i) * sin(k_x * x_i + k_y * y_i)
        C[i + 1] = C[i] - q_ip1 * exp( - k*z_ip1) * cos(k_x * x_ip1 + k_y * y_ip1)
        D[i + 1] = D[i] - q_ip1 * exp( - k*z_ip1) * sin(k_x * x_ip1 + k_y * y_ip1)


    F_long_x_sum2_val = np.zeros(NUM_particle)
    F_long_y_sum2_val = np.zeros(NUM_particle)

    for j in range(NUM_particle):
        l = z_list[j]
        q_j, x_j, y_j, z_j = POSCAR[l]

        xy_sum2_core = sin(k_x * x_j + k_y * y_j) * exp(-k*z_j) * A[j] - cos(k_x * x_j + k_y * y_j) * exp(-k*z_j) * B[j] + sin(k_x * x_j + k_y * y_j) * exp(k*z_j) * C[j] - cos(k_x * x_j + k_y * y_j) * exp(k*z_j) * D[j]

        F_long_x_sum2_val_j = - q_j * (k_x / (2 * k)) * xy_sum2_core
        F_long_y_sum2_val_j = - q_j * (k_y / (2 * k)) * xy_sum2_core

        z_sum2_core = cos(k_x * x_j + k_y * y_j) * exp(-k*z_j) * A[j] + sin(k_x * x_j + k_y * y_j) * exp(-k*z_j) * B[j] - cos(k_x * x_j + k_y * y_j) * exp(k*z_j) * C[j] - sin(k_x * x_j + k_y * y_j) * exp(k*z_j) * D[j]

        F_long_z_sum2_val_j = + (q_j / 2) * z_sum2_core

        F_long_x_sum2_val[l] = - F_long_x_sum2_val_j
        F_long_y_sum2_val[l] = - F_long_y_sum2_val_j

    return F_long_x_sum2_val, F_long_y_sum2_val


def F_long_z_sum_2(k_set, func_para):
    k_x, k_y, K, k = k_set
    POSCAR, INCAR, CELL_LIST = func_para

    NUM_particle = INCAR['NUM_particle']
    L_z = INCAR['L_z']

    Z_Cell = CELL_LIST['Z_Cell']
    z_list = []
    for room in Z_Cell:
        for member in room:
            z_list.append(member)

    z_0 = POSCAR[z_list[0]][3]
    equal_cell = [[z_list[0]]]
    equal_cell_num = 0
    z_pre = z_0
    for i in range(1, INCAR['NUM_particle']):
        z_mem = z_list[i]
        z_mem_pos = POSCAR[z_mem][3]
        if z_mem_pos > z_pre:
            equal_cell.append([z_mem])
            equal_cell_num += 1
            z_pre = z_mem_pos
        else:
            equal_cell[equal_cell_num].append(z_mem)

    equal_size = [0]
    for equal_cell_room in equal_cell:
        size = np.shape(equal_cell_room)[0]
        equal_size.append(equal_size[-1] + size)

    equal_cell_room_num = np.shape(equal_size)[0] - 1
    A_p = np.zeros(equal_cell_room_num)
    B_p = np.zeros(equal_cell_room_num)
    C_p = np.zeros(equal_cell_room_num)
    D_p = np.zeros(equal_cell_room_num)
    for i in range(0, equal_cell_room_num):
        for member in equal_cell[i]:
            q_i, x_i, y_i, z_i = POSCAR[member]
            A_p[i] += q_i * exp(+k*z_i) * cos(k_x * x_i + k_y * y_i)
            B_p[i] += q_i * exp(+k*z_i) * sin(k_x * x_i + k_y * y_i)
            C_p[i] += q_i * exp(-k*z_i) * cos(k_x * x_i + k_y * y_i)
            D_p[i] += q_i * exp(-k*z_i) * sin(k_x * x_i + k_y * y_i)
    
    NUM_partilce = INCAR['NUM_particle']
    A = np.zeros(NUM_partilce)
    B = np.zeros(NUM_partilce)
    C = np.zeros(NUM_partilce)
    D = np.zeros(NUM_partilce)
    for i in range(1, equal_cell_room_num):
        C[0] += C_p[i]
        D[0] += D_p[i]
    flag = 0
    for i in range(equal_cell_room_num):
        for j in range(equal_size[i], equal_size[i + 1]):
            if j == equal_size[i]:
                if j==0:
                    A[j] = 0
                    B[j] = 0
                else:
                    A[j] = A_p[flag] + A[j - 1]
                    B[j] = B_p[flag] + B[j - 1]
                    C[j] = - C_p[flag + 1] + C[j - 1]
                    D[j] = - D_p[flag + 1] + D[j - 1]
                    flag += 1
            else:
                A[j] = A[j - 1]
                B[j] = B[j - 1]
                C[j] = C[j - 1]
                D[j] = D[j - 1]

    F_long_z_sum2_val = np.zeros(NUM_particle)

    for j in range(NUM_particle):
        l = z_list[j]
        q_j, x_j, y_j, z_j = POSCAR[l]


        z_sum2_core = cos(k_x * x_j + k_y * y_j) * exp(-k*z_j) * A[j] + sin(k_x * x_j + k_y * y_j) * exp(-k*z_j) * B[j] - cos(k_x * x_j + k_y * y_j) * exp(k*z_j) * C[j] - sin(k_x * x_j + k_y * y_j) * exp(k*z_j) * D[j]

        F_long_z_sum2_val_j = + (q_j / 2) * z_sum2_core

        F_long_z_sum2_val[l] = - F_long_z_sum2_val_j
    
    return F_long_z_sum2_val

def F_long_z_sum_0(func_para):
    POSCAR, INCAR, CELL_LIST = func_para

    NUM_particle = INCAR['NUM_particle']
    L_z = INCAR['L_z']

    Z_Cell = CELL_LIST['Z_Cell']
    z_list = []
    for room in Z_Cell:
        for member in room:
            z_list.append(member)

    z_0 = POSCAR[z_list[0]][3]
    equal_cell = [[z_list[0]]]
    equal_cell_num = 0
    z_pre = z_0
    for i in range(1, INCAR['NUM_particle']):
        z_mem = z_list[i]
        z_mem_pos = POSCAR[z_mem][3]
        if z_mem_pos > z_pre:
            equal_cell.append([z_mem])
            equal_cell_num += 1
            z_pre = z_mem_pos
        else:
            equal_cell[equal_cell_num].append(z_mem)

    equal_size = [0]
    for equal_cell_room in equal_cell:
        size = np.shape(equal_cell_room)[0]
        equal_size.append(equal_size[-1] + size)

    equal_cell_room_num = np.shape(equal_size)[0] - 1
    Q_ap = np.zeros(equal_cell_room_num)
    Q_bp = np.zeros(equal_cell_room_num)
    for i in range(0, equal_cell_room_num):
        for member in equal_cell[i]:
            q_i, x_i, y_i, z_i = POSCAR[member]
            Q_ap[i] += q_i
            Q_bp[i] += q_i
    
    NUM_partilce = INCAR['NUM_particle']
    Q_a = np.zeros(NUM_partilce)
    Q_b = np.zeros(NUM_partilce)
    for i in range(1, equal_cell_room_num):
        Q_b[0] += Q_bp[i]
    flag = 0
    for i in range(equal_cell_room_num):
        for j in range(equal_size[i], equal_size[i + 1]):
            if j == equal_size[i]:
                if j==0:
                    Q_a[j] = 0
                else:
                    Q_a[j] = Q_ap[flag] + Q_a[j - 1]
                    Q_b[j] = - Q_bp[flag + 1] + Q_b[j - 1]
                    flag += 1
            else:
                Q_a[j] = Q_a[j - 1]
                Q_b[j] = Q_b[j - 1]

    F_long_z_sum0_val = np.zeros(NUM_particle)


    for j in range(NUM_particle):
        l = z_list[j]
        q_j, x_j, y_j, z_j = POSCAR[l]


        z_sum2_core = Q_a[j] - Q_b[j]

        F_long_z_sum0_val_j = + q_j * z_sum2_core

        F_long_z_sum0_val[l] = F_long_z_sum0_val_j
    
    return F_long_z_sum0_val


def Force_long_z_self(k_set, func_para):
    k_x, k_y, K, k = k_set
    POSCAR, INCAR, CELL_LIST = func_para

    NUM_particle = INCAR['NUM_particle']
    L_z = INCAR['L_z']

    Force_long_z_self_val = np.zeros(NUM_particle)
    for j in range(NUM_particle):
        q_j, x_j, y_j, z_j = POSCAR[j]
        Gamma = Gamma_func(k, INCAR, POSCAR, j, j)
        p_Gamma = partial_Gamma_func(k, INCAR, POSCAR, j, j)
        Gamma_para_val = Gamma_para(k, INCAR, POSCAR, j, j)
        gamma_val = gamma(k, INCAR, POSCAR, j, j)

        Force_long_z_self_val_j = Gamma_para_val[0] * gamma_val * exp(-2*k*L_z) * exp(k * (z_j + z_j)) - Gamma_para_val[2] * gamma_val * exp(-k * (z_j + z_j))

        Force_long_z_self_val[j] = Force_long_z_self_val_j * q_j**2

    return Force_long_z_self_val