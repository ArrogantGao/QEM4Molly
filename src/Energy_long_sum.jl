export energy_sort_1, energy_sort_2, energy_non_sort_1, energy_non_sort_2, direct_sum_1, direct_sum_2, direct_sum_3, direct_sum_4

function energy_sort_1(k_set, charge, z_list, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    q = charge
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 0.5 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

    # there are two terms to be summed
    # sum_1 = beta(k) exp(-k z_n)
    # sum_2 = beta(k) g_1 g_2 exp(-k(2 * L_z - z_n))

    # first for sum_1
    # C1 for the forward process and C2 for the backward, same for S1/2
    # C1[i] = \sum_{j<i} q_j exp(+k z_j) cos(k \rho_j)
    # C2[i] = \sum_{j>i} q_j exp(-k z_j) cos(k \rho_j)
    # S1[i] = \sum_{j<i} q_j exp(+k z_j) sin(k \rho_j)
    # S2[i] = \sum_{j>i} q_j exp(-k z_j) sin(k \rho_j)
    C1, C2, S1, S2 = [zeros(n_atoms) for i in 1:4]


    for i in 1:n_atoms-1
        #forward process
        lf = z_list[i]
        q_lf = q[lf]
        x_lf, y_lf, z_lf = coords[lf]
        forward_val = q_lf * exp(k * z_lf)
        C1[i + 1] = C1[i] + forward_val * cos(k_x * x_lf + k_y * y_lf)
        S1[i + 1] = S1[i] + forward_val * sin(k_x * x_lf + k_y * y_lf)
    
        #backward process
        back_i = n_atoms - i
        lb = z_list[back_i + 1]
        q_lb = q[lb]
        x_lb, y_lb, z_lb = coords[lb]
        backward_val = q_lb * exp(-k * z_lb)
        C2[back_i] = C2[back_i + 1] + backward_val * cos(k_x * x_lb + k_y * y_lb)
        S2[back_i] = S2[back_i + 1] + backward_val * sin(k_x * x_lb + k_y * y_lb)
    end

    sum = 0
    # sum = \sum_i q_i cos(k \rho_i)(exp(k z_i) C1[i] + exp(-k z_i) C2[i]) +
    #              q_i sin(k \rho_i)(exp(k z_i) S1[i] + exp(-k z_i) S2[i])
    for i in 1:n_atoms
        l = z_list[i]
        q_l = q[l]
        x_l, y_l, z_l = coords[l]
        sum += q_l * (
            cos(k_x * x_l + k_y * y_l) * (
                exp(-k * z_l) * C1[i] + exp(+k * z_l) * C2[i]
            ) + 
            sin(k_x * x_l + k_y * y_l) * (
                exp(-k * z_l) * S1[i] + exp(+k * z_l) * S2[i]
            ) + 
            q_l
        )
    end

    return sum * beta / k
end

function direct_sum_1(k_set, charge, z_list, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    q = charge
    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 0.5 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)
    alpha = element.alpha
    
    sum = 0
    for i in 1:n_atoms
        for j in 1:n_atoms
            qi = q[i]
            qj = q[j]
            xi, yi, zi = coords[i]
            xj, yj, zj = coords[j]
            sum += qi * qj * exp( - k * abs(zi - zj)) * cos(k_x * (xi - xj) + k_y * (yi - yj))
        end
    end

    return sum * beta / k
end

function energy_sort_2(k_set, charge, z_list, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    q = charge
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 0.5 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

    # there are two terms to be summed
    # sum_1 = sum_ij beta(k) exp(-k z_n)
    # sum_2 = sum_ij beta(k) g_1 g_2 exp(-k(2 * L_z - z_n))

    # here we will handle sum_2
    # sum_2 = sum_i q_i (sum_{j<i} q_j exp(-k(2*L_z - (z_i - z_j))) + sum_{j>i} q_j exp(-k(2*L_z - (z_j - z_i))))
    # = sum_i q_i (exp(-k(2*L_z - z_i)) sum_{j<i} q_j exp(-k * z_j) + exp(-k * z_i) sum_{j>i} q_j exp(-k(2*L_z - z_j)))
    # C1 for the forward process and C2 for the backward, same for S1/2
    # C1[i] = \sum_{j<i} q_j exp(-k z_j) cos(k \rho_j)
    # C2[i] = \sum_{j>i} q_j exp(-k (2L_z - z_j)) cos(k \rho_j)
    # S1[i] = \sum_{j<i} q_j exp(-k z_j) sin(k \rho_j)
    # S2[i] = \sum_{j>i} q_j exp(-k (2L_z - z_j)) sin(k \rho_j)
    C1, C2, S1, S2 = [zeros(n_atoms) for i in 1:4]

    for i in 1:n_atoms-1
        #forward process
        lf = z_list[i]
        q_lf = q[lf]
        x_lf, y_lf, z_lf = coords[lf]
        forward_val = q_lf * exp(- k * z_lf)
        C1[i + 1] = C1[i] + forward_val * cos(k_x * x_lf + k_y * y_lf)
        S1[i + 1] = S1[i] + forward_val * sin(k_x * x_lf + k_y * y_lf)
    
        #backward process
        back_i = n_atoms - i
        lb = z_list[back_i + 1]
        q_lb = q[lb]
        x_lb, y_lb, z_lb = coords[lb]
        backward_val = q_lb * exp(-k * (2 * L_z - z_lb))
        C2[back_i] = C2[back_i + 1] + backward_val * cos(k_x * x_lb + k_y * y_lb)
        S2[back_i] = S2[back_i + 1] + backward_val * sin(k_x * x_lb + k_y * y_lb)
    end

    sum = 0
    # sum = \sum_i q_i cos(k \rho_i)(exp(k z_i) C1[i] + exp(-k z_i) C2[i]) +
    #              q_i sin(k \rho_i)(exp(k z_i) S1[i] + exp(-k z_i) S2[i])
    for i in 1:n_atoms
        l = z_list[i]
        q_l = q[l]
        x_l, y_l, z_l = coords[l]
        sum += q_l * (
            cos(k_x * x_l + k_y * y_l) * (
                exp(-k * (2 * L_z - z_l)) * C1[i] + exp(-k * z_l) * C2[i]
            ) + 
            sin(k_x * x_l + k_y * y_l) * (
                exp(-k * (2 * L_z - z_l)) * S1[i] + exp(-k * z_l) * S2[i]
            ) + 
            q_l * exp(-2 * k * L_z)
        )
    end

    return g_1 * g_2 * sum * beta / k
end

function direct_sum_2(k_set, charge, z_list, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    q = charge
    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 0.5 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)
    alpha = element.alpha
    
    sum = 0
    for i in 1:n_atoms
        for j in 1:n_atoms
            qi = q[i]
            qj = q[j]
            xi, yi, zi = coords[i]
            xj, yj, zj = coords[j]
            sum += qi * qj * exp( - k * (2 * L_z - abs(zi - zj))) * cos(k_x * (xi - xj) + k_y * (yi - yj))
        end
    end

    return g_1 * g_2 * sum * beta / k
end


function energy_non_sort_1(k_set, charge, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    q = charge
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 0.5 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

    # in this function, the summation
    # sum = \sum_{i!=j} beta(k) \gamma_1 q_i q_j exp(- k * z_p) cos(k \rho_ij)
    #     = \sum_{i!=j} beta(k) \gamma_1 q_i q_j exp(- k * z_p)[cos(k rho_i) cos(k rho_j) + sin(k rho_i) sin(k rho_j)]
    #     = beta(k) * \gamma_1 * sum_{i} q_i exp(-k * z_i) {cos(k rho_i) [sum_{j!=i} q_j exp(- k * z_j) cos(k rho_j)] + sin(k rho_i) [sum_{j!=i} q_j exp(- k * z_j) sin(k rho_j)]}
    
    C = 0
    S = 0
    for i in 1:n_atoms
        qi = q[i]
        x_i, y_i, z_i = coords[i]
        val = qi * exp(- k * z_i)
        C += val * cos(k_x * x_i + k_y * y_i)
        S += val * sin(k_x * x_i + k_y * y_i)
    end

    sum = 0
    for i in 1:n_atoms
        qi = q[i]
        x_i, y_i, z_i = coords[i]
        val = qi * exp(- k * z_i)
        ci = cos(k_x * x_i + k_y * y_i)
        si = sin(k_x * x_i + k_y * y_i)
        sum += val * (ci * C + si * S)
    end
    return beta * g_1 * sum / k
end

function direct_sum_3(k_set, charge, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    q = charge
    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 0.5 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)
    alpha = element.alpha
    
    sum = 0
    for i in 1:n_atoms
        for j in 1:n_atoms
            qi = q[i]
            qj = q[j]
            xi, yi, zi = coords[i]
            xj, yj, zj = coords[j]
            sum += qi * qj * exp( - k * (zi + zj)) * cos(k_x * (xi - xj) + k_y * (yi - yj))
        end
    end

    return g_1 * sum * beta / k
end

function energy_non_sort_2(k_set, charge, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    q = charge
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 0.5 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

    # in this function, the summation
    # sum = \sum_{i!=j} beta(k) \gamma_1 q_i q_j exp(- k * (2L_z - z_p)) cos(k \rho_ij)
    #     = \sum_{i!=j} beta(k) \gamma_1 q_i q_j exp(- k * (2L_z - z_p))[cos(k rho_i) cos(k rho_j) + sin(k rho_i) sin(k rho_j)]
    #     = beta(k) * \gamma_1 * sum_{i} q_i exp(-k * (L_z - z_i)) {cos(k rho_i) [sum_{j!=i} q_j exp(- k * (L_z - z_j)) cos(k rho_j)] + sin(k rho_i) [sum_{j!=i} q_j exp(- k * (L_z - z_j)) sin(k rho_j)]}
    
    C = 0
    S = 0
    for i in 1:n_atoms
        qi = q[i]
        x_i, y_i, z_i = coords[i]
        val = qi * exp(- k * (L_z - z_i))
        C += val * cos(k_x * x_i + k_y * y_i)
        S += val * sin(k_x * x_i + k_y * y_i)
    end

    sum = 0
    for i in 1:n_atoms
        qi = q[i]
        x_i, y_i, z_i = coords[i]
        val = qi * exp(- k * (L_z - z_i))
        ci = cos(k_x * x_i + k_y * y_i)
        si = sin(k_x * x_i + k_y * y_i)
        sum += val * (ci * C + si * S)
    end
    return beta * g_2 * sum / k
end

function direct_sum_4(k_set, charge, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    q = charge
    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 0.5 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)
    alpha = element.alpha
    
    sum = 0
    for i in 1:n_atoms
        for j in 1:n_atoms
            qi = q[i]
            qj = q[j]
            xi, yi, zi = coords[i]
            xj, yj, zj = coords[j]
            sum += qi * qj * exp( - k * (2 * L_z - zi - zj)) * cos(k_x * (xi - xj) + k_y * (yi - yj))
        end
    end

    return g_2 * sum * beta / k
end