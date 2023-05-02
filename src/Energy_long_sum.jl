export direct_sum_k, direct_sum_k0, energy_k0_sum, energy_k_sum_1, energy_k_sum_2, energy_k_sum_3, energy_k_sum_4, energy_k_sum_total, energy_sum_total


function direct_sum_k0(q, coords, L_x, L_y)
    n_atoms = size(coords)[1]

    # k = 0 part
    sum_k0 = 0
    for i in 1:n_atoms
        for j in 1:n_atoms
            sum_k0 += q[i] * q[j] * abs(coords[i][3] - coords[j][3])
        end
    end

    return sum_k0 / (4 * L_x * L_y)
end

function direct_sum_k(q, coords, L_x, L_y, L_z, kc, alpha, g_1, g_2)
    n_atoms = size(coords)[1]
    
    sum_k = 0
    n_x_max = trunc(Int, kc * L_x / 2 * π)
    n_y_max = trunc(Int, kc * L_y / 2 * π)

    for n_x in - n_x_max : n_x_max
        for n_y in - n_y_max : n_y_max
            k_x = 2 * π * n_x / L_x
            k_y = 2 * π * n_y / L_y
            k = sqrt(k_x^2 + k_y^2)
            if k < kc && k != 0
                beta = (g_1 * g_2 * exp(-2 * k * L_z) - 1)
                sum_1 = 0
                sum_2 = 0
                sum_3 = 0
                sum_4 = 0
                for i in 1:n_atoms
                    for j in 1:n_atoms
                        xi, yi, zi = coords[i]
                        xj, yj, zj = coords[j]
                        qc = q[i] * q[j] *  cos(k_x * (xi - xj) + k_y * (yi - yj)) / (beta * k)
                        sum_1 += qc * exp(-k * abs(zi - zj)) 
                        sum_2 += g_1 * qc * exp(-k * (zi + zj)) 
                        sum_3 += g_2 * qc * exp(-k * (2 * L_z - zi - zj))
                        sum_4 += g_1 * g_2 * qc * exp(-k * (2 * L_z - abs(zi - zj)))
                    end
                end
                sum_k += (sum_1 + sum_2 + sum_3 + sum_4) * exp(-k^2 / (4 * alpha))
            end
        end
    end

    return sum_k / (4 * L_x * L_y)
end

function energy_k0_sum(q, coords, z_list)
    n_atoms = size(coords)[1]

    Q_1, Q_2 = [zeros(n_atoms) for i in 1:2]

    for i in 2:n_atoms
        l = z_list[i - 1]
        Q_1[i] = Q_1[i - 1] + q[l]
        Q_2[i] = Q_2[i - 1] + q[l] * coords[l][3]
    end

    sum_k0 = 0
    for i in 1:n_atoms
        l = z_list[i]
        q_i = q[l]
        z_i = coords[l][3]
        sum_k0 += q_i * z_i * Q_1[i] - q_i * Q_2[i]
    end

    return 2 * sum_k0
end

function energy_k_sum_1(k_set, q, coords, z_list, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 1 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

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

function energy_k_sum_2(k_set, q, coords, z_list, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 1 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

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

function energy_k_sum_3(k_set, q, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 1 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

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

function energy_k_sum_4(k_set, q, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 1 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

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

function energy_k_sum_total(k_set, q, coords, z_list, green_element::greens_element)
    n_atoms = size(coords)[1]
    # green_ele = greens_element_init(g_1, g_2, L_z, alpha)

    sum_k_total = 
        energy_k_sum_1(k_set, q, coords, z_list, green_element) + 
        energy_k_sum_2(k_set, q, coords, z_list, green_element) + 
        energy_k_sum_3(k_set, q, coords, green_element) + 
        energy_k_sum_4(k_set, q, coords, green_element)

    return sum_k_total
end


function energy_sum_total(q, coords, alpha, L_x, L_y, L_z, g_1, g_2, eps_0, k_c)
    n_atoms = size(coords)[1]
    z_coords = [coords[i][3] for i in 1:n_atoms]
    z_list = sortperm(z_coords)

    sum_k0 = energy_k0_sum(q, coords, z_list)
    
    sum_total = 0
    n_x_max = trunc(Int, k_c * L_x / 2 * π)
    n_y_max = trunc(Int, k_c * L_y / 2 * π)

    green_element = greens_element_init(g_1, g_2, L_z, alpha)

    for n_x in - n_x_max : n_x_max
        for n_y in - n_y_max : n_y_max
            k_x = 2 * π * n_x / L_x
            k_y = 2 * π * n_y / L_y
            k = sqrt(k_x^2 + k_y^2)
            k_set = (k_x, k_y, k)
            if k < k_c && k != 0
                sum_k = energy_k_sum_total(k_set, q, coords, z_list, green_element)
                sum_total += sum_k * exp(-k^2 / (4 * alpha))
            end
        end
    end

    return - sum_k0 / (4 * L_x * L_y * eps_0) - sum_total / (4 * L_x * L_y * eps_0)
end