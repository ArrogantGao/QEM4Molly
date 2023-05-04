export force_k_sum_total, force_k0_sum, force_sum_total, direct_sum_F, direct_Fxy, direct_Fz

function force_k_sum_1(k_set, q, coords, z_list, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 1 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

    # sum_xy = beta(k) exp(-k z_n) sin(kx (xi - xj) + ky(yi - yj)) = beta(k) exp(-k zn) (sin(k ri) cos(k rj) - cos(k ri) sin(k rj))
    # sum_z = beta(k) -sign(z_n) exp(-k z_n) cos(kx (xi - xj) + ky(yi - yj)) = beta(k) -sign(z_n) exp(-k zn) (cos(k ri) cos(k rj) + sin(k ri) sin(k rj))

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

    sum_x = zeros(n_atoms)
    sum_y = zeros(n_atoms)
    sum_z = zeros(n_atoms)
    # sum_x[i] = q_i sin(k \rho_i)(exp(k z_i) C1[i] + exp(-k z_i) C2[i]) -
    #            q_i cos(k \rho_i)(exp(k z_i) S1[i] + exp(-k z_i) S2[i])
    for i in 1:n_atoms
        l = z_list[i]
        q_l = q[l]
        x_l, y_l, z_l = coords[l]
        sum_ri = q_l * (
            sin(k_x * x_l + k_y * y_l) * (
                exp(-k * z_l) * C1[i] + exp(+k * z_l) * C2[i]
            ) -
            cos(k_x * x_l + k_y * y_l) * (
                exp(-k * z_l) * S1[i] + exp(+k * z_l) * S2[i]
            )
        )
        sum_zi = q_l * (
            cos(k_x * x_l + k_y * y_l) * ( - exp(-k * z_l) * C1[i] + exp(+k * z_l) * C2[i]) +
            sin(k_x * x_l + k_y * y_l) * ( - exp(-k * z_l) * S1[i] + exp(+k * z_l) * S2[i]) )

        sum_x[l] = k_x * sum_ri
        sum_y[l] = k_y * sum_ri
        sum_z[l] = sum_zi
    end

    return sum_x .* (beta / k), sum_y .* (beta / k), sum_z .* beta
end

function force_k_sum_2(k_set, q, coords, z_list, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 1 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

    # there are two terms to be summed
    # sum_xy = sum_ij beta(k) g_1 g_2 exp(-k(2 * L_z - z_n)) cos(kx(xi - xj) + ky(xi - xj))

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

    sum_x = zeros(n_atoms)
    sum_y = zeros(n_atoms)
    sum_z = zeros(n_atoms)
    # sum = \sum_i q_i sin(k \rho_i)(exp(k z_i) C1[i] + exp(-k z_i) C2[i]) +
    #              q_i cos(k \rho_i)(exp(k z_i) S1[i] + exp(-k z_i) S2[i])
    for i in 1:n_atoms
        l = z_list[i]
        q_l = q[l]
        x_l, y_l, z_l = coords[l]
        sum_ri = q_l * (
            sin(k_x * x_l + k_y * y_l) * (exp(-k * (2 * L_z - z_l)) * C1[i] + exp(-k * z_l) * C2[i]) -
            cos(k_x * x_l + k_y * y_l) * (exp(-k * (2 * L_z - z_l)) * S1[i] + exp(-k * z_l) * S2[i]))

        sum_zi = q_l * (
            cos(k_x * x_l + k_y * y_l) * (exp(-k * (2 * L_z - z_l)) * C1[i] - exp(-k * z_l) * C2[i]) +
            sin(k_x * x_l + k_y * y_l) * (exp(-k * (2 * L_z - z_l)) * S1[i] - exp(-k * z_l) * S2[i]))

        sum_x[l] = k_x * sum_ri
        sum_y[l] = k_y * sum_ri
        sum_z[l] = sum_zi
    end

    return sum_x .* (g_1 * g_2 * beta / k), sum_y .* (g_1 * g_2 * beta / k), sum_z .* (g_1 * g_2 * beta)
end

function force_k_sum_3(k_set, q, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 1 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

    # in this function, the summation
    # sum_xy = \sum_{i!=j} beta(k) \gamma_1 q_i q_j exp(- k * z_p) sin(k \rho_ij)
    #     = \sum_{i!=j} beta(k) \gamma_1 q_i q_j exp(- k * z_p)[sin(k rho_i) cos(k rho_j) - cos(k rho_i) sin(k rho_j)]
    #     = beta(k) * \gamma_1 * sum_{i} q_i exp(-k * z_i) {sin(k rho_i) [sum_{j!=i} q_j exp(- k * z_j) cos(k rho_j)] - cos(k rho_i) [sum_{j!=i} q_j exp(- k * z_j) sin(k rho_j)]}
    # sum_z = \sum_{j} - g1 qi qj exp(-k(zi + zj)) cos(k rho_ij)

    C = 0
    S = 0
    for i in 1:n_atoms
        qi = q[i]
        x_i, y_i, z_i = coords[i]
        val = qi * exp(- k * z_i)
        C += val * cos(k_x * x_i + k_y * y_i)
        S += val * sin(k_x * x_i + k_y * y_i)
    end

    sum_x = zeros(n_atoms)
    sum_y = zeros(n_atoms)
    sum_z = zeros(n_atoms)
    for i in 1:n_atoms
        qi = q[i]
        x_i, y_i, z_i = coords[i]
        val = qi * exp(- k * z_i)
        ci = cos(k_x * x_i + k_y * y_i)
        si = sin(k_x * x_i + k_y * y_i)
        sum_ri = val * (si * C - ci * S)
        sum_x[i] = k_x * sum_ri
        sum_y[i] = k_y * sum_ri
        sum_z[i] = - val * (ci * C + si * S)
    end
    return sum_x .* (beta * g_1 / k), sum_y .* (beta * g_1 / k), sum_z .* (beta * g_1)
end

function force_k_sum_4(k_set, q, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z
    # q = [atoms.charge[i] for i in 1:n_atoms]
    alpha = element.alpha

    g_1 = element.gamma_1
    g_2 = element.gamma_2
    beta = 1 / (g_1 * g_2 * exp(- 2 * k * L_z) - 1)

    # in this function, the summation
    # sum = \sum_{i!=j} beta(k) \gamma_1 q_i q_j exp(- k * (2L_z - z_p)) sin(k \rho_ij)
    #     = \sum_{i!=j} beta(k) \gamma_1 q_i q_j exp(- k * (2L_z - z_p))[sin(k rho_i) cos(k rho_j) - cos(k rho_i) sin(k rho_j)]
    #     = beta(k) * \gamma_1 * sum_{i} q_i exp(-k * (L_z - z_i)) {sin(k rho_i) [sum_{j!=i} q_j exp(- k * (L_z - z_j)) cos(k rho_j)] - cos(k rho_i) [sum_{j!=i} q_j exp(- k * (L_z - z_j)) sin(k rho_j)]}
    
    C = 0
    S = 0
    for i in 1:n_atoms
        qi = q[i]
        x_i, y_i, z_i = coords[i]
        val = qi * exp(- k * (L_z - z_i))
        C += val * cos(k_x * x_i + k_y * y_i)
        S += val * sin(k_x * x_i + k_y * y_i)
    end

    sum_x = zeros(n_atoms)
    sum_y = zeros(n_atoms)
    sum_z = zeros(n_atoms)
    for i in 1:n_atoms
        qi = q[i]
        x_i, y_i, z_i = coords[i]
        val = qi * exp(- k * (L_z - z_i))
        ci = cos(k_x * x_i + k_y * y_i)
        si = sin(k_x * x_i + k_y * y_i)
        sum_ri = val * (si * C - ci * S)
        sum_x[i] = k_x * sum_ri
        sum_y[i] = k_y * sum_ri
        sum_z[i] = val * (ci * C + si * S)
    end
    return sum_x .* (beta * g_2 / k), sum_y .* (beta * g_2 / k), sum_z .* (beta * g_2)
end

function force_k_sum_total(k_set, q, coords, z_list, green_element::greens_element)

    sum_k_total = 
        force_k_sum_1(k_set, q, coords, z_list, green_element) .+ 
        force_k_sum_2(k_set, q, coords, z_list, green_element) .+ 
        force_k_sum_3(k_set, q, coords, green_element) .+ 
        force_k_sum_4(k_set, q, coords, green_element)

    return sum_k_total
end

# this part is for the k = 0 part summation for Fz
function force_k0_sum(q, z_list)

    # this function is used to calculate the summation given by
    # q_i sum_j q_j sign(z_i - z_j) = q_i (sum_{j < i} q_j - sum_{j > i} q_j)

    n_atoms = size(q)[1]
    
    Q1 = zeros(n_atoms)
    Q2 = zeros(n_atoms)
    for i in 2:n_atoms
        lf = z_list[i - 1]
        Q1[i] = Q1[i - 1] + q[lf]

        ib = n_atoms - i + 1
        lb = z_list[ib + 1]
        Q2[ib] = Q2[ib + 1] + q[lb]
    end

    sum_k0 = zeros(n_atoms)
    for i in 1:n_atoms
        l = z_list[i]
        sum_k0[l] = q[l] * (Q1[i] - Q2[i])
    end
    return sum_k0
end

function force_sum_total(q, coords, alpha, L_x, L_y, L_z, g_1, g_2, eps_0, k_c)
    n_atoms = size(coords)[1]
    z_coords = [coords[i][3] for i in 1:n_atoms]
    z_list = sortperm(z_coords)

    sum_x = zeros(n_atoms)
    sum_y = zeros(n_atoms)
    sum_z = zeros(n_atoms)

    sum_k0 = force_k0_sum(q, z_list)
    
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
                sum_kx, sum_ky, sum_kz = force_k_sum_total(k_set, q, coords, z_list, green_element)
                sum_x .+= sum_kx .* exp(-k^2 / (4 * alpha))
                sum_y .+= sum_ky .* exp(-k^2 / (4 * alpha))
                sum_z .+= sum_kz .* exp(-k^2 / (4 * alpha))
            end
        end
    end

    Fx = - sum_x ./ (2 * L_x * L_y * eps_0)
    Fy = - sum_y ./ (2 * L_x * L_y * eps_0)
    Fz = (sum_k0 .+ sum_z) ./ (2 * L_x * L_y * eps_0)

    return Fx, Fy, Fz
end

# the methods given below are the direct summation methods.

function direct_sum_F(k_set, q, coords, L_x, L_y, L_z, kc, alpha, g_1, g_2)
    
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]


    beta = (g_1 * g_2 * exp(-2 * k * L_z) - 1)
    sum_1 = zeros(n_atoms)
    sum_2 = zeros(n_atoms)
    sum_3 = zeros(n_atoms)
    sum_4 = zeros(n_atoms)
    sumz_1 = zeros(n_atoms)
    sumz_2 = zeros(n_atoms)
    sumz_3 = zeros(n_atoms)
    sumz_4 = zeros(n_atoms)
    for i in 1:n_atoms
        for j in 1:n_atoms
            xi, yi, zi = coords[i]
            xj, yj, zj = coords[j]
            if j != i
                qs = q[i] * q[j] *  sin(k_x * (xi - xj) + k_y * (yi - yj)) / (beta * k)
                sum_1[i] += qs * exp(-k * abs(zi - zj)) 
                sum_2[i] += g_1 * qs * exp(-k * (zi + zj)) 
                sum_3[i] += g_2 * qs * exp(-k * (2 * L_z - zi - zj))
                sum_4[i] += g_1 * g_2 * qs * exp(-k * (2 * L_z - abs(zi - zj)))
            end
            qc = q[i] * q[j] *  cos(k_x * (xi - xj) + k_y * (yi - yj)) / beta
            sumz_1[i] += - sign(zi - zj) * qc * exp(-k * abs(zi - zj)) 
            sumz_2[i] += - g_1 * qc * exp(-k * (zi + zj)) 
            sumz_3[i] += + g_2 * qc * exp(-k * (2 * L_z - zi - zj))
            sumz_4[i] += + sign(zi - zj) *g_1 * g_2 * qc * exp(-k * (2 * L_z - abs(zi - zj)))
        end
    end
    
    sum_x = k_x .* (sum_1 .+ sum_4 .+ sum_2 .+ sum_3)
    sum_y = k_y .* (sum_1 .+ sum_4 .+ sum_2.+ sum_3)
    sum_z = (sumz_1 .+ sumz_4 .+ sumz_2.+ sumz_3)

    return sum_x, sum_y, sum_z
end

function direct_Fxy(q, coords, L_x, L_y, L_z, kc, alpha, g_1, g_2)
    
    n_atoms = size(coords)[1]
    sum_x = zeros(n_atoms)
    sum_y = zeros(n_atoms)

    n_x_max = trunc(Int, kc * L_x / 2 * π)
    n_y_max = trunc(Int, kc * L_y / 2 * π)

    for n_x in - n_x_max : n_x_max
        for n_y in - n_y_max : n_y_max
            k_x = 2 * π * n_x / L_x
            k_y = 2 * π * n_y / L_y
            k = sqrt(k_x^2 + k_y^2)
            if k < kc && k != 0
                beta = (g_1 * g_2 * exp(-2 * k * L_z) - 1)
                sum_1 = zeros(n_atoms)
                sum_2 = zeros(n_atoms)
                sum_3 = zeros(n_atoms)
                sum_4 = zeros(n_atoms)
                for i in 1:n_atoms
                    for j in 1:n_atoms
                        if j != i
                            xi, yi, zi = coords[i]
                            xj, yj, zj = coords[j]
                            qc = q[i] * q[j] *  sin(k_x * (xi - xj) + k_y * (yi - yj)) / (beta * k)
                            sum_1[i] += qc * exp(-k * abs(zi - zj)) 
                            sum_2[i] += g_1 * qc * exp(-k * (zi + zj)) 
                            sum_3[i] += g_2 * qc * exp(-k * (2 * L_z - zi - zj))
                            sum_4[i] += g_1 * g_2 * qc * exp(-k * (2 * L_z - abs(zi - zj)))
                        end
                    end
                end
                sum_k = (sum_1 .+ sum_2 .+ sum_3 .+ sum_4) .* exp(-k^2 / (4 * alpha))
                sum_x .+= k_x .* sum_k
                sum_y .+= k_y .* sum_k
            end
        end
    end

    Fx = - sum_x ./ (2 * L_x * L_y)
    Fy = - sum_y ./ (2 * L_x * L_y)

    return Fx, Fy
end

# here I will compare the result of direct sum and compare that with the result of ICM 

function direct_Fz(q, coords, L_x, L_y, L_z, kc, alpha, g_1, g_2)
    n_atoms = size(coords)[1]

    sum_z = zeros(n_atoms)

    sum_0 = zeros(n_atoms)
    for i in 1:n_atoms
        for j in 1:n_atoms
            xi, yi, zi = coords[i]
            xj, yj, zj = coords[j]
            sum_0[i] += q[i] * q[j] * sign(zi - zj)
        end
    end

    sum_z .+= sum_0

    n_x_max = trunc(Int, kc * L_x / 2 * π)
    n_y_max = trunc(Int, kc * L_y / 2 * π)
    for n_x in - n_x_max : n_x_max
        for n_y in - n_y_max : n_y_max
            k_x = 2 * π * n_x / L_x
            k_y = 2 * π * n_y / L_y
            k = sqrt(k_x^2 + k_y^2)
            if k < kc && k != 0
                beta = (g_1 * g_2 * exp(-2 * k * L_z) - 1)
                sum_1 = zeros(n_atoms)
                sum_2 = zeros(n_atoms)
                sum_3 = zeros(n_atoms)
                sum_4 = zeros(n_atoms)
                for i in 1:n_atoms
                    for j in 1:n_atoms
                        # if j != i
                            xi, yi, zi = coords[i]
                            xj, yj, zj = coords[j]
                            qc = q[i] * q[j] *  cos(k_x * (xi - xj) + k_y * (yi - yj)) / (beta)
                            sum_1[i] += - sign(zi - zj) * qc * exp(-k * abs(zi - zj)) 
                            sum_2[i] += - g_1 * qc * exp(-k * (zi + zj)) 
                            sum_3[i] += + g_2 * qc * exp(-k * (2 * L_z - zi - zj))
                            sum_4[i] += + sign(zi - zj) * g_1 * g_2 * qc * exp(-k * (2 * L_z - abs(zi - zj)))
                        # end
                    end
                end
                sum_z .+= (sum_1 .+ sum_2 .+ sum_3 .+ sum_4) .* exp(-k^2 / (4 * alpha))
            end
        end
    end

    Fz = sum_z ./ (2 * L_x * L_y)

    return Fz
end