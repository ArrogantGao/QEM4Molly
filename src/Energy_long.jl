export E_long

function E_long(sys, inter::QEM_long)
    g_1 = inter.gamma_1
    g_2 = inter.gamma_2
    eps_0 = inter.eps_0
    L_x, L_y, L_z = inter.L
    alpha = inter.alpha
    n_atoms = size(sys.coords)[1]
    coords = sys.coords

    E_long_val = 0

    # here we set z_i = z_j = rho_ij = 0
    green_element = greens_element_init(g_1, g_2, L_z, alpha)
    z_coords = [coords[i][3] for i in 1:n_atoms]
    z_list = sortperm(z_coords)
    q = [charge(sys.atoms[i]) for i in 1:n_atoms]

    sum_k0 = - energy_k0_sum(q, coords, z_list) / (4 * L_x * L_y * eps_0)
    E_long_val += sum_k0

    sum_total = 0
    k_c = inter.k_cutoff
    # this part is related to k, so here we choose rbe_mode on/off
    if inter.rbe_mode == false
        n_x_max = trunc(Int, k_c * L_x / 2 * π)
        n_y_max = trunc(Int, k_c * L_y / 2 * π)
        for n_x in - n_x_max : n_x_max
            for n_y in - n_y_max : n_y_max
                k_x = 2 * π * n_x / L_x
                k_y = 2 * π * n_y / L_y
                k = sqrt(k_x^2 + k_y^2)
                if k < k_c && k != 0
                    k_set = (k_x, k_y, k)
                    sum_k = energy_k_sum_total(k_set, q, coords, z_list, green_element) * exp(-k^2 / (4 * alpha))
                    sum_total += - sum_k / (4 * L_x * L_y * eps_0)
                end
            end
        end
    elseif inter.rbe_mode == true
        K_p = sample(inter.K_set, inter.Prob, inter.rbe_p)
        for k_set in K_p
            sum_k = energy_k_sum_total(k_set, q, coords, z_list, green_element)* (inter.sum_K / inter.rbe_p)
            sum_total += - sum_k / (4 * L_x * L_y * eps_0)
        end
    end

    E_long_val += sum_total
    
    return E_long_val
end