export F_long

function F_long(sys, inter::QEM_long)
    gamma_1 = inter.gamma_1
    gamma_2 = inter.gamma_2
    eps_0 = inter.eps_0
    L_x, L_y, L_z = inter.L
    alpha = inter.alpha
    n_atoms = size(sys.coords)[1]
    q = [charge(sys.atoms[i]) for i in 1:n_atoms]
    coords = sys.coords

    # this is a naive version to generate the z_list, costing time O(NlogN), the result will be given in ascending order
    z_coord = [coord[3] for coord in sys.coords]
    z_list = sortperm(z_coord)

    # here we set z_i = z_j = rho_ij = 0
    green_element = greens_element_init(gamma_1, gamma_2, L_z, alpha)


    sum_k0 = force_k0_sum(q, z_list)
    sum_x = zeros(n_atoms)
    sum_y = zeros(n_atoms)
    sum_z = zeros(n_atoms)
    
    # this part is related to k, so here we choose rbe_mode on/off
    if inter.rbe_mode == false
        n_x_max = trunc(Int, inter.k_cutoff * L_x / 2 * π)
        n_y_max = trunc(Int, inter.k_cutoff * L_y / 2 * π)
        for n_x in - n_x_max : n_x_max
            for n_y in - n_y_max : n_y_max
                k_x = 2 * π * n_x / L_x
                k_y = 2 * π * n_y / L_y
                k = sqrt(k_x^2 + k_y^2)
                if k < inter.k_cutoff && k != 0
                    k_set = (k_x, k_y, k)
                    sum_kx, sum_ky, sum_kz = force_k_sum_total(k_set, q, coords, z_list, green_element)
                    sum_x .+= sum_kx .* exp(-k^2 / (4 * alpha))
                    sum_y .+= sum_ky .* exp(-k^2 / (4 * alpha))
                    sum_z .+= sum_kz .* exp(-k^2 / (4 * alpha))
                end
            end
        end
    elseif inter.rbe_mode == true
        K_p = sample(inter.K_set, inter.Prob, inter.rbe_p)
        for k_set in K_p
            sum_kx, sum_ky, sum_kz = force_k_sum_total(k_set, q, coords, z_list, green_element) 
            sum_x .+= sum_kx .* (inter.sum_K / inter.rbe_p)
            sum_y .+= sum_ky .* (inter.sum_K / inter.rbe_p)
            sum_z .+= sum_kz .* (inter.sum_K / inter.rbe_p)
        end
    end

    Fx = - sum_x ./ (2 * L_x * L_y * eps_0)
    Fy = - sum_y ./ (2 * L_x * L_y * eps_0)
    Fz = (sum_k0 .+ sum_z) ./ (2 * L_x * L_y * eps_0)

    F_long_val = [[Fx[i], Fy[i], Fz[i]] for i in 1:n_atoms]

    return F_long_val
end


