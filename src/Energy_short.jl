export E_short_ij, E_short_i, E_s_gauss_c, E_s_point_c

# here I will define the integrands of the short range interaction potential_energy

function E_s_gauss_c(k, element::greens_element; l::Int = 0)
    E_s_g = Gamma_1(k, element; l = l) * exp(- k^2 / (4 * element.alpha)) * besselj0(k * element.rho_ij)
    return E_s_g
end

function E_s_point_c(k, element::greens_element; l::Int = 0)
    E_s_p = Gamma_2(k, element; l = l) * besselj0(k * element.rho_ij)
    return E_s_p
end

function E_short_ij(q_i, q_j, coord_i, coord_j, inter::QEM_short; single::Bool = false)
    gamma_1 = inter.gamma_1
    gamma_2 = inter.gamma_2
    eps_0 = inter.eps_0
    L_x, L_y, L_z = inter.L


    x_i, y_i, z_i = coord_i
    x_j, y_j, z_j = coord_j

    #using a pairwise periodic neighborlist to get rho_ij
    n_list = neighborlist([[x_i, y_i], [x_j, y_j]], inter.r_cutoff; unitcell = [L_x, L_y])

    E_ij = 0.0
    Gauss_para = inter.Gauss_para

    for (i, j, rho_ij) in n_list
        element = greens_element_ij_init(gamma_1, gamma_2, coord_i[3], coord_j[3], rho_ij, L_z, inter.alpha, inter.accuracy)
        k_f1 = maximum(element.k_f1)
        k_f2 = maximum(element.k_f2)

        if rho_ij != 0

            E_s_p_ij_1 = Gauss_int(E_s_point_c, Gauss_para, element, region = (0.0, k_f2))
            E_s_p_ij_2 = 0.5 * sum(element.b[l] / sqrt(element.a[l]^2 + rho_ij^2) for l in 1:4)

            if single == false
                E_s_g_ij = Gauss_int(E_s_gauss_c, Gauss_para, element, region = (0.0, k_f1))
            else
                E_s_g_ij = 0.0
            end
            # println("julia: ", E_s_p_ij_1, ' ', E_s_p_ij_2, ' ', E_s_g_ij)

            E_s_ij = - E_s_p_ij_1 + E_s_p_ij_2 + E_s_g_ij
        else
            E_s_ij = 0.0
        end

        E_ij += q_i * q_j / (2 * π * eps_0) * E_s_ij
    end

    return E_ij
end

# this function is used to calculate the self interaction energy of particle i

function E_short_i(q_i, coord_i, inter; single::Bool = false)
    gamma_1 = inter.gamma_1
    gamma_2 = inter.gamma_2
    eps_0 = inter.eps_0
    L_x, L_y, L_z = inter.L


    x_i, y_i, z_i = coord_i

    Gauss_para = inter.Gauss_para

    element = greens_element_i_init(gamma_1, gamma_2, z_i, L_z, inter.alpha, inter.accuracy)
    k_f1 = maximum(element.k_f1)
    k_f2 = maximum(element.k_f2)

    E_s_p_i_1 = Gauss_int(E_s_point_c, Gauss_para, element, region = (0.0, k_f2))
    E_s_p_i_2 = 0.5 * sum(element.b[l] / (element.a[l]) for l in [2, 3, 4])
    if single == false
        E_s_g_i = Gauss_int(E_s_gauss_c, Gauss_para, element, region = (0.0, k_f1))
    else
        E_s_g_i = 0
    end

    E_s_i = - E_s_p_i_1 + E_s_p_i_2 + E_s_g_i
    E_i = q_i^2 / (4 * π * eps_0) * E_s_i

    return E_i
end