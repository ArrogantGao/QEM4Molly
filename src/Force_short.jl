export F_short_ij, F_short_i

# to check the cell label for two particles
function distance_check(coord_i, coord_j, L_x, L_y, r_c)
    xi, yi, zi = coord_i
    xj, yj, zj = coord_j
    for mx in [0, -1, 1]
        for my in [0, -1, 1]
            rho = sqrt((xi - xj + mx * L_x)^2 + (yi - yj + my * L_y)^2)
            if rho < r_c
                return [xi + mx * L_x, yi + my * L_y, zi], [xj, yj, zj], rho
            end
        end
    end
    return nothing
end

#here I will define the integrand of the short range forces
function F_sr_gauss_c(k, element::greens_element; l::Int = 0)
    f_sr_g = k * Gamma_1(k, element; l = l) * exp(- k^2 / (4 * element.alpha)) * besselj1(k * element.rho_ij)
    return f_sr_g
end

function F_sr_point_c(k, element::greens_element; l::Int = 0)
    f_sr_p = k * Gamma_2(k, element; l = l) * besselj1(k * element.rho_ij)
    return f_sr_p
end

function F_sz_gauss_c(k, element::greens_element; l::Int = 0)
    f_sz_g = dz_Gamma_1(k, element; l = l) * exp(- k^2 / (4 * element.alpha)) * besselj0(k * element.rho_ij)
    return f_sz_g
end

function F_sz_point_c(k, element::greens_element; l::Int = 0)
    f_sz_p = dz_Gamma_2(k, element; l = l) * besselj0(k * element.rho_ij)
    return f_sz_p
end

# the function to calculate the pairwise interaction between i and j
function F_short_ij(q_i, q_j, coord_i, coord_j, inter::QEM_short; single::Bool = false)
    gamma_1 = inter.gamma_1
    gamma_2 = inter.gamma_2
    eps_0 = inter.eps_0
    L_x, L_y, L_z = inter.L
    r_c = inter.r_cutoff


    x_i, y_i, z_i = coord_i
    x_j, y_j, z_j = coord_j

    n_list = distance_check(coord_i, coord_j, L_x, L_y, r_c)

    #using a pairwise periodic neighborlist to get rho_ij
    # n_list = neighborlist([[x_i, y_i], [x_j, y_j]], inter.r_cutoff; unitcell = [L_x, L_y])

    #init the F_i and F_j
    F_i = [0.0, 0.0, 0.0]
    F_j = [0.0, 0.0, 0.0]

    Gauss_para = inter.Gauss_para

    if n_list !== nothing
        coord_i, coord_j, rho_ij = n_list
        element_i = greens_element_ij_init(gamma_1, gamma_2, coord_i[3], coord_j[3], rho_ij, L_z, inter.alpha, inter.accuracy)
        element_j = greens_element_ij_init(gamma_1, gamma_2, coord_j[3], coord_i[3], rho_ij, L_z, inter.alpha, inter.accuracy)

        k_f1 = maximum(element_i.k_f1)
        k_f2 = maximum(element_i.k_f2)


        
        if rho_ij != 0

            F_sr_p_ij_1 = Gauss_int(F_sr_point_c, Gauss_para, element_i, region = (0.0, k_f2))
            F_sr_p_ij_2 = 0.5 * sum(element_i.b[l] * rho_ij / (element_i.a[l]^2 + rho_ij^2)^1.5 for l in 1:4)
            if single == false
                F_sr_g_ij = Gauss_int(F_sr_gauss_c, Gauss_para, element_i, region = (0.0, k_f1))
            else
                F_sr_g_ij = 0
            end

            F_sr_ij = - F_sr_p_ij_1 + F_sr_p_ij_2 + F_sr_g_ij
            F_sx_ij = F_sr_ij * (coord_i[1] - coord_j[1])/rho_ij
            F_sy_ij = F_sr_ij * (coord_i[2] - coord_j[2])/rho_ij
        else
            F_sx_ij = 0
            F_sy_ij = 0
        end

        # here I found that the force in z is different for particle i and j.
        # so 
        F_sz_p_i_1 = Gauss_int(F_sz_point_c, Gauss_para, element_i, region = (0.0, k_f2))
        F_sz_p_i_2 = 0.5 * sum(element_i.b[l] * element_i.a[l] * element_i.sign_a[l] / (element_i.a[l]^2 + rho_ij^2)^1.5 for l in 1:4)
        if single == false
            F_sz_g_i = Gauss_int(F_sz_gauss_c, Gauss_para, element_i, region = (0.0, k_f1))
        else
            F_sz_g_i = 0
        end
        F_sz_i = + F_sz_p_i_1 - F_sz_p_i_2 - F_sz_g_i

        F_sz_p_j_1 = Gauss_int(F_sz_point_c, Gauss_para, element_j, region = (0.0, k_f2))
        F_sz_p_j_2 = 0.5 * sum(element_j.b[l] * element_j.a[l] * element_j.sign_a[l] / (element_j.a[l]^2 + rho_ij^2)^1.5 for l in 1:4)
        if single == false
            F_sz_g_j = Gauss_int(F_sz_gauss_c, Gauss_para, element_j, region = (0.0, k_f1))
        else
            F_sz_g_j = 0
        end
        F_sz_j = + F_sz_p_j_1 - F_sz_p_j_2 - F_sz_g_j

        F_i = q_i * q_j / (2 * π * eps_0) * [F_sx_ij, F_sy_ij, F_sz_i]
        F_j = q_i * q_j / (2 * π * eps_0) * [ - F_sx_ij, - F_sy_ij, F_sz_j]
    end

    return F_i, F_j
end

# this function is used to calculate the self interaction in z direction on particle i

function F_short_i(q_i, coord_i, inter; single::Bool = false)
    gamma_1 = inter.gamma_1
    gamma_2 = inter.gamma_2
    eps_0 = inter.eps_0
    L_x, L_y, L_z = inter.L


    x_i, y_i, z_i = coord_i

    element = greens_element_i_init(gamma_1, gamma_2, z_i, L_z, inter.alpha, inter.accuracy)
    k_f1 = maximum(element.k_f1)
    k_f2 = maximum(element.k_f2)

    Gauss_para = inter.Gauss_para

    F_sz_p_i_1 = Gauss_int(F_sz_point_c, Gauss_para, element, region = (0.0, k_f2))
    F_sz_p_i_2 = 0.5 * sum(element.b[l] * element.a[l] * element.sign_a[l] / (element.a[l]^3) for l in [2, 3])
    if single == false
        F_sz_g_i = Gauss_int(F_sz_gauss_c, Gauss_para, element, region = (0.0, k_f1))
    else
        F_sz_g_i = 0
    end

    F_sz_i = - F_sz_p_i_1 + F_sz_p_i_2 + F_sz_g_i
    F_i = - q_i^2 / (2 * π * eps_0) * [0, 0, F_sz_i]

    return F_i
end