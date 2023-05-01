export El_sort_k0, El_sort, El_nonsort, E_long

function El_sort_k0(atoms, z_list, coords, element::greens_element)
    n_atoms = size(coords)[1]

    Q_1, Q_2 = [zeros(n_atoms) for i in 1:2]

    for i in 2:n_atoms 
        l = z_list[i]
        Q_2[1] += atoms[l].charge
    end

    for i in 1:(n_atoms - 1)
        l = z_list[i]
        q_i = atoms[l].charge
        Q_1[i + 1] = Q_1[i] + q_i
    end

    for i in 2:n_atoms 
        l = z_list[i]
        q_i = atoms[l].charge
        Q_2[i] = Q_2[i - 1] - q_i
    end

    El_sort_k0_val = 0

    for j in 1:n_atoms
        l = z_list[j]
        q_i = atoms[l].charge
        z_i = coords[l][3]
        El_sort_k0_val += q_i * z_i * (Q_1[j] - Q_2[j])
    end

    return El_sort_k0_val
end

function El_nonsort(k_set, atoms, coords, element::greens_element)
    k_x, k_y, k = k_set
    @assert k_x^2 + k_y^2 ≈ k^2
    n_atoms = size(coords)[1]
    L_z = element.L_z
    
    f_func = zeros(4)
    for i in 1 : n_atoms
        q_i = atoms[i].charge
        x_i, y_i, z_i = coords[i]
        f_func[1] += q_i * cos(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func[2] += q_i * sin(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func[3] += q_i * cos(k_x * x_i + k_y * y_i) * exp( - k * z_i)
        f_func[4] += q_i * sin(k_x * x_i + k_y * y_i) * exp( - k * z_i)
    end

    gamma_val = 0.5 / (element.gamma_1 * element.gamma_2 * exp(- 2 * k * L_z) - 1)
    Gamma_para = [element.gamma_2, 1, element.gamma_1, element.gamma_1 * element.gamma_2]

    El_nonsort_val = gamma_val*(
        Gamma_para[1] * (f_func[1]^2 + f_func[2]^2) * exp(-2*k*L_z) + 
        Gamma_para[3] * (f_func[3]^2 + f_func[4]^2) + 
        Gamma_para[4] * (f_func[1] * f_func[3] + f_func[2] * f_func[4]) * exp(-2*k*L_z)
    ) / k

    return El_nonsort_val
end

function El_sort(k_set, atoms, z_list, coords, element::greens_element)
    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z

    A, B, C, D = [zeros(n_atoms) for i in 1:4]

    for i in 2:n_atoms
        l = z_list[i]
        q_i = atoms[l].charge
        x_i, y_i, z_i = coords[l]
        C[1] += q_i * exp( - k * z_i) * cos(k_x * x_i + k_y * y_i)
        D[1] += q_i * exp( - k * z_i) * sin(k_x * x_i + k_y * y_i)
    end

    for i in 1:n_atoms - 1
        l = z_list[i]
        lp1 = z_list[i + 1]
        q_i = atoms[l].charge
        x_i, y_i, z_i = coords[l]
        q_ip1 = atoms[lp1].charge
        x_ip1, y_ip1, z_ip1 = coords[lp1]

        A[i + 1] = A[i] + q_i * exp( + k*z_i) * cos(k_x * x_i + k_y * y_i)
        B[i + 1] = B[i] + q_i * exp( + k*z_i) * sin(k_x * x_i + k_y * y_i)
        C[i + 1] = C[i] - q_ip1 * exp( - k*z_ip1) * cos(k_x * x_ip1 + k_y * y_ip1)
        D[i + 1] = D[i] - q_ip1 * exp( - k*z_ip1) * sin(k_x * x_ip1 + k_y * y_ip1)
    end
    
    El_sort_val = 0
    for i in 1:n_atoms
        l = z_list[i]
        q_i = atoms[l].charge
        x_i, y_i, z_i = coords[l]
        El_sort_val += - q_i^2
        El_sort_val += - q_i * exp( - k * z_i) * cos(k_x * x_i + k_y * y_i) * A[i]
        El_sort_val += - q_i * exp( - k * z_i) * sin(k_x * x_i + k_y * y_i) * B[i]
        El_sort_val += - q_i * exp( + k * z_i) * cos(k_x * x_i + k_y * y_i) * C[i]
        El_sort_val += - q_i * exp( + k * z_i) * sin(k_x * x_i + k_y * y_i) * D[i]
    end

    return El_sort_val/(2 * k)
    
end

# function E_l_total(k_set, sys, inter, element::greens_element)
#     L_x, L_y, L_z = inter.L
#     eps_0 = inter.eps_0

#     El_nonsort_val = El_nonsort(k_set, sys.atoms, sys.coords, element)
#     El_sort_val = El_sort(k_set, sys.atoms, inter.z_list, sys.coords, element)

#     El_k = - (El_nonsort_val + El_sort_val) / (2 * L_x * L_y * eps_0)

#     return El_k
# end

function E_l_total(k_set, sys, inter, element::greens_element; direct_sum::Bool = false)
    coords = sys.coords
    z_list = inter.z_list
    n_atoms = size(coords)[1]
    charge_vec = [charge(sys.atoms[i]) for i in 1:n_atoms]
    L_x, L_y, L_z = inter.L
    eps_0 = inter.eps_0

    if direct_sum == false
        sum = energy_sort_1(k_set, charge_vec, z_list, coords, element) +
        energy_sort_2(k_set, charge_vec, z_list, coords, element) +
        energy_non_sort_1(k_set, charge_vec, coords, element) +
        energy_non_sort_2(k_set, charge_vec, coords, element)
    elseif direct_sum == true
        sum = direct_sum_1(k_set, charge_vec, z_list, coords, element) +
        direct_sum_2(k_set, charge_vec, z_list, coords, element) +
        direct_sum_3(k_set, charge_vec, coords, element) +
        direct_sum_4(k_set, charge_vec, coords, element)
    end

    El_k = - sum / (2 * L_x * L_y * eps_0)
    return El_k
end

function E_long(sys, inter::QEM_long; direct_sum::Bool = false)
    gamma_1 = inter.gamma_1
    gamma_2 = inter.gamma_2
    eps_0 = inter.eps_0
    L_x, L_y, L_z = inter.L
    alpha = inter.alpha
    n_atoms = size(sys.coords)[1]

    E_long_val = 0

    # here we set z_i = z_j = rho_ij = 0
    element = greens_element_init(gamma_1, gamma_2, L_z, inter.alpha)

    El_sort_k0_val = - El_sort_k0(sys.atoms, inter.z_list, sys.coords, element) / (2 * eps_0 * L_x * L_y)
    E_long_val += El_sort_k0_val

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
                    E_long_val += E_l_total(k_set, sys, inter, element; direct_sum = direct_sum) * exp(-k^2/(4 * alpha))
                end
            end
        end
    elseif inter.rbe_mode == true
        K_p = sample(inter.K_set, inter.Prob, inter.rbe_p)
        for k_set in K_p
            E_long_val += E_l_total(k_set, sys, inter, element) * (inter.sum_K / inter.rbe_p)
        end
    end
    
    return E_long_val
end