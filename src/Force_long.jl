export Fl_non_sort, Fl_sort, Flz_sort_k0, Flz_self, F_l_total, F_long

# Important: The Gamma_para defined in Fl_non_sort here is different from the b defined in the greens.elements

function Fl_non_sort(k_set, atoms, coords, element::greens_element)
    k_x, k_y, k = k_set
    @assert k_x^2 + k_y^2 ≈ k^2
    n_atoms = size(coords)[1]
    L_z = element.L_z

    f_func_0 = zeros(4)
    for i in 1:n_atoms
        q_i = atoms[i].charge
        x_i, y_i, z_i = coords[i]
        f_func_0[1] += q_i * cos(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func_0[2] += q_i * sin(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func_0[3] += q_i * cos(k_x * x_i + k_y * y_i) * exp( - k * z_i)
        f_func_0[4] += q_i * sin(k_x * x_i + k_y * y_i) * exp( - k * z_i)
    end

    gamma_val = 0.5 / (element.gamma_1 * element.gamma_2 * exp(- 2 * k * L_z) - 1)
    Gamma_para = [element.gamma_2, 1, element.gamma_1, element.gamma_1 * element.gamma_2]

    Fl_non_sort_val = [zeros(3) for i in 1:n_atoms]


    for i in 1:n_atoms
        q_i = atoms[i].charge
        x_i, y_i, z_i = coords[i]
        
        # this part is to cancel out effect from particle j to itself
        f_func = zeros(4)
        f_func[1] = f_func_0[1] - q_i * cos(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func[2] = f_func_0[2] - q_i * sin(k_x * x_i + k_y * y_i) * exp( + k * z_i)
        f_func[3] = f_func_0[3] - q_i * cos(k_x * x_i + k_y * y_i) * exp( - k * z_i)
        f_func[4] = f_func_0[4] - q_i * sin(k_x * x_i + k_y * y_i) * exp( - k * z_i)
        

        sum_xy = q_i * gamma_val * 
        (cos(k_x * x_i + k_y * y_i) * (
            Gamma_para[1] * exp(-2*k*L_z) * exp(k * z_i) * f_func[2] + 
            Gamma_para[3] * exp( - k * z_i) * f_func[4] + 
            Gamma_para[4] * exp(-2*k*L_z) * exp(-k * z_i) * f_func[2] + 
            Gamma_para[4] * exp(-2*k*L_z) * exp(k * z_i) * f_func[4]) - 
        sin(k_x * x_i + k_y * y_i) * (
            Gamma_para[1] * exp(-2*k*L_z) * exp(k * z_i) * f_func[1] + 
            Gamma_para[3] * exp( - k * z_i) * f_func[3] + 
            Gamma_para[4] * exp(-2*k*L_z) * exp(-k * z_i) * f_func[1] + 
            Gamma_para[4] * exp(-2*k*L_z) * exp(k * z_i) * f_func[3]))

        sum_x = (k_x / k) * sum_xy
        sum_y = (k_y / k) * sum_xy

        sum_z = q_i * gamma_val * 
        (cos(k_x * x_i + k_y * y_i) * (
            Gamma_para[1] * exp(-2*k*L_z) * exp(k * z_i) * f_func[1] - 
            Gamma_para[3] * exp( - k * z_i) * f_func[3] - 
            Gamma_para[4] * exp(-2*k*L_z) * exp(-k * z_i) * f_func[1] + 
            Gamma_para[4] * exp(-2*k*L_z) * exp(k * z_i) * f_func[3]) + 
        sin(k_x * x_i + k_y * y_i) * (
            Gamma_para[1] * exp(-2*k*L_z) * exp(k * z_i) * f_func[2] - 
            Gamma_para[3] * exp( - k * z_i) * f_func[4] - 
            Gamma_para[4] * exp(-2*k*L_z) * exp(-k * z_i) * f_func[2] + 
            Gamma_para[4] * exp(-2*k*L_z) * exp(k * z_i) * f_func[4]))

        Fl_non_sort_val[i] = [sum_x, sum_y, sum_z]
    end

    return Fl_non_sort_val
end

function Fl_sort(k_set, atoms, z_list, coords, element::greens_element)
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

    Fl_sort_val = [zeros(3) for i in 1:n_atoms]
    
    for i in 1:n_atoms
        l = z_list[i]
        q_i = atoms[l].charge
        x_i, y_i, z_i = coords[l]

        sum_xy = 
        sin(k_x * x_i + k_y * y_i) * exp(-k * z_i) * A[i] - 
        cos(k_x * x_i + k_y * y_i) * exp(-k * z_i) * B[i] + 
        sin(k_x * x_i + k_y * y_i) * exp( k * z_i) * C[i] - 
        cos(k_x * x_i + k_y * y_i) * exp( k * z_i) * D[i]

        sum_x = q_i * k_x / (2 * k) * sum_xy
        sum_y = q_i * k_y / (2 * k) * sum_xy

        sum_z = (q_i / 2) * (
        cos(k_x * x_i + k_y * y_i) * exp(-k * z_i) * A[i] + 
        sin(k_x * x_i + k_y * y_i) * exp(-k * z_i) * B[i] - 
        cos(k_x * x_i + k_y * y_i) * exp( k * z_i) * C[i] - 
        sin(k_x * x_i + k_y * y_i) * exp( k * z_i) * D[i])

        Fl_sort_val[l] = [sum_x, sum_y, sum_z]
    end

    return Fl_sort_val
end

# this function define the summation of q_i * q_j * abs(z_i - z_j) / 2 term
function Flz_sort_k0(atoms, z_list, coords, element::greens_element)
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

    Fl_sort_k0_val = [zeros(3) for i in 1:n_atoms]

    for j in 1:n_atoms
        l = z_list[j]
        q_i = atoms[l].charge
        Fl_sort_k0_val[l][3] = q_i * (Q_1[j] - Q_2[j])
    end

    return Fl_sort_k0_val
end

# this function is used to calculate the self interaction of the particle
function Flz_self(k_set, atoms, coords, element::greens_element)
    gamma_1 = element.gamma_1
    gamma_2 = element.gamma_2

    k_x, k_y, k = k_set
    n_atoms = size(coords)[1]
    L_z = element.L_z

    gamma_val = gamma_1 * gamma_2 * exp(- 2 * k * L_z) - 1

    Flz_self_val = [zeros(3) for i in 1:n_atoms]

    for i in 1:n_atoms
        q_i = atoms[i].charge
        x_i, y_i, z_i = coords[i]

        Flz_self_val[i][3] = q_i^2 * (gamma_2 * exp(- 2* k * (L_z - z_i)) - gamma_1 * exp(-2 * k * z_i)) / (2 * gamma_val)
    end

    return Flz_self_val
end

function F_l_total(k_set, sys, inter, element::greens_element)
    L_x, L_y, L_z = inter.L

    Fl_non_sort_val = Fl_non_sort(k_set, sys.atoms, sys.coords, element)
    Fl_sort_val = Fl_sort(k_set, sys.atoms, inter.z_list, sys.coords, element)
    Flz_self_val = Flz_self(k_set, sys.atoms, sys.coords, element)

    Fl_k = (Fl_non_sort_val + Fl_sort_val + Flz_self_val) / (L_x * L_y * inter.eps_0)

    return Fl_k
end

function F_long(sys, inter::QEM_long)
    gamma_1 = inter.gamma_1
    gamma_2 = inter.gamma_2
    eps_0 = inter.eps_0
    L_x, L_y, L_z = inter.L
    alpha = inter.alpha
    n_atoms = size(sys.coords)[1]

    # here we set z_i = z_j = rho_ij = 0
    element = greens_element_init(gamma_1, gamma_2, L_z, inter.alpha)

    F_long_val = [zeros(3) for i in 1:n_atoms]

    # this part is unrelated to k
    Flz_sort_k0_val = Flz_sort_k0(sys.atoms, inter.z_list, sys.coords, element) / (2 * L_x * L_y * inter.eps_0)

    F_long_val += Flz_sort_k0_val
    
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
                    F_long_val += F_l_total(k_set, sys, inter, element) * exp(-k^2/(4 * alpha))
                end
            end
        end
    elseif inter.rbe_mode == true
        K_p = sample(inter.K_set, inter.Prob, inter.rbe_p)
        for k_set in K_p
            F_long_val += F_l_total(k_set, sys, inter, element) * (inter.sum_K / inter.rbe_p)
        end
    end

    return F_long_val
end


