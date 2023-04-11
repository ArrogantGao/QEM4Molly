export QEM_inter

function QEM_inter(n_atoms, L; s = 1.0, accuracy = 1e-6, rbe_mode = true, rbe_p = 30, eps_0 = 1.0, gamma_1 = 0.0, gamma_2 = 0.0, N_t = 30, iter_period = 100)
    L_x, L_y, L_z = L
    if rbe_mode == true
        alpha = n_atoms / (L_x * L_y)
        r_cutoff = s / sqrt(alpha)
        k_cutoff = 2 * s * sqrt(alpha)

        QEM_short_inter = QEM_short(L, r_cutoff, alpha; iter_period = iter_period, gamma_1 = gamma_1, gamma_2 = gamma_2, eps_0 = eps_0, accuracy = accuracy, N_t = N_t)

        QEM_long_inter = QEM_long(L; k_cutoff = k_cutoff, z_list = [], gamma_1 = gamma_1, gamma_2 = gamma_2, eps_0 = eps_0, accuracy = accuracy, alpha = alpha, rbe_mode = true, rbe_p = rbe_p)
    else
        alpha = sqrt(n_atoms) / (L_x * L_y)
        r_cutoff = s / sqrt(alpha)
        k_cutoff = 2 * s * sqrt(alpha)

        QEM_short_inter = QEM_short(L, r_cutoff, alpha; iter_period = iter_period, gamma_1 = gamma_1, gamma_2 = gamma_2, eps_0 = eps_0, accuracy = accuracy, N_t = N_t)

        QEM_long_inter = QEM_long(L; k_cutoff = k_cutoff, z_list = [], gamma_1 = gamma_1, gamma_2 = gamma_2, eps_0 = eps_0, accuracy = accuracy, alpha = alpha, rbe_mode = false, rbe_p = 0)
    end

    return QEM_short_inter, QEM_long_inter
end