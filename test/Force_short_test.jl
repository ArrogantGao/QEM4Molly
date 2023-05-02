@testset "Force short range, compare with python ICM lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    ICM = pyimport("ICM")

    np = pyimport("numpy")


    for try_step in 1:10
        
        # using the python package ICM.py
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)
        N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = 2, 1e-9, 10, 10, 2, e_1, 1, e_3, .1, 1e-9
        INCAR = [N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff]
        # print(INCAR)
        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [1, 5.1, 5.2, 0.15]])

        N_image = 100::Int
        N_real = 0::Int
        POS = np.zeros([2 * N_image + 1, N, 4])

        Fx_icm, Fy_icm, Fz_icm, Fxs_icm, Fys_icm, Fzs_icm = ICM.F_cal(INCAR, POSCAR, POS, N_image, N_real)
        
        # using the julia libary

        q_i = 1
        q_j = 1
        coord_i = [5.0, 5.0, 0.2]
        coord_j = [5.1, 5.2, 0.15]
        QEM_short_inter = QEM_short((10, 10, 2), 1.0, 1; iter_period = 100, gamma_1 = g_1, gamma_2 = g_2, eps_0 = 1, accuracy = 10^(-6), N_t = 30)

        F_ij = F_short_ij(q_i, q_j, coord_i, coord_j, QEM_short_inter; single = true)
        F_self_i = F_short_i(q_i, coord_i, QEM_short_inter; single = true)
        F_self_j = F_short_i(q_j, coord_j, QEM_short_inter; single = true)
        
        # println(F_x, F_y, F_z, F_x_self, F_y_self, F_z_self)
        @test isapprox(F_ij[1][1], Fx_icm[1], atol = 1e-5)
        @test isapprox(F_ij[1][2], Fy_icm[1], atol = 1e-5)
        @test isapprox(F_ij[1][3], Fz_icm[1], atol = 1e-5)
        @test isapprox(F_self_i[3], Fzs_icm[1], atol = 1e-5)
        @test isapprox(F_ij[2][1], Fx_icm[2], atol = 1e-5)
        @test isapprox(F_ij[2][2], Fy_icm[2], atol = 1e-5)
        @test isapprox(F_ij[2][3], Fz_icm[2], atol = 1e-5)
        @test isapprox(F_self_j[3], Fzs_icm[2], atol = 1e-5)
        # @test F_z ≈ Fz_icm[1] ≈ 1/(4 * π * (0.05)^2)
    end
end

@testset "Force short range, compare with python QEM lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    Fs_py = pyimport("Force_short_range_njit")

    np = pyimport("numpy")

    for try_step in 1:10
        # initialize the System
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)
        N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = 2, 1e-9, 10, 10, 2, e_1, 1, e_3, .1, 1e-9
        INCAR = [N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff]
        # print(INCAR)
        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [1, 5.1, 5.2, 0.15]])
        q_i = 1
        q_j = 1
        coord_i = [5.0, 5.0, 0.2]
        coord_j = [5.1, 5.2, 0.15]

        # this is the python lib
        Gauss_para_1 = np.polynomial.legendre.leggauss(30)
        Gauss_para_2 = np.polynomial.legendre.leggauss(30)
        pos_info = np.array([1, 1, 5.0, 5.1, 5.0, 5.2, 0.2, 0.15])
        F_rs = Fs_py.Force_rhos_ji(pos_info, INCAR, Gauss_para_1, Gauss_para_2, 1)
        F_zs_other = Fs_py.Force_zs_ji(pos_info, INCAR, Gauss_para_1, Gauss_para_2, 1)
        pos_info_self = np.array([1, 5.0, 5.0, 0.2])
        F_zs_self = Fs_py.Force_zs_self_ji(pos_info_self, INCAR, Gauss_para_1, Gauss_para_2, 1)

        F_xs = - 0.1 / sqrt(0.1^2 + 0.2^2) * F_rs
        F_ys = - 0.2 / sqrt(0.1^2 + 0.2^2) * F_rs


        # this is the julia lib
        QEM_short_inter = QEM_short((10, 10, 2), 1.0, .1; iter_period = 100, gamma_1 = g_1, gamma_2 = g_2, eps_0 = 1, accuracy = 10^(-6), N_t = 30)
        F_i, F_j = F_short_ij(q_i, q_j, coord_i, coord_j, QEM_short_inter; single = false)
        F_xi_self, F_yi_self, F_zi_self = F_short_i(q_i, coord_i, QEM_short_inter; single = false)
        F_xj_self, F_yj_self, F_zj_self = F_short_i(q_j, coord_j, QEM_short_inter; single = false)

        @test isapprox(F_i[1], F_xs, atol = 1e-5)
        @test isapprox(F_i[2], F_ys, atol = 1e-5)
        @test isapprox(F_i[3], F_zs_other, atol = 1e-5)
        @test isapprox(F_zi_self, F_z_self, atol = 1e-5)
    end
end