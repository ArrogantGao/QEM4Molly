@testset "Force short range, compare with python lib" begin
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
        QEM_short_inter = QEM_short((10, 10, 2), 1.0, 1; iter_period = 100, gamma_1 = g_1, gamma_2 = g_2, eps_0 = 1, accuracy = 10^(-9), N_t = 30)

        F_x, F_y, F_z = F_short_ij(q_i, q_j, coord_i, coord_j, QEM_short_inter; single = true)
        F_x_self, F_y_self, F_z_self = F_short_i(q_i, coord_i, QEM_short_inter; single = true)
        
        # println(F_x, F_y, F_z, F_x_self, F_y_self, F_z_self)
        @test isapprox(F_x, Fx_icm[1], atol = 1e-6)
        @test isapprox(F_y, Fy_icm[1], atol = 1e-6)
        @test isapprox(F_z, Fz_icm[1], atol = 1e-6)
        # @test isapprox(F_x_self, Fxs_icm[1], atol = 1e-6)
        # @test isapprox(F_y_self, Fys_icm[1], atol = 1e-6)
        @test isapprox(F_z_self, Fzs_icm[1], atol = 1e-6)
        # @test F_z ≈ Fz_icm[1] ≈ 1/(4 * π * (0.05)^2)
    end
end