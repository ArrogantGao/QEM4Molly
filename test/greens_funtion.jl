@testset "compare the green's function with python lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    g_py = pyimport("green_norm_njit")

    # this test is to make sure the python lib has been properly loaded
    @test g_py.gamma(1, 1, 1, 1, 1) == -1.0

    # then 100 random test will be generated
    for i in 1:100
        k, eps_1, eps_2, eps_3, L_z = rand(5)
        gamma_1 = (eps_2 - eps_1) / (eps_2 + eps_1)
        gamma_2 = (eps_2 - eps_3) / (eps_2 + eps_3)

        L = (20, 20, 10)
        L_x, L_y, L_z = L

        x_i, y_i, z_i = L .* rand(3)
        x_j, y_j, z_j = L .* rand(3)

        rho_ij = 1.0
        alpha = 1.0

        # this will test the green's function between ij pair
        element_ij = greens_element_ij_init(gamma_1, gamma_2, z_i, z_j, rho_ij, L_z, alpha)
        @test g_py.Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) ≈ Gamma_1(k, element_ij)
        @test g_py.Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) ≈ Gamma_2(k, element_ij)
        @test g_py.dz_Gamma(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) ≈ dz_Gamma_1(k, element_ij)
        @test g_py.dz_Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_j, z_i) ≈ dz_Gamma_2(k, element_ij)

        # this will test the self interaction on i itself
        element_i = greens_element_i_init(gamma_1, gamma_2, z_i, L_z, alpha)
        @test g_py.Gamma(k, eps_1, eps_2, eps_3, L_z, z_i, z_i) ≈ Gamma_1(k, element_i)
        @test g_py.Gamma_a(k, eps_1, eps_2, eps_3, L_z, z_i, z_i) ≈ Gamma_2(k, element_i)
        @test g_py.dz_Gamma_self(k, eps_1, eps_2, eps_3, L_z, z_i, z_i) ≈ dz_Gamma_1(k, element_i)
        @test g_py.dz_Gamma_a_self(k, eps_1, eps_2, eps_3, L_z, z_i, z_i) ≈ dz_Gamma_2(k, element_i)

    end
end