# @testset "Energy short range, compare with python QEM lib" begin
#     sys = pyimport("sys")
#     sys.path = push!(sys.path, "../src/python_lib/")
#     Es_py = pyimport("Energy_short_range_njit")

#     np = pyimport("numpy")

#     for try_step in 1:10
#         # initialize the System
#         g_1, g_2 = rand(2)
#         # g_1 = 0.9
#         # g_2 = 0.8
#         eps_0 = 1
#         e_1 = (1 - g_1) / (1 + g_1)
#         e_3 = (1 - g_2) / (1 + g_2)
#         N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = 2, 1e-9, 10, 10, 2, e_1, 1, e_3, .1, 1e-6
#         k_f1 = sqrt( - 4 * alpha * log(cutoff))
#         k_f2 = - log(cutoff) / (2 * L_z)

#         INCAR = [N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff]
#         # print(INCAR)
#         POSCAR = np.array([[1, 5.0, 5.0, 0.2], [-1, 5.1, 5.2, 0.15]])
#         q_i, x_i, y_i, z_i = 1, 5.0, 5.0, 0.2
#         q_j, x_j, y_j, z_j = - 1, 5.1, 5.2, 0.15
#         coord_i = [5.0, 5.0, 0.2]
#         coord_j = [5.1, 5.2, 0.15]
#         rho_ij = sqrt(0.1^2 + 0.2^2)

#         # this is the python lib
#         Gauss_para_1 = np.polynomial.legendre.leggauss(50)
#         Gauss_para_2 = np.polynomial.legendre.leggauss(50)

#         np_neighbor_list_j = np.array([[- 1, 5.1, 5.2, 0.15, 0.2236], [9999.0, 5.1, 5.2, 0.15, 1.0]])

#         E_ij_py = Es_py.Energy_short_other_j(q_i, x_i, y_i, z_i, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha)

#         E_i_py = Es_py.Energy_short_self_j(q_i, x_i, y_i, z_i, np_neighbor_list_j, Gauss_para_1, k_f1, Gauss_para_2, k_f2, eps_0, eps_1, eps_2, eps_3, L_z, alpha)


#         # this is the julia lib
#         QEM_short_inter = QEM_short((10, 10, 2), 1.0, alpha; iter_period = 100, gamma_1 = g_1, gamma_2 = g_2, eps_0 = 1, accuracy = 10^(-6), N_t = 30)
#         E_ij = E_short_ij(q_i, q_j, coord_i, coord_j, QEM_short_inter; single = false)
#         E_i = E_short_i(q_i, coord_i, QEM_short_inter; single = false)

#         element = greens_element_ij_init(g_1, g_2, coord_i[3], coord_j[3], rho_ij, L_z, alpha, 10^(-6))

#         k = 0.1
#         @test isapprox(Es_py.gauss_charge_intcore(k, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ij, alpha), sum(E_s_gauss_c(k, element; l = i) for i in 1:4))
#         @test isapprox(Es_py.point_charge_intcore(k, eps_1, eps_2, eps_3, L_z, z_j, z_i, rho_ij, alpha), sum(E_s_point_c(k, element; l = i) for i in 1:4))

#         @test isapprox(0.5 * E_ij, E_ij_py, atol = 1e-5)
#         @test isapprox(E_i, E_i_py, atol = 1e-5)
#     end
# end

@testset "Energy short range, compare with python ICM lib" begin
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
        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [ - 1, 5.1, 5.2, 0.15]])

        N_image = 100::Int
        N_real = 0::Int
        POS = np.zeros([2 * N_image + 1, N, 4])

        E_icm = ICM.U_cal(INCAR, POSCAR, POS, N_image, N_real)
        
        # using the julia libary

        q_i = 1
        q_j =  - 1
        coord_i = [5.0, 5.0, 0.2]
        coord_j = [5.1, 5.2, 0.15]
        QEM_short_inter = QEM_short((10, 10, 2), 1.0, 1; iter_period = 100, gamma_1 = g_1, gamma_2 = g_2, eps_0 = 1, accuracy = 10^(-6), N_t = 30)

        E_ij = E_short_ij(q_i, q_j, coord_i, coord_j, QEM_short_inter; single = true)
        E_i = E_short_i(q_i, coord_i, QEM_short_inter; single = true)
        E_j = E_short_i(q_j, coord_j, QEM_short_inter; single = true)
        
        # println(F_x, F_y, F_z, F_x_self, F_y_self, F_z_self)
        @test isapprox(E_icm, E_ij + E_i + E_j, atol = 1e-5)
    end
end