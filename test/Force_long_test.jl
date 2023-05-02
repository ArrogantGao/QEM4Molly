@testset "function Fl_non_sort, compare with python lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    pylib = pyimport("Sigma_Gamma_func_njit")

    np = pyimport("numpy")


    for try_step in 1:10

        n_atoms = 4

        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [-1, 5.1, 5.2, 0.15], [1, 4.9, 4.9, 0.3], [-1, 5.5, 5.3, 0.4]])
        L_x, L_y, L_z = (10, 10, 2)
        
        k_x, k_y = rand(2)
        K = k_x^2 + k_y^2
        k = sqrt(K)
        k_set = np.array([k_x, k_y, K, k])

        # using the python package Sigma_Gamma_func_njit.py
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)

        F_long_x_sum1_val, F_long_y_sum1_val, F_long_z_sum1_val = pylib.F_long_sum_1(k_set, e_1, 1, e_3, POSCAR, L_z)
        
        
        # using the julia libary

        q_i = 1
        q_j = - 1
        atoms = [Atom(mass=1.0, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^(i + 1)) for i in 1:n_atoms]
        coords = [[5.0, 5.0, 0.2], [5.1, 5.2, 0.15], [4.9, 4.9, 0.3], [5.5, 5.3, 0.4]]

        QEM_long_inter = QEM_long((10, 10, 2) ; z_list = [2, 1, 3, 4], gamma_1 = g_1, gamma_2 = g_2)

        # element = greens_element_ij_init(g_1, g_2, 1, 1, 0, 2, 1)
        element = greens_element_init(g_1, g_2, 2, 1)

        Fl_non_sort_val = Fl_non_sort((k_x, k_y, sqrt(k_x^2 + k_y^2)), atoms, coords, element)
        
        for i in 1:n_atoms
            @test F_long_x_sum1_val[i] ≈ Fl_non_sort_val[i][1]
            @test F_long_y_sum1_val[i] ≈ Fl_non_sort_val[i][2]
            @test F_long_z_sum1_val[i] ≈ Fl_non_sort_val[i][3]
        end
    end
end

@testset "function Fl_sort, compare with the python lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    pylib = pyimport("Sigma_Gamma_func_njit")

    np = pyimport("numpy")

    for try_step in 1:10

        n_atoms = 4

        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [-1, 5.1, 5.2, 0.15], [1, 4.9, 4.9, 0.3], [-1, 5.5, 5.3, 0.4]])
        L_x, L_y, L_z = (10, 10, 2)
        
        k_x, k_y = rand(2)
        K = k_x^2 + k_y^2
        k = sqrt(K)
        k_set = np.array([k_x, k_y, K, k])

        # using the python package Sigma_Gamma_func_njit.py
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)

        z_list_py = np.array([1, 0, 2, 3])
        F_long_x_sum2_val, F_long_y_sum2_val = pylib.F_long_xy_sum_2(k_set, POSCAR, z_list_py)

        equal_cell = np.array([[1], [0], [2], [3]])
        equal_size = np.array([1, 2, 3, 4])
        equal_cell_room_num = 4
        F_long_z_sum0_val, F_long_z_sum2_val = pylib.F_long_z_sum_2(k_set, POSCAR, z_list_py, equal_cell, equal_size, equal_cell_room_num)

        F_long_z_self_sum_val = pylib.F_long_z_self_sum(k_set, POSCAR, e_1, 1, e_3, L_z)
        
        
        # using the julia libary

        atoms = [Atom(mass=1.0, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^(i + 1)) for i in 1:n_atoms]
        coords = [[5.0, 5.0, 0.2], [5.1, 5.2, 0.15], [4.9, 4.9, 0.3], [5.5, 5.3, 0.4]]

        QEM_long_inter = QEM_long((10, 10, 2) ; z_list = [2, 1, 3, 4], gamma_1 = g_1, gamma_2 = g_2)

        # element = greens_element_ij_init(g_1, g_2, 1, 1, 0, 2, 1)
        element = greens_element_init(g_1, g_2, 2, 1)

        z_list = [2, 1, 3, 4]
        Fl_sort_val = Fl_sort([k_x, k_y, k], atoms, z_list, coords, element)
        
        for i in 1:n_atoms
            @test F_long_x_sum2_val[i] ≈ Fl_sort_val[i][1]
            @test F_long_y_sum2_val[i] ≈ Fl_sort_val[i][2]
            # here notice that the sign of Fl_sort_val has been changed compare to the python libary
            @test F_long_z_sum2_val[i] ≈ - Fl_sort_val[i][3]
        end
    end
end

@testset "function Fl_sort_k0_val, compare with the python lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    pylib = pyimport("Sigma_Gamma_func_njit")

    np = pyimport("numpy")

    for try_step in 1:10

        n_atoms = 4

        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [-1, 5.1, 5.2, 0.15], [1, 4.9, 4.9, 0.3], [-1, 5.5, 5.3, 0.4]])
        L_x, L_y, L_z = (10, 10, 2)
        
        k_x, k_y = rand(2)
        K = k_x^2 + k_y^2
        k = sqrt(K)
        k_set = np.array([k_x, k_y, K, k])

        # using the python package Sigma_Gamma_func_njit.py
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)

        z_list_py = np.array([1, 0, 2, 3])
        equal_cell = np.array([[1], [0], [2], [3]])
        equal_size = np.array([1, 2, 3, 4])
        equal_cell_room_num = 4

        F_long_z_sum0_val, F_long_z_sum2_val = pylib.F_long_z_sum_2(k_set, POSCAR, z_list_py, equal_cell, equal_size, equal_cell_room_num)
        
        
        # using the julia libary

        atoms = [Atom(mass=1.0, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^(i + 1)) for i in 1:n_atoms]
        coords = [[5.0, 5.0, 0.2], [5.1, 5.2, 0.15], [4.9, 4.9, 0.3], [5.5, 5.3, 0.4]]

        QEM_long_inter = QEM_long((10, 10, 2) ; z_list = [2, 1, 3, 4], gamma_1 = g_1, gamma_2 = g_2)

        # element = greens_element_ij_init(g_1, g_2, 1, 1, 0, 2, 1)
        element = greens_element_init(g_1, g_2, 2, 1)

        z_list = [2, 1, 3, 4]
        Flz_sort_k0_val = Flz_sort_k0(atoms, z_list, coords, element)
        
        for i in 1:n_atoms
            @test F_long_z_sum0_val[i] ≈ Flz_sort_k0_val[i][3]
        end
    end
end

@testset "function Flz_self, compare with the python lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    pylib = pyimport("Sigma_Gamma_func_njit")

    np = pyimport("numpy")

    for try_step in 1:10

        n_atoms = 4

        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [-1, 5.1, 5.2, 0.15], [1, 4.9, 4.9, 0.3], [-1, 5.5, 5.3, 0.4]])
        L_x, L_y, L_z = (10, 10, 2)
        
        k_x, k_y = rand(2)
        K = k_x^2 + k_y^2
        k = sqrt(K)
        k_set = np.array([k_x, k_y, K, k])

        # using the python package Sigma_Gamma_func_njit.py
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)

        F_long_z_self_sum_val = pylib.F_long_z_self_sum(k_set, POSCAR, e_1, 1, e_3, L_z)
        
        
        # using the julia libary

        atoms = [Atom(mass=1.0, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^(i + 1)) for i in 1:n_atoms]
        coords = [[5.0, 5.0, 0.2], [5.1, 5.2, 0.15], [4.9, 4.9, 0.3], [5.5, 5.3, 0.4]]

        QEM_long_inter = QEM_long((10, 10, 2) ; z_list = [2, 1, 3, 4], gamma_1 = g_1, gamma_2 = g_2)

        # element = greens_element_ij_init(g_1, g_2, 1, 1, 0, 2, 1)
        element = greens_element_init(g_1, g_2, 2, 1)

        Flz_self_val = Flz_self([k_x, k_y, k], atoms, coords, element)
        
        for i in 1:n_atoms
            @test F_long_z_self_sum_val[i] ≈ Flz_self_val[i][3]
        end
    end
end

@testset "function Fl_total, compare with the python lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    pylib = pyimport("Sigma_Gamma_func_njit")

    np = pyimport("numpy")

    for try_step in 1:10

        n_atoms = 4

        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [-1, 5.1, 5.2, 0.15], [1, 4.9, 4.9, 0.3], [-1, 5.5, 5.3, 0.4]])
        L_x, L_y, L_z = (10, 10, 2)
        
        k_x, k_y = rand(2)
        K = k_x^2 + k_y^2
        k = sqrt(K)
        k_set = np.array([k_x, k_y, K, k])

        # using the python package Sigma_Gamma_func_njit.py
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)

        z_list_py = np.array([1, 0, 2, 3])
        equal_cell = np.array([[1], [0], [2], [3]])
        equal_size = np.array([1, 2, 3, 4])
        equal_cell_room_num = 4

        eps_1 = e_1
        eps_2 = 1
        eps_3 = e_3

        F_x_sum1_val, F_y_sum1_val, F_z_sum1_val = pylib.F_long_sum_1(k_set, eps_1, eps_2, eps_3, POSCAR, L_z)
        F_x_sum2_val, F_y_sum2_val = pylib.F_long_xy_sum_2(k_set, POSCAR, z_list_py)
        F_z_sum0_val, F_z_sum2_val = pylib.F_long_z_sum_2(k_set, POSCAR, z_list_py, equal_cell, equal_size, equal_cell_room_num)
        F_z_self_val = pylib.F_long_z_self_sum(k_set, POSCAR, eps_1, eps_2, eps_3, L_z)

        F_py = [zeros(3) for i in 1:n_atoms]

        for j in 1:n_atoms
            F_py[j][1] += (1/(L_x * L_y * eps_2)) * (F_x_sum1_val[j] + F_x_sum2_val[j])
            F_py[j][2] += (1/(L_x * L_y * eps_2)) * (F_y_sum1_val[j] + F_y_sum2_val[j])
            F_py[j][3] += (1/(L_x * L_y * eps_2)) * (F_z_sum1_val[j] - F_z_sum2_val[j] + F_z_self_val[j])
        end
        
        
        # using the julia libary
        atom_mass = 1.0u"NoUnits"
        atoms = [Atom(mass=1.0, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^(i + 1)) for i in 1:n_atoms]
        coords = [[5.0, 5.0, 0.2], [5.1, 5.2, 0.15], [4.9, 4.9, 0.3], [5.5, 5.3, 0.4]]

        boundary = CubicBoundary(L_x * u"NoUnits", L_y * u"NoUnits", Inf * u"NoUnits")
        temp = 1.0u"NoUnits"
        velocities = [velocity(atom_mass, temp, 1) for i in 1:n_atoms]
        pairwise_inters = (LennardJones(;energy_units = u"NoUnits", force_units = u"NoUnits",),)

        sys = System(
            atoms=atoms,
            pairwise_inters=pairwise_inters,
            coords=coords,
            velocities=velocities,
            boundary=boundary,
            energy_units = u"NoUnits",
            force_units = u"NoUnits",
            k = 1.0
        )

        QEM_long_inter = QEM_long((10, 10, 2) ; z_list = [2, 1, 3, 4], gamma_1 = g_1, gamma_2 = g_2)

        # element = greens_element_ij_init(g_1, g_2, 1, 1, 0, 2, 1)
        element = greens_element_init(g_1, g_2, 2, 1)

        F_l_total_val = F_l_total([k_x, k_y, k], sys, QEM_long_inter, element)
        
        @test F_l_total_val ≈ F_py
        
    end
end