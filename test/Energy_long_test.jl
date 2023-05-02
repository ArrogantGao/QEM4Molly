@testset "function El_non_sort, compare with python lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    pylib = pyimport("Sigma_Gamma_func_njit")

    np = pyimport("numpy")


    for try_step in 1:10

        n_atoms = 4

        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [-1, 5.1, 5.2, 0.15], [1, 4.9, 4.9, 0.3], [-1, 5.5, 5.3, 0.4]])
        L_x, L_y, L_z = (10, 10, 2)
        
        k_x = rand()
        k_y = rand()
        K = k_x^2 + k_y^2
        k = sqrt(K)
        k_set = np.array([k_x, k_y, K, k])

        # using the python package Sigma_Gamma_func_njit.py
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)

        non_sort_val_py = pylib.Sigma_Gamma_func_s(k_set, e_1, 1, e_3, POSCAR, L_z)

        q_i = 1
        q_j = - 1
        atoms = [Atom(mass=1.0, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^(i + 1)) for i in 1:n_atoms]
        coords = [[5.0, 5.0, 0.2], [5.1, 5.2, 0.15], [4.9, 4.9, 0.3], [5.5, 5.3, 0.4]]

        QEM_long_inter = QEM_long((10, 10, 2) ; z_list = [2, 1, 3, 4], gamma_1 = g_1, gamma_2 = g_2)

        # element = greens_element_ij_init(g_1, g_2, 1, 1, 0, 2, 1)
        element = greens_element_init(g_1, g_2, 2, 1)

        non_sort_val_julia = El_nonsort((k_x, k_y, sqrt(k_x^2 + k_y^2)), atoms, coords, element)
        
        @test non_sort_val_py ≈ non_sort_val_julia
    end
end

@testset "function El_sort, compare with the python lib" begin
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
        sort_val_py = pylib.Sigma_Gamma_func_ns(k_set, POSCAR, z_list_py)


        # using the julia libary

        atoms = [Atom(mass=1.0, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^(i + 1)) for i in 1:n_atoms]
        coords = [[5.0, 5.0, 0.2], [5.1, 5.2, 0.15], [4.9, 4.9, 0.3], [5.5, 5.3, 0.4]]

        QEM_long_inter = QEM_long((10, 10, 2) ; z_list = [2, 1, 3, 4], gamma_1 = g_1, gamma_2 = g_2)

        # element = greens_element_ij_init(g_1, g_2, 1, 1, 0, 2, 1)
        element = greens_element_init(g_1, g_2, 2, 1)

        z_list = [2, 1, 3, 4]
        sort_val_julia = El_sort([k_x, k_y, k], atoms, z_list, coords, element)
        
        @test sort_val_py ≈ sort_val_julia
    end
end

@testset "function El_sort_k0_val, compare with the python lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    pylib = pyimport("Sigma_Gamma_func_njit")

    np = pyimport("numpy")

    for try_step in 1:1

        n_atoms = 4

        POSCAR = np.array([[1, 5.0, 5.0, 0.2], [-1, 5.1, 5.2, 0.15], [1, 4.9, 4.9, 0.3], [-1, 5.5, 5.3, 0.4]])
        L_x, L_y, L_z = (10, 10, 2)
        

        # using the python package Sigma_Gamma_func_njit.py
        g_1, g_2 = rand(2)
        # g_1 = 0.9
        # g_2 = 0.8
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)

        z_list_py = np.array([1, 0, 2, 3])

        k0_val_py = pylib.Sigma_Gamma_func_k0(POSCAR, z_list_py)

        
        # using the julia libary

        atoms = [Atom(mass=1.0, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^(i + 1)) for i in 1:n_atoms]
        coords = [[5.0, 5.0, 0.2], [5.1, 5.2, 0.15], [4.9, 4.9, 0.3], [5.5, 5.3, 0.4]]

        QEM_long_inter = QEM_long((10, 10, 2) ; z_list = [2, 1, 3, 4], gamma_1 = g_1, gamma_2 = g_2)

        # element = greens_element_ij_init(g_1, g_2, 1, 1, 0, 2, 1)
        element = greens_element_init(g_1, g_2, 2, 1)

        z_list = [2, 1, 3, 4]
        k0_val_julia = El_sort_k0(atoms, z_list, coords, element)
        
        @test 2 * k0_val_julia ≈ k0_val_py
    end
end