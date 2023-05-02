# in this script, I will compare the total Energy result of julia code and python code

@testset "Total force of a 100 partices system, compare with python ICM lib" begin
    sys = pyimport("sys")
    sys.path = push!(sys.path, "../src/python_lib/")
    ICM = pyimport("ICM")

    np = pyimport("numpy")


    for try_step in 1:2

        g_1, g_2 = rand(2)
        e_1 = (1 - g_1) / (1 + g_1)
        e_3 = (1 - g_2) / (1 + g_2)

        # the system is defined here
        n_atoms = 100
        atom_mass = 1.0u"NoUnits"
        atoms = [Atom(mass=atom_mass, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^i) for i in 1:n_atoms]

        L_x = 50.0
        L_y = 50.0
        L_z = 10.0

        boundary = CubicBoundary(L_x * u"NoUnits", L_y * u"NoUnits", Inf * u"NoUnits") # Periodic boundary conditions
        boundary_place = CubicBoundary(L_x * u"NoUnits", L_y * u"NoUnits", 9 * u"NoUnits")
        coords_temp = place_atoms(n_atoms, boundary_place; min_dist=0.3u"NoUnits") # Random placement without clashing

        coords = [SVector{3, typeof(x)}(x, y, z + .5) for (x, y, z) in coords_temp]

        temp = 1.0u"NoUnits"
        velocities = [velocity(atom_mass, temp, 1) for i in 1:n_atoms]

        pairwise_inters = (LennardJones(;cutoff = DistanceCutoff(3.5u"NoUnits"), energy_units = u"NoUnits", force_units = u"NoUnits", nl_only = true),)
        
        s = 0.5
        e = 1
        Substrate_LJ_inter_1 = Substrate_LJ(;cutoff = 1.0, sigma = s , eps = e, sub_pos = 0.0, direction = +1)
        Substrate_LJ_inter_2 = Substrate_LJ(;cutoff = 1.0, sigma = s, eps = e, sub_pos = 10, direction = -1)

        QEM_short_inter, QEM_long_inter = QEM_inter(n_atoms, (L_x, L_y, L_z); gamma_1 = g_1, gamma_2 = g_2, rbe_mode = false, rbe_p = 10, N_t = 30, s = 1.0)

        coords_2d = [[coords[i][1], coords[i][2]] for i in 1:n_atoms]
        QEM_short_inter.neighbor_list = neighborlist(coords_2d, QEM_short_inter.r_cutoff; unitcell = [QEM_short_inter.L[1], QEM_short_inter.L[2]])

        general_inters = (Substrate_LJ_inter_1, Substrate_LJ_inter_2, QEM_short_inter, QEM_long_inter, )

        # general_inters = (Substrate_LJ_inter_1, Substrate_LJ_inter_2, )

        sys = System(
            atoms=atoms,
            pairwise_inters=pairwise_inters,
            general_inters = general_inters,
            coords=coords,
            velocities=velocities,
            boundary=boundary,
            neighbor_finder=DistanceNeighborFinder(
                nb_matrix=trues(n_atoms, n_atoms),
                n_steps=100,
                dist_cutoff=3.5u"NoUnits",
            ),
            loggers=(
                temp=TemperatureLogger(typeof(1.0), 10),
                coords=CoordinateLogger(typeof(1.0), 10),
            ),
            energy_units = u"NoUnits",
            force_units = u"NoUnits",
            k = 1.0
        )
        
        
        # using the julia libary
        F_short = Molly.forces(QEM_short_inter, sys, nothing)
        F_long = Molly.forces(QEM_long_inter, sys, nothing)

        F_QEM = F_short + F_long

        # using the python package ICM.py
        alpha = QEM_short_inter.alpha
        N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff = 100, 1e-9, 50, 50, 10, e_1, 1, e_3, alpha, 1e-9

        INCAR = [N, scale, L_x, L_y, L_z, eps_1, eps_2, eps_3, alpha, cutoff]
        # print(INCAR)
        POSCAR = np.array([[atoms[i].charge, coords[i][1], coords[i][2], coords[i][3]] for i in 1:n_atoms])

        N_image = 40::Int
        N_real = 40::Int
        POS = np.zeros([2 * N_image + 1, N, 4])

        Fx_array, Fy_array, Fz_array, Fx_self_array, Fy_self_array, Fz_self_array = ICM.F_cal(INCAR, POSCAR, POS, N_image, N_real)

        for i in 1:n_atoms
            @test isapprox(Fx_array[i] + Fx_self_array[i], F_QEM[i][1], atol = 1e-4)
            @test isapprox(Fy_array[i] + Fy_self_array[i], F_QEM[i][2], atol = 1e-4)
            @test isapprox(Fz_array[i] + Fz_self_array[i], F_QEM[i][3], atol = 1e-4)
        end
    end
end