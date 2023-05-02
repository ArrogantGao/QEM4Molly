@testset "compare E_long by sorting and direct summation" begin
    for test_step in 1:10
        n_atoms = 100
        L_x, L_y, L_z = 10, 10, 10
        coords = [(10, 10, 10) .* rand(3) for i in 1:100]
        q = 2 .* rand(100) .- 1
        # z_coords = [coords[i][3] for i in 1:100]
        # z_list = sortperm(z_coords)
        alpha = 0.1
        g_1, g_2 = rand(2)
        eps_0 = 1
        # green_ele = greens_element_init(g_1, g_2, L_z, alpha)
        k_c = 5


        QEM_long_inter = QEM_long((L_x, L_y, L_z) ;k_cutoff = k_c, gamma_1 = g_1, gamma_2 = g_2, eps_0 = 1, alpha = alpha, rbe_mode = false)
        atom_mass = 1.0u"NoUnits"
        atoms = [Atom(mass=atom_mass, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = q[i]) for i in 1:n_atoms]
        boundary = CubicBoundary(L_x * u"NoUnits", L_y * u"NoUnits", Inf * u"NoUnits")
        temp = 1.0u"NoUnits"
        velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]
        sys = System(
            atoms=atoms,
            coords=coords,
            velocities=velocities,
            boundary=boundary,
            neighbor_finder=nothing,
            loggers=(
                temp=TemperatureLogger(typeof(1.0), 10),
                coords=CoordinateLogger(typeof(1.0), 10),
            ),
            energy_units = u"NoUnits",
            force_units = u"NoUnits",
            k = 1.0
        )
        E_long_Molly = E_long(sys, QEM_long_inter)

        E_long_total = energy_sum_total(q, coords, alpha, L_x, L_y, L_z, g_1, g_2, eps_0, k_c)

        sum_direct = direct_sum_k(q, coords, L_x, L_y, L_z, k_c, alpha, g_1, g_2)
        sum_direct_k0 = direct_sum_k0(q, coords, L_x, L_y)
        @test E_long_total ≈ - sum_direct - sum_direct_k0
        @test E_long_Molly ≈ E_long_total
    end
end
