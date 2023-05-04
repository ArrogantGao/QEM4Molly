@testset "compare direct sum with sorted sum" begin
    for step in 1:10
        g1, g2 = rand(2)
        n_atoms = 100
        Lx, Ly, Lz = 10, 10, 10
        coords = [(Lx, Ly, Lz) .* rand(3) for i in 1:n_atoms]
        q = 2 .* rand(n_atoms) .- 1
        z_coords = [coords[i][3] for i in 1:n_atoms]
        z_list = sortperm(z_coords)

        kc = 1
        alpha = 1

        kx, ky = rand(2)
        k = sqrt(kx^2 + ky^2)
        k_set = (kx, ky, k)

        element = greens_element_init(g1, g2, Lz, alpha)

        sum_x_direct, sum_y_direct, sum_z_direct = direct_sum_F(k_set, q, coords, Lx, Ly, Lz, kc, alpha, g1, g2)
        sum_x, sum_y, sum_z = force_k_sum_total(k_set, q, coords, z_list, element)
        
        @test sum_x_direct ≈ sum_x
        @test sum_y_direct ≈ sum_y
        @test sum_z_direct ≈ sum_z
    end
end

@testset "compare the k space summation result with direct one" begin
    for step in 1:10
        g1, g2 = rand(2)
        eps_0 = 1

        alpha = 1
        kc = 5

        n_atoms = 100
        Lx, Ly, Lz = 10, 10, 10
        coords = [(Lx, Ly, Lz) .* rand(3) for i in 1:n_atoms]
        q = 2 .* rand(n_atoms) .- 1
        z_coords = [coords[i][3] for i in 1:n_atoms]
        z_list = sortperm(z_coords)

        F_x_direct, F_y_direct = direct_Fxy(q, coords, Lx, Ly, Lz, kc, alpha, g1, g2)
        F_z_direct = direct_Fz(q, coords, Lx, Ly, Lz, kc, alpha, g1, g2)
        F_long_direct = [[F_x_direct[i], F_y_direct[i], F_z_direct[i]] for i in 1:n_atoms]

        QEM_long_inter = QEM_long((Lx, Ly, Lz) ;k_cutoff = kc, gamma_1 = g1, gamma_2 = g2, eps_0 = 1, alpha = alpha, rbe_mode = false)
        atom_mass = 1.0u"NoUnits"
        atoms = [Atom(mass=atom_mass, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = q[i]) for i in 1:n_atoms]
        boundary = CubicBoundary(Lx * u"NoUnits", Ly * u"NoUnits", Inf * u"NoUnits")
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
        F_long_Molly = F_long(sys, QEM_long_inter)


        @test F_long_Molly ≈ F_long_direct
    end
end