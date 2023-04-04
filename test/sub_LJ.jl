@testset "substrate_LJ Q2D simulation" begin
    n_atoms = 100
    atom_mass = 1.0u"NoUnits"
    atoms = [Atom(mass=atom_mass, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits") for i in 1:n_atoms]

    L_x = 10.0
    L_y = 10.0
    L_z = 10.0

    boundary = CubicBoundary(L_x * u"NoUnits", L_y * u"NoUnits", Inf * u"NoUnits") # Periodic boundary conditions
    boundary_place = CubicBoundary(L_x * u"NoUnits", L_y * u"NoUnits", 9 * u"NoUnits")
    coords = place_atoms(n_atoms, boundary_place; min_dist=0.3u"NoUnits") # Random placement without clashing

    temp = 1.0u"NoUnits"
    velocities = [velocity(atom_mass, temp, 1) for i in 1:n_atoms]

    pairwise_inters = (LennardJones(;energy_units = u"NoUnits", force_units = u"NoUnits",),)
    s = 0.5
    e = 1
    general_inters = (Substrate_LJ(;cutoff = 1.0, sigma = s , eps = e, sub_pos = -0.5, direction = +1), Substrate_LJ(;cutoff = 1.0, sigma = s, eps = e, sub_pos = 9.5, direction = -1),)

    sys = System(
        atoms=atoms,
        pairwise_inters=pairwise_inters,
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        loggers=(
            temp=TemperatureLogger(typeof(1.0), 10),
            coords=CoordinateLogger(typeof(1.0), 10),
        ),
        energy_units = u"NoUnits",
        force_units = u"NoUnits",
        k = 1.0
    )

    simulator = VelocityVerlet(
        dt=0.002u"NoUnits",
        coupling=AndersenThermostat(temp, 1.0u"NoUnits"),
    )

    simulate!(sys, simulator, 1_0000)

    @test sys.loggers.temp.n_steps == 10

end