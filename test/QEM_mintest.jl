using QEM4Molly
using Profile
using Molly
using Unitful
using StatProfilerHTML
using BenchmarkTools
using LinearAlgebra
using SpecialFunctions

n_atoms = 100
atom_mass = 1.0u"NoUnits"
atoms = [Atom(mass=atom_mass, σ=0.3u"NoUnits", ϵ=0.2u"NoUnits", charge = (-1)^i) for i in 1:n_atoms]

L_x = 100.0
L_y = 100.0
L_z = 50.0

boundary = CubicBoundary(L_x * u"NoUnits", L_y * u"NoUnits", Inf * u"NoUnits") # Periodic boundary conditions
boundary_place = CubicBoundary(L_x * u"NoUnits", L_y * u"NoUnits", 9 * u"NoUnits")
coords_temp = place_atoms(n_atoms, boundary_place; min_dist=0.3u"NoUnits") # Random placement without clashing

coords = [SVector{3, typeof(x)}(x, y, z + .5) for (x, y, z) in coords_temp]

temp = 1.0u"NoUnits"
velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

pairwise_inters = (LennardJones(;cutoff = DistanceCutoff(3.5u"NoUnits"), energy_units = u"NoUnits", force_units = u"NoUnits"),)

s = 0.5
e = 1
Substrate_LJ_inter_1 = Substrate_LJ(;cutoff = 1.0, sigma = s , eps = e, sub_pos = 0.0, direction = +1)
Substrate_LJ_inter_2 = Substrate_LJ(;cutoff = 1.0, sigma = s, eps = e, sub_pos = 10, direction = -1)

QEM_short_inter, QEM_long_inter = QEM_inter(n_atoms, (L_x, L_y, L_z); gamma_1 = 0.9, gamma_2 = 0.9, rbe_mode = false, rbe_p = 30, N_t = 10)

general_inters = (QEM_short_inter, QEM_long_inter, )

# general_inters = (Substrate_LJ_inter_1, Substrate_LJ_inter_2, )

sys = System(
    atoms=atoms,
    pairwise_inters=pairwise_inters,
    general_inters = general_inters,
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

Force = Molly.forces(QEM_long_inter, sys) +  Molly.forces(QEM_short_inter, sys)


coords_1 = [5.0, 5.0, 5.0]
coords_2 = [6.0, 5.0, 5.0]
F_s = F_short_ij(1, 1, coords_1, coords_2, QEM_short_inter)

Gauss_para = Gauss_parameter(30)

ge_ij_1 = greens_element_ij_init(0.5, 0.8, 2.2, .9, 1.0, 2.2, 10.0, 0.1)

function F_sz_gauss_c(k::T, element::greens_element; l::Int = 0) where T<:Number
    f_sz_g = dz_Gamma_1(k, element; l = l) * exp(- k^2 / (4 * element.alpha)) * besselj0(k * element.rho_ij)
    return f_sz_g
end

I = Gauss_int(F_sz_gauss_c, Gauss_para, ge_ij_1, region = (0.0, 1.0))

