using PyCall
using Revise
using Test
using Plots

include("../src/types.jl")
include("../src/ICM.jl")

sys = pyimport("sys")
sys.path = push!(sys.path, "src/python_lib/")
ICM = pyimport("ICM")
np = pyimport("numpy")


N = 10
γ_up = 0.10
γ_down = - 0.15

L = (1.0, 1.0, 1.0)
position = [L .* rand(3) for _=1:N]
charge = [(-1.0)^i for i in 1:N]


N_img = 5
N_real = 20

sys = IcmSys((γ_up, γ_down), L, N_real, N_img)
ref_pos, ref_charge = IcmSysInit(sys, position, charge)

energy_jl = IcmEnergy(sys, position, charge, ref_pos, ref_charge)
force_jl = IcmForce(sys, position, charge, ref_pos, ref_charge)

e_1 = (1 - γ_down) / (1 + γ_down)
e_3 = (1 - γ_up) / (1 + γ_up)
INCAR = [N, 1.0, L[1], L[2], L[3], e_1, 1.0, e_3, 1, 1e-9]

POSCAR = np.array([[charge[i], position[i][1], position[i][2], position[i][3]] for i in 1:N])
POS = np.zeros([2 * N_img + 1, N, 4])

energy_py = ICM.U_cal(INCAR, POSCAR, POS, N_img, N_real)
force_py = ICM.F_cal(INCAR, POSCAR, POS, N_img, N_real)

@test energy_py ≈ energy_jl
@testset "force compare with py lib" begin
    for i in 1:N
        @test force_jl[i].coo[1] ≈ force_py[1][i]
        @test force_jl[i].coo[2] ≈ force_py[2][i]
        @test force_jl[i].coo[3] ≈ force_py[3][i] + force_py[6][i]
    end
end

energy_Array = Vector{Float64}()
N_Array = Vector{Int}()

for N_num in [400]
    N_real = N_num
    N_img = 5
    sys = IcmSys((γ_up, γ_down), L, N_real, N_img)
    ref_pos, ref_charge = IcmSysInit(sys, position, charge)
    @time push!(energy_Array, IcmEnergy(sys, position, charge, ref_pos, ref_charge))
    push!(N_Array, N_num)
end

plot(N_Array, energy_Array)