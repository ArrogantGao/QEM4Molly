export QEM_short


mutable struct QEM_short{TF, TI, NL, GP}
    r_cutoff::TF # cutoff is the cutoff length of the short range interaction
    # cell_list::CL # cell list is the InPlaceNeighborList
    neighbor_list::NL
    n_steps::TI # n_steps is the number of the currect step
    iter_period::TI # after iter_period step of iterations, update the cell list
    gamma_1::TF
    gamma_2::TF
    eps_0::TF
    L::NTuple{3, TF}
    accuracy::TF # the accuracy needed for integral
    N_t::TI # number of point used in Integrator
    alpha::TF
    Gauss_para::GP
end

function QEM_short(L, r_cutoff, alpha; iter_period = 100, gamma_1 = 0.0, gamma_2 = 0.0, eps_0 = 1.0, accuracy = 10^(-6), N_t = 30)
    
    neighbor_list = Tuple{Int64, Int64, Float64}[]
    n_steps = 0

    Gauss_para = Gauss_parameter(N_t)

    return QEM_short{typeof(r_cutoff), typeof(N_t), typeof(neighbor_list), Gauss_parameter}(r_cutoff, neighbor_list, n_steps, iter_period, gamma_1, gamma_2, eps_0, L, accuracy, N_t, alpha, Gauss_para)
end

function Molly.forces(inter::QEM_short, sys, neighbors=nothing)

    n_atoms = size(sys.coords)[1]

    # update the number of step in the inter, and if the number is n times period, update the neighborlist
    inter.n_steps += 1
    if (inter.n_steps - 1) % inter.iter_period == 0
        coords_q2d = [[x_i[1], x_i[2]] for x_i in sys.coords]
        inter.neighbor_list = neighborlist(coords_q2d, inter.r_cutoff + 0.1; unitcell = [inter.L[1], inter.L[2]])
    end

    F_short = [zeros(Float64, 3) for i in 1:n_atoms]

    # here we will compute the short range pairwise interaction
    for (i, j, rho_fake) in inter.neighbor_list
        q_i, q_j = sys.atoms[i].charge, sys.atoms[j].charge
        coord_i = sys.coords[i]
        coord_j = sys.coords[j]
        F_i, F_j = F_short_ij(q_i, q_j, coord_i, coord_j, inter)
        F_short[i] += F_i
        F_short[j] += F_j
    end

    # here we will compute the short range self interaction (only in z direction)
    for i in 1:n_atoms
        q_i = sys.atoms[i].charge
        coord_i = sys.coords[i]
        F_i_z = F_short_i(q_i, coord_i, inter)
        F_short[i] .+= F_i_z
    end

    return F_short
end

function Molly.potential_energy(inter::QEM_short, sys, neighbors=nothing)

    n_atoms = size(sys.coords)[1]

    E_short = 0.0

    # here we will compute the short range pairwise interaction
    for (i, j, rho_fake) in inter.neighbor_list
        q_i, q_j = sys.atoms[i].charge, sys.atoms[j].charge
        coord_i = sys.coords[i]
        coord_j = sys.coords[j]
        E_ij = E_short_ij(q_i, q_j, coord_i, coord_j, inter)
        E_short += E_ij
    end

    # here we will compute the short range self interaction
    for i in 1:n_atoms
        q_i = sys.atoms[i].charge
        coord_i = sys.coords[i]
        E_i = E_short_i(q_i, coord_i, inter)
        E_short += E_i
    end

    return E_short
end