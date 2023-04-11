export QEM_short


mutable struct QEM_short{C, NL, NS, IP, G1, G2, EP, LA, AC, NT, AL}
    r_cutoff::C # cutoff is the cutoff length of the short range interaction
    # cell_list::CL # cell list is the InPlaceNeighborList
    neighbor_list::NL
    n_steps::NS # n_steps is the number of the currect step
    iter_period::IP # after iter_period step of iterations, update the cell list
    gamma_1::G1 
    gamma_2::G2
    eps_0::EP
    L::LA # L = (L_x, L_y, L_z)
    accuracy::AC # the accuracy needed for integral
    N_t::NT # number of point used in Integrator
    alpha::AL
end

function QEM_short(L, r_cutoff, alpha; iter_period = 100, gamma_1 = 0, gamma_2 = 0, eps_0 = 1, accuracy = 10^(-6), N_t = 30)
    
    neighbor_list = []
    n_steps = 0

    return QEM_short{typeof(r_cutoff), typeof(neighbor_list), typeof(n_steps), typeof(iter_period), typeof(gamma_1), typeof(gamma_2), typeof(eps_0), typeof(L), typeof(accuracy), typeof(N_t), typeof(alpha)}(r_cutoff, neighbor_list, n_steps, iter_period, gamma_1, gamma_2, eps_0, L, accuracy, N_t, alpha)
end

function Molly.force(inter::QEM_short, sys, neighbors=nothing)

    n_atoms = size(sys.coords)[1]

    # update the number of step in the inter, and if the number is n times period, update the neighborlist
    inter.n_steps += 1
    if (inter.n_steps - 1) % iter_period == 0
        coords_q2d = [[x_i[1], x_i[2]] for x_i in sys.coords]
        inter.neighbor_list = neighborlist(coords_q2d, inter.r_cutoff; unitcell = inter.unitcell)
    end

    F_short = [zeros(Float64, 3) for i in 1:n_atoms]

    # here we will compute the short range pairwise interaction
    for (i, j, rho_fake) in inter.neighbor_list
        q_i, q_j = sys.atoms[i].charge, sys.atoms[j].charge
        coord_i = sys.coords[i]
        coord_j = sys.coords[j]
        F_ij = F_short_ij(q_i, q_j, coord_i, coord_j, inter)
        F_short[i] += F_ij
        F_short[j] -= F_ij
    end

    # here we will compute the short range self interaction (only in z direction)
    for i in 1:n_atoms
        q_i = sys.atoms[i].charge
        coord_i = sys.coords[i]
        F_i_z = F_short_i(q_i, coord_i, inter)
        F_short[i][3] += F_i_z
    end

    return F_short
end