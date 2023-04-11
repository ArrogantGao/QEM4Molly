export QEM_long

mutable struct QEM_long{KC, ZL, G, EP, LA, AC, AL, RM, RP, S, KS}
    k_cutoff::KC # the cufoff length in the k space
    z_list::ZL
    gamma_1::G 
    gamma_2::G
    eps_0::EP
    L::LA # L = (L_x, L_y, L_z)
    accuracy::AC # the accuracy needed for integral
    alpha::AL 
    rbe_mode::RM # a Bool value, true for rbe_mode on and false for rbe_mode off
    rbe_p::RP # if rbe_mode on, p is the batch size
    sum_k::S # summation of distribution 
    K_set::KS
end

function QEM_long(; k_cutoff = 1.0, z_list = [], gamma_1 = 0, gamma_2 = 0, eps_0 = 1, L = (1, 1, 1), accuracy = 1e-6, alpha = 1, rbe_mode = true, rbe_p = 30, sum_k = 1, K_set = [])
    return QEM_long{typeof(k_cutoff), typeof(z_list), typeof(gamma_1), typeof(eps_0), typeof(L), typeof(accuracy), typeof(alpha), typeof(rbe_mode), typeof(rbe_p), typeof(sum_k), typeof(K_set)}(k_cutoff, z_list, gamma_1, gamma_2, eps_0, L, accuracy, alpha, rbe_mode, rbe_p, sum_k, K_set)
end

function Molly.force(inter::QEM_long, sys, neighbors=nothing)
    n_atoms = size(sys.coords)[1]

    # update the z_list via bulket sorting (will be updated later)
    # inter.z_list = bulket_sorting(copy(sys.coord), inter.L)

    # this is a naive version to generate the z_list, costing time O(NlogN), the result will be given in ascending order
    z_coord = [coord[3] for coord in sys.coords]
    inter.z_list = sortperm(z_coord)

    return F_long(sys, inter)

end