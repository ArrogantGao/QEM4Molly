export QEM_long

mutable struct QEM_long{KC, ZL, G, EP, LA, AC, AL, RM, RP, S, KS, PW}
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
    sum_K::S # summation of distribution 
    K_set::KS
    Prob::PW
end

function QEM_long(L ; k_cutoff = 1.0, z_list = [], gamma_1 = 0, gamma_2 = 0, eps_0 = 1, accuracy = 1e-6, alpha = 1, rbe_mode = true, rbe_p = 30)

    if rbe_mode == true
        K_set, Prob, sum_K = K_set_generator(L[1], L[2], alpha, accuracy)
    else
        K_set = []
        Prob = []
        sum_K = 0
    end

    return QEM_long{typeof(k_cutoff), typeof(z_list), typeof(gamma_1), typeof(eps_0), typeof(L), typeof(accuracy), typeof(alpha), typeof(rbe_mode), typeof(rbe_p), typeof(sum_K), typeof(K_set), typeof(Prob)}(k_cutoff, z_list, gamma_1, gamma_2, eps_0, L, accuracy, alpha, rbe_mode, rbe_p, sum_K, K_set, Prob)
end

function Molly.forces(inter::QEM_long, sys, neighbors=nothing)
    return F_long(sys, inter)

end

function Molly.potential_energy(inter::QEM_long, sys, neighbors=nothing)
    
    return E_long(sys, inter)
end