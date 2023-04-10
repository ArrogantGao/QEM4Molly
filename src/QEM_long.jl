export QEM_long

mutable struct QEM_long{KC, ZL, G1, G2, EP, LA, AC, RM, RP}
    k_cutoff::KC # the cufoff length in the k space
    z_list::ZL
    gamma_1::G1 
    gamma_2::G2
    eps_0::EP
    L::LA # L = (L_x, L_y, L_z)
    accuracy::AC # the accuracy needed for integral
    alpha::AL 
    rbe_mode::RM # a Bool value, true for rbe_mode on and false for rbe_mode off
    rbe_p::RP # if rbe_mode on, p is the batch size
end

function ()
    
end