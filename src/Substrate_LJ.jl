# using Molly


export Substrate_LJ


struct Substrate_LJ{C, S, E, SP, D}
    cutoff::C
    sigma::S
    eps::E
    sub_pos::SP
    direction::D
end

function Substrate_LJ(;cutoff = 1.0, sigma = 0.5, eps = 1.0, sub_pos = 0.0, direction = +1)
    return Substrate_LJ{typeof(cutoff), typeof(sigma), typeof(eps), typeof(sub_pos), typeof(direction)}(cutoff, sigma, eps, sub_pos, direction)
end

@fastmath function substrate_LJ_force(invr2, sigma2, epsilon)
    six_term = (sigma2 * invr2) ^ 3
    return (24 * epsilon * invr2) * (2 * six_term ^ 2 - six_term)
end

@inline @inbounds function Molly.forces(inter::Substrate_LJ, sys, neighbors=nothing)
    sigma = inter.sigma
    sigma2 = sigma^2
    epsilon = inter.eps
    z_slab = inter.sub_pos

    n_atoms = size(sys.atoms)[1]

    F = [zeros(typeof(sigma), 3) for i in 1:n_atoms]

    for i in 1:n_atoms
        z_i = sys.coords[i][3]
        dz = z_i - z_slab
        @assert inter.direction * dz > 0 "Particle run out of the box!"
        if dz < inter.cutoff
            invr2 = 1 / (dz)^2
            fz = inter.direction * substrate_LJ_force(invr2, sigma2, epsilon)
            F[i][3] = fz
        end
    end

    return F
end