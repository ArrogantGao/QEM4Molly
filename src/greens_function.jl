export greens_element, greens_element_i_init, greens_element_ij_init, Gamma_1, Gamma_2, dz_Gamma_1, dz_Gamma_2

# this structure defines the elements need calculate the interaction between a pair wise interaction
struct greens_element{G1, G2, RHO, A, B, SA, LZ, AL}
    γ_1::G1
    γ_2::G2
    rho_ij::RHO
    a::A
    b::B
    sign_a::SA
    L_z::LZ
    alpha::AL
end

function greens_element_ij_init(γ_1, γ_2, z_i, z_j, rho_ij, L_z, alpha)

    z_n = abs(z_i - z_j)
    z_p = z_i + z_j
    a = (z_n, z_p, 2 * L_z - z_p, 2 * L_z - z_n)
    b = (1, γ_1, γ_2, γ_1 * γ_2)

    sign_a = ( - sign(z_i - z_j), -1, +1, sign(z_i - z_j))

    return greens_element{typeof(γ_1), typeof(γ_2), typeof(rho_ij), typeof(a), typeof(b), typeof(sign_a), typeof(L_z), typeof(alpha)}(γ_1, γ_2, rho_ij, a, b, sign_a, L_z, alpha)
end

# the part below defines the sturcture for the self_interaction
function greens_element_i_init(γ_1, γ_2, z_i, L_z, alpha)

    rho_ij = 0

    z_n = 0
    z_p = 2 * z_i
    a = (z_n, z_p, 2 * L_z - z_p, 2 * L_z - z_n)
    b = (1, γ_1, γ_2, γ_1 * γ_2)

    sign_a = (0, -1, +1, 0)

    return greens_element{typeof(γ_1), typeof(γ_2), typeof(rho_ij), typeof(a), typeof(b), typeof(sign_a), typeof(L_z), typeof(alpha)}(γ_1, γ_2, rho_ij, a, b, sign_a, L_z, alpha)
end


# these functions define the Gamma1/2, which is used in calculation of the potential
function Gamma_1(k, element::greens_element)
    green_u = 0.5 * sum(element.b[i] * exp(- k * element.a[i]) for i in 1:4)
    green_d = element.γ_1 * element.γ_2 * exp(- 2 * k * element.L_z) - 1
    G_1 = green_u / green_d
    return G_1
end

function Gamma_2(k, element::greens_element)
    green_u = 0.5 * sum(element.b[i] * exp(- k * element.a[i]) for i in 1:4)
    green_d = element.γ_1 * element.γ_2 * exp(- 2 * k * element.L_z) - 1
    G_2 = element.γ_1 * element.γ_2 * green_u * exp(- 2 * k * element.L_z) / green_d
    return G_2
end


# these functions define the dz_Gamma1/2, which are used to calculate the forces
function dz_Gamma_1(k, element::greens_element)
    dz_green_u = 0.5 * sum(k * element.sign_a[i] * element.b[i] * exp(- k * element.a[i]) for i in 1:4)
    green_d = element.γ_1 * element.γ_2 * exp(- 2 * k * element.L_z) - 1

    dz_G_1 = dz_green_u / green_d
    return dz_G_1
end

function dz_Gamma_2(k, element::greens_element)
    dz_green_u = 0.5 * sum(k * element.sign_a[i] * element.b[i] * exp(- k * element.a[i]) for i in 1:4)
    green_d = element.γ_1 * element.γ_2 * exp(- 2 * k * element.L_z) - 1

    dz_G_2 = element.γ_1 * element.γ_2 * dz_green_u * exp(- 2 * k * element.L_z) / green_d
    return dz_G_2
end



