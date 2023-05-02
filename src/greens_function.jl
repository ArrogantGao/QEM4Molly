export greens_element, greens_element_i_init, greens_element_ij_init, greens_element_init, Gamma_1, Gamma_2, dz_Gamma_1, dz_Gamma_2

# this structure defines the elements need calculate the interaction between a pair wise interaction
struct greens_element{G1, G2, RHO, A, B, SA, LZ, AL, KF}
    gamma_1::G1
    gamma_2::G2
    rho_ij::RHO
    a::A
    b::B
    sign_a::SA
    L_z::LZ
    alpha::AL
    k_f1::KF
    k_f2::KF
end

function greens_element_ij_init(gamma_1, gamma_2, z_i, z_j, rho_ij, L_z, alpha, accuracy)

    z_n = abs(z_i - z_j)
    z_p = z_i + z_j
    a = (z_n, z_p, 2 * L_z - z_p, 2 * L_z - z_n)
    b = (1, gamma_1, gamma_2, gamma_1 * gamma_2)

    sign_a = ( - sign(z_i - z_j), -1, +1, sign(z_i - z_j))

    k_f1 = sqrt.(4 * alpha^2 .* a.^2 .- 4 * alpha * log(accuracy)) .- 2 * alpha .* a
    k_f2 = - log(accuracy) ./ (2 * L_z .+ a)

    return greens_element{typeof(gamma_1), typeof(gamma_2), typeof(rho_ij), typeof(a), typeof(b), typeof(sign_a), typeof(L_z), typeof(alpha), typeof(k_f1)}(gamma_1, gamma_2, rho_ij, a, b, sign_a, L_z, alpha, k_f1, k_f2)
end

# the part below defines the sturcture for the self_interaction
function greens_element_i_init(gamma_1, gamma_2, z_i, L_z, alpha, accuracy)

    rho_ij = 0

    z_n = 0
    z_p = 2 * z_i
    a = (z_n, z_p, 2 * L_z - z_p, 2 * L_z - z_n)
    b = (1, gamma_1, gamma_2, gamma_1 * gamma_2)

    sign_a = (0, -1, +1, 0)

    k_f1 = sqrt.(4 * alpha^2 .* a.^2 .- 4 * alpha * log(accuracy)) .- 2 * alpha .* a
    k_f2 = - log(accuracy) ./ (2 * L_z .+ a)

    return greens_element{typeof(gamma_1), typeof(gamma_2), typeof(rho_ij), typeof(a), typeof(b), typeof(sign_a), typeof(L_z), typeof(alpha), typeof(k_f1)}(gamma_1, gamma_2, rho_ij, a, b, sign_a, L_z, alpha, k_f1, k_f2)
end

# the part below defines the sturcture for long range interaction
function greens_element_init(gamma_1, gamma_2, L_z, alpha)

    rho_ij = 0

    z_n = 0
    z_p = 0
    a = (z_n, z_p, 2 * L_z - z_p, 2 * L_z - z_n)
    b = (1, gamma_1, gamma_2, gamma_1 * gamma_2)

    sign_a = (0, -1, +1, 0)
    k_f1 = []
    k_f2 = []

    return greens_element{typeof(gamma_1), typeof(gamma_2), typeof(rho_ij), typeof(a), typeof(b), typeof(sign_a), typeof(L_z), typeof(alpha), typeof(k_f1)}(gamma_1, gamma_2, rho_ij, a, b, sign_a, L_z, alpha, k_f1, k_f2)
end

# these functions define the Gamma1/2, which is used in calculation of the potential
function Gamma_1(k, element::greens_element; l::Int = 0)
    if l == 0
        green_u = 0.5 * sum(element.b[i] * exp(- k * element.a[i]) for i in 1:4)
    else
        green_u = 0.5 * element.b[l] * exp(- k * element.a[l])
    end
    green_d = element.gamma_1 * element.gamma_2 * exp(- 2 * k * element.L_z) - 1
    G_1 = green_u / green_d
    return G_1
end

function Gamma_2(k, element::greens_element; l::Int = 0)
    if l == 0
        green_u = 0.5 * sum(element.b[i] * exp(- k * element.a[i]) for i in 1:4)
    else
        green_u = 0.5 * element.b[l] * exp(- k * element.a[l])
    end
    green_d = element.gamma_1 * element.gamma_2 * exp(- 2 * k * element.L_z) - 1
    G_2 = element.gamma_1 * element.gamma_2 * green_u * exp(- 2 * k * element.L_z) / green_d
    return G_2
end


# these functions define the dz_Gamma1/2, which are used to calculate the forces
function dz_Gamma_1(k, element::greens_element; l::Int = 0)
    if l == 0
        dz_green_u = 0.5 * sum(k * element.sign_a[i] * element.b[i] * exp(- k * element.a[i]) for i in 1:4)
    else
        dz_green_u = 0.5 * k * element.sign_a[l] * element.b[l] * exp(- k * element.a[l])
    end

    green_d = element.gamma_1 * element.gamma_2 * exp(- 2 * k * element.L_z) - 1

    dz_G_1 = dz_green_u / green_d
    return dz_G_1
end

function dz_Gamma_2(k, element::greens_element; l::Int = 0)
    if l == 0
        dz_green_u = 0.5 * sum(k * element.sign_a[i] * element.b[i] * exp(- k * element.a[i]) for i in 1:4)
    else
        dz_green_u = 0.5 * k * element.sign_a[l] * element.b[l] * exp(- k * element.a[l])
    end

    green_d = element.gamma_1 * element.gamma_2 * exp(- 2 * k * element.L_z) - 1
    dz_G_2 = element.gamma_1 * element.gamma_2 * dz_green_u * exp(- 2 * k * element.L_z) / green_d

    return dz_G_2
end



