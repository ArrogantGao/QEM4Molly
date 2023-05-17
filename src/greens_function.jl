export greens_element, greens_element_i_init, greens_element_ij_init, greens_element_init, Gamma_1, Gamma_2, dz_Gamma_1, dz_Gamma_2

# this structure defines the elements need calculate the interaction between a pair wise interaction

struct greens_element{T}
    gamma_1::T
    gamma_2::T
    rho_ij::T
    a::NTuple{4, T}
    b::NTuple{4, T}
    sign_a::NTuple{4, T}
    L_z::T
    alpha::T
    k_f1::NTuple{4, T}
    k_f2::NTuple{4, T}
end

function greens_element_init(gamma_1::T, gamma_2::T, L_z::T, alpha::T) where T <: Number

    rho_ij = 0.0

    z_n = 0.0
    z_p = 0.0
    a = (z_n, z_p, 2.0 * L_z - z_p, 2.0 * L_z - z_n)
    b = (1.0, gamma_1, gamma_2, gamma_1 * gamma_2)

    sign_a = (0.0, -1.0, +1.0, 0.0)
    k_f1 = (0.0, 0.0, 0.0, 0.0)
    k_f2 = (0.0, 0.0, 0.0, 0.0)

    return greens_element{typeof(gamma_1)}(gamma_1, gamma_2, rho_ij, a, b, sign_a, L_z, alpha, k_f1, k_f2)
end

function greens_element_ij_init(gamma_1::T, gamma_2::T, z_i::T, z_j::T, rho_ij::T, L_z::T, alpha::T, accuracy::T) where T <: Number

    z_n = abs(z_i - z_j)
    z_p = z_i + z_j
    a = (z_n, z_p, 2 * L_z - z_p, 2 * L_z - z_n)
    b = (1.0, gamma_1, gamma_2, gamma_1 * gamma_2)

    sign_a = ( - sign(z_i - z_j), -1.0, +1.0, sign(z_i - z_j))

    k_f1 = sqrt.(4.0 * alpha^2 .* a.^2 .- 4 * alpha * log(accuracy)) .- 2 * alpha .* a
    k_f2 = - log(accuracy) ./ (2 * L_z .+ a)

    return greens_element{typeof(gamma_1)}(gamma_1, gamma_2, rho_ij, a, b, sign_a, L_z, alpha, k_f1, k_f2)
end

# the part below defines the sturcture for the self_interaction
function greens_element_i_init(gamma_1::T, gamma_2::T, z_i::T, L_z::T, alpha::T, accuracy::T) where T <: Number

    rho_ij = zero(T)

    z_n = zero(T)
    z_p = 2.0 * z_i
    a = (z_n, z_p, 2.0 * L_z - z_p, 2.0 * L_z - z_n)
    b = (1.0, gamma_1, gamma_2, gamma_1 * gamma_2)

    sign_a = (0.0, -1.0, +1.0, 0.0)

    k_f1 = sqrt.(4.0 * alpha^2 .* a.^2 .- 4.0 * alpha * log(accuracy)) .- 2.0 * alpha .* a
    k_f2 = - log(accuracy) ./ (2.0 * L_z .+ a)

    return greens_element{typeof(gamma_1)}(gamma_1, gamma_2, rho_ij, a, b, sign_a, L_z, alpha, k_f1, k_f2)
end

# these functions define the Gamma1/2, which is used in calculation of the potential
function Gamma_1(k::T, element::greens_element; l::Int = 0) where T<:Number
    green_u = zero(T)
    if l == 0
        for i in 1:4
            green_u += 0.5 * element.b[i] * exp(- k * element.a[i])
        end
    else
        green_u = 0.5 * element.b[l] * exp(- k * element.a[l])
    end
    green_d = element.gamma_1 * element.gamma_2 * exp(- 2 * k * element.L_z) - 1.0
    G_1 = green_u / green_d
    return G_1
end

function Gamma_2(k::T, element::greens_element; l::Int = 0) where T<:Number
    green_u = zero(T)
    if l == 0
        for i in 1:4
            green_u += 0.5 * element.b[i] * exp(- k * element.a[i])
        end
    else
        green_u = 0.5 * element.b[l] * exp(- k * element.a[l])
    end
    green_d = element.gamma_1 * element.gamma_2 * exp(- 2 * k * element.L_z) - 1.0
    G_2 = element.gamma_1 * element.gamma_2 * green_u * exp(- 2.0 * k * element.L_z) / green_d
    return G_2
end


# these functions define the dz_Gamma1/2, which are used to calculate the forces
function dz_Gamma_1(k::T, element::greens_element; l::Int = 0) where T<:Number
    dz_green_u = zero(T)
    if l == 0
        for i in 1:4
            dz_green_u += 0.5 * k * element.sign_a[i] * element.b[i] * exp(- k * element.a[i])
        end
    else
        dz_green_u = 0.5 * k * element.sign_a[l] * element.b[l] * exp(- k * element.a[l])
    end

    green_d = element.gamma_1 * element.gamma_2 * exp(- 2.0 * k * element.L_z) - 1.0

    dz_G_1 = dz_green_u / green_d
    return dz_G_1
end

function dz_Gamma_2(k::T, element::greens_element; l::Int = 0) where T<:Number
    dz_green_u = zero(T)
    if l == 0
        for i in 1:4
            dz_green_u += 0.5 * k * element.sign_a[i] * element.b[i] * exp(- k * element.a[i])
        end
    else
        dz_green_u = 0.5 * k * element.sign_a[l] * element.b[l] * exp(- k * element.a[l])
    end

    green_d = element.gamma_1 * element.gamma_2 * exp(- 2 * k * element.L_z) - 1.0
    dz_G_2 = element.gamma_1 * element.gamma_2 * dz_green_u * exp(- 2.0 * k * element.L_z) / green_d

    return dz_G_2
end



