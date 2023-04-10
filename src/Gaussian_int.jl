export finite_integral
export Gauss_Legendre
export Gauss_Laguree
export Gauss_Hermite

# the finite integral is defined by \int_a^b f(x, para) \w(x) dx = \sum_{i = 1}^{step} f(x_i, para) w_i, where (x_i, w_i) is given by the package GaussQuadrature.jl


function Guass_int(integrand::Function; para = nothing, region::Tuple{Real, Real} = (-1, 1), Step::Int = 10, Quadrature::Function = legendre)

    x, w = Quadrature(Step)
    a, b = region
    x = (b + a) / 2 .+ (b - a) .* x / 2
    w = (b - a) .* w / 2

    if para == nothing
        result = sum(integrand(x[i]) * w[i] for i in 1:Step)
    else
        result = sum(integrand(x[i], para) * w[i] for i in 1:Step)
    end

    return result
end

# for integrals with form of \int_a^b f(x, para) dx
function Gauss_Legendre(integrand::Function; para = nothing, region::Tuple{Real, Real} = (-1, 1), Step::Int = 10)
    return Guass_int(integrand; para = para, region = region, Step = Step, Quadrature = legendre)
end

# for integrals with form of \int_a^b f(x, para) exp(- x) dx
function Gauss_Laguree(integrand::Function; para = nothing, region::Tuple{Real, Real} = (-1, 1), Step::Int = 10)
    result = Guass_int(integrand; para = para, region = region, Step = Step, Quadrature = laguerre) 
    return result
end

# for integrals with form of \int_a^b f(x, para) exp(- x^2) dx
function Gauss_Hermite(integrand::Function; para = nothing, region::Tuple{Real, Real} = (-1, 1), Step::Int = 10)
    return Guass_int(integrand; para = para, region = region, Step = Step, Quadrature = hermite)
end