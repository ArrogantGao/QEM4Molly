export Gauss_int
# export Gauss_Legendre
export Gauss_parameter

# the finite integral is defined by \int_a^b f(x, para) \w(x) dx = \sum_{i = 1}^{step} f(x_i, para) w_i, where (x_i, w_i) is given by the package GaussQuadrature.jl

struct Gauss_parameter{N, T}
    s::NTuple{N, T}
    w::NTuple{N, T}
end

Gauss_parameter(Step::Int) = Gauss_parameter{Step, Float64}(tuple(legendre(Step)[1]...), tuple(legendre(Step)[2]...))
Gauss_parameter(s::T, w::T) where T <: AbstractVector = Gauss_parameter{length(s), eltype(s)}(tuple(s...), tuple(w...))

# here by profiling I found that half of the time cost is caused by function laguree(), so generating the parameter once will be better


@inline function Gauss_int(integrand::Function, Gaussian::Gauss_parameter, para::greens_element; region::NTuple{2, T} = (-1.0, 1.0)) where {T <: Number}
    a, b = region
    s = Gaussian.s
    w = Gaussian.w

    s = (b + a) / 2 .+ (b - a) .* s ./ 2.0
    w = (b - a) .* w ./ 2.0

    result = 0
    for i in axes(s)[1]
        result += integrand(s[i], para; l = 0) * w[i]
    end

    return result
end

# function Guass_int(integrand::Function; para = nothing, region::Tuple{Real, Real} = (-1, 1), Step::Int = 10, Quadrature::Function = legendre, l::Int = 0)

#     x, w = Quadrature(Step)
#     a, b = region
#     x = (b + a) / 2 .+ (b - a) .* x / 2
#     w = (b - a) .* w / 2

#     if para == nothing
#         result = sum(integrand(x[i]; l = l) * w[i] for i in 1:Step)
#     else
#         result = sum(integrand(x[i], para; l = l) * w[i] for i in 1:Step)
#     end

#     return result
# end

# # for integrals with form of \int_a^b f(x, para) dx
# function Gauss_Legendre(integrand::Function; para = nothing, region::Tuple{Real, Real} = (-1, 1), Step::Int = 10, l::Int = 0)
#     return Guass_int(integrand; para = para, region = region, Step = Step, Quadrature = legendre, l = l)
# end
