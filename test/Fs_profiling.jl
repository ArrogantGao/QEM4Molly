using QEM4Molly, Test, LinearAlgebra, Test, Unitful, Molly, CellListMap, PyCall, Revise, JET, BenchmarkTools, SpecialFunctions

ge_ij_1 = greens_element_ij_init(0.5, 0.8, 2.2, .9, 1.0, 2.2, 10.0, 0.1)

function F_sz_gauss_c(k::T, element::greens_element; l::Int = 0) where T<:Number
    f_sz_g::T = dz_Gamma_1(k, element; l = l) * exp(- k^2 / (4 * element.alpha)) * besselj0(k * element.rho_ij)
    return f_sz_g
end


@report_opt dz_Gamma_1(1.0, ge_ij_1)
@report_opt dz_Gamma_2(1.0, ge_ij_1)

struct Force_single{N,T}
    coo::NTuple{N,T}
end
Force_single(arg::T, args::T...) where T<:Number = Force_single((arg, args...))
Base.:(+)(x::Force_single{N,T}, y::Force_single{N,T}) where {N, T} = Force_single(x.coo .+ y.coo)
Base.:(-)(x::Force_single{N,T}, y::Force_single{N,T}) where {N, T} = Force_single(x.coo .- y.coo)
Base.:(-)(x::Force_single{N,T}) where {N, T} = Force_single(Base.:(-).(x.coo))
Base.adjoint(x::Force_single) = x
Base.:(*)(x::Number, y::Force_single) = Force_single(y.coo .* x)
Base.:(*)(y::Force_single, x::Number) = Force_single(y.coo .* x)
Base.iterate(x::Force_single, args...) = Base.iterate(x.coo, args...)
Base.getindex(x::Force_single, i::Int) = x.coo[i]

# this seems to be unnesscary
# struct Force_collect{N, T}
#     F_coo::NTuple{N, T}
# end

# Force_collect(arg::T, args::T...) where T<:Force_single = Force_collect((arg, args...))

# function creat_force(N::Int)
#     return tuple(Force_single(rand(3)...))
# end

function distance_check(cooi::T, cooj::T, Lx::F, Ly::F, r_c::F) where{T<:Point, F<:Number}
    for mx in (0.0, -1.0, 1.0)
        for my in (0.0, -1.0, 1.0)
            r = dist2(cooi + Point(mx * Lx, my * Ly, 0.0), cooj)
            println(r)
            if r < r_c
                return (cooi + Point(mx * Lx, my * Ly, 0.0), cooj, r)
            end
        end
    end
    return nothing
end

a = Point(4.9, 4.9, 1.0)
b = Point(0.1, 0.1, 1.0)

c = distance_check(a, b, 5.0, 5.0, 1.0)