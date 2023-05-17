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

struct Point{N,T}
    coo::NTuple{N,T}
end
Point(arg::T, args::T...) where T<:Number = Point((arg, args...))
Base.:(+)(x::Point{N,T}, y::Point{N,T}) where {N, T} = Point(x.coo .+ y.coo)
Base.:(-)(x::Point{N,T}, y::Point{N,T}) where {N, T} = Point(x.coo .- y.coo)
Base.:(-)(x::Point{N,T}) where {N, T} = Point(Base.:(-).(x.coo))
Base.adjoint(x::Point) = x
Base.:(*)(x::Number, y::Point) = Point(y.coo .* x)
Base.:(*)(y::Point, x::Number) = Point(y.coo .* x)
Base.iterate(x::Point, args...) = Base.iterate(x.coo, args...)
Base.getindex(x::Point, i::Int) = x.coo[i]

dist2(x::Number, y::Number) = abs2(x - y)
dist2(x::Point, y::Point) = sum(abs2, x - y)
