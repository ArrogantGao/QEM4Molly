function func_1(x)
    return 1
end

function func_2(x)
    return x
end

struct parameter{K, N}
    k::K
    n::N
end

@testset "finite integral, no parameter test" begin
    for i in 1:10
        a, b = rand(2)
        a, b = min(a, b), max(a, b)
        @test Gauss_Legendre(func_1; region = (a, b), Step = 20) ≈ b - a
        @test Gauss_Legendre(func_2; region = (a, b), Step = 20) ≈ (b^2 - a^2) / 2
    end
end

function func_3(x, para)
    k = para[1]
    return k
end

function func_4(x, para)
    k, n = para
    return k * x^n
end

function func_5(x, para::parameter)
    k = para.k
    n = para.n
    return k * x^n
end

@testset "finite integral, with vector parameter test" begin
    for i in 1:10
        a, b = rand(2)
        a, b = min(a, b), max(a, b)
        k, n = abs.(rand(Int8, 2)) .% 5 .+ 1
        @test Gauss_Legendre(func_3; para = [k], region = (a, b), Step = 20) ≈ k * (b - a)
        @test Gauss_Legendre(func_4; para = [k, n], region = (a, b), Step = 20) ≈ k * (b^(n + 1) - a^(n + 1)) / (n + 1)
        @test Gauss_Legendre(func_5; para = parameter{Int, Int}(k, n), region = (a, b), Step = 20) ≈ k * (b^(n + 1) - a^(n + 1)) / (n + 1)
    end
end