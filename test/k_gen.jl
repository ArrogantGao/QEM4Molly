@testset "test the k generator function" begin

    K, P, sum_K = K_set_generator(1, 1, 0.1, 1e-6)

    @test sum(P) ≈ 1.0

end