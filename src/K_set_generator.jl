export K_set_generator

# this function is used to generate the a distribution of (k_x, k_y) under Probability given by exp(-k^2/4alpha), where k^2 = k_x^2 + k_y^2 and k != 0

function K_set_generator(L_x::L, L_y::L, alpha::AL, accuracy::AC) where{L<:Real, AL<:Real, AC<:Real}

    k_c = 2 * sqrt(- 4 * alpha * log(accuracy))
    mx_max = ceil(Int, k_c * L_x / 2π) + 1
    my_max = ceil(Int, k_c * L_y / 2π) + 1
    S_x::Float64 = 0.0
    S_y::Float64 = 0.0

    for m_x in - mx_max : mx_max
		if m_x != 0
	        k_x = m_x * 2π / L_x
	        S_x += exp(- k_x^2 / (4 * alpha))
		end
    end

    for m_y in - my_max : my_max
		if m_y != 0
	        k_y = m_y * 2π / L_y
	        S_y += exp(-k_y^2 / (4 * alpha))
		end
    end

    # In this way we can aviod problem cased by small S_x and S_y
    sum_K = S_x + S_y + S_x * S_y
    K = []
    P = Float32[]

    for m_x in - mx_max : mx_max
        for m_y in - my_max : my_max
            k_x = m_x * 2π / L_x
            k_y = m_y * 2π / L_y
            k = sqrt(k_x^2 + k_y^2)
            if k != 0
                K = push!(K, [k_x, k_y, k])
                P = push!(P, exp(-k^2 / (4 * alpha))/sum_K)
            end
        end
    end

    return K, ProbabilityWeights(P), sum_K
end