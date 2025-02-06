using SimplePWS

using Statistics
using Test
using StatsBase

n=4
sigma = 1.0
coeffs = SimplePWS.generate_stable_ar_coefficients(n)

input_model = ARModel(coeffs, sigma)
output_model = NonlinearModel(50.0, 10.0, 50.0, 1.0, 1e-2)

s = rand(1_000, 10000)

input_model(true; s)


@test isapprox(mean(s), 0.0, atol=0.1)
@test all(isapprox.(mean(pacf(s', n+1:n+10; method=:yulewalker), dims=2), 0.0; atol=0.05))
