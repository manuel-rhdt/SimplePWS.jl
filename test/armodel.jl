using SimplePWS

using Statistics
using Test
using StatsBase

using JSON

n = 3
coeffs = [0.5, -0.3, 0.2]  # Example coefficients for AR(3) model
# coeffs = SimplePWS.generate_stable_ar_coefficients(n)

input_model = ARModel(coeffs, 1.0)

s = zeros(10_000)
logp = input_model(true, s=s)
logp2 = input_model(false; s=s)
@test logp == logp2

output_model = LogisticModel(1.0, 0.2, 0.5)

x = similar(s)
output_model(true, s=s, x=x)
x

mean(x)
@test isapprox(mean(s), 0.0, atol=0.1)

mutual_info = mutual_information(input_model, output_model; num_steps=50, dt=1.0, mc_samples=1024, n_particles=2^10)

params = Dict(
    :gain => output_model.gain,
    :decay => output_model.decay,
    :output_noise => output_model.noise,
    :ar_coefs => input_model.coefficients,
    :ar_noise => input_model.sigma,
)

mutual_info[:params] = params

open("simulation_results.json", "w") do f
    JSON.print(f, mutual_info, 4)  # 4 spaces for indentation
end

# @test all(isapprox.(mean(pacf(s', n+1:n+10; method=:yulewalker), dims=2), 0.0; atol=0.05))
