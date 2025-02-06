using SimplePWS

# Helper functions
function make_models(kappa::Float64, tau::Float64, gain::Float64, dt::Float64)
    input_model = BirthDeathModel(kappa, 1.0, dt)
    rho = 2 * kappa / tau
    mu = 1 / tau
    n = 2 * gain
    K = kappa
    output_model = NonlinearModel(rho, n, K, mu, dt)
    return input_model, output_model
end

function run_simulation(kappa::Float64, tau::Float64, gain::Float64, dt::Float64; kwargs...)
    input_model, output_model = make_models(kappa, tau, gain, dt)
    data = mutual_information(input_model, output_model, dt=dt; kwargs...)
    
    params = Dict(
        :gain => gain,
        :rho => output_model.rho,
        :kappa => kappa,
        :tau => tau,
        :mu => output_model.mu,
        :n => output_model.n,
        :K => output_model.K
    )

    return merge(data, Dict(:Parameters => params))
end

run_simulation(100.0, 1.0, 5.0, 1e-2)