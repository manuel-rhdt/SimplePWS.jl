module MutualInformation

using StatsFuns
using Distributions

using ..Models

export mutual_information

# Define the resample_state function
function resample_state(inp_state::Tuple{Val, Vector{Float64}, Vector{Float64}}, log_weights::Vector{Float64})
    weights = softmax(log_weights)
    resampled_indices = rand(Categorical(weights), size(log_weights))
    state = map(x -> x[resampled_indices], inp_state[2:end])
    return (inp_state[1], state...)
end

# Define the smc_estimate function
function smc_estimate(input_model::StochasticModel, output_model::StochasticModel, x::Array{Float64}, n_particles::Int = 128)
    function smc_run(input_model::StochasticModel, output_model::StochasticModel, x::AbstractVector{Float64})
        s = zeros(n_particles, size(x, 1))
        inp_state = Models.initial_state(input_model, true; s)
        out_state = Models.initial_state(output_model, false; s, x=reshape(x, (1, length(x))))

        log_marginals, ess_vals = Float64[0.0], Float64[n_particles]

        itr = Iterators.drop(zip(eachcol(s), eachcol(x')), 1)
        foldl(itr, init=(inp_state, out_state)) do state, v
            inp_state, out_state = state
            s, x = v
            inp_state = Models.step(input_model, inp_state; s)
            out_state = Models.step(output_model, out_state; s, x)
            log_weights = out_state[2]
            log_marginal_estimate = logsumexp(log_weights) - log(n_particles)
            ess = 1 / sum(softmax(log_weights) .^ 2) / n_particles

            if ess < 0.5
                inp_state = resample_state(inp_state, log_weights)
                log_weights .= log_marginal_estimate
            end

            push!(log_marginals, log_marginal_estimate)
            push!(ess_vals, ess)

            (inp_state, out_state)
        end

        return log_marginals, ess_vals
    end

    result = map(x -> smc_run(input_model, output_model, x), eachrow(x))
    log_marginals = mapreduce(x -> x[1], hcat, result)'
    ess = mapreduce(x -> x[2], hcat, result)'
    log_marginals, ess
end

# Define mutual_information function
function mutual_information(input_model::StochasticModel, output_model::StochasticModel; num_steps::Int = 500, dt::Float64 = 1e-2, mc_samples::Int = 1000, n_particles::Int = 1024)
    s = zeros(mc_samples, num_steps)
    input_model(true; s=s)
    x = similar(s)
    output_model(true; s=s, x=x)

    log_c = log_probability(output_model; s, x)
    log_m, ess = smc_estimate(input_model, output_model, x, n_particles)
    time = collect(1:num_steps) .* dt
    mi = mean(log_c .- log_m, dims=1)[1, :]
    variance = var(log_c .- log_m, dims=1)[1, :]
    return Dict(
        :Time => time,
        :MutualInformation => mi,
        :Variance => variance,
        :MC_Samples => mc_samples,
        :SEM => sqrt.(variance ./ mc_samples),
        :SMC_particles => n_particles
    )
end

end