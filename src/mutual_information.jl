module MutualInformation

using Statistics
using StatsFuns
using Distributions

using ..Models

export mutual_information

# Define the resample_state function
function resample_indices(log_weights::Vector{Float64})
    weights = softmax(log_weights)
    rand(Categorical(weights), size(log_weights))
end

# Define the smc_estimate function
function smc_estimate(input_model::StochasticModel, output_model::StochasticModel, x::Array{Float64}, n_particles::Int=128)
    function smc_run(input_model::StochasticModel, output_model::StochasticModel, x::AbstractVector{Float64})
        inp_states = [Models.initial_state(input_model, true) for _ in 1:n_particles]
        out_states = [Models.initial_state(output_model, false) for _ in 1:n_particles]

        log_weights = zeros(n_particles)
        log_marginals, ess_vals = Float64[], Float64[]
        prev_log_marginal_estimate = 0.0

        for i in eachindex(x)
            for j in 1:n_particles
                inp_state, (_, out) = Models.step(input_model, inp_states[j])
                out_state, logp = Models.step(output_model, out_states[j]; x=x[i], out...)

                inp_states[j] = inp_state
                out_states[j] = out_state

                log_weights[j] += logp
            end

            log_marginal_estimate = logsumexp(log_weights) - log(n_particles)
            ess = 1 / sum(softmax(log_weights) .^ 2) / n_particles

            push!(log_marginals, prev_log_marginal_estimate + log_marginal_estimate)
            push!(ess_vals, ess)

            if ess < 0.5
                println("Resampling at step $i with ESS: $ess")
                indices = resample_indices(log_weights)
                inp_states = [Models.copy_state(input_model, inp_states[idx]) for idx in indices]
                out_states = [Models.copy_state(output_model, out_states[idx]) for idx in indices]
                prev_log_marginal_estimate = log_marginals[end]
                log_weights .= 0.0
            end
        end

        return log_marginals, ess_vals
    end

    result = map(x -> smc_run(input_model, output_model, x), eachcol(x))
    log_marginals = mapreduce(x -> x[1], hcat, result)
    ess = mapreduce(x -> x[2], hcat, result)
    log_marginals, ess
end

# Define mutual_information function
function mutual_information(input_model::StochasticModel, output_model::StochasticModel; num_steps::Int=500, dt::Float64=1e-2, mc_samples::Int=1000, n_particles::Int=1024)
    s = zeros(num_steps, mc_samples)
    x = similar(s)
    log_c = similar(s)
    for i in 1:mc_samples
        s_view = @view s[:, i]
        x_view = @view x[:, i]
        input_model(true; s=s_view)
        log_c[:, i] .= output_model(true; s=s_view, x=x_view)
    end

    log_m, ess = smc_estimate(input_model, output_model, x, n_particles)
    time = collect(1:num_steps) .* dt
    mi = dropdims(mean(log_c .- log_m, dims=2); dims=2)
    variance = dropdims(var(log_c .- log_m, dims=2); dims=2)
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