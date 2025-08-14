module logistic_model

using StatsFuns
using Distributions

using ..Models

import ..Models: step, initial_state, copy_state

struct LogisticModel <: StochasticModel
    gain::Float64
    decay::Float64
    noise::Float64
end

function initial_state(::LogisticModel, generate::Bool=false)
    return Val(generate), 0.0
end

function copy_state(::LogisticModel, state::Tuple{Val, Float64})
    return (state[1], state[2])
end

function step(model::LogisticModel, carry::Tuple{Val,Float64}; s::Float64, x::Union{Nothing,Float64}=nothing)
    generate, x_prev = carry
    bias = logistic(s * model.gain)
    noise_dist = Normal(0, model.noise)

    if generate === Val(true)
        noise = rand(noise_dist)
        x = bias + x_prev * model.decay + noise
        logp = logpdf(noise_dist, noise)
        return (generate, x::Float64), (logp, (; x))
    else
        logp = logpdf(noise_dist, x - bias - x_prev * model.decay)
        return (generate, x::Float64), logp
    end
end

end