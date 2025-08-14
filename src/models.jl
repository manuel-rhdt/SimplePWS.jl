module Models

using Random, Distributions, LinearAlgebra
using StatsFuns

export BirthDeathModel, NonlinearModel, StochasticModel, log_probability, initial_state, step

abstract type StochasticModel end

initial_state(model::StochasticModel, generate::Bool) = error("unimplemented")
copy_state(model::StochasticModel, state) = error("unimplemented")

function (model::StochasticModel)(generate::Bool = false; vals...)
    init_state = initial_state(model, generate)

    len = minimum(length, values(vals))

    if generate
        logp = zeros(len)
        state = foldl(range(; length=len); init=init_state) do state, i
            v = map(v -> v[i], (; vals...))
            state, (l, out) = step(model, state; v...)
            logp[i] = l
            for (k, v) in pairs(out)
                vals[k][i] = v
            end
            state
        end
        logp
    else
        logp = zeros(len)
        state = foldl(range(; length=len); init=init_state) do state, i
            v = map(v -> v[i], (; vals...))
            state, l = step(model, state; v...)
            logp[i] = l
            state
        end
        logp
    end
end

struct BirthDeathModel <: StochasticModel
    birth::Float64
    death::Float64
    dt::Float64
end

struct NonlinearModel <: StochasticModel
    rho::Float64
    n::Float64
    K::Float64
    mu::Float64
    dt::Float64
end

function initial_state(model::BirthDeathModel, generate::Bool = false; s::AbstractMatrix{Float64})
    mu = model.birth / model.death
    if generate
        for i in axes(s, 1)
            s[i, 1] = randn() * sqrt(mu)
        end
    end

    s0 = s[:, 1]
    logp_init = logpdf(Normal(0, sqrt(mu)), s0)
    return Val(generate), logp_init, s0
end

function step(model::BirthDeathModel, carry::Tuple{Val, Vector{Float64}, Vector{Float64}}; s::AbstractArray{Float64})
    generate, logp, s_prev = carry
    sigma = sqrt(2 * model.birth * model.dt)
    mu = model.birth / model.death

    @simd for i in axes(s, 1)
        @inbounds ds = model.dt * (model.birth - model.death * (s_prev[i] + mu))
        if generate === Val(true)
            @inbounds s[i] = s_prev[i] + ds + sigma * randn()
        end
        @inbounds logp[i] += logpdf(Normal(0, sigma), s[i] - (s_prev[i] + ds))
    end

    s_prev .= s
    return (generate, logp, s_prev)
end


function initial_state(::NonlinearModel, generate::Bool = false; s::AbstractMatrix{Float64}, x::AbstractMatrix{Float64})
    if generate
        x[:, 1] .= zero(eltype(x))
    end
    x_prev = x[:, 1]
    logp = zeros(max(size(x, 1), size(s, 1)))
    return Val(generate), logp, x_prev
end

# Define the step function for NonlinearModel
function step(
    model::NonlinearModel, 
    carry::Tuple{Val, Vector{Float64}, Vector{Float64}};
    s::AbstractVector{Float64},
    x::AbstractVector{Float64}
    )
    mean_x = 0.5 * model.rho / model.mu
    sigma = sqrt(model.rho * model.dt)
    generate, logp, x_prev = carry
    dim = max(size(s, 1), size(x, 1))

    function drift(s, x)
        a = logistic(model.n * log1p(s / model.K))
        a = isnan(a) ? 0.0 : a
        a * model.rho - model.mu * (x + mean_x)
    end

    for i in 1:dim
        ix = min(i, size(x, 1))
        is = min(i, size(s, 1))

        @inbounds if generate === Val(true)
            dx = model.dt * drift(s[is], x_prev[ix])
            dW = sigma * randn()
            x[ix] = x_prev[ix] + dx + dW
            logp[i] += logpdf(Normal(0, sigma), dW)
        else
            dx = model.dt * drift(s[is], x_prev[ix])
            logp[i] += logpdf(Normal(0, sigma), x_prev[ix] + dx - x[ix])
        end
    end
    
    x_prev .= x
    return (generate, logp, x_prev)
end


function log_probability(model::StochasticModel; vals...)
    state = initial_state(model, false; vals...)

    batch = max(size.(values((; vals...)), 1)...)
    length = size(pairs(vals)[1], 2)

    logp = zeros(batch, length)

    for i in range(2, length)
        v = map(v -> (@view v[:, i]), (; vals...))
        state = step(model, state; v...)
        logp[:, i] .= state[2]
    end

    logp
end

end