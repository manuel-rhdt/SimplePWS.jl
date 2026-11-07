module Models

using Random, Distributions, LinearAlgebra
using StatsFuns

export BirthDeathModel, NonlinearModel, StochasticModel, log_probability, initial_state, step

abstract type StochasticModel end

initial_state(model::StochasticModel, generate::Bool) = error("unimplemented")
copy_state(model::StochasticModel, state) = error("unimplemented")

# Step protocol used everywhere (see MutualInformation.smc_run):
#
#   initial_state(model, generate)                -> carry
#   step(model, carry)                            -> (carry, (logp, (; s = value)))
#   step(model, carry; x = observation, s = ...)  -> (carry, logp)
#   copy_state(model, carry)                      -> carry
#
# A carry is immutable and holds everything needed for the next step, so
# copy_state is the identity for the scalar models below.
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

"""
    BirthDeathModel(birth, death, dt)

Linear birth-death process in the chemical Langevin approximation. `s` is
centred: it has mean zero and stationary variance `birth / death`.
"""
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

variance_scale(model::BirthDeathModel) = model.birth / model.death

# Same convention as ARModel: the carry starts from a fixed value in both
# generate and score mode, so a generated series scores to the same log
# likelihood that generated it. The process is centred, so s0 = 0 is the
# stationary mean; only the first few steps see the transient.
function initial_state(model::BirthDeathModel, generate::Bool = false; kw...)
    return Val(generate), 0.0, 0.0
end

copy_state(::BirthDeathModel, state::Tuple) = state

function step(model::BirthDeathModel, carry::Tuple{Val, Float64, Float64}; s = nothing)
    generate, logp_init, s_prev = carry
    sigma = sqrt(2 * model.birth * model.dt)
    mu = variance_scale(model)
    # drift of the centred variable: relaxes to zero, stationary variance mu
    ds = model.dt * (model.birth - model.death * (s_prev + mu))

    if generate === Val(true)
        s_new = s_prev + ds + sigma * randn()
        logp = logpdf(Normal(0, sigma), s_new - (s_prev + ds))
        return (generate, logp_init, s_new), (logp, (; s = s_new))
    else
        s_new = s === nothing ? s_prev : Float64(s)
        logp = logpdf(Normal(0, sigma), s_new - (s_prev + ds))
        return (generate, logp_init, s_new), logp
    end
end

function initial_state(::NonlinearModel, generate::Bool = false; kw...)
    return Val(generate), 0.0
end

copy_state(::NonlinearModel, state::Tuple) = state

# saturating response; log1p throws for s/K < -1, where the response is 0
function response(model::NonlinearModel, s)
    z = s / model.K
    z <= -1 && return 0.0
    a = logistic(model.n * log1p(z))
    return isnan(a) ? 0.0 : a
end

drift(model::NonlinearModel, s, x) =
    response(model, s) * model.rho - model.mu * (x + 0.5 * model.rho / model.mu)

# Define the step function for NonlinearModel
function step(model::NonlinearModel, carry::Tuple{Val, Float64}; s, x = nothing)
    generate, x_prev = carry
    sigma = sqrt(model.rho * model.dt)
    dx = model.dt * drift(model, Float64(s), x_prev)

    if generate === Val(true)
        dW = sigma * randn()
        x_new = x_prev + dx + dW
        logp = logpdf(Normal(0, sigma), dW)
        return (generate, x_new), (logp, (; x = x_new))
    else
        x_new = x === nothing ? x_prev : Float64(x)
        logp = logpdf(Normal(0, sigma), x_new - (x_prev + dx))
        return (generate, x_new), logp
    end
end


"""
    log_probability(model; s = trajectories[, x = trajectories])

Joint log density of the given trajectories under the model, one column per
trajectory. Returns a `time x trajectories` matrix.
"""
function log_probability(model::StochasticModel; vals...)
    first_series = first(values(vals))
    ncols = size(first_series, 2)
    logp = zeros(size(first_series, 1), ncols)

    for j in 1:ncols
        v = map(v -> view(v, :, j), (; vals...))
        logp[:, j] .= (model)(false; v...)
    end

    logp
end

end
