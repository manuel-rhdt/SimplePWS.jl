module armodel

using Distributions
using Random, Polynomials

using ..Models

import ..Models: step, initial_state, copy_state
    
struct ARModel <: StochasticModel
    coefficients::Vector{Float64}  # Coefficients of the AR(n) process
    sigma::Float64  # Standard deviation of noise
end

function initial_state(model::ARModel, generate::Bool = false)
    n = length(model.coefficients)
    return Val(generate), zeros(Float64, n)
end

function copy_state(::ARModel, state::Tuple{Val, Vector{Float64}})
    return (state[1], copy(state[2]))
end

function step(model::ARModel, carry::Tuple{Val, Vector{Float64}}; s::Union{Nothing, Float64} = nothing)
    generate, s_prev = carry
    n = length(model.coefficients)
    noise_dist = Normal(0, model.sigma)
    
    pred = sum(model.coefficients[j] * s_prev[n - j + 1] for j in 1:n)
    noise = rand(noise_dist)

    if generate === Val(true)
        s = pred + noise
    end

    logp = logpdf(noise_dist, s - pred)
    
    for i in 1:n-1
        s_prev[i] = s_prev[i + 1]
    end
    s_prev[n] = s

    if generate === Val(true)
        return (generate, s_prev), (logp, (; s))
    else
        return (generate, s_prev), logp
    end
end

const min_magnitude = 1.1
const max_magnitude = 2.0

function generate_stable_ar_coefficients(n, seed=0)
    """
    Generates stable AR(n) coefficients.

    Parameters:
        n (Int): Order of the AR process.
        seed (Int, optional): Random seed for reproducibility.

    Returns:
        Vector{Float64}: Stable AR coefficients of length n.
    """
    rng = Random.MersenneTwister(seed)

    # Generate random roots outside the unit circle
    roots = Complex{Float64}[]
    
    while length(roots) < n
        remaining = n - length(roots)
        add_complex = remaining >= 2 && rand(rng) > 0.5

        if add_complex
            # Generate a complex conjugate pair
            magnitude = rand(rng, Uniform(min_magnitude, max_magnitude))
            angle = rand(rng, Uniform(0, 2π))
            root = magnitude * exp(im * angle)
            push!(roots, root, conj(root))
        else
            # Generate a real root
            magnitude = rand(rng, Uniform(min_magnitude, max_magnitude))
            sign = rand(rng, (-1, 1))
            root = sign * magnitude
            push!(roots, root)
        end
    end

    # Convert roots to polynomial coefficients
    p = fromroots(roots)
    coefficients = coeffs(p) |> real

    # Normalize to make the leading coefficient 1
    coefficients /= coefficients[1]

    # AR coefficients are the negatives of the remaining coefficients
    return -coefficients[2:end]
end


end