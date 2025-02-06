module armodel

using Distributions
using Random, Polynomials

using ..Models

import ..Models: step, initial_state
    
struct ARModel <: StochasticModel
    coefficients::Vector{Float64}  # Coefficients of the AR(n) process
    sigma::Float64  # Standard deviation of noise
end

function initial_state(model::ARModel, generate::Bool = false; s::AbstractMatrix{Float64})
    n = length(model.coefficients)
    if generate
        for i in axes(s, 1)
            s[i, 1] = randn() * model.sigma
        end
    end
    
    s0 = zeros(size(s, 1), n)
    logp_init = logpdf(Normal(0, model.sigma), s[:, 1])
    return Val(generate), logp_init, s0
end

function step(model::ARModel, carry::Tuple{Val, Vector{Float64}, Matrix{Float64}}; s::AbstractVector{Float64})
    generate, logp, s_prev = carry
    n = length(model.coefficients)
    noise_dist = Normal(0, model.sigma)
    
    for i in axes(s, 1)
        pred = sum(model.coefficients[j] * s_prev[i, j] for j in 1:n)
        noise = rand(noise_dist)
        
        @inbounds if generate === Val(true)
            s[i] = pred + noise
        end
        
        @inbounds logp[i] += logpdf(noise_dist, s[i] - pred)
    end
    
    s_prev[:, 1:n-1] .= s_prev[:, 2:n]
    s_prev[:, n] .= s
    return (generate, logp, s_prev)
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
    return -coefficients[2:end] |> reverse
end


end