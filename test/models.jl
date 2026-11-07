using SimplePWS
using Test

# Fast settings: this file has to run in CI on every push.
const STEPS = 100
const DT = 0.1
const MC = 10
const PARTICLES = 16

@testset "models" begin
    input = BirthDeathModel(10.0, 1.0, DT)
    output = NonlinearModel(20.0, 10.0, 10.0, 1.0, DT)

    # Generating a trajectory and then scoring it must give the same log density,
    # otherwise the conditional density and the sampled data disagree.
    s = zeros(STEPS)
    x = zeros(STEPS)
    @test input(true, s = s) ≈ input(false, s = s)
    @test output(true, s = s, x = x) ≈ output(false, s = s, x = x)

    # The generated signals are finite (this is what the log1p guard protects)
    @test all(isfinite, s)
    @test all(isfinite, x)
    @test extrema(s) != (0.0, 0.0)

    # The estimator runs, returns one value per time step, and they are finite
    result = mutual_information(input, output; num_steps = STEPS, dt = DT,
                                mc_samples = MC, n_particles = PARTICLES)
    mi = result[:MutualInformation]
    @test length(mi) == STEPS
    @test all(isfinite, mi)

    # Without coupling (logistic(0) = 1/2 for every s) x carries no information
    # about s, so the estimate must come out at zero.
    decoupled = NonlinearModel(20.0, 0.0, 10.0, 1.0, DT)
    zero_mi = mutual_information(input, decoupled; num_steps = STEPS, dt = DT,
                                 mc_samples = MC, n_particles = PARTICLES)[:MutualInformation]
    @test abs(zero_mi[end]) < 1.0

    # log_probability is defined for a batch of trajectories
    batch = zeros(STEPS, 3)
    for j in 1:3
        input(true, s = view(batch, :, j))
    end
    logp = SimplePWS.Models.log_probability(input; s = batch)
    @test size(logp) == (STEPS, 3)
    @test all(isfinite, logp)
end
