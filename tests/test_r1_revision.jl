using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "scripts", "r1_revision_common.jl"))
using .R1Revision
using .R1Revision.B

@testset "Revised protocol and training-only quality gate" begin
    c = revision_config()
    @test (c.train_trajectories, c.validation_trajectories, c.test_trajectories) == (512, 128, 128)
    @test c.seed != default_config().seed
    @test c.dmax == 3 && c.niter == 10
    rng = MersenneTwister(17)
    X = 2 .* rand(rng, 3, 128) .- 1
    theta, _ = B.get_θd(X, 3)
    @test require_quality(quality_metrics(theta), c)
    bad_theta, _ = B.get_θd(ones(3, 128), 3)
    @test_throws ErrorException require_quality(quality_metrics(bad_theta), c)
end

@testset "Validation selection and honest numerical ties" begin
    candidates = [
        (lambda=0.001, rho=0.0, validation_normalized_error=1e-20, nonzero_coefficients=10),
        (lambda=0.03, rho=0.0, validation_normalized_error=2e-20, nonzero_coefficients=10),
        (lambda=0.1, rho=0.0, validation_normalized_error=0.1, nonzero_coefficients=5),
        (lambda=0.03, rho=1e-8, validation_normalized_error=1e-19, nonzero_coefficients=10)]
    selected = choose_candidate(candidates, 1e-12)
    @test selected.lambda == 0.03 && selected.rho == 0.0
    @test choose_candidate(reverse(candidates), 1e-12) == selected
    @test selected.nonzero_coefficients == 10 # More sparse but inaccurate model must lose.
    @test_throws ErrorException choose_candidate([(lambda=0.1, rho=0.0, validation_normalized_error=NaN, nonzero_coefficients=1)], 1e-12)
end

@testset "New independent short trajectories" begin
    c = merge(revision_config(), (nodes=3, train_trajectories=8, validation_trajectories=2, test_trajectories=2))
    data = generate_dataset(c)
    @test size(data.X) == (3, 36)
    @test sort(unique(data.times)) == [0.0, 0.05, 0.1]
    for id in unique(data.trajectory_ids)
        @test length(unique(data.splits[findall(==(id), data.trajectory_ids)])) == 1
    end
    @test data.maximum_state <= 1 + 1e-10
end

@testset "Original STLSQ selected using validation response only" begin
    rng = MersenneTwister(41)
    X_train, X_val = 2 .* rand(rng, 3, 128) .- 1, 2 .* rand(rng, 3, 64) .- 1
    model = make_model(3)
    Y_train = hcat([rhs(x, model) for x in eachcol(X_train)]...)
    Y_val = hcat([rhs(x, model) for x in eachcol(X_val)]...)
    c = merge(revision_config(), (nodes=3, lambda_grid=[0.001, 0.1], rho_grid=[0.0, 1.0]))
    rows, selected = fit_candidates(X_train, Y_train, X_val, Y_val, c)
    @test length(rows) == 4
    @test selected.lambda == 0.001 && selected.rho == 0.0
    @test selected.validation_normalized_error < 1e-20
end

println("R1_REVISION_TESTS_PASS")
