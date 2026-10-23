# R0 only: fixed array fixtures; no trajectories or structure-recovery experiment.
using Test, LinearAlgebra, Combinatorics, DelimitedFiles, EDF, FFTW, Statistics
include(joinpath(@__DIR__, "..", "upstream", "THIS", "this.jl"))

@test VERSION.major == 1 && VERSION.minor == 9
println("Julia=", VERSION)
X = [1.0 2.0 3.0 4.0 5.0;
     2.0 1.0 0.5 1.5 2.5;
     0.5 1.5 2.5 1.0 2.0]
@testset "THIS basic features and calls" begin
    for degree in (2, 3)
        θ, d = get_θd(X, degree)
        P = binomial(size(X, 1) + degree, degree)
        @test size(θ) == (P, size(X, 2))
        @test size(d) == (P, degree)
        @test d[1, :] == zeros(Int, degree)
        @test θ[1, :] == ones(size(X, 2))
        for row in axes(d, 1)
            nodes = d[row, d[row, :] .> 0]
            expected = isempty(nodes) ? ones(size(X, 2)) : vec(prod(X[nodes, :]; dims=1))
            @test θ[row, :] ≈ expected
        end
        println("dmax=", degree, " X=", size(X), " theta=", size(θ), " d=", size(d))
        if degree == 2
            println("Actual ordered d (each row aligns to theta):")
            show(stdout, "text/plain", d)
            println()
            # Dummy response for interface checks; not a physical derivative.
            Y = copy(X)
            coeff, err = mySINDy(θ, Y, 0.1, 1.0, 3)
            @test size(coeff) == (3, P)
            @test all(isfinite, coeff) && isfinite(err)
            Ainf, coeff2, relerr = this(X, Y, [2, 3], degree, 0.1, 1.0, 3)
            @test size(coeff2) == (3, P)
            @test isfinite(relerr)
            @test sort(collect(keys(Ainf))) == collect(1:degree+1)
            println("mySINDy and this basic calls passed; no recovery metrics computed.")
        end
    end
end
println("R0_JULIA_SMOKE_PASS")
