using Test, LinearAlgebra, Random, TOML
include(joinpath(@__DIR__, "..", "scripts", "r1_common.jl"))
using .R1Baseline

@testset "R1 prerequisites" begin
    @test check_prerequisites()
    c = default_config()
    @test (c.nodes, c.dmax, c.lambda, c.rho, c.niter) == (8, 3, 0.1, 1.0, 10)
end

@testset "Hand-calculated 3-node Taylor library" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    theta, d = R1Baseline.get_θd(X, 2)
    @test size(theta) == (10, 2)
    @test d == [0 0; 0 1; 1 1; 2 1; 3 1; 0 2; 2 2; 3 2; 0 3; 3 3]
    @test theta == [1.0 1.0; 1.0 2.0; 1.0 4.0; 3.0 8.0; 5.0 12.0;
                   3.0 4.0; 9.0 16.0; 15.0 24.0; 5.0 6.0; 25.0 36.0]
    theta3, d3 = R1Baseline.get_θd(X, 3)
    @test size(theta3) == (20, 2)
    @test size(d3) == (20, 3)
    for row in axes(d3, 1)
        nodes = filter(!=(0), d3[row, :])
        @test theta3[row, :] == (isempty(nodes) ? ones(2) : vec(prod(X[nodes, :]; dims=1)))
    end
end

@testset "Known polynomial coefficients and original THIS decoding" begin
    # Complete small Cartesian grid: quadratic library has full row rank.
    points = collect(Iterators.product([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]))
    X = hcat([collect(point) for point in vec(points)]...)
    theta, d = R1Baseline.get_θd(X, 2)
    @test rank(theta) == 10
    C = zeros(3, 10)
    for (node, sources, value) in [(1, Int[], 0.1), (1, [1], -0.8), (1, [2], 0.4),
            (1, [2, 3], 0.3), (1, [2, 2], 0.2), (1, [1, 2], 0.15), (2, [2], -0.7), (3, [3], -0.9)]
        C[node, feature_index(d, sources)] = value
    end
    Y = C * theta
    # Numerical mapping fixture only: rho=0 and tiny threshold; not baseline tuning.
    Ainf, fitted, relerr = R1Baseline.this(X, Y, [2, 3], 2, 1e-8, 0.0, 10)
    @test size(fitted) == (3, 10)
    @test fitted ≈ C atol=1e-12 rtol=1e-12
    @test relerr < 1e-12
    @test inferred_edges(Ainf, 2) == Set([(1, 2)])
    @test inferred_edges(Ainf, 3) == Set([(1, 2, 3)])
    @test size(Ainf[1], 1) == 1
    @test only(Ainf[3][:, end]) ≈ 0.3 atol=1e-12
    @test fitted[1, feature_index(d, [2, 2])] ≈ 0.2 atol=1e-12
    @test fitted[1, feature_index(d, [1, 2])] ≈ 0.15 atol=1e-12
end

@testset "Stable trajectories, analytic derivatives and independent splits" begin
    for n in (8, 12)
        model = make_model(n)
        @test minimum(cube_margins(model)) > 0
        @test all(all(source < edge.target for source in edge.sources) for edge in model.interactions)
    end
    c = merge(default_config(; nodes=3), (train_trajectories=3, validation_trajectories=1,
        test_trajectories=1, integration_steps=20))
    model = make_model(3)
    data, again = generate_dataset(c, model), generate_dataset(c, model)
    @test data.X == again.X && data.Y == again.Y
    @test size(data.X) == (3, 25)
    @test data.maximum_state <= 1 + 1e-10
    theta, d = R1Baseline.get_θd(data.X, c.dmax)
    @test data.Y ≈ true_coefficients(model, d) * theta atol=1e-12
    for trajectory in 1:5
        columns = findall(==(trajectory), data.trajectory_ids)
        @test length(unique(data.splits[columns])) == 1
        @test data.X[1, columns] ≈ data.initials[1, trajectory] .* exp.(-model.decay[1] .* data.times[columns]) atol=1e-9
    end
    ids = Dict(split => Set(data.trajectory_ids[findall(==(split), data.splits)]) for split in ("train", "validation", "test"))
    @test isempty(intersect(ids["train"], ids["test"]))
    @test isempty(intersect(ids["train"], ids["validation"]))
    @test isempty(intersect(ids["validation"], ids["test"]))
end

@testset "Edge metrics and serialization" begin
    universe = edge_universe(3, 2)
    @test length(universe) == 6
    @test length(edge_universe(8, 3)) == 168
    m = confusion_metrics(Set([(1, 2), (2, 3)]), Set([(1, 2), (1, 3)]), universe)
    @test (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 3)
    @test m["precision"] == m["recall"] == m["F1"] == 0.5
    @test m["MCC"] == 0.25
    empty = confusion_metrics(Set([(1, 2)]), Set{Tuple}(), universe)
    @test empty["precision"] == 0 && empty["recall"] == 0
    @test isnan(empty["MCC"])
    mktempdir() do path
        A = [pi nextfloat(1.0); -1e-18 1e16]
        write_matrix(joinpath(path, "float.csv"), A)
        @test read_matrix(joinpath(path, "float.csv")) == A
        d = [0 0; 3 2]
        write_matrix(joinpath(path, "d.csv"), d)
        @test read_matrix(joinpath(path, "d.csv"), Int64) == d
        ainf = Dict(1 => [1.0 0.1], 2 => zeros(0, 3), 3 => [1.0 2.0 3.0 -0.2])
        write_table(joinpath(path, "edges.csv"), ["order", "target", "s1", "s2", "s3", "coefficient"], inferred_rows(ainf))
        @test inferred_rows(load_inferred(joinpath(path, "edges.csv"), 2)) == inferred_rows(ainf)
    end
end

println("R1_UNIT_TESTS_PASS")
