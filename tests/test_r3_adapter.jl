using Test, TOML, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "this_adapter.jl"))
using .THISAdapter
include(joinpath(@__DIR__, "..", "scripts", "r1_common.jl"))

const PYTHON = joinpath(THISAdapter.ROOT, ".venv-r0", "bin", "python")
const BRIDGE = joinpath(THISAdapter.ROOT, "scripts", "r3_bridge.py")

@testset "R3 native decoding, including threshold and exclusions" begin
    X = randn(MersenneTwister(31), 4, 120)
    Y = [-0.8 .* X[1, :]'; (0.2 .* X[1, :] .- 0.9 .* X[2, :])';
         (0.08 .* X[1, :] .* X[2, :] .- X[3, :])'; (-1.1 .* X[4, :])']
    for (lambda, rho) in ((0.03, 0.), (.1, 1.))
        nativeA, nativeC, nativeR = THISAdapter.this(X, Y, [2, 3], 3, lambda, rho, 10)
        wrapped = run_this_solver(X, Y, [2, 3], 3; lambda, rho)
        @test wrapped.coeff == nativeC
        @test wrapped.Ainf == nativeA
        @test wrapped.relerr == nativeR
    end
    _, d = get_θd(X, 3)
    C = zeros(4, size(d, 1))
    index(s) = R1Baseline.feature_index(d, s)
    C[3, index([1, 2])] = .2
    C[3, index([1, 1])] = 5. # repeated source: exclude
    C[3, index([1, 3])] = 6. # target among sources: exclude
    C[3, index([1])] = 1e-8 # strict threshold: exclude equality
    C[4, index([1])] = nextfloat(1e-8)
    C[1, 1] = .3 # native o=1 return is retained
    C[4, index([1, 2, 3])] = .4 # native o=4 return is retained
    A = decode_coefficients(C, d, 4, 3)
    @test A[3] == [3. 1. 2. .2]
    @test A[2] == [4. 1. nextfloat(1e-8)]
    @test A[1] == [1. .3]
    @test A[4] == [4. 1. 2. 3. .4]
    @test_throws ErrorException decode_coefficients(C[:, 2:end], d, 4, 3)
end

@testset "R3 cross-language CSV precision and singleton axes" begin
    for (n, p, k) in ((1, 1, 1), (1, 2, 3), (3, 1, 1))
        mktempdir() do temp
            theta = fill(.12345678901234567, p, k);theta[1, :] .= 1.
            Y = reshape(collect(1.:n*k), n, k) ./ 7
            d = zeros(Int, p, 1);p > 1 && (d[2, 1] = 1)
            meta = export_request(temp, theta, Y, d; solver="lasso", config=Dict("selection" => "fixed", "alpha" => .01))
            run(addenv(`$PYTHON $BRIDGE $temp --echo`, "PYTHONDONTWRITEBYTECODE" => "1", "OPENBLAS_NUM_THREADS" => "1"))
            @test read_matrix(joinpath(temp, "roundtrip_theta.csv")) == theta
            @test read_matrix(joinpath(temp, "roundtrip_Y.csv")) == Y
            @test read_matrix(joinpath(temp, "roundtrip_indices.csv"), Int) == d
            @test file_hash(joinpath(temp, "roundtrip_indices.csv")) == meta["feature_order_hash"]
            @test TOML.parsefile(joinpath(temp, "response.toml"))["feature_entry_count"] == p
            @test_throws ErrorException export_request(temp, theta, Y, d)
        end
    end
end

@testset "R3 Python coefficients return to the correct Julia feature" begin
    X = randn(MersenneTwister(731), 3, 160)
    theta, d = get_θd(X, 3)
    truth = R1Baseline.true_coefficients(R1Baseline.make_model(3), d)
    Y = truth*theta
    for solver in ("lasso", "adaptive_lasso")
        mktempdir() do temp
            out = run_this_solver(X, Y, [2, 3], 3; solver, work_dir=temp,
                config=Dict("selection" => "fixed", "alpha" => 1e-8))
            @test size(out.coeff) == size(truth)
            @test isapprox(out.coeff, truth; atol=5e-6, rtol=0)
            @test out.d == d && out.theta == theta
            j = R1Baseline.feature_index(d, [1, 2])
            @test out.coeff[3, j] ≈ .08 atol=5e-6
            @test any(row -> row[1:3] == [3, 1, 2], eachrow(out.Ainf[3]))
            request_hash = file_hash(joinpath(temp, "metadata.toml"))
            C, diag = read_response(temp; expected_request_hash=request_hash)
            @test C == out.coeff
            @test length(diag["nodes"]) == 3
            response_path = joinpath(temp, "response.toml")
            response = TOML.parsefile(response_path)
            response["feature_order_hash"] = repeat("0", 64)
            open(io -> TOML.print(io, response), response_path, "w")
            @test_throws ErrorException read_response(temp; expected_request_hash=request_hash)
            response["feature_order_hash"] = file_hash(joinpath(temp, "monomial_indices.csv"))
            response["coefficient_shape"] = [3, size(truth, 2)+1]
            open(io -> TOML.print(io, response), response_path, "w")
            @test_throws ErrorException read_response(temp; expected_request_hash=request_hash)
            response["coefficient_shape"] = collect(size(truth))
            open(io -> TOML.print(io, response), response_path, "w")
            open(io -> println(io, "0"), joinpath(temp, "coefficients.csv"), "a")
            @test_throws ErrorException read_response(temp; expected_request_hash=request_hash)
        end
    end
end

@testset "R3 R1 truth mapping and native coefficient parity" begin
    dir = joinpath(THISAdapter.ROOT, "data", "r1", "revision")
    X = read_matrix(joinpath(dir, "states.csv"))
    Y = read_matrix(joinpath(dir, "derivatives.csv"))
    manifest = readlines(joinpath(dir, "sample_manifest.csv"))[2:end]
    train = findall(line -> split(line, ',')[3] == "train", manifest)
    theta, d = get_θd(X, 3)
    @test theta == read_matrix(joinpath(dir, "theta.csv"))
    @test d == read_matrix(joinpath(dir, "monomial_indices.csv"), Int)
    truth = read_matrix(joinpath(dir, "true_coefficients.csv"))
    @test R1Baseline.true_coefficients(R1Baseline.make_model(8), d) == truth
    @test isapprox(truth*theta, Y; atol=1e-15, rtol=1e-14)
    saved = read_matrix(joinpath(THISAdapter.ROOT, "results", "r1", "revision", "selected", "coefficients.csv"))
    nativeA, C, _ = THISAdapter.this(X[:, train], Y[:, train], [2, 3], 3, .03, 0., 10)
    @test C == saved
    @test decode_coefficients(C, d, 8, 3) == nativeA
    for node in 1:8
        @test isapprox(C[node, :], truth[node, :]; atol=1e-12, rtol=1e-12)
    end
end

println("R3_ADAPTER_TESTS_PASS")
