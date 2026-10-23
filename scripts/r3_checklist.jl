# Six-item R3 acceptance recheck. Uses one fixed seed, temporary bridge files,
# and existing artifacts; no new experiment selection or R4 execution.
using TOML, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "this_adapter.jl"))
include(joinpath(@__DIR__, "r1_common.jl"))
using .THISAdapter

function check_r3()
    root = THISAdapter.ROOT
    directory = joinpath(root, "results", "r3")
    data = joinpath(root, "data", "r1", "revision")
    python = joinpath(root, ".venv-r0", "bin", "python")
    bridge = joinpath(root, "scripts", "r3_bridge.py")
    m = TOML.parsefile(joinpath(data, "metadata.toml"))
    R1Baseline.verify_data_hashes(data, m)
    X = read_matrix(joinpath(data, "states.csv"))
    Y = read_matrix(joinpath(data, "derivatives.csv"))
    rows = [split(row, ',') for row in readlines(joinpath(data, "sample_manifest.csv"))[2:end]]
    train = findall(row -> row[3] == "train", rows)
    validation = findall(row -> row[3] == "validation", rows)
    groups = [parse(Int, row[2]) for row in rows]
    Xt, Yt, Xv, Yv = X[:, train], Y[:, train], X[:, validation], Y[:, validation]
    theta, d = get_θd(Xt, 3)
    @assert size(theta) == (165, 1536) && size(Yt) == (8, 1536) && size(d) == (165, 3)
    @assert theta == read_matrix(joinpath(data, "theta.csv"))[:, train]
    @assert d == read_matrix(joinpath(data, "monomial_indices.csv"), Int)
    @assert all(x -> 0 <= x <= 8, d) && all(d[1, :] .== 0)
    println("CHECK_1_PASS theta=[165,1536]; Python design=[1536,165]; Y=[8,1536]; coeff=[8,165]; node IDs=1:8; padding=0")

    roundtrip = Dict{String,Any}()
    for (name, theta_part, Ypart) in (("train", theta, Yt), ("validation", first(get_θd(Xv, 3)), Yv))
        mktempdir() do temp
            metadata = export_request(temp, theta_part, Ypart, d; solver="lasso")
            run(addenv(`$python $bridge $temp --echo`, "PYTHONDONTWRITEBYTECODE" => "1", "OPENBLAS_NUM_THREADS" => "1"))
            returned_theta = read_matrix(joinpath(temp, "roundtrip_theta.csv"))
            returned_Y = read_matrix(joinpath(temp, "roundtrip_Y.csv"))
            returned_d = read_matrix(joinpath(temp, "roundtrip_indices.csv"), Int)
            @assert returned_theta == theta_part && returned_Y == Ypart && returned_d == d
            @assert file_hash(joinpath(temp, "roundtrip_indices.csv")) == metadata["feature_order_hash"] == m["feature_order_hash"]
            roundtrip[name] = Dict("theta_max_abs_error" => maximum(abs, returned_theta-theta_part),
                "Y_max_abs_error" => maximum(abs, returned_Y-Ypart), "indices_exact" => true,
                "feature_order_hash" => metadata["feature_order_hash"], "feature_entry_count" => metadata["feature_entry_count"])
        end
    end
    println("CHECK_2_PASS full N=8 train and validation matrices: Julia -> Python -> Julia max_abs_error=0")
    println("CHECK_4_PASS all 165 monomial rows / 495 node entries and order hashes exactly preserved")

    saved_protocol = TOML.parsefile(joinpath(directory, "protocol.toml"))
    @assert saved_protocol["seed"] == 21261016
    locked = saved_protocol["stlsq"]
    nativeA, nativeC, _ = THISAdapter.this(Xt, Yt, [2, 3], 3, locked["lambda"], locked["rho"], locked["niter"])
    @assert decode_coefficients(nativeC, d, 8, 3) == nativeA
    @assert nativeC == read_matrix(joinpath(directory, "stlsq", "coefficients.csv"))
    @assert R1Baseline.inferred_rows(nativeA) == R1Baseline.inferred_rows(R1Baseline.load_inferred(
        joinpath(directory, "stlsq", "inferred_structure.csv"), 3))
    println("CHECK_5_PASS native STLSQ coefficients and all o=1:4 Ainf matrices match adapter decoding exactly")

    replays = Dict{String,Any}()
    trace = TOML.parsefile(joinpath(directory, "feature_trace.toml"))
    feature = trace["feature_julia"]
    @assert feature == 138 && trace["feature_python"] == feature-1 && d[feature, :] == [0, 6, 5]
    @assert all(theta[feature, :] .== Xt[6, :] .* Xt[5, :])
    for solver in ("stlsq", "lasso", "adaptive_lasso")
        mktempdir() do temp
            out = if solver == "stlsq"
                run_this_solver(Xt, Yt, [2, 3], 3; solver,
                    lambda=locked["lambda"], rho=locked["rho"], niter=locked["niter"])
            else
                run_this_solver(Xt, Yt, [2, 3], 3; solver, config=Dict("selection" => "validation"),
                    validation_X=Xv, validation_Y=Yv, train_groups=groups[train],
                    validation_groups=groups[validation], work_dir=temp)
            end
            saved_C = read_matrix(joinpath(directory, solver, "coefficients.csv"))
            @assert out.d == d && out.theta == theta && size(out.coeff) == (8, 165)
            @assert isapprox(out.coeff, saved_C; atol=1e-12, rtol=1e-12)
            saved_edges = R1Baseline.inferred_rows(R1Baseline.load_inferred(joinpath(directory, solver, "inferred_structure.csv"), 3))
            actual_edges = R1Baseline.inferred_rows(out.Ainf)
            @assert length(actual_edges) == length(saved_edges)
            @assert all(a[1:5] == b[1:5] && isapprox(a[6], b[6]; atol=1e-12, rtol=1e-12)
                        for (a, b) in zip(actual_edges, saved_edges))
            triple = only([collect(row) for row in eachrow(out.Ainf[3]) if row[1:3] == [7, 5, 6]])
            @assert triple[end] == out.coeff[7, feature]
            @assert isapprox(triple[end], trace[solver]["coefficient"]; atol=1e-12, rtol=1e-12)
            replays[solver] = Dict("coefficient_max_abs_difference" => maximum(abs, out.coeff-saved_C),
                "coefficient_replay_tolerance" => 1e-12, "theta_and_d_exact" => true,
                "all_order_edges_equal_within_tolerance" => true,
                "example_coefficient" => triple[end], "example_decoded_row" => triple)
            println("REPLAY solver=", solver, " coefficient_max_abs_difference=", maximum(abs, out.coeff-saved_C))
            println("FEATURE_TO_EDGE solver=", solver, " Julia_feature=138 Python_feature=137 d=[0,6,5] target=7 coefficient=",
                    out.coeff[7, feature], " Ainf[3]_row=", triple)
        end
    end
    println("CHECK_3_PASS same fixed R1 train/validation inputs; all three solvers reproduced saved coefficients and decoded edges")
    repeated = trace["repeated_feature_julia"]
    @assert d[repeated, :] == [0, 5, 5]
    test_coeff = zeros(8, size(d, 1));test_coeff[7, repeated] = 1.
    @assert isempty(decode_coefficients(test_coeff, d, 8, 3)[3])
    println("CHECK_6_PASS printed x5*x6 -> coefficient -> 7<-{5,6}; x5^2 does not become a two-source edge")
    R1Baseline.verify_data_hashes(data, m)
    result = Dict("client_review_date" => "2026-10-17", "all_six_checks_passed" => true,
        "seed" => 21261016, "roundtrip" => roundtrip, "replays" => replays,
        "test_samples_evaluated" => false, "r4_executed" => false)
    path = joinpath(directory, "review_20261017", "checklist_results.toml")
    ispath(path) && error("Review output already exists; preserve previous evidence")
    R1Baseline.save_toml(path, result)
    println("R3_SIX_ITEM_RECHECK_PASS")
end

check_r3()
