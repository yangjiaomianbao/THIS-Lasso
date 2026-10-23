# R3: one fixed R1 seed; interface checks only, no performance grid.
include(joinpath(@__DIR__, "..", "src", "this_adapter.jl"))
include("r1_common.jl")
using .THISAdapter
using TOML, LinearAlgebra

root = THISAdapter.ROOT
data_dir = joinpath(root, "data", "r1", "revision")
result_dir = joinpath(root, "results", "r3")
protocol_path = joinpath(result_dir, "protocol.toml")
ispath(protocol_path) && error("R3 demo already run; preserve results and use a separate run directory")
metadata = TOML.parsefile(joinpath(data_dir, "metadata.toml"))
R1Baseline.verify_data_hashes(data_dir, metadata)
X = THISAdapter.read_matrix(joinpath(data_dir, "states.csv"))
Y = THISAdapter.read_matrix(joinpath(data_dir, "derivatives.csv"))
rows = [split(line, ',') for line in readlines(joinpath(data_dir, "sample_manifest.csv"))[2:end]]
train = findall(row -> row[3] == "train", rows)
validation = findall(row -> row[3] == "validation", rows)
groups = [parse(Int, row[2]) for row in rows]
isempty(intersect(groups[train], groups[validation])) || error("Trajectory split overlap")
theta, d = get_θd(X, 3)
theta == THISAdapter.read_matrix(joinpath(data_dir, "theta.csv")) || error("Feature values changed")
d == THISAdapter.read_matrix(joinpath(data_dir, "monomial_indices.csv"), Int) || error("Feature order changed")
locked = TOML.parsefile(joinpath(root, "results", "r1", "revision", "selection_locked.toml"))
protocol = Dict("stage" => "R3 single-seed interface check only", "seed" => metadata["config"]["seed"],
    "data_metadata_sha256" => THISAdapter.file_hash(joinpath(data_dir, "metadata.toml")),
    "feature_order_hash" => metadata["feature_order_hash"], "feature_count" => size(theta, 1),
    "N" => size(X, 1), "train_K" => length(train), "validation_K" => length(validation),
    "train_trajectories" => length(unique(groups[train])), "validation_trajectories" => length(unique(groups[validation])),
    "evaluation_orders" => [2, 3], "returned_orders" => [1, 2, 3, 4], "decode_abs_threshold_strict" => 1e-8,
    "test_evaluated" => false, "structural_truth_used_for_parameter_selection" => false,
    "stlsq" => Dict("lambda" => locked["lambda"], "rho" => locked["rho"], "niter" => locked["niter"],
        "source" => "R1 previously locked validation choice; no R3 retuning"),
    "lasso" => Dict("selection" => "validation", "objective_units" => "original feature units", "intercept" => "unpenalized constant column"),
    "adaptive_lasso" => Dict("selection" => "validation", "initialization" => "ridge", "ridge_rho" => 1., "gamma" => 1., "eps" => 1e-6))
R1Baseline.save_toml(protocol_path, protocol)
println("R3_PROTOCOL_SAVED_BEFORE_FITTING: seed=", protocol["seed"])
outputs = Dict{String,Any}()
for solver in ("stlsq", "lasso", "adaptive_lasso")
    dir = joinpath(result_dir, solver)
    output = if solver == "stlsq"
        run_this_solver(X[:, train], Y[:, train], [2, 3], 3; solver,
            lambda=locked["lambda"], rho=locked["rho"], niter=locked["niter"])
    else
        run_this_solver(X[:, train], Y[:, train], [2, 3], 3; solver,
            config=Dict("selection" => "validation"), work_dir=dir,
            validation_X=X[:, validation], validation_Y=Y[:, validation],
            train_groups=groups[train], validation_groups=groups[validation])
    end
    outputs[solver] = output
    mkpath(dir)
    solver == "stlsq" && THISAdapter.write_matrix(joinpath(dir, "coefficients.csv"), output.coeff)
    coefficient_rows = Tuple[]
    println("\nSOLVER=", solver, " NONZERO_TAYLOR_COEFFICIENTS abs>1e-8 (original decoding threshold)")
    println("target,feature_julia,feature_python,d_1,d_2,d_3,coefficient,eligible_interaction")
    for node in axes(output.coeff, 1), feature in axes(output.coeff, 2)
        value = output.coeff[node, feature]
        if abs(value) > 1e-8
            sources = filter(!=(0), d[feature, :])
            eligible = length(sources) == length(unique(sources)) && !(node in sources)
            row = (node, feature, feature-1, d[feature, :]..., value, eligible)
            push!(coefficient_rows, row)
            println(join(row, ','))
        end
    end
    R1Baseline.write_table(joinpath(dir, "nonzero_taylor.csv"),
        ["target", "feature_julia", "feature_python", "d_1", "d_2", "d_3", "coefficient", "eligible_interaction"], coefficient_rows)
    edge_rows = R1Baseline.inferred_rows(output.Ainf)
    R1Baseline.write_table(joinpath(dir, "inferred_structure.csv"),
        ["order", "target", "source_1", "source_2", "source_3", "coefficient"], edge_rows)
    println("DECODED_2_3_INTERACTIONS order,target,source_1,source_2,source_3,coefficient")
    for row in edge_rows
        row[1] in (2, 3) && println(join(row, ','))
    end
end

# All solver choices are finished before loading structure truth for evaluation.
truth = THISAdapter.read_matrix(joinpath(data_dir, "true_coefficients.csv"))
model = R1Baseline.make_model(size(X, 1))
R1Baseline.true_coefficients(model, d) == truth || error("R1 truth correspondence changed")
summary = Dict{String,Any}()
for solver in ("stlsq", "lasso", "adaptive_lasso")
    out = outputs[solver]
    metrics = R1Baseline.evaluate_structure(model, out.Ainf, size(X, 1))
    metrics["train_original_this_relerr"] = out.relerr
    metrics["train_MSE"] = sum(abs2, Y[:, train]-out.coeff*theta[:, train])/length(Y[:, train])
    metrics["validation_MSE"] = sum(abs2, Y[:, validation]-out.coeff*theta[:, validation])/length(Y[:, validation])
    metrics["ignored_o1_returned_rows"] = size(out.Ainf[1], 1)
    metrics["ignored_o4_returned_rows"] = size(out.Ainf[4], 1)
    metrics["interpretation"] = "Interface check only; no general performance conclusion"
    R1Baseline.save_toml(joinpath(result_dir, solver, "metrics.toml"), metrics)
    summary[solver] = metrics
    println("INTERFACE_METRICS solver=", solver, " pairwise=", metrics["pairwise"], " triple=", metrics["triple"])
end
R1Baseline.save_toml(joinpath(result_dir, "summary.toml"), summary)
feature = R1Baseline.feature_index(d, [5, 6])
repeated = R1Baseline.feature_index(d, [5, 5])
trace = Dict{String,Any}("target" => 7, "sources" => [5, 6], "feature_julia" => feature,
    "feature_python" => feature-1, "d_row" => d[feature, :], "true_coefficient" => truth[7, feature],
    "repeated_feature_julia" => repeated, "repeated_d_row" => d[repeated, :],
    "repeated_source_is_not_three_body" => true)
for solver in ("stlsq", "lasso", "adaptive_lasso")
    out = outputs[solver]
    matches = [collect(row) for row in eachrow(out.Ainf[3]) if row[1:3] == [7, 5, 6]]
    length(matches) == 1 || error("Example triple not uniquely mapped for $solver")
    trace[solver] = Dict("coefficient" => out.coeff[7, feature], "decoded_Ainf_row" => only(matches))
end
R1Baseline.save_toml(joinpath(result_dir, "feature_trace.toml"), trace)
println("FEATURE_TRACE=", trace)
R1Baseline.verify_data_hashes(data_dir, metadata)
println("R3_DEMO_PASS; NO_TEST_EVALUATION; NO_R4_GRID; UPSTREAM_AND_R2_READ_ONLY")
