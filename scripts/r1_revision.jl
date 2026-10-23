include("r1_revision_common.jl")
using .R1Revision
using .R1Revision.B
using TOML, Dates, LinearAlgebra

check_prerequisites()
config = revision_config()
data_dir = joinpath(ROOT, "data", "r1", "revision")
result_dir = joinpath(ROOT, "results", "r1", "revision")
ispath(data_dir) && error("Revised data already exists; never overwrite a locked experiment")
isfile(joinpath(result_dir, "protocol.toml")) && error("Revised protocol already exists")
mkpath(data_dir)
mkpath(result_dir)
protocol = Dict{String,Any}("created_at" => string(now(UTC) + Hour(8), "+08:00"),
    "config" => Dict{String,Any}(string(k) => v for (k, v) in pairs(config)),
    "this_commit" => THIS_COMMIT,
    "model" => "Exactly the original R1 model, coefficients and complete dmax=3 feature library",
    "data_design" => "512/128/128 independent short trajectories, t=0,0.05,0.10; fixed new seed namespace",
    "quality_rule" => "Use training X only; require full rank and raw/unit-RMS conditions <=1000/300; failure stops, no seed search",
    "normalization" => "none in fitting; unit-RMS condition is diagnostic only",
    "selection_rule" => "Minimum validation normalized squared derivative error; within absolute 1e-12 prefer fewer coefficients, then larger lambda, then smaller rho",
    "selection_inputs" => "Training and validation X/Y only; no structure labels or test samples",
    "test_rule" => "Write selection_locked.toml before evaluating any revised test metrics; no selection after test disclosure",
    "acceptance_rule" => "On the clean held-out fixture, both orders Precision/Recall>=0.95 and normalized test error<=1e-8; report failure honestly if unmet",
    "scope" => "R1 revision authorized by user; original THIS unchanged, no R2-R5")
save_toml(joinpath(result_dir, "protocol.toml"), protocol)

model = make_model(config.nodes)
data = generate_dataset(config, model)
theta, d = B.get_θd(data.X, config.dmax)
true_C = true_coefficients(model, d)
@assert true_C == read_matrix(joinpath(ROOT, "data", "r1", "true_coefficients.csv"))
@assert d == read_matrix(joinpath(ROOT, "data", "r1", "monomial_indices.csv"), Int64)
@assert data.maximum_state <= 1 + 1e-10
@assert isapprox(data.Y, true_C * theta; atol=1e-12, rtol=1e-12)
columns = Dict(split => findall(==(split), data.splits) for split in ("train", "validation", "test"))
quality = quality_metrics(theta[:, columns["train"]])
save_toml(joinpath(result_dir, "training_quality.toml"), quality)
require_quality(quality, config)
for (name, matrix) in (("states", data.X), ("derivatives", data.Y), ("theta", theta),
        ("monomial_indices", d), ("true_coefficients", true_C), ("initial_conditions", data.initials))
    write_matrix(joinpath(data_dir, name * ".csv"), matrix)
end
write_table(joinpath(data_dir, "sample_manifest.csv"), ["sample_id", "trajectory_id", "split", "time"],
    ((k, data.trajectory_ids[k], data.splits[k], data.times[k]) for k in axes(data.X, 2)))
write_table(joinpath(data_dir, "true_structure.csv"), ["order", "target", "source_1", "source_2", "coefficient"],
    ((length(e.sources) + 1, e.target, e.sources[1], length(e.sources) == 2 ? e.sources[2] : 0, e.coefficient) for e in model.interactions))
files = sort(filter(name -> endswith(name, ".csv"), readdir(data_dir)))
metadata = Dict("config" => protocol["config"], "this_commit" => THIS_COMMIT,
    "state_shape" => collect(size(data.X)), "theta_shape" => collect(size(theta)), "d_shape" => collect(size(d)),
    "index_base" => 1, "padding_node_id" => 0, "float_format" => "%.17g Float64; numeric matrices have no headers",
    "trajectory_seeds" => config.seed .+ (1:size(data.initials, 2)),
    "split_rule" => "trajectory 1:512 train, 513:640 validation, 641:768 test; independent initial conditions",
    "times_per_trajectory" => [0.0, 0.05, 0.10], "derivatives" => "analytic rhs(saved X); no noise, no finite differences",
    "normalization" => "none", "original_truth_and_feature_order_unchanged" => true,
    "maximum_abs_state_all_integration_steps" => data.maximum_state,
    "file_sha256" => Dict(name => file_hash(joinpath(data_dir, name)) for name in files),
    "feature_order_hash" => file_hash(joinpath(data_dir, "monomial_indices.csv")), "training_quality" => quality)
save_toml(joinpath(data_dir, "metadata.toml"), metadata)
println("REVISED_TRAINING_QUALITY=", quality)
println("DATA_SHAPES X=", size(data.X), " theta=", size(theta), " split_counts=", Dict(k => length(v) for (k, v) in columns))

train, validation = columns["train"], columns["validation"]
candidates, selected = fit_candidates(data.X[:, train], data.Y[:, train], data.X[:, validation], data.Y[:, validation], config)
write_table(joinpath(result_dir, "validation_candidates.csv"),
    ["lambda", "rho", "validation_normalized_error", "nonzero_coefficients", "selected"],
    ((row.lambda, row.rho, row.validation_normalized_error, row.nonzero_coefficients, Int(row == selected)) for row in candidates))
selection = Dict("locked_at" => string(now(UTC) + Hour(8), "+08:00"),
    "lambda" => selected.lambda, "rho" => selected.rho, "niter" => config.niter,
    "validation_normalized_error" => selected.validation_normalized_error,
    "nonzero_coefficients" => selected.nonzero_coefficients,
    "candidate_file_sha256" => file_hash(joinpath(result_dir, "validation_candidates.csv")),
    "data_metadata_sha256" => file_hash(joinpath(data_dir, "metadata.toml")),
    "test_metrics_evaluated_before_lock" => false,
    "structural_truth_used_for_selection" => false)
save_toml(joinpath(result_dir, "selection_locked.toml"), selection)
println("PARAMETERS_LOCKED_BEFORE_TEST=", selection)

# Only after the selection lock do we compute held-out metrics or inspect edge truth.
all_metrics = Dict{String,Any}()
for (method, lambda, rho) in (("default", 0.1, 1.0), ("selected", selected.lambda, selected.rho))
    output = joinpath(result_dir, method)
    mkpath(output)
    Ainf, C, relerr = B.this(data.X[:, train], data.Y[:, train], [2, 3], config.dmax, lambda, rho, config.niter)
    write_matrix(joinpath(output, "coefficients.csv"), C)
    write_table(joinpath(output, "inferred_structure.csv"),
        ["order", "target", "source_1", "source_2", "source_3", "coefficient"], inferred_rows(Ainf))
    write_table(joinpath(output, "edge_audit.csv"),
        ["order", "target", "source_1", "source_2", "truth", "predicted", "status", "true_coefficient", "inferred_coefficient"],
        edge_audit(model, Ainf, config.nodes))
    metrics = evaluate_structure(model, Ainf, config.nodes)
    metrics["regression"] = Dict(split => merge(regression_metrics(data.Y[:, cols], C, theta[:, cols]),
        Dict("normalized_squared_error" => normalized_error(data.Y[:, cols], C, theta[:, cols]))) for (split, cols) in columns)
    metrics["lambda"], metrics["rho"], metrics["niter"] = lambda, rho, config.niter
    metrics["original_this_relerr_train"] = relerr
    metrics["raw_Ainf_rows_by_order"] = Dict(string(o) => size(Ainf[o], 1) for o in keys(Ainf))
    save_toml(joinpath(output, "metrics.toml"), metrics)
    all_metrics[method] = metrics
    println(method, " PAIRWISE=", metrics["pairwise"], " TRIPLE=", metrics["triple"])
end
chosen_metrics = all_metrics["selected"]
accepted = all(chosen_metrics[name]["precision"] >= config.minimum_precision &&
    chosen_metrics[name]["recall"] >= config.minimum_recall for name in ("pairwise", "triple")) &&
    chosen_metrics["regression"]["test"]["normalized_squared_error"] <= config.max_test_normalized_error
save_toml(joinpath(result_dir, "acceptance.toml"), Dict("passed" => accepted,
    "scope" => "One fixed clean N=8 fixture only; no guarantee for noise, other sizes or seeds",
    "test_evaluated_at" => string(now(UTC) + Hour(8), "+08:00"),
    "minimum_precision" => config.minimum_precision, "minimum_recall" => config.minimum_recall,
    "max_test_normalized_error" => config.max_test_normalized_error,
    "observed_test_normalized_error" => chosen_metrics["regression"]["test"]["normalized_squared_error"],
    "selection_lock_sha256" => file_hash(joinpath(result_dir, "selection_locked.toml"))))
println("REVISED_ACCEPTANCE_PASSED=", accepted)
println("R1_REVISION_COMPLETE: original model and old baseline retained; no test-driven reselection")
