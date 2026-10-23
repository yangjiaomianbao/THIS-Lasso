include("r1_common.jl")
using .R1Baseline
using TOML

check_prerequisites()
data_dir, result_dir = joinpath(ROOT, "data", "r1"), joinpath(ROOT, "results", "r1")
metadata = TOML.parsefile(joinpath(data_dir, "metadata.toml"))
protocol = TOML.parsefile(joinpath(result_dir, "protocol.toml"))
config = metadata["config"]
@assert config == protocol["config"]
@assert metadata["this_commit"] == THIS_COMMIT
verify_data_hashes(data_dir, metadata)
X, Y = read_matrix(joinpath(data_dir, "states.csv")), read_matrix(joinpath(data_dir, "derivatives.csv"))
theta = read_matrix(joinpath(data_dir, "theta.csv"))
d = read_matrix(joinpath(data_dir, "monomial_indices.csv"), Int64)
rebuilt_theta, rebuilt_d = R1Baseline.get_θd(X, config["dmax"])
@assert theta == rebuilt_theta && d == rebuilt_d
samples = readlines(joinpath(data_dir, "sample_manifest.csv"))[2:end]
train = findall(line -> split(line, ',')[3] == "train", samples)
@assert size(X) == size(Y) && size(X, 2) == length(samples)
Ainf, C, relerr = R1Baseline.this(X[:, train], Y[:, train], [2, 3], config["dmax"],
    config["lambda"], config["rho"], config["niter"])
saved_C = read_matrix(joinpath(result_dir, "coefficients.csv"))
saved_Ainf = load_inferred(joinpath(result_dir, "inferred_structure.csv"), config["dmax"])
rows, saved_rows = inferred_rows(Ainf), inferred_rows(saved_Ainf)
@assert isapprox(C, saved_C; atol=1e-12, rtol=1e-12)
@assert length(rows) == length(saved_rows)
@assert all(a[1:5] == b[1:5] && isapprox(a[6], b[6]; atol=1e-12, rtol=1e-12) for (a, b) in zip(rows, saved_rows))
saved_metrics = TOML.parsefile(joinpath(result_dir, "metrics.toml"))
@assert isapprox(relerr, saved_metrics["original_this_relerr_train"]; atol=1e-12, rtol=1e-12)
new_metrics = evaluate_structure(make_model(config["nodes"]), Ainf, config["nodes"])
for name in ("pairwise", "triple"), key in ("TP", "FP", "FN", "TN")
    @assert new_metrics[name][key] == saved_metrics[name][key]
end
output = Dict("passed" => true, "coefficient_max_abs_difference" => maximum(abs, C - saved_C),
    "feature_values_exactly_equal" => theta == rebuilt_theta, "monomial_indices_exactly_equal" => d == rebuilt_d,
    "Ainf_canonical_edges_equal" => true, "Ainf_rows" => length(rows),
    "coefficient_tolerance" => 1e-12, "data_hashes_verified" => true, "confusion_counts_equal" => true,
    "source" => "Saved X/Y/theta/d; no regenerated trajectories and no parameter changes")
save_toml(joinpath(result_dir, "replay_check.toml"), output)
println(output)
println("R1_REPLAY_PASS")
