include("r1_revision_common.jl")
using .R1Revision
using .R1Revision.B
using TOML, DelimitedFiles

check_prerequisites()
data_dir, result_dir = joinpath(ROOT, "data", "r1", "revision"), joinpath(ROOT, "results", "r1", "revision")
metadata = TOML.parsefile(joinpath(data_dir, "metadata.toml"))
selection = TOML.parsefile(joinpath(result_dir, "selection_locked.toml"))
config = revision_config()
@assert metadata["config"] == Dict(string(k) => v for (k, v) in pairs(config))
verify_data_hashes(data_dir, metadata)
@assert file_hash(joinpath(data_dir, "metadata.toml")) == selection["data_metadata_sha256"]
@assert file_hash(joinpath(result_dir, "validation_candidates.csv")) == selection["candidate_file_sha256"]
candidate_matrix = readdlm(joinpath(result_dir, "validation_candidates.csv"), ',', Float64; skipstart=1)
candidates = [(lambda=row[1], rho=row[2], validation_normalized_error=row[3], nonzero_coefficients=Int(row[4])) for row in eachrow(candidate_matrix)]
winner = choose_candidate(candidates, config.validation_tolerance)
@assert winner.lambda == selection["lambda"] && winner.rho == selection["rho"]
X, Y = read_matrix(joinpath(data_dir, "states.csv")), read_matrix(joinpath(data_dir, "derivatives.csv"))
theta, d = B.get_θd(X, config.dmax)
@assert theta == read_matrix(joinpath(data_dir, "theta.csv"))
@assert d == read_matrix(joinpath(data_dir, "monomial_indices.csv"), Int64)
manifest = readlines(joinpath(data_dir, "sample_manifest.csv"))[2:end]
train = findall(row -> split(row, ',')[3] == "train", manifest)
validation = findall(row -> split(row, ',')[3] == "validation", manifest)
differences = Dict{String,Float64}()
for (method, lambda, rho) in (("default", 0.1, 1.0), ("selected", winner.lambda, winner.rho))
    output = joinpath(result_dir, method)
    Ainf, C, _ = B.this(X[:, train], Y[:, train], [2, 3], config.dmax, lambda, rho, config.niter)
    saved_C = read_matrix(joinpath(output, "coefficients.csv"))
    saved_Ainf = load_inferred(joinpath(output, "inferred_structure.csv"), config.dmax)
    @assert isapprox(C, saved_C; atol=1e-12, rtol=1e-12)
    a, b = inferred_rows(Ainf), inferred_rows(saved_Ainf)
    @assert length(a) == length(b)
    @assert all(x[1:5] == y[1:5] && isapprox(x[6], y[6]; atol=1e-12, rtol=1e-12) for (x, y) in zip(a, b))
    differences[method] = maximum(abs, C - saved_C)
    if method == "selected"
        @assert isapprox(normalized_error(Y[:, validation], C, theta[:, validation]), selection["validation_normalized_error"]; atol=1e-14, rtol=1e-12)
    end
end
save_toml(joinpath(result_dir, "replay_check.toml"), Dict("passed" => true,
    "coefficient_max_abs_difference" => differences, "feature_values_and_indices_exact" => true,
    "selected_parameters_match_saved_validation_table" => true,
    "original_model_and_truth_unchanged" => read_matrix(joinpath(data_dir, "true_coefficients.csv")) == read_matrix(joinpath(ROOT, "data", "r1", "true_coefficients.csv"))))
println("REPLAY_COEFFICIENT_DIFFERENCES=", differences)
println("R1_REVISION_REPLAY_PASS")
