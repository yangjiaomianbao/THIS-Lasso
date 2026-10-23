include("r1_common.jl")
using .R1Baseline
using TOML, LinearAlgebra

function main(args)
    options = Dict("--nodes" => "8", "--seed" => "20261016",
        "--data-dir" => joinpath(ROOT, "data", "r1"),
        "--result-dir" => joinpath(ROOT, "results", "r1"))
    iseven(length(args)) || error("Options require values")
    for i in 1:2:length(args)
        haskey(options, args[i]) || error("Unknown option $(args[i])")
        options[args[i]] = args[i + 1]
    end
    n, seed = parse(Int, options["--nodes"]), parse(Int, options["--seed"])
    n in (8, 12) || error("This R1 entry supports N=8 or N=12")
    config = default_config(; nodes=n, seed=seed)
    data_dir, result_dir = abspath(options["--data-dir"]), abspath(options["--result-dir"])
    check_prerequisites()
    ispath(data_dir) && error("Data directory already exists; never overwrite saved benchmark data")
    isfile(joinpath(result_dir, "protocol.toml")) && error("Protocol already exists; do not overwrite a trial")
    mkpath(data_dir)
    mkpath(result_dir)
    config_dict = Dict{String,Any}(string(k) => v for (k, v) in pairs(config))
    protocol = Dict("config" => config_dict, "this_commit" => THIS_COMMIT,
        "model_selection" => "Original defaults lambda=0.1 rho=1 niter=10, fixed before data and results; no tuning",
        "fit_split" => "train", "scored_orders" => [2, 3], "pearson_filter" => false,
        "normalization" => "none", "data_dir" => data_dir,
        "execution_scope" => (n == 8 ? "one seed, N=8 primary baseline; no R2-R5 implementation" :
            "one seed, N=12 extension; no R2-R5 implementation"))
    save_toml(joinpath(result_dir, "protocol.toml"), protocol)
    model = make_model(n)
    @assert minimum(cube_margins(model)) > 0
    data = generate_dataset(config, model)
    @assert data.maximum_state <= 1 + 1e-10
    theta, d = R1Baseline.get_θd(data.X, config.dmax)
    true_C = true_coefficients(model, d)
    @assert isapprox(data.Y, true_C * theta; atol=1e-12, rtol=1e-12)
    train = findall(==("train"), data.splits)
    feature_stats = feature_diagnostics(theta[:, train])
    for (name, matrix) in (("states", data.X), ("derivatives", data.Y), ("theta", theta),
            ("monomial_indices", d), ("true_coefficients", true_C), ("initial_conditions", data.initials))
        write_matrix(joinpath(data_dir, name * ".csv"), matrix)
    end
    write_table(joinpath(data_dir, "sample_manifest.csv"),
        ["sample_id", "trajectory_id", "split", "time"],
        ((k, data.trajectory_ids[k], data.splits[k], data.times[k]) for k in axes(data.X, 2)))
    write_table(joinpath(data_dir, "true_structure.csv"),
        ["order", "target", "source_1", "source_2", "coefficient"],
        ((length(e.sources) + 1, e.target, e.sources[1], length(e.sources) == 2 ? e.sources[2] : 0, e.coefficient)
            for e in model.interactions))
    filenames = sort(filter(name -> endswith(name, ".csv"), readdir(data_dir)))
    metadata = Dict("config" => config_dict, "this_commit" => THIS_COMMIT,
        "rng" => "Julia MersenneTwister(seed + trajectory_id), independent initialization per trajectory",
        "trajectory_seeds" => seed .+ (1:size(data.initials, 2)),
        "integration" => "fixed-step classical RK4, analytic Y=rhs(saved X); no finite differences",
        "concatenation" => "trajectory_id ascending, then time ascending; no differences across boundaries",
        "split_rule" => "trajectory 1:24 train, 25:32 validation, 33:40 test; no trajectory crosses splits",
        "initial_distribution" => "independent Uniform[-1,1] per node", "normalization" => "none",
        "index_base" => 1, "padding_node_id" => 0, "float_format" => "%.17g Float64, numeric matrices have no headers",
        "state_shape" => collect(size(data.X)), "theta_shape" => collect(size(theta)), "d_shape" => collect(size(d)),
        "coefficient_shape" => collect(size(true_C)), "self_decay" => model.decay,
        "cube_invariance_margins" => cube_margins(model), "maximum_abs_state_all_integration_steps" => data.maximum_state,
        "feature_order_hash" => file_hash(joinpath(data_dir, "monomial_indices.csv")),
        "file_sha256" => Dict(name => file_hash(joinpath(data_dir, name)) for name in filenames),
        "train_feature_diagnostics" => feature_stats,
        "upstream_source_sha256" => Dict(name => file_hash(joinpath(ROOT, "upstream", "THIS", name)) for name in ("this.jl", "this-tools.jl", "LICENSE")))
    save_toml(joinpath(data_dir, "metadata.toml"), metadata)
    println("PREDECLARED_CONFIG=", config)
    println("X=", size(data.X), " theta=", size(theta), " d=", size(d), " train_K=", length(train))
    println("TRAIN_FEATURE_DIAGNOSTICS=", feature_stats)
    println("MAX_ABS_STATE=", data.maximum_state, " CUBE_MIN_MARGIN=", minimum(cube_margins(model)))
    started = time_ns()
    Ainf, C, relerr = R1Baseline.this(data.X[:, train], data.Y[:, train], [2, 3],
        config.dmax, config.lambda, config.rho, config.niter)
    elapsed = (time_ns() - started) / 1e9
    write_matrix(joinpath(result_dir, "coefficients.csv"), C)
    write_table(joinpath(result_dir, "inferred_structure.csv"),
        ["order", "target", "source_1", "source_2", "source_3", "coefficient"], inferred_rows(Ainf))
    write_table(joinpath(result_dir, "edge_audit.csv"),
        ["order", "target", "source_1", "source_2", "truth", "predicted", "status", "true_coefficient", "inferred_coefficient"],
        edge_audit(model, Ainf, n))
    metrics = evaluate_structure(model, Ainf, n)
    metrics["regression"] = Dict(split => begin
        columns = findall(==(split), data.splits)
        regression_metrics(data.Y[:, columns], C, theta[:, columns])
    end for split in ("train", "validation", "test"))
    metrics["original_this_relerr_train"] = relerr
    metrics["fit_seconds_including_JIT"] = elapsed
    metrics["raw_Ainf_rows_by_order"] = Dict(string(o) => size(Ainf[o], 1) for o in keys(Ainf))
    save_toml(joinpath(result_dir, "metrics.toml"), metrics)
    for name in ("pairwise", "triple")
        println(name, "=", metrics[name])
    end
    println("REGRESSION=", metrics["regression"])
    println("R1_BASELINE_COMPLETE (any FP/FN are retained; no parameter search)")
end

main(ARGS)
