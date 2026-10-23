# Same R1 physical network; new predeclared trajectories are written only in R4.
using TOML, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "src", "this_adapter.jl"))
include("r1_common.jl")
using .THISAdapter
include(joinpath(@__DIR__, "..", "src", "solvers", "stlsq.jl"))
using .THISSTLSQ

function prepare(config_path, output)
    c = TOML.parsefile(config_path)
    c["phase"] == "A" && length(c["seeds"]) == 2 || error("Only two-seed phase A allowed")
    for seed in c["seeds"]
        dir = joinpath(output, "seed_$(seed)", "data")
        ispath(dir) && error("Never overwrite a trial")
        mkpath(dir)
        cfg = merge(R1Baseline.default_config(; nodes=c["nodes"], seed), (
            train_trajectories=c["train_trajectories"], validation_trajectories=c["validation_trajectories"],
            test_trajectories=c["test_trajectories"], integration_dt=c["integration_dt"],
            integration_steps=c["integration_steps"], save_every=c["save_every"]))
        model = R1Baseline.make_model(c["nodes"])
        data = R1Baseline.generate_dataset(cfg, model)
        theta, d = get_θd(data.X, c["dmax"])
        truth = R1Baseline.true_coefficients(model, d)
        @assert truth == read_matrix(joinpath(THISAdapter.ROOT, "data/r1/revision/true_coefficients.csv"))
        @assert d == read_matrix(joinpath(THISAdapter.ROOT, "data/r1/revision/monomial_indices.csv"), Int)
        @assert data.maximum_state <= 1+1e-10
        @assert isapprox(truth*theta, data.Y; atol=1e-14, rtol=1e-13)
        train = findall(==("train"), data.splits)
        z, means, scales = standardize_taylor(theta, train;
            degenerate_message="Degenerate nonconstant feature; retain failure rather than redraw")
        for (name, matrix) in (("states", data.X), ("Y", data.Y), ("theta", theta),
                ("standardized_theta", z), ("monomial_indices", d), ("true_coefficients", truth),
                ("initial_conditions", data.initials), ("feature_means", reshape(means, :, 1)),
                ("feature_scales", reshape(scales, :, 1)))
            write_matrix(joinpath(dir, name*".csv"), matrix)
        end
        R1Baseline.write_table(joinpath(dir, "sample_manifest.csv"), ["sample_id", "trajectory_id", "split", "time"],
            ((k, data.trajectory_ids[k], data.splits[k], data.times[k]) for k in axes(data.X, 2)))
        files = filter(name -> endswith(name, ".csv"), readdir(dir))
        metadata = Dict("seed" => seed, "N" => c["nodes"], "P" => size(theta, 1), "K_total" => size(theta, 2),
            "train_K" => length(train), "dmax" => c["dmax"], "index_base" => 1, "padding_node_id" => 0,
            "feature_order_hash" => file_hash(joinpath(dir, "monomial_indices.csv")),
            "raw_train_rank" => rank(theta[:, train]), "raw_train_condition" => cond(theta[:, train]),
            "standardized_train_rank" => rank(z[:, train]), "standardized_train_condition" => cond(z[:, train]),
            "normalization_fit_split" => "train only; population RMS, constant unchanged",
            "this_commit" => THISAdapter.THIS_COMMIT, "file_sha256" => Dict(name => file_hash(joinpath(dir, name)) for name in files))
        R1Baseline.save_toml(joinpath(dir, "metadata.toml"), metadata)
        for method in ("M0", "M0s")
            trial = joinpath(output, "seed_$(seed)", method);mkpath(trial)
            coefficients = zeros(c["nodes"], size(theta, 1))
            elapsed = if method == "M0"
                @elapsed begin
                    out = fit_stlsq_raw(data.X[:, train], data.Y[:, train], c["dmax"];
                        lambda=c["stlsq"]["lambda"], rho=c["stlsq"]["rho"], niter=c["stlsq"]["niter"])
                    coefficients = out.coeff
                end
            else
                @elapsed begin
                    out = fit_stlsq_standardized(z[:, train], data.Y[:, train], means, scales;
                        lambda=c["stlsq"]["lambda"], rho=c["stlsq"]["rho"], niter=c["stlsq"]["niter"])
                    Cz = out.standardized_coefficients
                    write_matrix(joinpath(trial, "standardized_coefficients.csv"), Cz)
                    coefficients = out.coefficients
                    @assert isapprox(coefficients*theta, Cz*z; atol=1e-12, rtol=1e-12)
                end
            end
            write_matrix(joinpath(trial, "coefficients.csv"), coefficients)
            R1Baseline.save_toml(joinpath(trial, "status.toml"), Dict("status" => "OK", "method" => method,
                "seed" => seed, "fit_seconds" => elapsed, "preprocessing" => method == "M0" ? "raw" : "common standardized",
                "runtime_scope" => "Julia fit/control wrapper, first-call JIT included; not a benchmark",
                "solver_parameters" => c["stlsq"], "data_metadata_sha256" => file_hash(joinpath(dir, "metadata.toml"))))
        end
        println("R4_PREPARED seed=", seed, " train_K=", length(train), " rank=", metadata["raw_train_rank"],
            " condition=", metadata["raw_train_condition"], " standardized_condition=", metadata["standardized_train_condition"])
    end
end

prepare(ARGS...)
