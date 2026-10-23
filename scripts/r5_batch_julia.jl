# Full E1 only; generation and inference use the accepted R1/R3 code unchanged.
using TOML, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "src", "this_adapter.jl"))
include("r1_common.jl")
using .THISAdapter
include(joinpath(@__DIR__, "..", "src", "solvers", "stlsq.jl"))
using .THISSTLSQ

function main(task, config_path, output)
    c = TOML.parsefile(config_path);base = TOML.parsefile(joinpath(THISAdapter.ROOT, c["base_config"]))
    c["experiment"] == "E1" && c["phase"] == "full" && c["full_30_seed_authorized"] || error("Full E1 authorization required")
    c["seeds"] == collect(22261017:22261046) || error("Seed protocol changed")
    !c["reuse_pilot_clean_data"] && c["trajectory_seed_stride"] == 10000 || error("Disjoint clean trajectory namespaces required")
    clean_root = joinpath(output, "clean")
    for seed in c["seeds"]
        data_dir = joinpath(clean_root, "seed_$(seed)", "data")
        if task == "prepare"
            if ispath(data_dir)
                R1Baseline.verify_data_hashes(data_dir, TOML.parsefile(joinpath(data_dir, "metadata.toml")))
                println("CLEAN_PILOT_COPY_VERIFIED seed=", seed)
                continue
            end
            mkpath(data_dir)
            seed_base = c["trajectory_seed_origin"] + findfirst(==(seed), c["seeds"])*c["trajectory_seed_stride"]
            cfg = merge(R1Baseline.default_config(; nodes=8, seed=seed_base), (
                train_trajectories=base["train_trajectories"], validation_trajectories=base["validation_trajectories"],
                test_trajectories=base["test_trajectories"], integration_dt=base["integration_dt"],
                integration_steps=base["integration_steps"], save_every=base["save_every"]))
            model = R1Baseline.make_model(8);data = R1Baseline.generate_dataset(cfg, model)
            theta, d = get_θd(data.X, 3);truth = R1Baseline.true_coefficients(model, d)
            @assert truth == read_matrix(joinpath(THISAdapter.ROOT, "data/r1/revision/true_coefficients.csv"))
            @assert d == read_matrix(joinpath(THISAdapter.ROOT, "data/r1/revision/monomial_indices.csv"), Int)
            @assert data.maximum_state <= 1+1e-10 && isapprox(truth*theta, data.Y; atol=1e-14, rtol=1e-13)
            train = findall(==("train"), data.splits)
            z, means, scales = standardize_taylor(theta, train;
                degenerate_message="Degenerate features: retain seed failure, do not redraw")
            for (name, matrix) in (("states", data.X), ("Y", data.Y), ("theta", theta), ("standardized_theta", z),
                    ("monomial_indices", d), ("true_coefficients", truth), ("initial_conditions", data.initials),
                    ("feature_means", reshape(means, :, 1)), ("feature_scales", reshape(scales, :, 1)))
                write_matrix(joinpath(data_dir, name*".csv"), matrix)
            end
            R1Baseline.write_table(joinpath(data_dir, "sample_manifest.csv"), ["sample_id", "trajectory_id", "split", "time"],
                ((k, data.trajectory_ids[k], data.splits[k], data.times[k]) for k in axes(data.X, 2)))
            files = filter(name -> endswith(name, ".csv"), readdir(data_dir))
            metadata = Dict("seed" => seed, "N" => 8, "P" => size(theta, 1), "K_total" => size(theta, 2), "train_K" => length(train),
                "dmax" => 3, "index_base" => 1, "padding_node_id" => 0,
                "feature_order_hash" => file_hash(joinpath(data_dir, "monomial_indices.csv")),
                "raw_train_rank" => rank(theta[:, train]), "raw_train_condition" => cond(theta[:, train]),
                "standardized_train_rank" => rank(z[:, train]), "standardized_train_condition" => cond(z[:, train]),
                "normalization_fit_split" => "train only; population RMS, constant unchanged", "this_commit" => THISAdapter.THIS_COMMIT,
                "trajectory_seed_base" => seed_base,
                "trajectory_rng_seeds" => collect(seed_base+1:seed_base+size(data.initials, 2)),
                "file_sha256" => Dict(name => file_hash(joinpath(data_dir, name)) for name in files))
            R1Baseline.save_toml(joinpath(data_dir, "metadata.toml"), metadata)
            println("CLEAN_GENERATED seed=", seed, " rank=", metadata["raw_train_rank"], " condition=", metadata["raw_train_condition"])
            continue
        end
        X = read_matrix(joinpath(data_dir, "states.csv"))
        d = read_matrix(joinpath(data_dir, "monomial_indices.csv"), Int)
        manifest = [split(row, ',') for row in readlines(joinpath(data_dir, "sample_manifest.csv"))[2:end]]
        train = findall(row -> row[3] == "train", manifest)
        for ratio in c["noise_ratios"]
            slot = "noise_"*lpad(string(round(Int, ratio*1000)), 3, '0')
            cell = joinpath(output, "seed_$(seed)", slot)
            if task == "stlsq"
                trial = joinpath(cell, "M0");ispath(trial) && error("Trial already exists")
                mkpath(trial);response = joinpath(cell, "data", "Y_noisy.csv");Y = read_matrix(response)
                elapsed = @elapsed out = fit_stlsq_raw(X[:, train], Y[:, train], 3;
                    lambda=base["stlsq"]["lambda"], rho=base["stlsq"]["rho"], niter=base["stlsq"]["niter"])
                @assert out.theta == read_matrix(joinpath(data_dir, "theta.csv"))[:, train] && out.d == d
                write_matrix(joinpath(trial, "coefficients.csv"), out.coeff)
                R1Baseline.save_toml(joinpath(trial, "status.toml"), Dict("status" => "OK", "method" => "M0", "seed" => seed,
                    "noise_ratio" => ratio, "fit_seconds" => elapsed, "runtime_scope" => "Native Julia THIS; first-call JIT included",
                    "data_metadata_sha256" => file_hash(joinpath(data_dir, "metadata.toml")),
                    "noise_metadata_sha256" => file_hash(joinpath(cell, "data", "metadata.toml")),
                    "noisy_response_sha256" => file_hash(response)))
            elseif task == "decode"
                for method in c["method_order"]
                    trial = joinpath(cell, method)
                    TOML.parsefile(joinpath(trial, "status.toml"))["status"] == "OK" || continue
                    C = read_matrix(joinpath(trial, "coefficients.csv"))
                    A = decode_coefficients(C, d, 8, 3)
                    R1Baseline.write_table(joinpath(trial, "inferred_structure.csv"),
                        ["order", "target", "source_1", "source_2", "source_3", "coefficient"], R1Baseline.inferred_rows(A))
                end
            else
                error("Unknown full E1 Julia task")
            end
        end
        println("JULIA_", uppercase(task), "_DONE seed=", seed)
    end
end

main(ARGS...)
