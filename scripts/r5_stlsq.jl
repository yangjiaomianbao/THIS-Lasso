# R5 E1: raw X/feature library remain clean; only regression response is noisy.
using TOML
include(joinpath(@__DIR__, "..", "src", "this_adapter.jl"))
include("r1_common.jl")
using .THISAdapter

c = TOML.parsefile(ARGS[1]);base = TOML.parsefile(ARGS[2]);output = ARGS[3]
c["experiment"] == "E1" && c["phase"] == "smoke" && length(c["seeds"]) == 2 || error("Only two-seed E1 smoke")
for seed in c["seeds"]
    source = joinpath(THISAdapter.ROOT, c["source_root"], "seed_$(seed)", "data")
    X = read_matrix(joinpath(source, "states.csv"))
    rows = [split(row, ',') for row in readlines(joinpath(source, "sample_manifest.csv"))[2:end]]
    train = findall(row -> row[3] == "train", rows)
    for ratio in c["noise_ratios"]
        slot = "noise_"*lpad(string(round(Int, ratio*1000)), 3, '0')
        cell = joinpath(output, "seed_$(seed)", slot)
        response = joinpath(cell, "data", "Y_noisy.csv")
        Y = read_matrix(response)
        trial = joinpath(cell, "M0");ispath(trial) && error("Do not overwrite E1 trial")
        mkpath(trial)
        elapsed = @elapsed out = run_this_solver(X[:, train], Y[:, train], [2, 3], base["dmax"];
            solver="stlsq", lambda=base["stlsq"]["lambda"], rho=base["stlsq"]["rho"], niter=base["stlsq"]["niter"])
        @assert out.theta == read_matrix(joinpath(source, "theta.csv"))[:, train]
        @assert out.d == read_matrix(joinpath(source, "monomial_indices.csv"), Int)
        write_matrix(joinpath(trial, "coefficients.csv"), out.coeff)
        R1Baseline.save_toml(joinpath(trial, "status.toml"), Dict("status" => "OK", "method" => "M0", "seed" => seed,
            "noise_ratio" => ratio, "fit_seconds" => elapsed,
            "runtime_scope" => "Native Julia THIS fit/wrapper, first-call JIT included", "solver_parameters" => base["stlsq"],
            "data_metadata_sha256" => file_hash(joinpath(source, "metadata.toml")),
            "noise_metadata_sha256" => file_hash(joinpath(cell, "data", "metadata.toml")),
            "noisy_response_sha256" => file_hash(response)))
        println("R5_STLSQ seed=", seed, " ratio=", ratio, " seconds=", elapsed)
    end
end
