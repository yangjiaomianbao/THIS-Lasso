using TOML
include(joinpath(@__DIR__, "..", "src", "this_adapter.jl"))
include("r1_common.jl")
using .THISAdapter

c = TOML.parsefile(ARGS[1]);base = TOML.parsefile(ARGS[2]);output = ARGS[3]
for seed in c["seeds"], ratio in c["noise_ratios"], method in c["method_order"]
    slot = "noise_"*lpad(string(round(Int, ratio*1000)), 3, '0')
    trial = joinpath(output, "seed_$(seed)", slot, method)
    status = TOML.parsefile(joinpath(trial, "status.toml"))
    status["status"] == "OK" || continue
    source = joinpath(THISAdapter.ROOT, c["source_root"], "seed_$(seed)", "data")
    d = read_matrix(joinpath(source, "monomial_indices.csv"), Int)
    C = read_matrix(joinpath(trial, "coefficients.csv"))
    A = decode_coefficients(C, d, base["nodes"], base["dmax"])
    R1Baseline.write_table(joinpath(trial, "inferred_structure.csv"),
        ["order", "target", "source_1", "source_2", "source_3", "coefficient"], R1Baseline.inferred_rows(A))
    println("R5_DECODED seed=", seed, " ratio=", ratio, " method=", method,
        " pairwise=", size(A[2], 1), " triple=", size(A[3], 1))
end
