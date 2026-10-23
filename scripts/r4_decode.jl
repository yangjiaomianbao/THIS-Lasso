# Decode restored raw-unit R4 coefficients with the accepted R3 THIS rules.
using TOML
include(joinpath(@__DIR__, "..", "src", "this_adapter.jl"))
include("r1_common.jl")
using .THISAdapter

c = TOML.parsefile(ARGS[1]);output = ARGS[2]
for seed in c["seeds"], method in c["method_order"]
    trial = joinpath(output, "seed_$(seed)", method)
    status = TOML.parsefile(joinpath(trial, "status.toml"))
    status["status"] == "OK" || continue
    d = read_matrix(joinpath(output, "seed_$(seed)", "data", "monomial_indices.csv"), Int)
    C = read_matrix(joinpath(trial, "coefficients.csv"))
    A = decode_coefficients(C, d, c["nodes"], c["dmax"])
    R1Baseline.write_table(joinpath(trial, "inferred_structure.csv"),
        ["order", "target", "source_1", "source_2", "source_3", "coefficient"], R1Baseline.inferred_rows(A))
    println("R4_DECODED seed=", seed, " method=", method, " returned_rows=", Dict(string(o) => size(A[o], 1) for o in 1:4))
end
