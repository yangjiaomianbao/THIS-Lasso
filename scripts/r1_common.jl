module R1Baseline

using Random, LinearAlgebra, Statistics, Printf, DelimitedFiles, TOML, SHA
const ROOT = normpath(joinpath(@__DIR__, ".."))
const THIS_COMMIT = "865efc43fe8ec21966659b482ca5756396974425"
include(joinpath(ROOT, "upstream", "THIS", "this.jl"))

export ROOT, THIS_COMMIT, default_config, make_model, rhs, simulate_trajectory,
    generate_dataset, cube_margins, feature_index, true_coefficients,
    write_matrix, write_table, read_matrix, save_toml, file_hash,
    inferred_rows, load_inferred, edge_universe, true_edges, inferred_edges,
    confusion_metrics, evaluate_structure, edge_audit, feature_diagnostics,
    regression_metrics, check_prerequisites, verify_data_hashes

default_config(; nodes=8, seed=20261016) = (
    nodes=nodes, seed=seed, train_trajectories=24, validation_trajectories=8,
    test_trajectories=8, integration_dt=0.01, integration_steps=200,
    save_every=5, dmax=3, lambda=0.1, rho=1.0, niter=10,
)

struct Interaction
    target::Int
    sources::Vector{Int}
    coefficient::Float64
end

struct PolynomialModel
    decay::Vector{Float64}
    interactions::Vector{Interaction}
end

"""A feed-forward polynomial system, fixed before observing solver results."""
function make_model(n::Int)
    n >= 3 || error("At least three nodes are required")
    decay = [0.8 + 0.05 * (i - 1) for i in 1:n]
    interactions = Interaction[]
    for i in 2:n
        push!(interactions, Interaction(i, [i - 1], [0.20, -0.16, 0.12][mod1(i - 1, 3)]))
    end
    for i in 4:2:n
        push!(interactions, Interaction(i, [i - 3], 0.07 * (-1)^(i ÷ 2)))
    end
    for i in 3:2:n
        weight = [0.08, -0.14, 0.20][mod1((i - 1) ÷ 2, 3)]
        push!(interactions, Interaction(i, [i - 2, i - 1], weight))
    end
    PolynomialModel(decay, interactions)
end

function rhs(x::AbstractVector{Float64}, model::PolynomialModel)
    y = -model.decay .* x
    for edge in model.interactions
        y[edge.target] += edge.coefficient * prod(x[edge.sources])
    end
    y
end

function cube_margins(model::PolynomialModel)
    margin = copy(model.decay)
    for edge in model.interactions
        margin[edge.target] -= abs(edge.coefficient)
    end
    margin
end

function simulate_trajectory(x0::Vector{Float64}, model::PolynomialModel, config)
    steps, every, dt = config.integration_steps, config.save_every, config.integration_dt
    steps % every == 0 || error("Integration steps must be divisible by save_every")
    samples = steps ÷ every + 1
    X = zeros(length(x0), samples)
    Y = similar(X)
    x = copy(x0)
    X[:, 1], Y[:, 1] = x, rhs(x, model)
    maximum_state = maximum(abs, x)
    for step in 1:steps
        k1 = rhs(x, model)
        k2 = rhs(x + (dt / 2) * k1, model)
        k3 = rhs(x + (dt / 2) * k2, model)
        k4 = rhs(x + dt * k3, model)
        x += (dt / 6) * (k1 + 2k2 + 2k3 + k4)
        all(isfinite, x) || error("Nonfinite state; retain failure, do not alter truth")
        maximum_state = max(maximum_state, maximum(abs, x))
        if step % every == 0
            column = step ÷ every + 1
            X[:, column], Y[:, column] = x, rhs(x, model)
        end
    end
    X, Y, maximum_state
end

function generate_dataset(config, model=make_model(config.nodes))
    nt = config.train_trajectories + config.validation_trajectories + config.test_trajectories
    samples = config.integration_steps ÷ config.save_every + 1
    X = zeros(config.nodes, samples * nt)
    Y = similar(X)
    initials = zeros(config.nodes, nt)
    ids, splits, times = Int[], String[], Float64[]
    maximum_state = 0.0
    for trajectory in 1:nt
        split = trajectory <= config.train_trajectories ? "train" :
            trajectory <= config.train_trajectories + config.validation_trajectories ? "validation" : "test"
        # Each trajectory has an explicit independent RNG; seed never selects topology.
        initials[:, trajectory] = 2 .* rand(MersenneTwister(config.seed + trajectory), config.nodes) .- 1
        states, derivatives, bound = simulate_trajectory(initials[:, trajectory], model, config)
        columns = (trajectory - 1) * samples .+ (1:samples)
        X[:, columns], Y[:, columns] = states, derivatives
        append!(ids, fill(trajectory, samples))
        append!(splits, fill(split, samples))
        append!(times, (0:samples-1) .* (config.integration_dt * config.save_every))
        maximum_state = max(maximum_state, bound)
    end
    (; X, Y, initials, trajectory_ids=ids, splits, times, maximum_state)
end

function feature_index(d::Matrix{Int64}, sources::Vector{Int})
    matches = [i for i in axes(d, 1) if sort(filter(!=(0), d[i, :])) == sort(sources)]
    length(matches) == 1 || error("Expected one feature for $sources, found $(length(matches))")
    only(matches)
end

function true_coefficients(model::PolynomialModel, d::Matrix{Int64})
    C = zeros(length(model.decay), size(d, 1))
    for i in eachindex(model.decay)
        C[i, feature_index(d, [i])] = -model.decay[i]
    end
    for edge in model.interactions
        C[edge.target, feature_index(d, edge.sources)] += edge.coefficient
    end
    C
end

function print_cell(io, value)
    if value isa Integer
        print(io, value)
    elseif value isa AbstractFloat
        @printf(io, "%.17g", value)
    else
        text = string(value)
        any(c -> c in (',', '\n', '\r'), text) && error("Unsupported CSV text field")
        print(io, text)
    end
end

function write_table(path, header, rows)
    open(path, "w") do io
        isempty(header) || println(io, join(header, ','))
        for row in rows
            for (column, value) in enumerate(row)
                column == 1 || print(io, ',')
                print_cell(io, value)
            end
            println(io)
        end
    end
end

write_matrix(path, A::AbstractMatrix) = write_table(path, String[], eachrow(A))
read_matrix(path, T=Float64) = Matrix{T}(readdlm(path, ',', T))
save_toml(path, value) = open(io -> TOML.print(io, value; sorted=true), path, "w")
file_hash(path) = bytes2hex(sha256(read(path)))

"""Lossless Ainf rows, sorted only for serialization; Taylor features are untouched."""
function inferred_rows(Ainf)
    rows = Tuple[]
    for order in sort(collect(keys(Ainf)))
        for row in eachrow(Ainf[order])
            sources = zeros(Int, 3)
            sources[1:order-1] = Int.(row[2:order])
            push!(rows, (order, Int(row[1]), sources..., row[end]))
        end
    end
    sort!(rows; by=row -> row[1:5])
end

function load_inferred(path, dmax)
    Ainf = Dict(o => zeros(0, o + 1) for o in 1:dmax+1)
    length(readlines(path)) == 1 && return Ainf
    rows = readdlm(path, ',', Float64; skipstart=1)
    size(rows, 2) == 6 || error("Incorrect inferred_structure CSV width")
    for row in eachrow(rows)
        o = Int(row[1])
        indices = vcat([2], collect(3:o+1), [6])
        Ainf[o] = vcat(Ainf[o], permutedims(row[indices]))
    end
    Ainf
end

function edge_universe(n::Int, order::Int)
    keys = Set{Tuple}()
    for target in 1:n
        sources = setdiff(1:n, [target])
        for combination in combinations(sources, order - 1)
            push!(keys, (target, combination...))
        end
    end
    keys
end

true_edges(model, order) = Set{Tuple}((e.target, sort(e.sources)...) for e in model.interactions if length(e.sources) + 1 == order)
inferred_edges(Ainf, order) = Set{Tuple}(Tuple(Int.(row[1:order])) for row in eachrow(Ainf[order]))

function confusion_metrics(truth, predicted, universe)
    issubset(truth, universe) && issubset(predicted, universe) || error("Invalid directed edge")
    tp = length(intersect(truth, predicted))
    fp = length(setdiff(predicted, truth))
    fn = length(setdiff(truth, predicted))
    tn = length(setdiff(universe, union(truth, predicted)))
    denominator = sqrt(Float64((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    Dict{String,Any}("TP" => tp, "FP" => fp, "FN" => fn, "TN" => tn,
        "candidate_count" => length(universe), "true_count" => length(truth),
        "predicted_count" => length(predicted),
        "precision" => (tp + fp == 0 ? 0.0 : tp / (tp + fp)),
        "recall" => (tp + fn == 0 ? NaN : tp / (tp + fn)),
        "F1" => (2tp + fp + fn == 0 ? 0.0 : 2tp / (2tp + fp + fn)),
        "MCC" => (denominator == 0 ? NaN : (tp * tn - fp * fn) / denominator))
end

evaluate_structure(model, Ainf, n) = Dict{String,Any}(name => confusion_metrics(
    true_edges(model, o), inferred_edges(Ainf, o), edge_universe(n, o))
    for (name, o) in (("pairwise", 2), ("triple", 3)))

function edge_audit(model, Ainf, n)
    rows = Tuple[]
    for order in (2, 3)
        truth = true_edges(model, order)
        predicted = inferred_edges(Ainf, order)
        true_weight = Dict((e.target, sort(e.sources)...) => e.coefficient for e in model.interactions if length(e.sources) + 1 == order)
        pred_weight = Dict(Tuple(Int.(row[1:order])) => row[end] for row in eachrow(Ainf[order]))
        for edge in sort(collect(edge_universe(n, order)))
            t, p = edge in truth, edge in predicted
            status = t ? (p ? "TP" : "FN") : (p ? "FP" : "TN")
            source2 = order == 3 ? edge[3] : 0
            push!(rows, (order, edge[1], edge[2], source2, Int(t), Int(p), status,
                get(true_weight, edge, 0.0), get(pred_weight, edge, 0.0)))
        end
    end
    rows
end

function feature_diagnostics(theta)
    singular = svdvals(theta)
    tolerance = max(size(theta)...) * eps(Float64) * maximum(singular)
    Dict{String,Any}("rows" => size(theta, 1), "columns" => size(theta, 2),
        "rank" => count(>(tolerance), singular), "rank_tolerance" => tolerance,
        "largest_singular_value" => first(singular), "smallest_singular_value" => last(singular),
        "condition_number" => (last(singular) == 0 ? Inf : first(singular) / last(singular)))
end

regression_metrics(Y, C, theta) = Dict(
    "MSE" => mean(abs2, Y - C * theta),
    "relative_entrywise_L1" => norm(Y - C * theta, 1) / norm(Y, 1))

function check_prerequisites()
    VERSION.major == 1 && VERSION.minor == 9 || error("R1 requires Julia 1.9")
    required = ["docs/data_contract.md", "environment/r0/julia/Manifest.toml",
        "reports/R0_learning.md", "reports/r0_logs/julia_smoke.log", "reports/r0_logs/python_smoke.log"]
    all(p -> isfile(joinpath(ROOT, p)), required) || error("Missing R0 prerequisite")
    for filename in ("julia_smoke.log", "python_smoke.log")
        occursin("EXIT_CODE=0", read(joinpath(ROOT, "reports", "r0_logs", filename), String)) || error("R0 smoke not passed")
    end
    repo = joinpath(ROOT, "upstream", "THIS")
    strip(read(`git -C $repo rev-parse HEAD`, String)) == THIS_COMMIT || error("Unexpected THIS commit")
    isempty(strip(read(`git -C $repo status --porcelain`, String))) || error("Upstream THIS is modified")
    true
end

function verify_data_hashes(directory, metadata)
    for (filename, expected) in metadata["file_sha256"]
        file_hash(joinpath(directory, filename)) == expected || error("Data checksum mismatch: $filename")
    end
    true
end

end # module
