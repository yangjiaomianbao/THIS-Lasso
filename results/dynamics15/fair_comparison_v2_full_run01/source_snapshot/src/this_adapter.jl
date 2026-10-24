module THISAdapter

using DelimitedFiles, TOML, SHA, Printf, LinearAlgebra
const ROOT = normpath(joinpath(@__DIR__, ".."))
const BRIDGE_VERSION = "r3-csv-toml-1"
const THIS_COMMIT = "865efc43fe8ec21966659b482ca5756396974425"
include(joinpath(ROOT, "upstream", "THIS", "this.jl"))

export run_this_solver, decode_coefficients, export_request, read_response,
    write_matrix, read_matrix, file_hash, get_θd

file_hash(path) = bytes2hex(sha256(read(path)))
read_matrix(path, T=Float64) = Matrix{T}(readdlm(path, ',', T))
function write_matrix(path, A)
    ndims(A) == 2 || error("CSV requires a matrix")
    all(isfinite, A) || error("Nonfinite matrix")
    open(path, "w") do io
        for row in eachrow(A)
            for (j, x) in enumerate(row)
                j == 1 || print(io, ',')
                x isa Integer ? print(io, x) : @printf(io, "%.17g", x)
            end
            println(io)
        end
    end
end

"""Private extraction of upstream THIS this.jl decoding (GPL-3.0, pinned above).
Keeps its >1e-8 rule, repeated-source/target exclusion, dictionary and row order.
ooi is intentionally not used to filter; native THIS returns 1:dmax+1 as well.
"""
function decode_coefficients(coeff::Matrix{Float64}, d::Matrix{Int64}, n::Int, dmax::Int)
    size(coeff) == (n, size(d, 1)) || error("Coefficient/feature dimension mismatch")
    size(d, 2) == dmax || error("Index width mismatch")
    all(isfinite, coeff) || error("Nonfinite coefficients")
    all(x -> 0 <= x <= n, d) || error("Invalid Julia node ID")
    idx_mon = Dict{Int64,Vector{Int64}}()
    for i in 1:size(d, 1)
        mon = d[i, :][d[i, :] .!= 0]
        if length(mon) == length(union(mon))
            idx_mon[i] = sort(mon)
        end
    end
    Ainf = Dict{Int64,Matrix{Float64}}(o => zeros(0, o+1) for o in 1:dmax+1)
    idx_coeff = Dict{Int64,Vector{Int64}}()
    nz_idx = Int64[]
    for i in keys(idx_mon)
        aaa = setdiff((1:n)[abs.(coeff[:, i]) .> 1e-8], idx_mon[i])
        if !isempty(aaa)
            push!(nz_idx, i)
            idx_coeff[i] = aaa
        end
    end
    for id in nz_idx
        ii, jj = idx_coeff[id], idx_mon[id]
        o = length(jj)+1
        Ainf[o] = vcat(Ainf[o], [ii repeat(jj', length(ii), 1) coeff[ii, id]])
    end
    Ainf
end

function export_request(directory, theta, Y, d; solver="adaptive_lasso", config=Dict(),
                        validation_theta=nothing, validation_Y=nothing,
                        train_groups=nothing, validation_groups=nothing)
    isdir(directory) && !isempty(readdir(directory)) && error("Bridge directory must be empty")
    n, k = size(Y); p, kt = size(theta); degree = size(d, 2)
    k == kt && size(d, 1) == p && n > 0 && k > 0 && p > 0 && degree > 0 || error("Invalid bridge dimensions")
    eltype(d) <: Integer || error("d must contain integer Julia node IDs")
    all(x -> 0 <= x <= n, d) && all(d[1, :] .== 0) && all(theta[1, :] .== 1) || error("Invalid feature indices/constant")
    has_validation = validation_theta !== nothing
    if has_validation
        validation_Y !== nothing && train_groups !== nothing && validation_groups !== nothing || error("Validation requires Y and trajectory IDs")
        kv = size(validation_theta, 2)
        size(validation_theta, 1) == p && size(validation_Y) == (n, kv) || error("Validation dimensions mismatch")
        length(train_groups) == k && length(validation_groups) == kv || error("Trajectory ID count mismatch")
        eltype(train_groups) <: Integer && eltype(validation_groups) <: Integer || error("Trajectory IDs must be integers")
        isempty(intersect(train_groups, validation_groups)) || error("Trajectory overlap")
    elseif any(x -> x !== nothing, (validation_Y, train_groups, validation_groups))
        error("Incomplete validation request")
    end
    mkpath(directory)
    arrays = Dict("theta.csv" => theta, "Y.csv" => Y, "monomial_indices.csv" => d)
    if has_validation
        merge!(arrays, Dict("validation_theta.csv" => validation_theta, "validation_Y.csv" => validation_Y,
            "train_groups.csv" => reshape(train_groups, :, 1), "validation_groups.csv" => reshape(validation_groups, :, 1)))
    end
    for (name, A) in arrays
        write_matrix(joinpath(directory, name), A)
    end
    hashes = Dict(name => file_hash(joinpath(directory, name)) for name in keys(arrays))
    metadata = Dict{String,Any}("schema_version" => BRIDGE_VERSION, "N" => n, "K" => k, "P" => p,
        "dmax" => degree, "index_base" => 1, "padding_node_id" => 0,
        "feature_order_hash" => hashes["monomial_indices.csv"], "feature_entry_count" => length(d),
        "theta_shape" => collect(size(theta)), "Y_shape" => collect(size(Y)), "d_shape" => collect(size(d)),
        "float_format" => "%.17g", "has_validation" => has_validation,
        "solver" => string(solver), "config" => config, "this_commit" => THIS_COMMIT, "file_sha256" => hashes)
    has_validation && (metadata["validation_K"] = size(validation_theta, 2))
    open(io -> TOML.print(io, metadata; sorted=true), joinpath(directory, "metadata.toml"), "w")
    metadata
end

function read_response(directory; expected_request_hash)
    file_hash(joinpath(directory, "metadata.toml")) == expected_request_hash || error("Request metadata changed")
    m = TOML.parsefile(joinpath(directory, "metadata.toml"))
    response = TOML.parsefile(joinpath(directory, "response.toml"))
    response["schema_version"] == BRIDGE_VERSION && response["mode"] == "solve" || error("Invalid response mode/schema")
    response["request_metadata_sha256"] == expected_request_hash || error("Stale response")
    response["solver"] == m["solver"] || error("Solver mismatch")
    response["feature_order_hash"] == m["feature_order_hash"] || error("Feature order mismatch")
    response["feature_count"] == m["P"] && response["feature_entry_count"] == m["feature_entry_count"] || error("Feature count mismatch")
    for (name, expected) in m["file_sha256"]
        file_hash(joinpath(directory, name)) == expected || error("Request CSV changed: $name")
    end
    for (name, expected) in response["output_sha256"]
        file_hash(joinpath(directory, name)) == expected || error("Response CSV changed: $name")
    end
    haskey(response["output_sha256"], "coefficients.csv") && haskey(response["output_sha256"], "diagnostics.toml") || error("Incomplete response")
    C = read_matrix(joinpath(directory, "coefficients.csv"))
    size(C) == (m["N"], m["P"]) && response["coefficient_shape"] == [m["N"], m["P"]] || error("Returned coefficient shape mismatch")
    all(isfinite, C) || error("Nonfinite returned coefficients")
    C, TOML.parsefile(joinpath(directory, "diagnostics.toml"))
end

function run_this_solver(X::Matrix{Float64}, Y::Matrix{Float64}, ooi::Vector{Int64}, dmax::Int;
                         solver="stlsq", lambda=0.1, rho=1.0, niter=10, config=Dict(),
                         validation_X=nothing, validation_Y=nothing, train_groups=nothing,
                         validation_groups=nothing, work_dir=nothing,
                         python=joinpath(ROOT, ".venv-r0", "bin", "python"))
    size(X) == size(Y) && all(isfinite, X) && all(isfinite, Y) || error("X/Y must be matching finite [N,K] matrices")
    solver = string(solver)
    solver in ("stlsq", "lasso", "adaptive_lasso") || error("Unknown solver")
    dmax >= 1 || error("R3 requires dmax>=1")
    theta, d = get_θd(X, dmax)
    if solver == "stlsq"
        Ainf, C, relerr = this(X, Y, ooi, dmax, Float64(lambda), Float64(rho), niter)
        decode_coefficients(C, d, size(X, 1), dmax) == Ainf || error("Native decoding parity failed")
        return (; Ainf, coeff=C, relerr, theta, d,
            diagnostics=Dict("solver" => solver, "lambda" => lambda, "rho" => rho, "niter" => niter, "native_decode_exact" => true))
    end
    work_dir === nothing && error("Python solvers require an explicit fresh work_dir for auditability")
    tv = nothing
    if validation_X !== nothing
        size(validation_X, 1) == size(X, 1) || error("Validation node count mismatch")
        tv, dv = get_θd(validation_X, dmax)
        dv == d || error("Validation feature order changed")
    end
    export_request(work_dir, theta, Y, d; solver, config, validation_theta=tv,
        validation_Y, train_groups, validation_groups)
    request_hash = file_hash(joinpath(work_dir, "metadata.toml"))
    bridge = joinpath(ROOT, "scripts", "r3_bridge.py")
    cmd = `$python $bridge $work_dir`
    println("PYTHON_COMMAND=", cmd)
    run(addenv(cmd, "PYTHONDONTWRITEBYTECODE" => "1", "OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1"))
    C, diagnostics = read_response(work_dir; expected_request_hash=request_hash)
    Ainf = decode_coefficients(C, d, size(X, 1), dmax)
    relerr = norm(Y-C*theta, 1)/norm(Y, 1)
    (; Ainf, coeff=C, relerr, theta, d, diagnostics)
end

end # module
