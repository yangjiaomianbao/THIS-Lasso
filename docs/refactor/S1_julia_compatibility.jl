# S1 baseline audit only: read existing paired R5/R4 data; no output writes.
using Test, TOML, LinearAlgebra, Statistics
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "src", "this_adapter.jl"))
using .THISAdapter

function structure_rows(path)
    [split(line, ',') for line in readlines(path)[2:end]]
end

function decode_check(C, d, trial)
    actual = decode_coefficients(C, d, 8, 3)
    rows = structure_rows(joinpath(trial, "inferred_structure.csv"))
    for order in 1:4
        stored = Dict{Tuple,Float64}()
        for row in rows
            parse(Int, row[1]) == order || continue
            key = Tuple(parse.(Int, row[2:order+1]))
            stored[key] = parse(Float64, row[end])
        end
        inferred = Dict(Tuple(Int.(row[1:end-1])) => row[end] for row in eachrow(actual[order]))
        @test Set(keys(inferred)) == Set(keys(stored))
        @test all(isapprox(inferred[key], stored[key]; atol=1e-12, rtol=1e-10) for key in keys(stored))
    end
end

@testset "S1 formal paired Taylor, training scales, M0 and saved coefficient decoding" begin
    formal = joinpath(ROOT,"results","r5","e1_full_independent")
    data = joinpath(formal,"clean","seed_22261035","data")
    X = read_matrix(joinpath(data,"states.csv"))
    theta, d = get_θd(X,3)
    @test theta == read_matrix(joinpath(data,"theta.csv"))
    @test d == read_matrix(joinpath(data,"monomial_indices.csv"),Int)
    manifest = [split(row,',') for row in readlines(joinpath(data,"sample_manifest.csv"))[2:end]]
    train = findall(row -> row[3]=="train",manifest)
    mu = vec(mean(theta[:,train];dims=2));mu[1]=0.
    scales = sqrt.(vec(mean(abs2,theta[:,train].-mu;dims=2)));scales[1]=1.
    @test mu == vec(read_matrix(joinpath(data,"feature_means.csv")))
    @test scales == vec(read_matrix(joinpath(data,"feature_scales.csv")))
    @test (theta.-mu)./scales == read_matrix(joinpath(data,"standardized_theta.csv"))
    for slot in ("noise_000","noise_050")
        cell = joinpath(formal,"seed_22261035",slot)
        Y = read_matrix(joinpath(cell,"data","Y_noisy.csv"))
        nativeA,C,_ = THISAdapter.this(X[:,train],Y[:,train],[2,3],3,.1,1.,10)
        saved = read_matrix(joinpath(cell,"M0","coefficients.csv"))
        @test isapprox(C,saved;atol=1e-12,rtol=1e-10)
        @test (abs.(C).>1e-8) == (abs.(saved).>1e-8)
        @test decode_coefficients(C,d,8,3) == nativeA
        println("S1_M0_MAX_ABS_DIFF ",slot," ",maximum(abs.(C-saved)))
        for method in ("M0","M1","M2","M3")
            trial = joinpath(cell,method)
            status = TOML.parsefile(joinpath(trial,"status.toml"))
            if status["status"] == "NA"
                @test slot=="noise_000" && method in ("M1","M2")
                @test !isfile(joinpath(trial,"coefficients.csv"))
                continue
            end
            decode_check(read_matrix(joinpath(trial,"coefficients.csv")),d,trial)
        end
    end
end

@testset "S1 existing R4 M0s control preserved" begin
    base = joinpath(ROOT,"results","r4","dry_run","seed_22261017")
    data = joinpath(base,"data")
    Z = read_matrix(joinpath(data,"standardized_theta.csv"))
    Y = read_matrix(joinpath(data,"Y.csv"))
    d = read_matrix(joinpath(data,"monomial_indices.csv"),Int)
    manifest = [split(row,',') for row in readlines(joinpath(data,"sample_manifest.csv"))[2:end]]
    train = findall(row -> row[3]=="train",manifest)
    Cz,_ = THISAdapter.mySINDy(Z[:,train],Y[:,train],.1,1.,10)
    saved_z = read_matrix(joinpath(base,"M0s","standardized_coefficients.csv"))
    @test isapprox(Cz,saved_z;atol=1e-12,rtol=1e-10)
    mu = vec(read_matrix(joinpath(data,"feature_means.csv")))
    scales = vec(read_matrix(joinpath(data,"feature_scales.csv")))
    C = Cz./permutedims(scales)
    C[:,1] = Cz[:,1]-C[:,2:end]*mu[2:end]
    saved = read_matrix(joinpath(base,"M0s","coefficients.csv"))
    @test isapprox(C,saved;atol=1e-12,rtol=1e-10)
    @test (abs.(C).>1e-8) == (abs.(saved).>1e-8)
    decode_check(C,d,joinpath(base,"M0s"))
    println("S1_M0s_MAX_ABS_DIFF ",maximum(abs.(C-saved)))
end
println("S1_JULIA_COMPATIBILITY_PASS")
