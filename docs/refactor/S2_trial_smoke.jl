# One existing N8 noisy trial, end-to-end Julia -> Python -> Julia; new audit output only.
using Test, TOML, LinearAlgebra, Statistics
const ROOT = normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(ROOT,"src","this_adapter.jl"))
using .THISAdapter
const HAS_SHARED_STLSQ = isfile(joinpath(ROOT,"src","solvers","stlsq.jl"))
if HAS_SHARED_STLSQ
    include(joinpath(ROOT,"src","solvers","stlsq.jl"))
end

@testset "S2 one-trial cross-language M2 BIC and frozen structure" begin
    formal=joinpath(ROOT,"results","r5","e1_full_independent")
    data=joinpath(formal,"clean","seed_22261035","data")
    cell=joinpath(formal,"seed_22261035","noise_050")
    X=read_matrix(joinpath(data,"states.csv"))
    theta,d=get_θd(X,3)
    @test theta==read_matrix(joinpath(data,"theta.csv"))
    @test d==read_matrix(joinpath(data,"monomial_indices.csv"),Int)
    rows=[split(row,',') for row in readlines(joinpath(data,"sample_manifest.csv"))[2:end]]
    train=findall(r->r[3]=="train",rows)
    means=vec(read_matrix(joinpath(data,"feature_means.csv")))
    scales=vec(read_matrix(joinpath(data,"feature_scales.csv")))
    Z=read_matrix(joinpath(data,"standardized_theta.csv"))
    if HAS_SHARED_STLSQ
        z,mu,s=THISSTLSQ.standardize_taylor(theta,train)
        @test z==Z && mu==means && s==scales
        changed=copy(theta);changed[:,setdiff(axes(theta,2),train)].+=100.
        _,mu2,s2=THISSTLSQ.standardize_taylor(changed,train)
        @test mu2==means && s2==scales
    end
    Y=read_matrix(joinpath(cell,"data","Y_noisy.csv"))
    base=TOML.parsefile(joinpath(ROOT,"configs","r4_dry_run.toml"))
    config=merge(base["lasso"],base["adaptive"],Dict("selection"=>"bic"))
    request=joinpath(ARGS[1],"bridge_m2");mkpath(dirname(request))
    export_request(request,Z[:,train],Y[:,train],d;solver="adaptive_lasso",config)
    hash=file_hash(joinpath(request,"metadata.toml"))
    python=joinpath(ROOT,".venv-r0","bin","python")
    bridge=joinpath(ROOT,"scripts","r3_bridge.py")
    run(`$python -B $bridge $request`)
    Cz,diag=read_response(request;expected_request_hash=hash)
    # Bridge and direct batch used different memory layouts already in S1.
    # Require exact equality against the same frozen bridge path; keep the
    # pre-existing cross-path tolerance and exact support checks below.
    frozen_request=joinpath(ARGS[1],"bridge_m2_frozen");mkpath(frozen_request)
    for name in ("metadata.toml","theta.csv","Y.csv","monomial_indices.csv")
        cp(joinpath(request,name),joinpath(frozen_request,name))
    end
    frozen_script=joinpath(ROOT,"docs","refactor","S2_frozen_bridge.py")
    run(`$python -B $frozen_script $frozen_request`)
    @test Cz==read_matrix(joinpath(frozen_request,"coefficients.csv"))
    reference=Cz./permutedims(scales)
    reference[:,1]=Cz[:,1]-reference[:,2:end]*means[2:end]
    C=HAS_SHARED_STLSQ ? THISSTLSQ.restore_coefficients(Cz,means,scales) : reference
    @test C==reference
    saved=read_matrix(joinpath(cell,"M2","coefficients.csv"))
    @test isapprox(C,saved;atol=1e-12,rtol=1e-10)
    @test (abs.(C).>1e-8)==(abs.(saved).>1e-8)
    decoded=decode_coefficients(C,d,8,3)
    frozen=decode_coefficients(saved,d,8,3)
    for order in 1:4
        @test size(decoded[order])==size(frozen[order])
        @test isapprox(decoded[order],frozen[order];atol=1e-12,rtol=1e-10)
    end
    @test length(diag["nodes"])==8
    println("S2_BRIDGE_VS_FROZEN_BRIDGE_EXACT=true DIRECT_BATCH_RAW_MAX_DIFF=",maximum(abs.(C-saved)))
    if HAS_SHARED_STLSQ
        raw=THISSTLSQ.fit_stlsq_raw(X[:,train],Y[:,train],3)
        @test raw.coeff==read_matrix(joinpath(cell,"M0","coefficients.csv"))
        control=joinpath(ROOT,"results","r4","dry_run","seed_22261017")
        cdata=joinpath(control,"data")
        cm=[split(r,',') for r in readlines(joinpath(cdata,"sample_manifest.csv"))[2:end]]
        ct=findall(r->r[3]=="train",cm)
        cz=read_matrix(joinpath(cdata,"standardized_theta.csv"))
        cy=read_matrix(joinpath(cdata,"Y.csv"))
        mu=vec(read_matrix(joinpath(cdata,"feature_means.csv")))
        s=vec(read_matrix(joinpath(cdata,"feature_scales.csv")))
        out=THISSTLSQ.fit_stlsq_standardized(cz[:,ct],cy[:,ct],mu,s)
        @test out.standardized_coefficients==read_matrix(joinpath(control,"M0s","standardized_coefficients.csv"))
        @test out.coefficients==read_matrix(joinpath(control,"M0s","coefficients.csv"))
        @test isapprox(out.coefficients*read_matrix(joinpath(cdata,"theta.csv")),out.standardized_coefficients*cz;atol=1e-12,rtol=1e-12)
    end
end
println("S2_ONE_TRIAL_SMOKE_PASS")
