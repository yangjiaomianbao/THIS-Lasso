# V2 wrapper; original vector fields, sample generator and native THIS basis unchanged.
using TOML, Test, LinearAlgebra
include(joinpath(@__DIR__,"..","src","this_adapter.jl"));using .THISAdapter
include(joinpath(@__DIR__,"..","src","data","paper_dynamics.jl"))
include(joinpath(@__DIR__,"..","src","data","multisystem_dynamics.jl"));using .MultiSystemDynamics

function table(path,header,rows)
    open(path,"w") do io
        println(io,join(header,','));for row in rows;println(io,join(row,','));end
    end
end

c=TOML.parsefile(ARGS[1]);out=ARGS[2]
for system in c["generation_systems"], seed in c["seeds"]
    dir=joinpath(out,system,"seed_$seed","data")
    isdir(dir) && error("Refuse overwrite: $dir")
    mkpath(dir)
    data=generate_snapshots(system,seed,c);U=data.X.-data.center
    # Exercise exact replay and independence without fitting any model.
    replay=generate_snapshots(system,seed,c)
    @test data.X==replay.X && data.Y==replay.Y && data.rng_seeds==replay.rng_seeds
    probe=[1,2,3,size(U,2)];theta,d=get_θd(U[:,probe],c["dmax"])
    truth=local_coefficients_multi(system,c[system],d)
    @test size(theta)==(binomial(size(U,1)+3,3),4)
    @test length(unique(data.rng_seeds))==size(U,2)
    @test all(isfinite,data.Y)
    @test length(unique([Tuple(data.X[:,i]) for i in axes(data.X,2)]))==size(U,2)
    if system!="kuramoto"
        @test isapprox(truth*theta,data.Y[:,probe];atol=1e-11,rtol=1e-12)
    end
    for (name,A) in [("states",data.X),("centered_states",U),("Y",data.Y),
                     ("monomial_indices",d),("theta_probe",theta)]
        write_matrix(joinpath(dir,name*".csv"),A)
    end
    table(joinpath(dir,"sample_manifest.csv"),["sample_id","trajectory_id","split","time"],
        ((i,data.groups[i],data.splits[i],0.) for i in axes(U,2)))
    table(joinpath(dir,"local_taylor_sparse.csv"),["node","feature","coefficient"],
        ((i,j,truth[i,j]) for i in axes(truth,1) for j in axes(truth,2) if truth[i,j]!=0.))
    files=filter(x->endswith(x,".csv"),readdir(dir))
    meta=Dict("system"=>system,"seed"=>seed,"N"=>size(U,1),"P"=>size(theta,1),"K_total"=>size(U,2),
        "physical_nodes"=>15,"physical_node_of_variable"=>physical_ids_multi(system,c[system]),
        "probe_columns_julia"=>probe,"trajectory_rng_seeds"=>data.rng_seeds,
        "replay_exact"=>true,"sampling"=>"Independent t=0 states and analytic derivatives",
        "file_sha256"=>Dict(name=>file_hash(joinpath(dir,name)) for name in files))
    open(io->TOML.print(io,meta;sorted=true),joinpath(dir,"metadata.toml"),"w")
    println("V2_DATA_READY $system seed=$seed N=$(size(U,1)) P=$(size(theta,1))")
end
