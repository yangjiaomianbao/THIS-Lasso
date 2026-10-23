using TOML,Test,LinearAlgebra
include(joinpath(@__DIR__,"..","src","this_adapter.jl"));using .THISAdapter
include(joinpath(@__DIR__,"..","src","data","paper_dynamics.jl"))
include(joinpath(@__DIR__,"..","src","data","multisystem_dynamics.jl"));using .MultiSystemDynamics

function table(path,header,rows)
    open(path,"w") do io
        println(io,join(header,','));for row in rows;println(io,join(row,','));end
    end
end

config_path,out=ARGS;c=TOML.parsefile(config_path)
for system in c["systems"],seed in c["seeds"]
    dir=joinpath(out,system,"seed_$seed","data");!ispath(dir)||error("Dataset exists");mkpath(dir)
    data=generate_snapshots(system,seed,c);U=data.X.-data.center
    probe=[1,2,3,size(U,2)];theta,d=get_θd(U[:,probe],c["dmax"])
    truth=local_coefficients_multi(system,c[system],d)
    @testset "$system $seed data and expansion" begin
        @test size(theta)==(binomial(size(U,1)+3,3),4)
        @test length(unique(data.rng_seeds))==size(U,2)
        @test all(isfinite,data.Y)
        @test length(unique(data.groups))==size(U,2)
        if system!="kuramoto"
            @test isapprox(truth*theta,data.Y[:,probe];atol=1e-11,rtol=1e-12)
        end
    end
    for (name,A) in [("states",data.X),("centered_states",U),("Y",data.Y),("initial_conditions",data.X),
                     ("monomial_indices",d),("theta_probe",theta)]
        write_matrix(joinpath(dir,name*".csv"),A)
    end
    table(joinpath(dir,"sample_manifest.csv"),["sample_id","trajectory_id","split","time"],
        ((i,data.groups[i],data.splits[i],0.) for i in axes(U,2)))
    table(joinpath(dir,"local_taylor_sparse.csv"),["node","feature","coefficient"],
        ((i,j,truth[i,j]) for i in axes(truth,1) for j in axes(truth,2) if truth[i,j]!=0.))
    files=filter(x->endswith(x,".csv"),readdir(dir))
    meta=Dict("system"=>system,"seed"=>seed,"N"=>size(U,1),"P"=>size(theta,1),"K_total"=>size(U,2),
        "physical_nodes"=>c[system]["physical_nodes"],"physical_node_of_variable"=>physical_ids_multi(system,c[system]),
        "probe_columns_julia"=>probe,"trajectory_rng_seeds"=>data.rng_seeds,
        "sampling"=>"One independent initial-condition snapshot per sample; analytic derivative, t=0",
        "file_sha256"=>Dict(name=>file_hash(joinpath(dir,name)) for name in files))
    open(io->TOML.print(io,meta;sorted=true),joinpath(dir,"metadata.toml"),"w")
    println("DATA_READY $system $seed N=$(size(U,1)) P=$(size(theta,1)) K=$(size(U,2))")
end
