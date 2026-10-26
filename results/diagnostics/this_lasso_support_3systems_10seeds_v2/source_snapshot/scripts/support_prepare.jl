using TOML, Test
include(joinpath(@__DIR__,"..","src","this_adapter.jl")); using .THISAdapter
include(joinpath(@__DIR__,"..","src","data","paper_dynamics.jl"))
include(joinpath(@__DIR__,"..","src","data","multisystem_dynamics.jl")); using .MultiSystemDynamics

function table(path,header,rows)
    open(path,"w") do io
        println(io,join(header,','))
        for row in rows;println(io,join(row,','));end
    end
end

root=abspath(ARGS[1]);request=TOML.parsefile(joinpath(root,"prepare_request.toml"))
for block in request["instances"]
    c=TOML.parsefile(joinpath(root,block["config"]));seed=only(c["seeds"])
    for system in c["generation_systems"]
        dir=joinpath(root,block["folder"],system,"seed_$seed","data")
        if isfile(joinpath(dir,"metadata.toml"))
            println("REUSE_INPUT $system $seed");continue
        end
        mkpath(dir);data=generate_snapshots(system,seed,c);U=data.X.-data.center
        probe=[1,2,3,size(U,2)];theta,d=get_θd(U[:,probe],c["dmax"])
        truth=local_coefficients_multi(system,c[system],d)
        @test length(unique(data.rng_seeds))==1536
        @test length(unique(data.groups))==1536
        @test size(theta)==(816,4)
        @test all(isfinite,data.Y)
        if system!="kuramoto";@test isapprox(truth*theta,data.Y[:,probe];atol=1e-11,rtol=1e-12);end
        for (name,A) in [("states",data.X),("centered_states",U),("Y",data.Y),
                        ("initial_conditions",data.X),("monomial_indices",d),("theta_probe",theta)]
            write_matrix(joinpath(dir,name*".csv"),A)
        end
        table(joinpath(dir,"sample_manifest.csv"),["sample_id","trajectory_id","split","time"],
              ((i,data.groups[i],data.splits[i],0.) for i in axes(U,2)))
        table(joinpath(dir,"local_taylor_sparse.csv"),["node","feature","coefficient"],
              ((i,j,truth[i,j]) for i in axes(truth,1) for j in axes(truth,2) if truth[i,j]!=0.))
        files=filter(x->endswith(x,".csv"),readdir(dir))
        meta=Dict("system"=>system,"seed"=>seed,"N"=>15,"P"=>816,"K_total"=>1536,
                  "physical_nodes"=>15,"physical_node_of_variable"=>collect(1:15),
                  "probe_columns_julia"=>probe,"trajectory_rng_seeds"=>data.rng_seeds,
                  "sampling"=>"Independent t=0 preparations; analytic derivatives; no integration/burn-in",
                  "file_sha256"=>Dict(n=>file_hash(joinpath(dir,n)) for n in files))
        open(io->TOML.print(io,meta;sorted=true),joinpath(dir,"metadata.toml"),"w")
        println("DATA_READY $(block["folder"]) $system $seed")
    end
end
