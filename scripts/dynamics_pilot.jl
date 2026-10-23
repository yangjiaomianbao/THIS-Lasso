# New isolated experiment; pinned upstream files and historical R-stage outputs are read-only.
using TOML, LinearAlgebra, Statistics, Test
include(joinpath(@__DIR__,"..","src","this_adapter.jl"))
using .THISAdapter
include(joinpath(@__DIR__,"..","src","bridge","physical_decode.jl"))
include(joinpath(@__DIR__,"..","src","solvers","stlsq.jl"))
using .THISSTLSQ
include(joinpath(@__DIR__,"..","src","data","paper_dynamics.jl"))
using .PaperDynamics

function table(path,header,rows)
    open(path,"w") do io
        println(io,join(header,','))
        for row in rows;println(io,join(row,','));end
    end
end
save(path,value)=open(io->TOML.print(io,value;sorted=true),path,"w")
slot(ratio)="noise_"*lpad(string(round(Int,1000ratio)),3,'0')

function prepare(c,output)
    for system in c["systems"], seed in c["seeds"]
        dir=joinpath(output,system,"seed_$seed","data");!ispath(dir)||error("Data exists")
        mkpath(dir); data=generate_local_trajectories(system,seed,c)
        U=data.X.-data.center; theta,d=get_θd(U,c["dmax"])
        train=findall(==("train"),data.splits);z,means,scales=standardize_taylor(theta,train)
        # Exact/local truth only for auditing and later evaluation, never solver selection.
        truth=local_coefficients(system,c[system],d)
        for (name,A) in [("states",data.X),("centered_states",U),("Y",data.Y),("theta",theta),
                         ("standardized_theta",z),("monomial_indices",d),("initial_conditions",data.initials),
                         ("feature_means",reshape(means,:,1)),("feature_scales",reshape(scales,:,1)),
                         ("local_taylor_coefficients",truth)]
            write_matrix(joinpath(dir,name*".csv"),A)
        end
        table(joinpath(dir,"sample_manifest.csv"),["sample_id","trajectory_id","split","time"],
              [(k,data.groups[k],data.splits[k],data.times[k]) for k in axes(U,2)])
        conf=c[system]
        table(joinpath(dir,"physical_truth.csv"),["order","target","source_1","source_2","weight"],
              vcat([(2,Int(e[1]),Int(e[2]),0,e[3]) for e in conf["pair_edges"]],
                   [(3,Int(e[1]),Int(e[2]),Int(e[3]),e[4]) for e in conf["triple_edges"]]))
        @testset "$system seed=$seed analytic checks" begin
            @test length(unique(data.rng_seeds))==length(data.rng_seeds)
            @test size(theta,1)==binomial(size(U,1)+c["dmax"],c["dmax"])
            @test rank(theta[:,train])==size(theta,1)
            @test all(isfinite,theta) && all(isfinite,data.Y)
            @test isempty(intersect(data.groups[train],data.groups[findall(==("test"),data.splits)]))
            if system=="lorenz"
                @test isapprox(truth*theta,data.Y;atol=1e-12,rtol=1e-12)
            end
            # Finite-difference mixed derivative must match analytic Taylor coefficient.
            h=1e-4
            for edge in conf["triple_edges"]
                target,j,k=Int.(edge[1:3])
                if system=="lorenz";target,j,k=3target-2,3j-2,3k-2;end
                ej=zeros(length(data.center));ek=copy(ej);ej[j]=h;ek[k]=h
                f(x)=vector_field(system,x,conf)[target]
                fd=(f(data.center+ej+ek)-f(data.center+ej-ek)-f(data.center-ej+ek)+f(data.center-ej-ek))/(4h*h)
                feat=only(findall(i->sort(filter(!=(0),d[i,:]))==sort([j,k]),axes(d,1)))
                @test isapprox(fd,truth[target,feat];atol=2e-6,rtol=2e-5)
                @test abs(truth[target,feat])>1e-6
            end
        end
        names=filter(x->endswith(x,".csv"),readdir(dir))
        save(joinpath(dir,"metadata.toml"),Dict("system"=>system,"seed"=>seed,"N"=>size(U,1),"P"=>size(theta,1),
            "K_total"=>size(U,2),"dmax"=>c["dmax"],"center"=>data.center,
            "physical_node_of_variable"=>physical_ids(system,conf),"trajectory_rng_seeds"=>data.rng_seeds,
            "feature_order_hash"=>file_hash(joinpath(dir,"monomial_indices.csv")),
            "raw_train_rank"=>rank(theta[:,train]),"standardized_train_condition"=>cond(z[:,train]),
            "analytic_local_taylor_relative_error"=>norm(data.Y-truth*theta)/norm(data.Y),
            "file_sha256"=>Dict(name=>file_hash(joinpath(dir,name)) for name in names)))
        println("PREPARED system=$system seed=$seed N=$(size(U,1)) P=$(size(theta,1)) Ktrain=$(length(train))")
    end
end

function fit_stlsq(c,output)
    for system in c["systems"],seed in c["seeds"]
        root=joinpath(output,system,"seed_$seed");dir=joinpath(root,"data")
        U=read_matrix(joinpath(dir,"centered_states.csv"));Z=read_matrix(joinpath(dir,"standardized_theta.csv"))
        means=vec(read_matrix(joinpath(dir,"feature_means.csv")));scales=vec(read_matrix(joinpath(dir,"feature_scales.csv")))
        rows=[split(line,',') for line in readlines(joinpath(dir,"sample_manifest.csv"))[2:end]]
        train=findall(row->row[3]=="train",rows); params=c["stlsq"]
        for ratio in c["noise_ratios"],method in ["M0","M0s"]
            trial=joinpath(root,slot(ratio),method);!ispath(trial)||error("Trial exists");mkpath(trial)
            Y=read_matrix(joinpath(root,slot(ratio),"Y_noisy.csv"))
            elapsed=@elapsed C=method=="M0" ? fit_stlsq_raw(U[:,train],Y[:,train],c["dmax"];
                lambda=params["lambda"],rho=params["rho"],niter=params["niter"]).coeff :
                fit_stlsq_standardized(Z[:,train],Y[:,train],means,scales;
                lambda=params["lambda"],rho=params["rho"],niter=params["niter"]).coefficients
            write_matrix(joinpath(trial,"coefficients.csv"),C)
            save(joinpath(trial,"status.toml"),Dict("status"=>"OK","fit_seconds"=>elapsed,"method"=>method))
            println("FIT system=$system seed=$seed ratio=$ratio method=$method seconds=$elapsed")
        end
    end
end

function decode(c,output)
    isfile(joinpath(output,"selection_locked.toml"))||error("Must lock all choices before decode/evaluation")
    for system in c["systems"],seed in c["seeds"]
        root=joinpath(output,system,"seed_$seed");dir=joinpath(root,"data")
        d=read_matrix(joinpath(dir,"monomial_indices.csv"),Int);n=length(c[system]["center"])
        for ratio in c["noise_ratios"],method in c["method_order"]
            trial=joinpath(root,slot(ratio),method)
            TOML.parsefile(joinpath(trial,"status.toml"))["status"]=="OK"||continue
            C=read_matrix(joinpath(trial,"coefficients.csv"));A=decode_coefficients(C,d,n,c["dmax"])
            # Upstream aggregation with Dict-key sorting compatibility only; same max rule.
            physical=THISAdapter.one2dim_compat(A,system=="lorenz" ? 3 : 1)
            rows=Tuple[]
            for order in [2,3],row in eachrow(physical[order])
                ids=Int.(row[1:order]);length(unique(ids))==order||continue
                push!(rows,(order,ids[1],ids[2],order==3 ? ids[3] : 0,row[end]))
            end
            table(joinpath(trial,"native_physical_edges.csv"),["order","target","source_1","source_2","score"],rows)
        end
    end
end

task,config_path,output=ARGS
c=TOML.parsefile(config_path)
task=="prepare" ? prepare(c,output) : task=="stlsq" ? fit_stlsq(c,output) : task=="decode" ? decode(c,output) : error("Unknown task")
