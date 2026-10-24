module MultiSystemDynamics
using Random
using ..PaperDynamics
export vector_field_multi, local_coefficients_multi, generate_snapshots, physical_ids_multi

physical_ids_multi(system,c)=repeat(collect(1:c["physical_nodes"]),inner=system=="lorenz" ? 3 : system=="stuart_landau" ? 2 : 1)

"""Sparse directed benchmark variants; sources and normalization are declared in config/docs."""
function vector_field_multi(system,x,c)
    system in ("kuramoto","lorenz") && return PaperDynamics.vector_field(system,x,c)
    if system=="lotka_volterra"
        growth=c["growth"].-c["self_damping"].*x
        for (i,j,a) in c["pair_edges"];growth[Int(i)]+=a*x[Int(j)];end
        for (i,j,k,b) in c["triple_edges"];growth[Int(i)]+=b*x[Int(j)]*x[Int(k)];end
        return x.*growth
    elseif system=="sis"
        pressure=zeros(length(x))
        for (i,j,a) in c["pair_edges"];pressure[Int(i)]+=a*x[Int(j)];end
        for (i,j,k,b) in c["triple_edges"];pressure[Int(i)]+=b*x[Int(j)]*x[Int(k)];end
        return -c["recovery"].*x+(1 .-x).*pressure
    elseif system=="stuart_landau"
        y=zeros(length(x));w=c["omega"]
        for i in 1:c["physical_nodes"]
            a=2i-1;u,v=x[a:a+1];radial=c["growth"]-u^2-v^2
            y[a]=radial*u-w*v;y[a+1]=radial*v+w*u
        end
        for (ii,jj,a) in c["pair_edges"]
            i,j=2Int(ii)-1,2Int(jj)-1
            y[i]+=a*(x[j+1]-x[i]);y[i+1]+=a*(x[j]-x[i+1])
        end
        for (ii,jj,kk,b) in c["triple_edges"]
            i,j,k=2Int(ii)-1,2Int(jj)-1,2Int(kk)-1
            y[i]+=b*(x[j]*x[k+1]^2-x[i]^3)
            y[i+1]+=b*(x[j+1]*x[k]^2-x[i+1]^3)
        end
        return y
    end
    error("Unknown system $system")
end

"""Independent symbolic expansion about the common center, using existing polynomial algebra."""
function local_coefficients_multi(system,c,d)
    system in ("kuramoto","lorenz") && return PaperDynamics.local_coefficients(system,c,d)
    add=PaperDynamics.plus;mul=PaperDynamics.times;scale=PaperDynamics.scale
    con=PaperDynamics.constant;var=PaperDynamics.variable
    center=Float64.(c["center"]);n=length(center);xs=[var(i,center) for i in 1:n]
    polys=[con(0.) for _ in 1:n]
    if system=="lotka_volterra"
        growth=[add(con(c["growth"][i]),scale(xs[i],-c["self_damping"])) for i in 1:n]
        for (i,j,a) in c["pair_edges"];i,j=Int(i),Int(j);growth[i]=add(growth[i],scale(xs[j],a));end
        for (i,j,k,b) in c["triple_edges"];i,j,k=Int(i),Int(j),Int(k);growth[i]=add(growth[i],scale(mul(xs[j],xs[k]),b));end
        polys=[mul(xs[i],growth[i]) for i in 1:n]
    elseif system=="sis"
        pressure=[con(0.) for _ in 1:n]
        for (i,j,a) in c["pair_edges"];i,j=Int(i),Int(j);pressure[i]=add(pressure[i],scale(xs[j],a));end
        for (i,j,k,b) in c["triple_edges"];i,j,k=Int(i),Int(j),Int(k);pressure[i]=add(pressure[i],scale(mul(xs[j],xs[k]),b));end
        polys=[add(scale(xs[i],-c["recovery"]),mul(add(con(1.),scale(xs[i],-1.)),pressure[i])) for i in 1:n]
    elseif system=="stuart_landau"
        for i in 1:c["physical_nodes"]
            a=2i-1;u,v=xs[a:a+1]
            radial=add(con(c["growth"]),scale(add(mul(u,u),mul(v,v)),-1.))
            polys[a]=add(mul(radial,u),scale(v,-c["omega"]))
            polys[a+1]=add(mul(radial,v),scale(u,c["omega"]))
        end
        for (ii,jj,a) in c["pair_edges"]
            i,j=2Int(ii)-1,2Int(jj)-1
            polys[i]=add(polys[i],scale(add(xs[j+1],scale(xs[i],-1.)),a))
            polys[i+1]=add(polys[i+1],scale(add(xs[j],scale(xs[i+1],-1.)),a))
        end
        for (ii,jj,kk,b) in c["triple_edges"]
            i,j,k=2Int(ii)-1,2Int(jj)-1,2Int(kk)-1
            polys[i]=add(polys[i],scale(add(mul(xs[j],mul(xs[k+1],xs[k+1])),scale(mul(xs[i],mul(xs[i],xs[i])),-1.)),b))
            polys[i+1]=add(polys[i+1],scale(add(mul(xs[j+1],mul(xs[k],xs[k])),scale(mul(xs[i+1],mul(xs[i+1],xs[i+1])),-1.)),b))
        end
    else
        error("Unknown system")
    end
    keys=[Tuple(sort(filter(!=(0),collect(row)))) for row in eachrow(d)]
    C=zeros(n,length(keys))
    for i in 1:n,(j,key) in enumerate(keys);C[i,j]=get(polys[i],key,0.);end
    C
end

function generate_snapshots(system,seed,config)
    c=config[system];center=Float64.(c["center"]);n=length(center)
    counts=Int.([config["train_samples"],config["validation_samples"],config["test_samples"]])
    total=sum(counts);X=zeros(n,total);Y=similar(X);splits=String[];rng_seeds=Int[]
    code=findfirst(==(system),config["systems"])
    for sample in 1:total
        rng_seed=Int(seed)*1000000+code*100000+sample;push!(rng_seeds,rng_seed)
        x=center+c["box_width"]*(rand(MersenneTwister(rng_seed),n).-0.5)
        system=="sis" && !all(0 .<x.<1) && error("SIS outside probability domain")
        system=="lotka_volterra" && !all(x.>0) && error("Negative species abundance")
        X[:,sample]=x;Y[:,sample]=vector_field_multi(system,x,c)
        push!(splits,sample<=counts[1] ? "train" : sample<=counts[1]+counts[2] ? "validation" : "test")
    end
    (;X,Y,center,splits,rng_seeds,groups=collect(1:total))
end
end
