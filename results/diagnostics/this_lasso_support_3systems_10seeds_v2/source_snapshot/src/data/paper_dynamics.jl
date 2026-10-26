module PaperDynamics
using Random

export vector_field, generate_local_trajectories, local_coefficients, physical_ids

physical_ids(system, c) = system == "kuramoto" ? collect(1:c["physical_nodes"]) : repeat(collect(1:c["physical_nodes"]), inner=3)
xi(i) = 3Int(i)-2

"""Paper-inspired models, directed edge rows counted once as configured."""
function vector_field(system, x, c)
    if system == "kuramoto"
        y = Float64.(c["omega"])
        for row in c["pair_edges"]
            i,j = Int.(row[1:2]); y[i] += row[3]*sin(x[j]-x[i])
        end
        for row in c["triple_edges"]
            i,j,k = Int.(row[1:3]); y[i] += row[4]*sin(x[j]+x[k]-2x[i])
        end
    elseif system == "lorenz"
        y = zeros(length(x))
        for i in 1:c["physical_nodes"]
            a=xi(i); u,v,w=x[a:a+2]
            y[a]=c["sigma"]*(v-u)
            y[a+1]=u*(c["rho"]-w)-v
            y[a+2]=u*v-c["beta"]*w
        end
        for row in c["pair_edges"]
            i,j=xi.(Int.(row[1:2])); y[i] += row[3]*(x[j]-x[i])
        end
        for row in c["triple_edges"]
            i,j,k=xi.(Int.(row[1:3])); y[i] += row[4]*(x[j]*x[k]^2-x[i]^3)
        end
    else
        error("Unknown dynamics: $system")
    end
    y
end

function generate_local_trajectories(system, seed, config)
    c=config[system]; center=Float64.(c["center"]); n=length(center)
    counts=Int.([config["train_trajectories"],config["validation_trajectories"],config["test_trajectories"]])
    nt=sum(counts); steps=config["integration_steps"]; every=config["save_every"]
    steps % every == 0 || error("Invalid save interval")
    samples=steps÷every+1; X=zeros(n,nt*samples); Y=similar(X)
    initials=zeros(n,nt); groups=Int[]; splits=String[]; times=Float64[]; rng_seeds=Int[]
    code=system=="kuramoto" ? 1 : 2
    for t in 1:nt
        # Non-overlapping deterministic namespaces across trial seeds and systems.
        rng_seed=Int(seed)*10000+code*1000+t
        push!(rng_seeds,rng_seed)
        x=center+c["box_width"]*(rand(MersenneTwister(rng_seed),n).-0.5)
        initials[:,t]=x
        split=t<=counts[1] ? "train" : t<=sum(counts[1:2]) ? "validation" : "test"
        dt=c["integration_dt"]
        for step in 0:steps
            if step % every == 0
                column=(t-1)*samples+step÷every+1
                X[:,column]=x; Y[:,column]=vector_field(system,x,c)
                push!(groups,t); push!(splits,split); push!(times,step*dt)
            end
            step==steps && break
            k1=vector_field(system,x,c); k2=vector_field(system,x+dt*k1/2,c)
            k3=vector_field(system,x+dt*k2/2,c); k4=vector_field(system,x+dt*k3,c)
            x=x+dt*(k1+2k2+2k3+k4)/6
            all(isfinite,x) || error("Nonfinite trajectory; retain failure")
        end
    end
    (; X,Y,initials,groups,splits,times,rng_seeds,center)
end

# Small polynomial algebra computes analytic local Taylor truth independently of fitting.
const Poly=Dict{Tuple,Float64}
constant(v)=Poly(()=>Float64(v))
variable(i,center)=Poly(()=>Float64(center[i]),(i,)=>1.)
function plus(a,b)
    out=copy(a)
    for (k,v) in b; out[k]=get(out,k,0.)+v; end
    out
end
scale(a,s)=Poly(k=>s*v for (k,v) in a)
function times(a,b)
    out=Poly()
    for (ka,va) in a, (kb,vb) in b
        key=Tuple(sort([ka...;kb...]));out[key]=get(out,key,0.)+va*vb
    end
    out
end

function local_coefficients(system,c,d)
    center=Float64.(c["center"]); n=length(center); polynomials=[constant(0.) for _ in 1:n]
    if system=="lorenz"
        xs=[variable(i,center) for i in 1:n]
        for i in 1:c["physical_nodes"]
            a=xi(i);u,v,w=xs[a:a+2]
            polynomials[a]=scale(plus(v,scale(u,-1)),c["sigma"])
            polynomials[a+1]=plus(times(u,plus(constant(c["rho"]),scale(w,-1))),scale(v,-1))
            polynomials[a+2]=plus(times(u,v),scale(w,-c["beta"]))
        end
        for row in c["pair_edges"]
            i,j=xi.(Int.(row[1:2]))
            polynomials[i]=plus(polynomials[i],scale(plus(xs[j],scale(xs[i],-1)),row[3]))
        end
        for row in c["triple_edges"]
            i,j,k=xi.(Int.(row[1:3]))
            term=plus(times(xs[j],times(xs[k],xs[k])),scale(times(xs[i],times(xs[i],xs[i])),-1))
            polynomials[i]=plus(polynomials[i],scale(term,row[4]))
        end
    else
        for i in 1:n; polynomials[i]=constant(c["omega"][i]); end
        for order in [2,3], row in c[order==2 ? "pair_edges" : "triple_edges"]
            i=Int(row[1]); linear=Poly(); phase=0.
            ids=Int.(row[1:order]); weights=order==2 ? [-1.,1.] : [-2.,1.,1.]
            for (id,w) in zip(ids,weights)
                phase+=w*center[id];linear=plus(linear,Poly((id,)=>w))
            end
            term=constant(sin(phase));power=constant(1.)
            for degree in 1:3
                power=times(power,linear)
                term=plus(term,scale(power,sin(phase+degree*pi/2)/factorial(degree)))
            end
            polynomials[i]=plus(polynomials[i],scale(term,row[end]))
        end
    end
    keys=[Tuple(sort(filter(!=(0),collect(row)))) for row in eachrow(d)]
    C=zeros(n,length(keys))
    for i in 1:n, (j,key) in enumerate(keys);C[i,j]=get(polynomials[i],key,0.);end
    C
end
end
