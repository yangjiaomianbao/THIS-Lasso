using Test,Random,TOML
include(joinpath(@__DIR__,"..","src","this_adapter.jl"));using .THISAdapter
include(joinpath(@__DIR__,"..","src","data","paper_dynamics.jl"))
include(joinpath(@__DIR__,"..","src","data","multisystem_dynamics.jl"));using .MultiSystemDynamics

@testset "Five-system equation and contract checks" begin
    systems=["kuramoto","lorenz","lotka_volterra","stuart_landau","sis"]
    for system in systems
        group=system=="lorenz" ? 3 : system=="stuart_landau" ? 2 : 1;n=4group
        c=Dict{String,Any}("physical_nodes"=>4,"center"=>fill(.5,n),"box_width"=>.4,
            "pair_edges"=>[[1.,2.,.4],[1.,3.,.3]],"triple_edges"=>[[1.,2.,3.,.2]],
            "omega"=>(system=="kuramoto" ? fill(.1,4) : 1.),"sigma"=>10.,"rho"=>28.,"beta"=>8/3,
            "growth"=>(system=="lotka_volterra" ? fill(1.,4) : 1.),"self_damping"=>1.,"recovery"=>1.)
        config=Dict("systems"=>systems,"train_samples"=>24,"validation_samples"=>8,"test_samples"=>8,system=>c)
        data=generate_snapshots(system,2026102201,config)
        @test size(data.X)==(n,40)
        @test data.X==generate_snapshots(system,2026102201,config).X
        @test length(unique(data.rng_seeds))==40
        @test count(==("train"),data.splits)==24
        @test length(physical_ids_multi(system,c))==n
        U=data.X.-data.center;theta,d=get_θd(U,3)
        C=local_coefficients_multi(system,c,d)
        if system=="kuramoto"
            @test maximum(abs.(C*theta-data.Y))<1e-3
            @test vector_field_multi(system,data.X[:,1],c)==PaperDynamics.vector_field(system,data.X[:,1],c)
        else
            @test isapprox(C*theta,data.Y;atol=1e-11,rtol=1e-12)
        end
        system=="lorenz" && @test vector_field_multi(system,data.X[:,1],c)==PaperDynamics.vector_field(system,data.X[:,1],c)
        # Coordinate blocks must remain inside the same physical node after native aggregation.
        @test physical_ids_multi(system,c)==repeat(1:4,inner=group)
        if system=="sis"
            @test all(vector_field_multi(system,zeros(n),c).==0.)
            @test all(vector_field_multi(system,ones(n),c).==-1.)
        elseif system=="lotka_volterra"
            @test all(vector_field_multi(system,zeros(n),c).==0.)
        end
    end
end
