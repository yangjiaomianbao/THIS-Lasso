using Test,TOML
include(joinpath(@__DIR__,"..","src","data","paper_dynamics.jl"))
include(joinpath(@__DIR__,"..","src","data","multisystem_dynamics.jl"));using .MultiSystemDynamics
@testset "V2 independent preparations across networks, data seeds and systems" begin
    names=Set{Int}();states=Set{Tuple}()
    for file in ARGS
        c=TOML.parsefile(file)
        for system in c["systems"],seed in c["seeds"]
            a=generate_snapshots(system,seed,c);b=generate_snapshots(system,seed,c)
            @test a.X==b.X && a.Y==b.Y
            @test a.rng_seeds==b.rng_seeds
            @test isempty(intersect(names,Set(a.rng_seeds)))
            union!(names,a.rng_seeds)
            tuples=Set(Tuple(a.X[:,i]) for i in axes(a.X,2))
            @test length(tuples)==size(a.X,2)
            @test isempty(intersect(states,tuples));union!(states,tuples)
            @test count(==("train"),a.splits)==c["train_samples"]
            @test count(==("validation"),a.splits)==c["validation_samples"]
            @test count(==("test"),a.splits)==c["test_samples"]
            @test size(a.X,1)==15*c[system]["coordinates_per_node"]
            if system=="sis";@test all(0 .< a.X .< 1);end
            if system=="lotka_volterra";@test all(a.X .> 0);end
        end
    end
end
