using Test,Random
include(joinpath(@__DIR__,"..","src","this_adapter.jl"))
using .THISAdapter
include(joinpath(@__DIR__,"..","src","preprocessing","pearson_filter.jl"))
using .PearsonFilter

@testset "Native Pearson + cubic adaptation" begin
    rng=MersenneTwister(42);U=randn(rng,6,80);U[6,:]=U[1,:]+.001randn(rng,80)
    train=collect(1:60);physical=[1,1,1,2,2,2]
    neighborhoods,keep=candidate_neighborhoods(U,train,physical,1)
    @test keep==THISAdapter.keep_correlated(U[:,train],1)
    @test keep==[[1,6]]
    @test 6 in neighborhoods[1] && 1 in neighborhoods[6]
    @test all(i in neighborhoods[i] for i in 1:6)
    @test all(j in neighborhoods[i] for i in 1:6 for j in findall(==(physical[i]),physical))
    full,d=THISAdapter.get_θd(U,3)
    theta,ids,mapped=mapped_local_library(U,neighborhoods[1],3,d)
    @test theta==full[ids,:]
    @test mapped==d[ids,:]
    @test size(theta,1)==binomial(length(neighborhoods[1])+3,3)
    U2=copy(U);U2[:,61:end].=randn(rng,6,20)*100
    @test candidate_neighborhoods(U2,train,physical,1)==(neighborhoods,keep)
    # Document the pinned wrapper's incompatible quadratic keys at dmax=3; upstream untouched.
    @test_throws KeyError THISAdapter.this_filter(U[1:3,:],U[1:3,:],[2,3],3,3)
end
