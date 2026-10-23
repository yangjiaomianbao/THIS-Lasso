using Test
include(joinpath(@__DIR__,"..","src","this_adapter.jl"))
using .THISAdapter
include(joinpath(@__DIR__,"..","src","bridge","physical_decode.jl"))

@testset "Upstream physical aggregation compatibility" begin
    A=Dict(1=>zeros(0,2),2=>[1. 4. -.2; 2. 5. -.7; 5. 4. 3.],
           3=>[1. 4. 7. -.35;2. 5. 8. .1],4=>zeros(0,5))
    before=deepcopy(A)
    @test_throws MethodError THISAdapter.one2dim(A,3)
    B=THISAdapter.one2dim_compat(A,3)
    @test A==before
    @test B[2]==[1. 2. .7;2. 2. 3.]
    @test B[3]==[1. 2. 3. .35]
    @test size(B[4])==(0,5)
    @test THISAdapter.one2dim_compat(A,1)[3]==abs.(A[3])
end
