# Independent post-fit validation of sparse global-index coefficients.
using TOML, DelimitedFiles
include(joinpath(@__DIR__,"..","src","this_adapter.jl"))
using .THISAdapter
include(joinpath(@__DIR__,"..","src","bridge","physical_decode.jl"))

out=abspath(ARGS[1])
isfile(joinpath(out,"selection_locked.json")) || error("Selection must be locked")
request=TOML.parsefile(joinpath(out,"native_decode_request.toml"))
for block in request["datasets"]
    d=read_matrix(joinpath(out,block["data"],"monomial_indices.csv"),Int)
    for trial in block["trials"]
        dir=joinpath(out,trial)
        coordinates=read_matrix(joinpath(dir,"coefficient_coordinates.csv"))
        C=zeros(block["N"],size(d,1))
        for r in eachrow(coordinates)
            C[Int(r[1]),Int(r[2])]=r[3]
        end
        edges=THISAdapter.one2dim_compat(decode_coefficients(C,d,block["N"],3),block["group_size"])
        open(joinpath(dir,"native_physical_edges.csv"),"w") do io
            println(io,"order,target,source_1,source_2,score")
            for order in (2,3),row in eachrow(edges[order])
                ids=Int.(row[1:order])
                length(unique(ids))==order || continue
                println(io,join((order,ids[1],ids[2],order==3 ? ids[3] : 0,row[end]),','))
            end
        end
        println("NATIVE_DECODE_OK $trial")
    end
end
