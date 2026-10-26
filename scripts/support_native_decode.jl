# Validate both the unchanged native 1e-8 structure and unthresholded AP scores.
using TOML, DelimitedFiles
include(joinpath(@__DIR__,"..","src","this_adapter.jl"));using .THISAdapter
include(joinpath(@__DIR__,"..","src","bridge","physical_decode.jl"))

out=abspath(ARGS[1]);isfile(joinpath(out,"selection_locked.json"))||error("Selection must be locked")
request=TOML.parsefile(joinpath(out,"native_decode_request.toml"))
for block in request["datasets"]
    block["group_size"]==1||error("This support experiment contains scalar-state systems only")
    d=read_matrix(joinpath(out,block["data"],"monomial_indices.csv"),Int)
    for trial in block["trials"]
        dir=joinpath(out,trial);coordinates=read_matrix(joinpath(dir,"coefficient_coordinates.csv"))
        C=zeros(block["N"],size(d,1))
        for row in eachrow(coordinates);C[Int(row[1]),Int(row[2])]=row[3];end
        # Native THIS deliberately omits coefficients <= 1e-8.
        edges=THISAdapter.one2dim_compat(decode_coefficients(C,d,block["N"],3),1)
        open(joinpath(dir,"native_physical_edges.csv"),"w") do io
            println(io,"order,target,source_1,source_2,score")
            for order in (2,3),row in eachrow(edges[order])
                ids=Int.(row[1:order]);length(unique(ids))==order||continue
                println(io,join((order,ids[1],ids[2],order==3 ? ids[3] : 0,row[end]),','))
            end
        end
        # Separate continuous score export: preserve every nonzero, including
        # subthreshold values, so AP is independently checked without censoring.
        open(joinpath(dir,"raw_physical_scores.csv"),"w") do io
            println(io,"order,target,source_1,source_2,score")
            for feature in axes(d,1)
                sources=sort(filter(!=(0),collect(d[feature,:])))
                length(sources) in (1,2)||continue
                length(unique(sources))==length(sources)||continue
                order=length(sources)+1
                for target in axes(C,1)
                    target in sources&&continue
                    score=abs(C[target,feature]);score>0||continue
                    println(io,join((order,target,sources[1],order==3 ? sources[2] : 0,score),','))
                end
            end
        end
        println("NATIVE_AND_RAW_DECODE_OK $trial")
    end
end
