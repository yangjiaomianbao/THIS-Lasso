module PearsonFilter
using ..THISAdapter

export candidate_neighborhoods, mapped_local_library

"""Native global absolute-Pearson ranking; symmetric endpoint handling plus own physical coordinates."""
function candidate_neighborhoods(U,train,physical_ids,k0)
    keep=THISAdapter.keep_correlated(Matrix{Float64}(U[:,train]),Int(k0))
    neighborhoods=[findall(==(physical_ids[i]),physical_ids) for i in axes(U,1)]
    for pair in keep
        i,j=pair
        push!(neighborhoods[i],j);push!(neighborhoods[j],i)
    end
    [sort(unique(v)) for v in neighborhoods],keep
end

function mapped_local_library(U,variables,dmax,global_d;lookup=Dict(Tuple(row)=>i for (i,row) in enumerate(eachrow(global_d))))
    theta,local_d=THISAdapter.get_θd(Matrix{Float64}(U[variables,:]),dmax)
    mapped=map(x->x==0 ? 0 : variables[x],local_d)
    ids=[lookup[Tuple(row)] for row in eachrow(mapped)]
    theta,ids,mapped
end
end
