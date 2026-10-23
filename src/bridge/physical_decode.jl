# Compatibility derivative of GPL-3.0 upstream THIS this-tools.jl:one2dim.
# Preserve the original file and function; only materialize Dict keys before sort.
let
    path=joinpath(THISAdapter.ROOT,"upstream","THIS","this-tools.jl")
    source=read(path,String)
    marker="function one2dim(Ainf::Dict{Int64,Matrix{Float64}}, d::Int64=1)"
    start=findfirst(marker,source)
    start===nothing && error("Pinned one2dim signature changed")
    body=source[first(start):end] # one2dim is the final function in the pinned file.
    needle="ooi = sort(keys(Ainf))"
    length(findall(needle,body))==1 || error("Unexpected pinned one2dim source")
    compatible=replace(body,"function one2dim("=>"function one2dim_compat(",
                       needle=>"ooi = sort(collect(keys(Ainf)))")
    include_string(THISAdapter,compatible,"THIS_one2dim_keys_compat.jl")
end
