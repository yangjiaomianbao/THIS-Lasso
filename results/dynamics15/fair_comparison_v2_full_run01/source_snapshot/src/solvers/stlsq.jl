module THISSTLSQ
using ..THISAdapter
include(joinpath(@__DIR__, "..", "preprocessing", "taylor_scaling.jl"))
using .TaylorScaling

export fit_stlsq_raw, fit_stlsq_standardized, standardize_taylor, restore_coefficients

"""M0: native THIS on raw states and derivatives, with its existing decoding parity guard."""
function fit_stlsq_raw(X, Y, dmax; lambda=0.1, rho=1.0, niter=10)
    THISAdapter.run_this_solver(X, Y, [2, 3], dmax;
        solver="stlsq", lambda, rho, niter)
end

"""M0s: native mySINDy on the common standardized library; same constant penalty."""
function fit_stlsq_standardized(Z, Y, means, scales; lambda=0.1, rho=1.0, niter=10)
    Cz, _ = THISAdapter.mySINDy(Z, Y, lambda, rho, niter)
    coefficients = restore_coefficients(Cz, means, scales)
    (; coefficients, standardized_coefficients=Cz)
end

end # module
