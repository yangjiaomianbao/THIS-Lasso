module TaylorScaling
using Statistics

export standardize_taylor, restore_coefficients

"""Original R4/R5 train-only centering/RMS operations, leading constant unchanged."""
function standardize_taylor(theta, train; degenerate_message="Degenerate nonconstant feature; retain failure rather than redraw")
    means = vec(mean(theta[:, train]; dims=2));means[1] = 0.
    scales = sqrt.(vec(mean(abs2, theta[:, train].-means; dims=2)));scales[1] = 1.
    all(>(0), scales) || error(degenerate_message)
    z = (theta.-means)./scales
    z, means, scales
end

"""Original Julia M0s inverse scaling, with the same multiplication order."""
function restore_coefficients(Cz, means, scales)
    coefficients = Cz./permutedims(scales)
    coefficients[:, 1] = Cz[:, 1]-coefficients[:, 2:end]*means[2:end]
    coefficients
end

end # module
