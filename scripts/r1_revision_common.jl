module R1Revision

include("r1_common.jl")
using .R1Baseline
using LinearAlgebra, Statistics

const B = R1Baseline
export B, revision_config, quality_metrics, require_quality, normalized_error,
    choose_candidate, fit_candidates

revision_config() = merge(default_config(; nodes=8, seed=21261016), (
    train_trajectories=512, validation_trajectories=128, test_trajectories=128,
    integration_steps=10, save_every=5,
    lambda_grid=[1e-4, 1e-3, 1e-2, 3e-2, 1e-1],
    rho_grid=[0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0],
    validation_tolerance=1e-12, max_raw_condition=1000.0,
    max_unit_RMS_condition=300.0, minimum_precision=0.95,
    minimum_recall=0.95, max_test_normalized_error=1e-8,
))

function quality_metrics(theta)
    result = feature_diagnostics(theta)
    scale = sqrt.(vec(mean(abs2, theta; dims=2)))
    result["minimum_feature_RMS"] = minimum(scale)
    result["unit_RMS_condition_number"] = all(>(0), scale) ? cond(theta ./ scale) : Inf
    result
end

function require_quality(stats, config)
    stats["rank"] == stats["rows"] || error("Training library is rank deficient; stop without redrawing a seed")
    stats["condition_number"] <= config.max_raw_condition || error("Raw training condition gate failed")
    stats["unit_RMS_condition_number"] <= config.max_unit_RMS_condition || error("Unit-RMS training condition gate failed")
    true
end

normalized_error(Y, C, theta) = sum(abs2, Y - C * theta) / sum(abs2, Y)

"""Only validation error and coefficient sparsity enter selection, never truth/test data."""
function choose_candidate(rows, tolerance)
    isempty(rows) && error("No parameter candidates")
    all(row -> isfinite(row.validation_normalized_error), rows) || error("A candidate failed; do not silently omit it")
    minimum_error = minimum(row.validation_normalized_error for row in rows)
    tied = filter(row -> row.validation_normalized_error <= minimum_error + tolerance, rows)
    first(sort(tied; by=row -> (row.nonzero_coefficients, -row.lambda, row.rho)))
end

function fit_candidates(X_train, Y_train, X_validation, Y_validation, config)
    train_theta, train_d = B.get_θd(X_train, config.dmax)
    validation_theta, validation_d = B.get_θd(X_validation, config.dmax)
    train_d == validation_d || error("Feature order differs across splits")
    require_quality(quality_metrics(train_theta), config)
    rows = NamedTuple[]
    for rho in config.rho_grid, lambda in config.lambda_grid
        _, C, _ = B.this(X_train, Y_train, [2, 3], config.dmax, lambda, rho, config.niter)
        push!(rows, (lambda=lambda, rho=rho,
            validation_normalized_error=normalized_error(Y_validation, C, validation_theta),
            nonzero_coefficients=count(>(1e-8), abs.(C))))
    end
    rows, choose_candidate(rows, config.validation_tolerance)
end

end # module
