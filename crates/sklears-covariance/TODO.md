# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Completed (2026-07-04 session)
- [x] `GraphicalLasso`: fixed an even/odd `max_iter` parity bug in the coordinate-descent solver. The old update rule was an exact period-2 cycle from a zero seed, so every off-diagonal entry landed back at exactly `0` at the end of any even `max_iter` — including the crate's own default of 100 — silently degenerating every fit to the identity matrix regardless of `alpha` or the input data. Replaced with a proper Friedman/Hastie/Tibshirani (2008) block coordinate-descent implementation, with an explicit even-vs-odd regression test.
- [x] `GraphicalLasso::get_covariance()`: now returns the alpha-regularized covariance consistent with the fitted precision matrix, instead of always returning the raw unregularized empirical covariance regardless of `alpha`. `get_n_iter()` now returns the real converged sweep count instead of always echoing back `max_iter`.
- [x] `CovarianceHyperparameterTuner::compute_log_likelihood`: added the missing `tr(covariance⁻¹·S)` quadratic-form term — the score previously depended only on `det(covariance)` and ignored the data entirely. `compute_determinant`: replaced the "product of diagonal entries" fallback used for `n > 2` (silently wrong for any matrix with real off-diagonal structure) with a real Gaussian-elimination-with-partial-pivoting determinant. `ScoringMetric::LogLikelihood`/`PredictiveLikelihood`/`CrossValidationScore` are now meaningfully sensitive to hyperparameters (validated by a new test confirming CV selects an interior alpha optimum across a grid instead of a fixed boundary value).
- [x] `examples/comprehensive_cookbook.rs` and `examples/covariance_hyperparameter_tuning_demo.rs`: previously printed "under development" placeholder text with all real code dead inside a commented-out block; both now run real, working recipes against the live API.
- 285 tests passing.

## Known Gaps
- [ ] `CovarianceHyperparameterTuner`'s `compute_condition_number`, `compute_stein_loss`, and `compute_spectral_error` scoring functions remain simplified/placeholder implementations (documented as such in-source) — not addressed this session.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
