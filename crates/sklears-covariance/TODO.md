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
- [x] `CovarianceHyperparameterTuner`'s `compute_condition_number`, `compute_stein_loss`, and `compute_spectral_error` scoring functions: replaced all three placeholders with real eigen/determinant-based implementations (`compute_condition_number` via `eigvalsh` fold-based `λ_max/λ_min`; `compute_stein_loss` via the full `tr(Σ̂⁻¹Σ) − log det(Σ̂⁻¹Σ) − p` formula using `trace_of_product` + `det()` ratio, with a finite `-1e6` penalty instead of `NEG_INFINITY` to avoid poisoning CV mean/variance with NaN; `compute_spectral_error` via `‖Σ̂−Σ‖₂ = max|λ_i(Σ̂−Σ)|` with a Frobenius-error fallback on decomposition failure), each backed by a hand-computed unit test at a known optimum. (2026-07-07)
  - **Goal:** Replace all three placeholder scoring methods in `src/hyperparameter_tuning.rs` with mathematically correct implementations, each with a direct unit test proving correctness at a known optimum.
  - **Design:** Reuse the established f64-convert + `scirs2_linalg::compat` pattern already used by `compute_log_likelihood`/`compute_determinant` in this file. Add `use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};`.
    - `compute_condition_number(matrix)` → `matrix.eigvalsh(UPLO::Lower)`, `κ = λ_max / max(λ_min, 1e-15)` (template already in-crate at `testing_quality.rs:588`). Return `f64::INFINITY` on decomposition error.
    - `compute_stein_loss(estimated, true_cov)` → full formula `tr(A) − log det(A) − p` where `A = Σ̂⁻¹·Σ` (convert both to f64; `.inv()` on estimated; matrix-multiply; `.det()` for log det; trace via existing `trace_of_product`/diagonal sum). Keep the current `Ok(-stein_loss)` sign convention. Handle non-invertible/`det≤0` via a large penalty (mirrors `compute_log_likelihood`'s `NEG_INFINITY` guard style).
    - `compute_spectral_error(estimated, true_cov)` → spectral norm of the difference: `D = estimated − true_cov`, `‖D‖₂ = max|λ_i(D)|` via `D.eigvalsh(UPLO::Lower)`. Replaces the current pure delegate to `compute_frobenius_error`.
  - **Files:** `src/hyperparameter_tuning.rs`.
  - **Prerequisites:** none.
  - **Tests:** condition number on `I₃ → ≈1.0` and `diag(1,100) → ≈100`; Stein loss `estimated == true_cov ⇒ score ≈ 0`, mismatched pair ⇒ strictly negative, monotone-worse as estimate drifts; spectral error on a hand-built pair with a known largest-eigenvalue gap ⇒ exact value.
  - **Risk:** `eigvalsh` ignores `UPLO` in this scirs2 version (harmless); eigenvalues unsorted → use `fold` max/min, clamp tiny/negative to `1e-15`. No `unwrap()` — all via `?`/`ok_or_else`.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## Known Gaps (found during 2026-07-11 README audit)

- [x] README previously advertised a "DataFrame Integration" feature (`CovarianceDataFrame`, `DataFrameEstimator`, a `polars_dataframe_demo.rs` example, `fit_dataframe()`) that does not exist anywhere in `src/` — there is no `polars` dependency in `Cargo.toml` and zero occurrences of `DataFrame` in the crate. Removed the fabricated section, its code example, and its example-file reference from README.md. The "Automatic Hyperparameter Tuning" and "Automatic Model Selection" examples also used wrong type/field names (`ScoringMethod` vs. real `ScoringMetric`, a flat `n_cv_folds` vs. real nested `cv_config: CrossValidationConfig`, `CovarianceHyperparameterTuner::new(config)` vs. real `new(parameter_specs, config)`, `AutoCovarianceSelector::builder()/.add_estimator()/.select()` vs. real `::new()/.add_candidate()/.select_best()`); corrected to match `src/hyperparameter_tuning.rs` and `src/model_selection.rs`.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
