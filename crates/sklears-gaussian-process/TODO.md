# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Completed This Session
- [x] Re-enable `deep_gp` module — DONE: deep Gaussian Process layers were already fully implemented, just gated behind a stale TODO; now compiled and exported (`DeepGaussianProcessRegressor`, `DeepGPLayer`, `DeepGPConfig`).
- [x] Add `convolution_processes` module — DONE: from-scratch Convolution Process / dependent multi-output GP implementation (Álvarez & Lawrence style); verified to collapse exactly to a standard single-output GP in the degenerate case and to demonstrably share information across correlated outputs.
- [x] Implement `FitcGaussianProcessRegressor` — DONE: previously a one-line stub, now a full Snelson & Ghahramani sparse GP via inducing points, `O(n·m²)` fit/predict via the Woodbury identity; genuinely exported from the crate root.
- [x] Implement `KernelSelector`/`select_best_kernel` — DONE: previously a one-line stub, now AIC/BIC/log-marginal-likelihood/genuine k-fold-CV selection among arbitrary candidate kernels; genuinely exported from the crate root.
- [x] Implement `VariationalGaussianProcessClassifier` — DONE: previously a one-line stub, now sparse variational GP classification with Bernoulli-logit likelihood and Gauss-Hermite quadrature ELBO; genuinely exported from the crate root.
- [x] Fix `GaussianProcessRegressor::predict_with_std` predictive-variance bug — DONE: the quadratic form computed `v_i · v_i` (i.e. `k_*ᵀ K_reg⁻² k_*`) instead of `k_star_i · v_i` (the correct `k_*ᵀ K_reg⁻¹ k_*`), since the Cholesky solve already applies one full `K_reg⁻¹`; `src/gpr.rs` now computes `k_star_i.dot(&v_i)`.
- [x] 182 tests passing in this crate.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
