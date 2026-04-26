# sklears-linear TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod constrained_optimization` — Phase C-1 (already live in lib.rs; clippy fixes applied)
  - **Files:** `src/lib.rs`, `src/constrained_optimization.rs`

- [x] Re-enable `pub mod glm` — Phase C-1 (already live in lib.rs; clippy fixes applied)
  - **Files:** `src/lib.rs`, `src/glm.rs`

- [x] Re-enable `pub mod logistic_regression_cv` — Phase C-1 (already live in lib.rs; clippy fixes applied)
  - **Files:** `src/lib.rs`, `src/logistic_regression_cv.rs`

- [x] Re-enable `pub mod multi_output_regression` — Phase D-2: nalgebra→ndarray migration complete
  - **Files:** `src/lib.rs`, `src/multi_output_regression.rs`

- [x] Re-enable `pub mod quantile` — Phase C-1 (already live in lib.rs; clippy fixes applied)
  - **Files:** `src/lib.rs`, `src/quantile.rs`

- [x] Re-enable `pub mod serialization` — Phase C-1 (already live in lib.rs; clippy fixes applied, items-after-test-module resolved)
  - **Files:** `src/lib.rs`, `src/serialization.rs`

- [x] Re-enable `pub mod simd_optimizations` — Phase C-1 (already live in lib.rs; clippy fixes applied)
  - **Files:** `src/lib.rs`, `src/simd_optimizations.rs`

- [x] Re-enable `pub use irls::{IRLSConfig, IRLSEstimator, IRLSResult, ScaleEstimator, WeightFunction}` — Phase C-1 (already live in lib.rs; clippy fixes applied)
  - **Files:** `src/lib.rs`, `src/irls.rs`

## Phase C-4 / C-5 (completed)

- [x] Phase C-4: No blanket `#![allow(…)]` were present in lib.rs (already removed prior); all 89 lint errors fixed individually across 20 files
- [x] Phase C-5: Zero `.unwrap()` calls in production code; all `.expect()` calls carry documented invariant messages

## Source-level TODOs

- [x] `src/advanced_property_tests.rs:19` — Replace with scirs2-linalg: full nalgebra→ndarray port complete; module re-enabled in lib.rs; all 429 tests pass
- [x] `src/multi_output_regression.rs` — Migrated ~30 nalgebra call sites to ndarray (DMatrix→Array2, DVector→Array1, SVD→scirs2_linalg::compat::svd)
- [x] `src/solver_implementations.rs:308` — Random permutation implemented using `scirs2_core::random::prelude::{seeded_rng, thread_rng, SliceRandom}` with `random_permutation(n, seed)` helper
- [x] `src/bayesian.rs:1498` — BayesianRidge: SVD-based posterior rewrite (X=USVt → numerically stable γ and alpha/lambda updates); `#[ignore]` removed; test passes
- [x] `src/bayesian.rs:1516` — ARDRegression: fixed missing `lambda * xty` factor in posterior mean; added `gamma_i.clamp(0,1)` and sklearn-style empirical Bayes priors; `#[ignore]` removed; test passes
- [x] `src/sparse_linear_regression.rs:48` — `Validate` trait implemented for `LinearRegressionConfig` in `linear_regression.rs`; sparse config now delegates to `self.base_config.validate()`
- [x] `src/sparse.rs:291` — Implement sparse LASSO coordinate descent
- [x] `src/sparse.rs:298` — Implement sparse Ridge regression
- [x] `src/sparse.rs:305` — Implement sparse Elastic Net
- [x] `src/sparse.rs:319` — Implement sparse LASSO solving
- [x] `src/sparse.rs:334` — Implement sparse Elastic Net solving

---

See also: [Workspace roadmap](../../TODO.md)
