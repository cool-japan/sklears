# sklears-decomposition TODO

## Current Status (updated 2026-07-14)
This crate is part of the sklears v0.2.1 release. 380 tests passing (`cargo nextest run -p sklears-decomposition --all-features`; previously recorded as 344/349).

## README accuracy pass (2026-07-11)
- [x] Verified README code examples against real source and rewrote fabricated content:
  `TruncatedSVD`, `RandomizedSVD`, `RandomizedPCA`, `OutOfCorePCA`, `TensorPCA`, `Tucker`,
  `PARAFAC`, `SignalICA`, `EMD`, `VMD`, `MemoryEfficientNMF` do not exist as public types (real
  equivalents: `CPDecomposition`/`TuckerDecomposition` for tensor work, `EmpiricalModeDecomposition`
  for EMD, `FastICA` for blind source separation — see corrected `README.md`).
- [ ] `RobustPCA`, `SparsePCA`, `ProbabilisticPCA` in `src/pca.rs` are empty placeholder marker
  structs (`pub struct SparsePCA;` etc., no fields or methods) kept only for name compatibility —
  `pub use pca::*;` re-exports them at the crate root with no working functionality behind the
  name. `LowRankMatrixRecovery` (`matrix_completion.rs`, `RecoveryAlgorithm::RPCA`/`PCP`) is the
  real robust/low-rank-plus-sparse recovery implementation. Either implement real
  `RobustPCA`/`SparsePCA`/`ProbabilisticPCA` types or remove the placeholders so they stop shadowing
  the real names via the glob re-export.
- [ ] `PcaConfig::svd_solver: String` is set (default `"auto"`) but never read anywhere in the fit
  path — no working randomized-SVD option currently exists despite the config field suggesting one.
- [ ] `PCA`/`PcaTrained` does not implement `inverse_transform` (confirmed via
  `property_tests.rs`'s own comment: "PCA inverse_transform not yet implemented").

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod cca` — Phase D-1 complete (banners were stale)
- [x] Re-enable `pub mod format_support` — Phase D-1 complete (banners were stale)
- [x] Re-enable `pub mod incremental_pca` — Phase D-1 complete (banners were stale)
- [x] Re-enable `pub mod kernel_pca` — Phase D-1 complete (banners were stale)
- [x] Re-enable `pub mod manifold` — Phase D-1 complete (banners were stale)
- [x] Re-enable `pub mod matrix_completion` — Phase D-1 complete (banners were stale)
- [x] Re-enable `pub mod pls` — Phase D-1 complete (banners were stale)
- [x] Re-enable `pub mod tensor_decomposition` — Phase D-1 complete (banners were stale)
- [x] Re-enable `pub mod property_tests` (test module) — Phase D-1 complete

## Phase D-1 Summary (2026-04-24)

All 8 disabled-module banners were empirically verified as STALE (zero nalgebra usage
found in any module). All modules re-enabled. Blockers fixed:
- rand 0.10 API: `rng.gen()` → `rng.random::<f64>()` via `scirs2_core::random::RngExt`
- `StdRng::from_rng`: use `scirs2_core::random::rng()` not `thread_rng()`
- Matrix inversion: standalone `fn invert_matrix` via `ArrayLinalgExt::inv`
- 7 blanket `#[allow]` removed from lib.rs; all 100 surfaced lints fixed at root cause
- `cargo clippy -p sklears-decomposition --all-features --all-targets -- -D warnings`: ZERO warnings
- `cargo nextest run -p sklears-decomposition --all-features`: 334 passed, 3 skipped

Post-completion cleanup (2026-04-24):
- Added TODO comments to all `_field` struct renames explaining intended future use
- No files exceed 2000 lines (largest: validation.rs at 1932)
- No `.unwrap()` in production code (only in doc-comment examples and tests)
- `simd_signal` module retained: private infrastructure with working tests; `#[allow(dead_code)]`
  is appropriate as it is planned SIMD infrastructure not yet wired into public API
- `CACHE_LINE_SIZE` / `SIMD_ALIGNMENT` constants in performance.rs retained as documentation
  anchors for the magic numbers used in `CacheConfig::default()`

## Source-level TODOs

- [x] `src/kernel_pca.rs` — Support seeding for reproducibility (was already implemented; verified seeded RNG via `StdRng::seed_from_u64` at both Nyström and random-sampling paths)
- [x] `src/manifold.rs` — Support seeding for reproducibility (tsne, umap init, optimize_umap_embedding now use `self.random_state` via `StdRng`)
- [x] `src/tensor_decomposition.rs` — Support seeding for reproducibility (`thread_rng()` replaced with seeded `StdRng` in both CP and Tucker fit paths)
- [x] `src/matrix_completion.rs` — Fix PCP algorithm numerical stability, alternating minimization
- [x] `src/factor_analysis.rs` — EM algorithm fully implemented: E-step/M-step, factor scores transform, loadings plot data, generative sampling, AIC/BIC model comparison, `select_n_components` helper; 8 new tests (342 total, all passing)
- [x] `src/component_selection.rs` — Use seeded rng when available (was already implemented via `StdRng` at lines 183-187, 444-449, 989-995; verified)
- [x] `src/property_tests.rs` — FactorAnalysis and DictionaryLearning property tests implemented (2026-04-24):
  - `test_factor_analysis_properties`: verifies (n_samples × n_components) factor scores, finite values, loadings shape, positive noise variances, finite log-likelihood
  - `test_dictionary_learning_properties`: verifies dictionary shape (n_components × n_features), finite atoms, finite sparse codes
  - FactorAnalysis added to `test_decomposition_dimension_consistency`
  - `test_reconstruction_bounds`: now exercises forward pass (PCA inverse_transform not yet implemented)
  - Stale `property_tests.rs.disabled` file deleted; 349 tests pass, 0 skipped
- [x] `src/signal_processing/wavelet_transform.rs` — Improve reconstruction accuracy
- [x] `src/modular_framework.rs:420` — Implement proper fit/transform separation: added `apply_fitted` method to `PreprocessingStep` trait, `apply` dispatcher that checks `is_fitted`, updated `DecompositionPipeline::transform` to apply each fitted step without mutation; `StandardizationStep` fully implements both paths; all `fluent_api.rs` step impls updated

## Phase D-2 Summary (2026-04-24)

Priority 1 — Factor Analysis EM implementation:
- Full `FactorAnalysis<Untrained>` / `FactorAnalysis<TrainedFactorAnalysis>` with phantom-type pattern
- EM algorithm: E-step (M matrix, Ez posterior mean), M-step (W_new, Ψ_new), log-likelihood convergence test
- `transform`: computes posterior mean factor scores (n_samples × q)
- `loadings_plot_data`, `communalities` for visualization
- `sample(n, seed)` for generative inference (Box-Muller N(0,1) — no external rand_distr)
- `aic()`, `bic()`, `n_free_params()`, `select_n_components()` for model comparison
- 8 new unit tests; all 342 tests pass

Priority 2 — Seeded RNG:
- `manifold.rs`: 3 sites (tsne init, umap init, optimize_umap_embedding) — seeded via `self.random_state`
- `tensor_decomposition.rs`: 2 sites (CP fit, Tucker fit) — seeded via `self.random_state`
- `kernel_pca.rs` and `component_selection.rs`: already had seeded RNG; verified

Priority 3 — Modular framework fit/transform:
- Added `apply_fitted(&self, data) -> Result<Array2<Float>>` to `PreprocessingStep` trait
- `apply` provides the public immutable transform-only path with fitted-check
- `StandardizationStep` in both `modular_framework.rs` and `fluent_api.rs` fully implement it
- 6 pass-through steps in `fluent_api.rs` implement `apply_fitted` as identity
- `DecompositionPipeline::transform` now applies all preprocessing steps correctly

Final status: `cargo clippy -p sklears-decomposition --all-features --all-targets -- -D warnings` → ZERO warnings; 344 tests, 344 passed, 3 skipped.

Notes on log-likelihood implementation:
- Uses matrix determinant lemma: `log|Σ| = Σⱼ log(ψⱼ) + log|M|` (exact, not diagonal approximation)
- Uses Woodbury identity for `trace(Σ⁻¹ S)` with full sample covariance S
- Numerically verified against hand-computed values for q=1, p=2 case (test_log_likelihood_numerical)
- `log|M|` computed via Leibniz cofactor expansion (q ≤ 4) or partial-pivot Gaussian elimination (q > 4)

## OxiCUDA Migration (v0.2.0)

Status: fully migrated to oxicuda-backed GPU support (Wave A2). Phase 5 hardening
(dependency pruning) complete for the unused-dep item below.

- [x] (S) Remove unused optional deps `oxicuda-blas` and `half` from the `gpu` feature — confirmed
  via `rg` that neither `oxicuda_blas` nor `half::`/`f16`/`bf16` is referenced anywhere in crate code
  (GEMM already routes through `sklears_core::gpu`). Dropped `oxicuda-blas` and `half` from
  `[dependencies]` and from the `gpu` feature list, keeping `dep:oxicuda-solver`,
  `dep:oxicuda-memory`, and `sklears-core/gpu_support`. Verified with `cargo check -p
  sklears-decomposition` and `cargo check -p sklears-decomposition --features gpu` (both
  warning-free), plus `cargo nextest run -p sklears-decomposition --features gpu
  hardware_acceleration` (20/20 passed).

Version watch: oxicuda-* pins are now 0.5.0 workspace-wide (bumped 0.4.0 → 0.4.1 → 0.5.0 during the
0.2.0 cycle). 0.5.0 is a hard floor, not a preference — 0.4.1's PTX codegen emitted invalid f64
elementwise/reduction kernels (32-bit value registers for 64-bit ops) that `ptxas` rejects on real
hardware; 0.5.0 carries the fix. This crate's `gpu` feature deps (`oxicuda-solver`,
`oxicuda-memory`) follow the same workspace pin.

---

See also: [Workspace roadmap](../../TODO.md)
