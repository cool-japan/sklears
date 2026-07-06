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

## OxiCUDA Migration (v0.2.0)

Status: fully migrated to oxicuda-* (no scirs2-core GPU usage remains). Remaining items are optional hardening — the GPU plumbing is real, but several bookkeeping/telemetry layers are still CPU-side simulators.

- [ ] (M) Back `GpuMemoryPool` bookkeeping with oxicuda-memory device pools — both pool types are CPU-side accounting simulators (`src/gpu_acceleration.rs:487-520`, documented "CPU-side memory-pool simulator"; `src/advanced_gpu_acceleration.rs:122-220` best-fit offset bookkeeping with no `DeviceBuffer` behind it). Replace the (offset, size) ledger with real oxicuda-memory pool/arena allocations so `allocate()`/`deallocate()` reserve device memory, and report actual driver numbers via `GpuBackend::memory_info()` instead of the ledger.
  - **Files:** `src/gpu_acceleration.rs`, `src/advanced_gpu_acceleration.rs`
- [ ] (L) Real oxicuda-driver streams for `CudaStream`/`async_matrix_multiply` — `CudaStream` (`src/advanced_gpu_acceleration.rs:224-251`) is a bool-flag struct; `async_matrix_multiply` (`src/advanced_gpu_acceleration.rs:610-654`) is honestly documented as eager/synchronous. Create real streams via oxicuda-driver, launch GEMMs on per-stream oxicuda-blas handles, and make `AsyncGpuOperation::is_ready` query stream completion. Purely additive.
  - **Files:** `src/advanced_gpu_acceleration.rs`
- [ ] (S) Real counters in `GpuPerformanceStats` and device-info fields — `get_performance_stats` (`src/gpu_acceleration.rs:459-469`) returns all-zero stats; `get_device_info` (`src/advanced_gpu_acceleration.rs:331-340`) fabricates 8GB/CC 8.0/68 SMs without a GPU and hardcodes multiprocessor_count/max_threads/shared-memory on the real branch (lines 325-327); `GpuPerformanceMetrics` always records 0.0 bandwidth/occupancy (lines 425-426, 469-470, 550-551, 758-759). Track op/fallback counts and transfer bytes in `GpuLinearOps`; extend sklears-core `GpuUtils::device_properties` (oxicuda-driver attribute queries) to surface SM count, max threads/block, and shared memory.
  - **Files:** `src/gpu_acceleration.rs`, `src/advanced_gpu_acceleration.rs`

---

See also: [Workspace roadmap](../../TODO.md)
