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

- [x] (M) Back `GpuMemoryPool` bookkeeping with oxicuda-memory device pools — both `GpuMemoryPool` types (`gpu_acceleration.rs`, `advanced_gpu_acceleration.rs`) now reserve a real `oxicuda-memory::DeviceBuffer<u8>` arena of the pool's full capacity at construction (via `GpuBackend::detect`/`with_device_id`) when a GPU is present, so the existing host-side (offset, size) best-fit ledger indexes into genuinely allocated device memory instead of pure simulation; falls back to host-side-only accounting (never fabricating a reservation) when no GPU is detected or the allocation itself fails. New `is_device_backed()` on both types reports which case applies. `GpuLinearOps::get_memory_info()` already reported real driver numbers via `GpuBackend::memory_info()` prior to this pass (pre-existing, not re-touched).
  - **Files:** `src/gpu_acceleration.rs`, `src/advanced_gpu_acceleration.rs`
- [x] (L) Real oxicuda-driver streams for `CudaStream`/`async_matrix_multiply` — `CudaStream` now optionally wraps its own `oxicuda-driver::Stream` + per-stream `oxicuda-blas::BlasHandle` (`CudaStream::with_backend`, constructed in `AdvancedGpuOps::new` when a backend is detected). `async_matrix_multiply` launches the GEMM via `CudaStream::launch_matmul`: H2D upload, kernel, and D2H result copy (`copy_to_host_async`) are issued against that stream, with an `oxicuda_driver::Event` recorded right after; `AsyncGpuOperation::is_ready` performs a real non-blocking `cuEventQuery` via that event (materializing the result on first observed completion) instead of a hardcoded `true`. Falls back to the eager synchronous path (unchanged behavior) whenever no GPU is present or the real-stream launch itself errors.
  - **Files:** `src/advanced_gpu_acceleration.rs`
- [x] (S) Real counters in `GpuPerformanceStats` and device-info fields — `GpuLinearOps` now carries atomic `OpCounters` updated by every GPU/CPU dispatch decision (`matrix_multiply`, `matrix_vector_multiply`, `vector_dot`, `solve_linear_system`, `qr_decomposition`), and `get_performance_stats()` computes real averages/totals/transfer-MB from them instead of returning all-zero. `get_device_info` in `advanced_gpu_acceleration.rs` now queries `oxicuda_driver::Device::{get, info}` directly for `multiprocessor_count`/`max_threads_per_block`/`max_shared_memory_per_block` (real `cuDeviceGetAttribute` values) rather than hardcoding `68`/`1024`/`49152`, and its no-GPU fallback now honestly reports all-zero/labelled-placeholder values instead of a fabricated 8GB/CC-8.0 device. `GpuPerformanceMetrics.memory_bandwidth_gbps` is now real bytes-moved/duration (`memory_bandwidth_gbps` helper) at every call site. `occupancy_percentage` stays `0.0` (deferred 2026-07-06: computing real occupancy needs the launch's actual grid/block dimensions, which are internal to `oxicuda_blas::level3::gemm` and not surfaced to callers; would require a lower-level launch API or wiring `oxicuda-driver`'s `occupancy`/`occupancy_ext` modules through a kernel-launch path this crate does not have). Deliberately did *not* extend `sklears_core::gpu::GpuUtils::device_properties` (deferred 2026-07-06: that's a shared file other in-flight per-crate hardening passes reference too; querying `oxicuda_driver::Device` directly from this crate achieves the same real values without a cross-crate edit race).
  - **Files:** `src/gpu_acceleration.rs`, `src/advanced_gpu_acceleration.rs`

---

See also: [Workspace roadmap](../../TODO.md)
