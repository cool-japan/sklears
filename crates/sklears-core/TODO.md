# TODO - v0.2.0

## Current Status (updated 2026-07-14)
This crate is part of the sklears v0.2.0 release. 863 tests passing (`cargo nextest run -p sklears-core --all-features`).

## Completed in 0.2.0 (2026-07-14)
- [x] `trait_explorer::security_analysis` fully re-enabled — previously excluded from the build
  entirely (`// pub mod security_analysis; // Temporarily disabled due to compilation issues`),
  with `compliance_framework.rs`/`security_metrics.rs` calling methods on 22 types that had no
  implementation anywhere in the repository. Now implemented and compiling
  (`pub mod security_analysis;` in `src/trait_explorer/mod.rs`), including data-backed
  constructors for common regulatory frameworks and standards (GDPR, HIPAA, CCPA, SOX, FERPA, ISO
  27001, NIST CSF, COBIT, ITIL, CIS Controls) and a new dashboard-management layer.
  `compliance_framework.rs` (975 lines) and `security_metrics.rs` (1380 lines) were each split
  into a submodule directory via `splitrs` per the workspace's file-size policy
  (`src/trait_explorer/security_analysis/{compliance_framework,security_metrics}/`); all
  pre-existing public items carry forward unchanged. This is internal trait-analysis dev-tooling
  (not part of the public ML API), so it is not re-exported at the crate root.
- [x] `contract_testing` module re-enabled — previously disabled pending an ndarray-0.17
  associated-type-projection normalization mismatch between generic `where`-clause bounds and
  concrete impls; now `pub mod contract_testing;` in `src/lib.rs` (re-exported from the crate
  `prelude`). New `mock_objects` owned-array (`Array2<f64>`/`Array1<f64>`) `Fit`/`Predict`/
  `PredictProba`/`Transform` impls for `MockEstimator`/`TrainedMockEstimator`/`MockTransformer`
  (delegating to the existing view-based impls) so contract tests can exercise them directly.

## README accuracy pass (2026-07-11)
- [x] Verified README code examples against real source and fixed several fabricated
  imports/APIs that would not compile: `sklears_core::phantom::{Classification, Regression}`
  (module does not exist — the crate's phantom-type pattern is applied per-type via `Estimator<State>`,
  not a shared marker module), `#[derive(ValidatedConfig)]` + `RangeValidator { min, max }`
  (real API is `ValidatedConfig::new(cfg).validate()` and const-generic
  `RangeValidator<const MIN: i64, const MAX: i64>`), `SklearnEstimator::from_sklearn` /
  `array.to_numpy()` / `array.to_torch_tensor()` / `Dataset::from_polars()` (none exist; real
  interop is `compatibility::{serialization::CrossPlatformModel, numpy::NumpyArray,
  pytorch::ndarray_to_pytorch_tensor, pandas::DataFrame}`), `sklears_core::testing::{properties,
  MockEstimator}` (no `testing` module; real path is `mock_objects::MockEstimator` with a
  behavior-simulation builder, not an `.expect_fit().returning(...)` mock), `SimdOps::
  euclidean_distance_matrix` (real name `euclidean_distances_simd`), `sklears_core::memory::
  {MemoryPool, CacheOptimized}` (real path `types::memory_pool::MemoryPool`, no `CacheOptimized`/
  `CacheOptimizedAccumulator` anywhere), and the `quick_dataset!`/`define_ml_float_bounds!`/
  `estimator_test_suite!` macro invocations (real macros exist and work, but with different field
  names/arities than shown — see corrected `README.md`).

## Completed This Session
- [x] GPU backend foundation — DONE: added `sklears_core::gpu::{GpuBackend, GpuArray, GpuMatrixOps}`, wired directly to `oxicuda-driver` / `oxicuda-blas`, replacing the previous CPU-only stubs; `GpuBackend::detect()` gracefully returns `Ok(None)` when no GPU is present instead of faking a backend.
- [x] Re-enable `trait_explorer::graph_visualization` — DONE: module previously never compiled due to a raw-string-literal bug; now provides real graph algorithms (hub/bridge/bottleneck node detection, Newman modularity calculation, small-world coefficient).
- [x] `api_reference_generator` module — DONE: new module added.
- [x] Fix `trait_explorer::security_analysis` doc examples — DONE: examples were using incorrect `crate::` paths, now correctly use `sklears_core::` external paths.
- [x] 863 tests passing in this crate (`cargo nextest run -p sklears-core --all-features`, verified 2026-07-11; previously recorded as 855).

## OxiCUDA Migration (v0.2.0)

Goal: zero sklears-side usage of the SciRS2-Core crate's GPU submodule. This crate used to host the only manifest-level enablement of that upstream crate's `gpu` feature in the workspace (Phase 1 of the migration — blocking for all later phases), plus stale comments and a broken DSL codegen snippet.

- [x] (M) Remove the SciRS2 GPU-reporting opt-in feature; port trait-graph GPU reporting to the oxicuda-backed `crate::gpu` — DONE (2026-07-06): deleted that feature (which had enabled the upstream crate's `gpu` feature) and its comment block from `Cargo.toml`. In `src/trait_explorer/graph_visualization/graph_generator.rs` replaced the upstream `GpuBackend`/`GpuContext` import with `#[cfg(feature = "gpu_support")] use crate::gpu::GpuBackend`; `gpu_context` field is now `Option<crate::gpu::GpuBackend>`; `new()` now calls `crate::gpu::GpuBackend::detect()?` instead of the upstream `preferred()` + `GpuContext::new()` (the CPU-context fallback collapsed to storing `None`); `is_gpu_accelerated()` is now `gpu_context.as_ref().map(|b| b.is_gpu()).unwrap_or(false)`; all cfg sites, the `Debug` consumer, and the `PerformanceMetrics` producers now gate on `gpu_support` instead of the removed feature. Verified both cfg polarities build warning-free (`cargo check -p sklears-core`, `--features gpu_support`, `--all-features`) and all 14 `graph_generator` tests (incl. `test_gpu_acceleration_reporting_without_gpu`, which explicitly sets `enable_gpu: false` so it is honest rather than hardcoded) pass under both feature states on this no-GPU macOS host.
- [x] (S) Rewrite stale upstream-GPU-module comments in `src/types/advanced_numeric.rs` — DONE (2026-07-06): rewrote the two stale comment blocks to point at `crate::gpu::GpuBackend::is_available()` / `GpuBackend::detect()...device_id()` under feature `gpu_support`. Also did the optional stretch: `GpuFloat::gpu_available()` now calls `crate::gpu::GpuBackend::is_available()` and `GpuFloat::preferred_device()` now calls `crate::gpu::GpuBackend::detect().ok().flatten().map(|b| b.device_id())` under `cfg(feature = "gpu_support")`, falling back to `false`/`None` when the feature is off; `gpu_elementwise_op`/`gpu_matrix_mul`/`gpu_reduce_sum` CPU fallbacks were left untouched (out of scope). `advanced_numeric` tests pass under `--features gpu_support`.
- [x] (S) Fix DSL code-generator GPU snippet to emit the real oxicuda-backed API — DONE (2026-07-06): the `quote!` template at `src/dsl_impl/code_generators.rs` now emits `#[cfg(feature = "gpu_support")] { let _gpu = sklears_core::gpu::GpuBackend::detect()?; }` instead of the nonexistent `crate::gpu::initialize_gpu_context()` under the wrong `"gpu"` feature name.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
