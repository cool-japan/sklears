# sklears-preprocessing TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod gpu_acceleration` — DONE (2026-06-21): `scirs2_core::memory::BufferPool` exists in scirs2-core 0.5.0; re-enabled.
- [x] Re-enable `pub mod lazy_evaluation` — DONE (2026-06-21): BufferPool available; re-enabled.
- [x] Re-enable `pub mod memory_management` — DONE (2026-06-21): BufferPool available; re-enabled.

## Source-level TODOs

- [x] src/robust_preprocessing.rs:366 — Implement fit/transform for OutlierAwareImputer (already implemented; fixed NaN propagation in RobustScaler::fit)
- [x] src/robust_preprocessing.rs:457 — Implement proper RobustScaler with these methods (already implemented; root fix was filtering non-finite values before quantile computation)
- [x] src/robust_preprocessing.rs:459 — Implement fit for RobustScaler (resolved as part of NaN fix)
- [x] src/pipeline.rs:1038 — Fix this test - requires properly fitted transformers (added is_fitted() to ParallelBranches; uncommented test)
- [x] src/pipeline.rs:1058 — Fix this test - requires properly fitted transformers (added len()/is_empty() to AdvancedPipeline; uncommented test)
- [x] src/pipeline.rs:1071 — Fix this test - requires properly fitted transformers (DynamicPipeline already had len()/is_empty(); uncommented test)

## Phase C-4

- [x] Phase C-4: Removed all 20 blanket #![allow(...)] suppressors; 300 tests pass, 0 warnings

## OxiCUDA Migration (v0.2.0)

This crate is the last code-real `scirs2_core::gpu` consumer in the workspace (it compiles today only because scirs2-optimize/scirs2-sparse transitively enable `scirs2-core/gpu`). Swap it to the Wave A2 oxicuda-backed `sklears_core::gpu` module. Reference pattern: the completed sklears-discriminant-analysis migration.

- [x] (M) Port `src/gpu_acceleration.rs` off `scirs2_core::gpu::GpuBackend` — DONE (2026-07-06): removed `use scirs2_core::gpu::GpuBackend as ScirGpuBackend` and the `from_scir_backend`/`to_scir_backend` conversion pair. Kept the local serde-derived six-variant `GpuBackend` enum intact; only `Cuda` has a real detection path (via `sklears_core::gpu::GpuBackend::is_available`/`detect`, behind `cfg(feature = "gpu")`), the other four variants are hard-wired to `is_available() == false`. `Default`/`is_available()` reimplemented on top of that. `GpuContextManager` reworked: `backend_kind: GpuBackend` (always present) plus a `#[cfg(feature = "gpu")] gpu_backend: Option<sklears_core::gpu::GpuBackend>` live handle populated via `detect()` when `backend_kind == Cuda`; `is_gpu_available()` is `backend_kind != Cpu`. `should_use_gpu()` and the `scirs2_core::memory::BufferPool` field are unchanged. Module docs/doc-example updated to name OxiCUDA; public type names/re-exports in `src/lib.rs` unchanged.
- [x] (S) Add `gpu = ["sklears-core/gpu_support"]` to `Cargo.toml` `[features]` — DONE (2026-07-06): matches sklears-cross-decomposition's wiring (no direct `oxicuda-*` deps needed in this crate's `Cargo.toml`; `sklears_core::gpu` hides those behind `sklears-core/gpu_support`). No `scirs2-core/gpu` enablement added. Detection code gated with `cfg(feature = "gpu")` with an honest `false`/`Cpu` fallback when off.
- [ ] (L) Implement real oxicuda compute kernels for `GpuStandardScaler`/`GpuMinMaxScaler` (deferred 2026-07-06: out of scope for this pass — the acceptance bar for the 0.2.0 migration was eliminating the `scirs2_core::gpu` build dependency, which is done; `dispatch_compute_mean`/`dispatch_compute_variance`, `transform_gpu`, and `compute_min_max_gpu` remain documented CPU passthroughs behind the (now real) OxiCUDA detection layer). Follow-up: upload via `sklears_core::gpu::GpuArray` (oxicuda-memory DeviceBuffer); per-feature mean/variance and min/max reductions plus affine `(x - mean)/std` and `(x - min) * scale` maps via oxicuda-blas level-1 ops or oxicuda-launch/oxicuda-ptx kernels; wire `GpuConfig.stream_count`/`block_size` to real launch params and `GpuPerformanceStats` to real timings. File: `src/gpu_acceleration.rs`.
- [x] (S) Docs and disabled-example cleanup after the backend swap — DONE (2026-07-06): `README.md` now names OxiCUDA and the `gpu` feature in both the hardware-acceleration bullet and the roadmap note. `examples/preprocessing_pipeline_demo.rs.disabled:131` still calls `GpuContextManager::new(gpu_config)?`, whose signature (`Result<Self>`) is unchanged by this migration, so no edit was needed there.

---

See also: [Workspace roadmap](../../TODO.md)
