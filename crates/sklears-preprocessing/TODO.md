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

- [ ] (M) Port `src/gpu_acceleration.rs` off `scirs2_core::gpu::GpuBackend`: delete `use scirs2_core::gpu::GpuBackend as ScirGpuBackend` (line 51) and the `from_scir_backend`/`to_scir_backend` conversion pair (lines 88-111). Keep the local serde-derived six-variant `GpuBackend` enum intact (Serialize/Deserialize makes variant changes format-visible; scirs2 OpenCL/Wgpu have no oxicuda BackendKind equivalent) and hard-wire non-CUDA variants' `is_available()` to false. Re-implement `Default` (82-86) and `is_available()` (114-116) on `sklears_core::gpu::GpuBackend::detect()`/`is_available()` under `cfg(feature = "gpu")`, returning Cpu/false when the feature is off. Rework `GpuContextManager` (206-244): replace `backend: ScirGpuBackend` (207, fallback at 219) with `backend: Option<sklears_core::gpu::GpuBackend>` from `detect()`; `is_gpu_available()` (233) becomes `backend.is_some()`; keep `should_use_gpu()` threshold logic and the `scirs2_core::memory::BufferPool` field (memory_management feature, not the gpu module — out of scope). Update module docs (1-15) and the doc-example (28-49) to name OxiCUDA; keep public type names so `src/lib.rs` re-exports (99-102, 269-272) stay valid.
- [ ] (S) Add `gpu = ["sklears-core/gpu_support"]` to `Cargo.toml` `[features]` (crate currently has no gpu feature; match sklears-cross-decomposition's wiring — `gpu_support` pulls `dep:oxicuda-backend/-memory/-blas/-driver` via sklears-core). Do NOT add any `scirs2-core/gpu` enablement. Because `sklears_core::gpu` only exists under `gpu_support`, gate the new detection code internally with `cfg(feature = "gpu")` and a CPU-only fallback when off — preserving the unconditional public module/re-exports while eliminating the accidental reliance on transitive `scirs2-core/gpu` activation. Files: `Cargo.toml`, `src/gpu_acceleration.rs`.
- [ ] (L) Implement real oxicuda compute kernels for `GpuStandardScaler`/`GpuMinMaxScaler` (follow-up; optional for the 0.2.0 scirs2-removal goal): `dispatch_compute_mean`/`dispatch_compute_variance` (335-362), `transform_gpu` (410-413, 588-591), and `compute_min_max_gpu` (522-524) are documented CPU passthroughs. After the detection swap, add device paths: upload via `sklears_core::gpu::GpuArray` (oxicuda-memory DeviceBuffer); per-feature mean/variance and min/max reductions plus affine `(x - mean)/std` and `(x - min) * scale` maps via oxicuda-blas level-1 ops or oxicuda-launch/oxicuda-ptx kernels; wire `GpuConfig.stream_count`/`block_size` (131-134) to real launch params and `GpuPerformanceStats` (648-699) to real timings. File: `src/gpu_acceleration.rs`.
- [ ] (S) Docs and disabled-example cleanup after the backend swap: `README.md:19` ("optional GPU support") and `README.md:42` (GPU categorical encoders roadmap) should name OxiCUDA and the new `gpu` feature; `examples/preprocessing_pipeline_demo.rs.disabled:131` constructs `GpuContextManager` and must match the new Option-based constructor semantics if ever re-enabled. Files: `README.md`, `TODO.md`, `examples/preprocessing_pipeline_demo.rs.disabled`.

---

See also: [Workspace roadmap](../../TODO.md)
