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
- [x] (L) Implement real oxicuda compute kernels for `GpuStandardScaler`/`GpuMinMaxScaler` — DONE (2026-07-07): added direct optional deps `oxicuda-blas`/`oxicuda-memory`/`oxicuda-driver` (`{ workspace = true, optional = true }`) wired into `gpu = ["sklears-core/gpu_support", "dep:oxicuda-blas", "dep:oxicuda-memory", "dep:oxicuda-driver"]`, plus a non-optional `log` dep. `dispatch_compute_mean`/`dispatch_compute_min`/`dispatch_compute_max` now call `oxicuda_blas::reduction::reduce_axis` (`ReductionOp::Mean`/`Min`/`Max`) via a shared `gpu_reduce_axis_to_row` helper; `dispatch_compute_variance` computes `E[x²]` on-device (`elementwise::mul` + `reduce_axis(Mean)`) and combines with the host-side `mean` via `var_sample = (E[x²] - mean²) * n/(n-1)` (rescaling population→sample variance). Both scalers' `transform_gpu` now run a real 3-5 kernel on-device pipeline (`bias_add` for the broadcast subtract, `broadcast_axes` to expand the per-feature vector across rows, `mul`/`div`, and `fill`+`bias_add` for the min-max scaler's constant `feature_range.0` shift) instead of calling `transform_cpu`. Every dispatch site falls back to the existing CPU reference (refactored into small `mean_cpu`/`variance_cpu`/`min_cpu`/`max_cpu` helpers, deduplicating the old direct-CPU and GPU-fallback code paths) with `log::warn!` on any GPU-path error, and honestly skips the GPU attempt entirely under `#[cfg(not(feature = "gpu"))]` or when `ctx.oxicuda_backend()` is `None`. `GpuPerformanceStats::record_gpu`/`record_cpu_fallback` are real now (running-average timings, byte-accounted `gpu_memory_used`); `GpuContextManager` gained a `Mutex<GpuPerformanceStats>` + `performance_stats()` snapshot accessor, and both fitted scalers expose `performance_stats() -> Option<GpuPerformanceStats>`. Verified: `cargo build`/`clippy --all-targets -D warnings` clean both with and without `--features gpu`; `cargo nextest run --all-features` 377/377 passing including 6 new tests (dispatch-level CPU-fallback stats recording for mean/variance and min/max, `transform_gpu` vs `transform_cpu` numeric equivalence for both scalers, and an explicit `should_use_gpu()==false` honesty check). No `unwrap()` added; file is 1633 lines (well under the 2000-line policy limit).
  - **Goal:** Replace the four CPU-passthrough placeholders with real on-device reductions via oxicuda-blas, with honest CPU fallback and real `GpuPerformanceStats`. Must build + pass tests with and without the `gpu` feature; no fabricated availability.
  - **Design (mirror sklears-linear's `gpu_acceleration.rs` upload→op→download + stats+fallback pattern):**
    - Add direct deps `oxicuda-blas`/`oxicuda-memory`/`oxicuda-driver` (`{ workspace = true, optional = true }`) wired into `gpu = [...]`. Obtain `&BlasHandle` from `ctx.oxicuda_backend()?.blas()`; call `backend.context().set_current()` before uploads. Use `oxicuda_memory::DeviceBuffer::from_host` (not `sklears_core::gpu::GpuArray` — its buffer is private, inaccessible cross-crate; sklears-core re-exports no oxicuda crate).
    - Per-feature mean/min/max → one `oxicuda_blas::reduction::reduce_axis(handle, ReductionOp::{Mean,Min,Max}, outer=1, axis_len=n_samples, inner=n_features, &d_x, &mut d_out)` each (row-major `[n_samples, n_features]`).
    - Per-feature variance (no per-axis variance op exists): `reduce_axis(Sum)` on the elementwise-squared buffer (`elementwise::mul(x,x)`), then `E[x²] − mean²`, then ×`n/(n−1)` to preserve the current sample-variance numerics (`reduction::variance` is population ÷n and scalar-only — not a drop-in).
    - `transform_gpu` (both scalers) via elementwise `sub`/`div`/`scale`/`bias_add`, or keep the affine map on host after the reductions if per-feature broadcast is awkward — the reduction is the hot path.
    - Real stats: add `record_gpu(elapsed, bytes)`/`record_cpu_fallback()` to `GpuPerformanceStats` and call them on the GPU/fallback branches (currently inert). CPU-fallback-on-error with `log::warn!`, like sklears-linear.
    - cfg-gate: `#[cfg(feature="gpu")]` real bodies + `#[cfg(not(feature="gpu"))]` CPU fallbacks; both build configs must stay warning-free.
  - **Files:** `src/gpu_acceleration.rs`, `Cargo.toml`.
  - **Prerequisites:** none beyond the dep additions.
  - **Tests:** numeric-equivalence tests that run on the no-GPU host via the CPU path (feature on/off yield identical mean/std/min/max/transform); assert `should_use_gpu()==false` on no-GPU host. GPU kernel code compiled under `--all-features` (clippy-verified), not executed on this host.
  - **Risk:** variance denominator mismatch (population vs sample) — explicitly rescaled. Warning hygiene across gpu/non-gpu cfg. No `unwrap()` in src/.
- [x] (S) Docs and disabled-example cleanup after the backend swap — DONE (2026-07-06): `README.md` now names OxiCUDA and the `gpu` feature in both the hardware-acceleration bullet and the roadmap note. `examples/preprocessing_pipeline_demo.rs.disabled:131` still calls `GpuContextManager::new(gpu_config)?`, whose signature (`Result<Self>`) is unchanged by this migration, so no edit was needed there.

---

See also: [Workspace roadmap](../../TODO.md)
