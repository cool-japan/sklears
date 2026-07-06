# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Completed This Session
- [x] GPU backend foundation — DONE: added `sklears_core::gpu::{GpuBackend, GpuArray, GpuMatrixOps}`, wired directly to `oxicuda-driver` / `oxicuda-blas`, replacing the previous CPU-only stubs; `GpuBackend::detect()` gracefully returns `Ok(None)` when no GPU is present instead of faking a backend.
- [x] Re-enable `trait_explorer::graph_visualization` — DONE: module previously never compiled due to a raw-string-literal bug; now provides real graph algorithms (hub/bridge/bottleneck node detection, Newman modularity calculation, small-world coefficient).
- [x] `api_reference_generator` module — DONE: new module added.
- [x] Fix `trait_explorer::security_analysis` doc examples — DONE: examples were using incorrect `crate::` paths, now correctly use `sklears_core::` external paths.
- [x] 855 tests passing in this crate.

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
