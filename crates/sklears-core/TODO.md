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

Goal: zero sklears-side usage of `scirs2_core::gpu`. This crate hosts the only manifest-level `scirs2-core/gpu` enablement in the workspace (Phase 1 of the migration — blocking for all later phases), plus stale comments and a broken DSL codegen snippet.

- [ ] (M) Remove `scirs2-gpu-reporting` feature; port trait-graph GPU reporting to the oxicuda-backed `crate::gpu` — delete `scirs2-gpu-reporting = ["scirs2-core/gpu"]` (`Cargo.toml:110`, plus comment block 107-109; verified as the ONLY `scirs2-core/gpu` enablement in any workspace manifest, enabled by no crate or default). In `src/trait_explorer/graph_visualization/graph_generator.rs` replace `use scirs2_core::gpu::{GpuBackend, GpuContext}` (line 17) with `crate::gpu::GpuBackend` gated on the existing `gpu_support` feature; change field `gpu_context: Option<GpuContext>` (lines 35-36) to `Option<crate::gpu::GpuBackend>`; in `new()` (110-126) replace `GpuBackend::preferred()` + `GpuContext::new()` with `crate::gpu::GpuBackend::detect()` (`Result<Option<Self>>`; `None` = no accelerator, so the explicit CPU-context fallback collapses to storing `None`); rewrite `is_gpu_accelerated()` (1007-1013) as `gpu_context.as_ref().map(|b| b.is_gpu()).unwrap_or(false)`; update all 8 cfg sites (16, 35, 110, 147, 1007, 1019, 1027, 1035), the `Debug` consumer (93), `PerformanceMetrics` producers (322, 470), and the test asserting `!has_gpu_acceleration()` (1677). Verify both cfg polarities build warning-free. Behavior note: scirs2 `preferred()` always returned `Cpu` in production, so `gpu_accelerated = true` becomes genuinely possible on CUDA hosts for the first time.
- [ ] (S) Rewrite stale `scirs2_core::gpu` comments in `src/types/advanced_numeric.rs` — comment-only: lines 482-484 and 488-491 recommend future wiring to `scirs2_core::gpu::is_available()` / `preferred_device()`, free functions that never existed in scirs2-core 0.6.0 and must never be adopted. Rewrite to point at `crate::gpu::GpuBackend::is_available()` / `GpuBackend::detect().map(|b| b.device_id())` under feature `gpu_support`. Optional stretch: actually wire `GpuFloat::gpu_available` / `preferred_device` to those calls under `cfg(feature = "gpu_support")`, keeping CPU fallbacks for `gpu_elementwise_op` / `gpu_matrix_mul` / `gpu_reduce_sum` (495-525).
- [ ] (S) Fix DSL code-generator GPU snippet to emit the real oxicuda-backed API — the `quote!` template at `src/dsl_impl/code_generators.rs:422-429` emits generated code `#[cfg(feature = "gpu")] { crate::gpu::initialize_gpu_context()?; }`, but the feature is `gpu_support` (not `gpu`) and `initialize_gpu_context` does not exist anywhere in `crate::gpu`. Emit `#[cfg(feature = "gpu_support")] { let _gpu = sklears_core::gpu::GpuBackend::detect()?; }` (or `GpuUtils::is_gpu_available()`) so generated code compiles against the oxicuda-backed API.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
