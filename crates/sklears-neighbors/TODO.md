# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: partial. GPU compute already routes through `sklears_core::gpu`
(OxiCUDA GEMM) and `oxicuda-manifold` (HNSW build/search), but device detection is
still mocked and the backend surface pretends to be multi-backend.

- [ ] (M) Replace mock GPU device detection with real oxicuda-driver queries —
  `detect_gpu_devices` (`src/gpu_distance.rs:148-208`) fabricates `GpuDeviceInfo`
  entries ("NVIDIA GPU (Mock)", "OpenCL Device (Mock)", "Apple GPU (Mock)" with
  fixed memory/compute units); `is_cuda_available` (211-215) keys off
  `cfg!(target_os)`, `is_opencl_available` (218-221) unconditionally returns
  `true`, `is_metal_available` (224-227) likewise guesses. Replace with
  `sklears_core::gpu::GpuContext::detect()`/`with_device_id()` for presence and
  `context.memory_info()` plus oxicuda-driver device-attribute queries for
  name/memory/compute units (pattern: sklears-clustering `gpu_distances.rs:193`).
  Non-gpu builds report only the honest `CpuFallback` entry; keep the
  "no GPU => `Ok(None)`, not `Err`" contract (388-392).
- [ ] (M) Collapse decorative Cuda/OpenCl/Metal backend enum to the OxiCUDA
  reality — `GpuBackend::{Cuda, OpenCl, Metal}` (`src/gpu_distance.rs:21-30`) all
  dispatch to the identical oxicuda GEMM path (`compute_cuda/opencl/metal_distances`
  at 320-353 are copy-paste wrappers over `dispatch_gpu_distances` 356-380).
  Rework to `{Cuda, CpuFallback}` (or `OxiCuda`/`CpuFallback`), delete duplicate
  wrappers, update `GpuConfig::default`, `backend_distribution` stats keying (94),
  re-exports (`src/lib.rs:131-134`), and tests using the `OpenCl` variant
  (884-893, 1094-1117). Public API break — appropriate for 0.2.0.
- [ ] (S) Prune unused `dep:oxicuda-backend` from the `gpu` feature —
  `Cargo.toml:30` declares `oxicuda-backend` optional and `:53` enables it in
  `gpu = [dep:oxicuda-backend, dep:oxicuda-manifold, sklears-core/gpu_support]`,
  but there is zero `oxicuda_backend` usage in `src/` — GPU compute goes through
  `sklears-core/gpu_support` and `oxicuda-manifold` (`hnsw_build`/`hnsw_search`
  at 736, 743). Drop it, leaving
  `gpu = ["dep:oxicuda-manifold", "sklears-core/gpu_support"]`, unless the
  detection item ends up needing oxicuda-driver directly (then declare exactly
  what is used).
- [ ] (S) Align stale multi-backend docs and dead config knobs with the oxicuda
  implementation — module docs (`src/gpu_distance.rs:1-5`) claim "CUDA, OpenCL,
  and Metal" backends; `GpuMemoryStrategy`/`GpuConfig` fields
  `memory_strategy`/`max_memory_usage`/`enable_async` (43-63) are accepted but
  never consulted. During the enum rework, describe the real architecture
  (OxiCUDA GEMM via `sklears_core::gpu` + oxicuda-manifold HNSW, CPU fallback)
  and implement or remove the dead knobs.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
