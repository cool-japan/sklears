# TODO - v0.2.0

## Current Status
This crate is part of the sklears v0.2.0 release line (initially shipped in v0.1.0).

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: partial. GPU compute already routes through `sklears_core::gpu`
(OxiCUDA GEMM) and `oxicuda-manifold` (HNSW build/search), but device detection is
still mocked and the backend surface pretends to be multi-backend.

- [x] (M) Replace mock GPU device detection with real oxicuda-driver queries —
  `detect_gpu_devices` now dispatches to `detect_cuda_device`, which uses
  `sklears_core::gpu::GpuContext::detect()`/`with_device_id()` for presence,
  `context.memory_info()` for live free/total device memory, and
  `oxicuda_driver::Device::{name, multiprocessor_count, max_threads_per_block}`
  (read off `ctx.context().device()`) for the device name/compute-unit/
  work-group fields — no more fabricated "(Mock)" devices or
  `cfg!(target_os = ...)` guessing. Non-gpu builds (and non-gpu feature
  builds) honestly return `Ok(None)` for the `Cuda` backend; the
  `CpuFallback` entry now reports the real logical-core count via
  `std::thread::available_parallelism` instead of a hardcoded `8`/`16GB`.
  The "no GPU => `Ok(None)`, not `Err`" contract is preserved.
- [x] (M) Collapse decorative Cuda/OpenCl/Metal backend enum to the OxiCUDA
  reality — `GpuBackend` is now `{Cuda, CpuFallback}`. Deleted the
  `compute_opencl_distances`/`compute_metal_distances` copy-paste wrappers
  (both dispatched to the identical `dispatch_gpu_distances` path `Cuda`
  used), updated `GpuConfig::default`, the `pairwise_distances` match, the
  `lib.rs` re-export list, and the tests that referenced `OpenCl`/`Metal`.
  Public API break, as expected for 0.2.0.
- [x] (S) Prune unused `dep:oxicuda-backend` from the `gpu` feature — removed
  from `Cargo.toml`; `gpu = ["dep:oxicuda-manifold", "sklears-core/gpu_support"]`.
  The device-detection rework above only needed inherent methods on the
  `oxicuda_driver::Device` already reachable through
  `sklears_core::gpu::GpuContext`'s `context()` accessor, so no new
  dependency was required either.
- [x] (S) Align stale multi-backend docs and dead config knobs with the oxicuda
  implementation — rewrote the module doc comment to describe the real
  architecture (OxiCUDA GEMM via `sklears_core::gpu` for pairwise distances,
  `oxicuda-manifold` HNSW for approximate k-NN, CPU fallback, no OpenCL/Metal
  path ever existed). Removed the dead `GpuMemoryStrategy` enum and the
  `memory_strategy`/`max_memory_usage`/`enable_async` `GpuConfig` fields,
  none of which were ever consulted by any code in this module; `batch_size`
  (used by `batch_pairwise_distances`'s real tiling) is the one knob that
  was actually implemented, and is documented as such.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
