# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status (2026-07-06): done except one explicitly deferred sub-item.
`gpu_computing` no longer fabricates any GPU hardware; device enumeration and
the core array ops (`add`/`mul`/`matmul`) route through the oxicuda-backed
`sklears_core::gpu` module (plus a direct optional `oxicuda-driver` dep for
device metadata `sklears_core::gpu` doesn't expose, e.g. SM count) behind a
new `gpu` feature. Default build stays CPU-only Pure Rust; `cargo check -p
sklears-utils` and `--features gpu` both pass warning-free, and all 491 tests
(29 of them in `gpu_computing`) pass in both configurations.

- [x] (S) Decide `gpu_computing.rs` strategy: thin layer over `sklears_core::gpu` (preferred) or direct oxicuda deps. Implemented the preferred thin-layer approach: `Cargo.toml` adds `gpu = ["sklears-core/gpu_support", "dep:oxicuda-driver"]` (the direct `oxicuda-driver` dep is only for device metadata -- SM count, clock rate, memory bus width -- that `sklears_core::gpu::GpuDeviceProperties` doesn't expose; all actual GPU *compute* goes through `sklears_core::gpu::{GpuBackend, GpuArray, GpuMatrixOps}`). `gpu_computing` module docs now state the honesty contract explicitly. Files: `Cargo.toml`, `src/gpu_computing.rs`, `src/lib.rs` (no lib.rs changes were needed -- the public re-export list at `src/lib.rs:250-252` still resolves, `GpuDevice`'s field set changed but the type name didn't).
- [x] (M) Replace fabricated GPU device list with real oxicuda-driver enumeration. `GpuUtils::init_devices` now calls `enumerate_real_devices()`, which (behind feature `gpu`) walks `oxicuda_driver::Device::count()`/`Device::get(i).info()` for real `name`/`total_memory`/`free_memory`/`compute_capability`/`multiprocessor_count` (SM count, not the old fake "cores")/`clock_rate_mhz`, plus a real-hardware-derived `memory_bandwidth_bytes_per_sec` (`2 * mem_clock_hz * bus_width_bytes`, the standard deviceQuery formula -- documented as a derived estimate, not a measurement). Without the `gpu` feature, or with no CUDA driver/device (this dev machine), it returns an empty `Vec`, never fake hardware. Dropped the unfillable `is_integrated` field (no such flag in `oxicuda-driver`'s `DeviceInfo`) rather than guess at it; `get_best_device` no longer filters on it. Files: `src/gpu_computing.rs`.
- [x] (M, was scoped (L)) Back `GpuArrayOps::{add_arrays,multiply_arrays,matrix_multiply}` with oxicuda, keep CPU fallback. Each now tries `sklears_core::gpu::{GpuBackend::with_device_id, GpuArray, GpuMatrixOps}` (real oxicuda-blas GEMM/elementwise add/mul) first; `Ok(None)` from the GPU helper (no such device / `gpu` feature off) transparently falls back to the existing CPU code, matching the `sklears-svm` `DeviceNotAvailable`-skip pattern. A genuine GPU-side error (as opposed to "no GPU here") is surfaced as `GpuError::InitializationFailed`, not silently swallowed.
  - (deferred 2026-07-06: `apply_activation`/`reduce_sum`/`reduce_max` stay CPU-only -- there is no `oxicuda-primitives` elementwise/reduction kernel to wire them to yet, and building one is a materially larger task than this pass. `execute_kernel`'s `thread::sleep(1ms)` timing mock, the "optimized executor" variant, and the rest of the scheduling scaffolding (`MultiGpuCoordinator`, `AsyncGpuOps`, `GpuOptimizationAdvisor`, `GpuMemoryPool`) are unchanged: they have no real kernel payload attached anywhere in this abstraction (kernels are identified only by a name string + grid/block metadata, with no dispatch table), so there is nothing concrete to back with a device launch without a larger redesign. Doc comments were updated to say plainly that these remain CPU-side bookkeeping rather than implying real device execution.) The new doc comments/helpers pushed the file to 2110 lines, over the 2000-line policy limit, so it was split with `splitrs --split-test-modules` into `src/gpu_computing/mod.rs` (1323 lines, production code) and `src/gpu_computing/tests.rs` (455 lines); `src/gpu_computing.rs` no longer exists as a single file. `pub mod gpu_computing;` in `src/lib.rs` needed no change (directory modules resolve the same way).
- [x] (S) Record `distributed_computing` GPU fields as out of scope. Added module-doc notes (2026-07-06) to `src/distributed_computing/types.rs` and `src/distributed_computing/functions.rs` stating that `gpu_count`/`gpu_usage`/`min_gpu_count`/`gpu_time`/`total_gpu_count` are pure scheduling/capacity metadata with no GPU API calls, so future GPU audits don't re-flag them. Files: `src/distributed_computing/types.rs`, `src/distributed_computing/functions.rs`, `TODO.md`.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
