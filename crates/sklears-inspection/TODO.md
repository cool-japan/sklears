# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## OxiCUDA Migration (v0.2.0)

Current migration status: **rewired to `sklears_core::gpu`** — the `gpu` module now wraps the oxicuda-backed `GpuBackend`/`GpuArray`/`GpuUtils` types (behind `gpu_support`) instead of shipping null-pointer placeholders, and the `gpu` Cargo feature actually gates the module.

- [x] (S) Rewire the dead `gpu` feature from `dep:tokio` to sklears-core `gpu_support` (`Cargo.toml`, `src/lib.rs`)
  - Replaced `gpu = ["dep:tokio"]` with `gpu = ["sklears-core/gpu_support"]` and dropped the now-dead optional `tokio` dependency (tests still use the unconditional dev-dependency).
  - Gated `pub mod gpu` and the gpu re-exports behind `#[cfg(feature = "gpu")]` in `src/lib.rs` so the module (and its types) no longer compile into default builds.
- [x] (M) Replace placeholder `GpuContext`/`GpuBuffer`/`GpuDevice` with `sklears_core::gpu` (`src/gpu.rs`)
  - Deleted the null-pointer `GpuBuffer<T>`, the fake `detect_cuda/opencl/metal_devices` stack, `utils::check_*_available` (hardcoded `false`), and the `unsafe impl Send/Sync` over null-pointer types.
  - Rebuilt `GpuContext` as a thin wrapper over `Option<sklears_core::gpu::GpuBackend>` (honest `None` on `detect()` failure), `GpuBuffer<T>` as a wrapper over `GpuArray<T>`, and `GpuDevice` as a type alias for `GpuDeviceProperties`; device enumeration goes through `GpuUtils::device_count()` / `device_properties()`.
  - Dropped the OpenCL/Metal enum variants entirely (OxiCUDA is CUDA-first) and updated the module doc accordingly.
- [ ] (L) GPU-stage SHAP / permutation-importance perturbation matrices where feasible (`src/gpu.rs`) — optional follow-up, not required for 0.2.0
  - (deferred 2026-07-06: `compute_shap_parallel` / `compute_permutation_importance` take a host `predict_fn` closure, so full on-GPU evaluation is impossible generically, and building the batched-linear-model GEMM fast path is a genuinely separate, larger design (deciding when a closure is "linear enough" to dispatch through `oxicuda-blas` GEMM safely) rather than a mechanical port. Left as CPU-only for this pass; the existing CPU estimators are correct and unchanged, with ChaCha8Rng-seeded reproducibility preserved.)

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
