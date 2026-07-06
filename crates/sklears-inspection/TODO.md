# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## OxiCUDA Migration (v0.2.0)

Current migration status: **simulated-gpu** — the `gpu` module ships placeholder types (null-pointer buffers, hardcoded "no device" detection) and the `gpu` Cargo feature does not gate any code. Goal: route all GPU claims through the oxicuda-backed `sklears_core::gpu` module (behind `gpu_support`) or honestly report absence.

- [ ] (S) Rewire the dead `gpu` feature from `dep:tokio` to sklears-core `gpu_support` (`Cargo.toml`, `src/lib.rs`)
  - Replace `gpu = ["dep:tokio"]` (Cargo.toml:59) with `gpu = ["sklears-core/gpu_support"]` — no `#[cfg(feature = "gpu")]` exists anywhere in the crate and the optional tokio dependency is never imported (tests use the dev-dependency).
  - Gate `pub mod gpu` (src/lib.rs:34) and the gpu re-exports (src/lib.rs:175-177) behind the feature so simulated GPU types stop compiling into default builds (semver-visible change, acceptable for 0.2.0).
- [ ] (M) Replace placeholder `GpuContext`/`GpuBuffer`/`GpuDevice` with `sklears_core::gpu` (`src/gpu.rs`)
  - Delete the null-pointer `GpuBuffer<T>` (src/gpu.rs:65-129, incl. `ptr::null_mut` at line 84), the fake detection stack `detect_cuda/opencl/metal_devices` (lines 187-206, all `return Ok(Vec::new())`), `utils::check_*_available` hardcoded to `false` (lines 528-547), and the `unsafe impl Send/Sync` over null-pointer types (lines 128-129, 273-274).
  - Rebuild on `sklears_core::gpu::GpuBackend::detect()` / `GpuContext` / `GpuArray` / `GpuUtils` (device enumeration and memory via `GpuUtils::device_count()` / `device_properties()`).
  - Drop OpenCL/Metal enum variants (OxiCUDA is CUDA-first); soften module doc line 12's "Support for both CUDA and OpenCL backends".
- [ ] (L) GPU-stage SHAP / permutation-importance perturbation matrices where feasible (`src/gpu.rs`) — optional follow-up, not required for 0.2.0
  - `compute_shap_parallel` / `compute_permutation_importance` (src/gpu.rs:317-424) take a host `predict_fn` closure, so full on-GPU evaluation is impossible generically. Scope honestly: build the perturbed batch matrices (lines 389-393, 479-488) as `GpuArray` uploads and use oxicuda-blas GEMM for the batched linear-model fast path.
  - Keep the existing correct CPU estimators as fallback when `detect()` returns `None`; preserve ChaCha8Rng-seeded reproducibility.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
