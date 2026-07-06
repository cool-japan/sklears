# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: simulated-gpu. The `gpu_computing` module currently simulates GPU
hardware; for 0.2.0 all GPU claims must route through the oxicuda-backed
`sklears_core::gpu` module (Wave A2) or honestly report absence. Default build
stays CPU-only Pure Rust.

- [ ] (S) Decide `gpu_computing.rs` strategy: thin layer over `sklears_core::gpu` (preferred) or direct oxicuda deps. Preferred: add `gpu = ["sklears-core/gpu_support"]` and make `gpu_computing` a utility layer over `sklears_core::gpu` rather than a second GPU stack; alternative is optional direct `oxicuda-driver`/`oxicuda-blas`/`oxicuda-memory` deps. Note the module is compiled unconditionally today (`src/lib.rs:42`; re-exports at `src/lib.rs:250-252` are semver-visible — 0.2.0 permits breakage). Files: `Cargo.toml`, `src/gpu_computing.rs`, `src/lib.rs`
- [ ] (M) Replace fabricated GPU device list with real oxicuda-driver enumeration. `GpuUtils::init_devices` (`src/gpu_computing.rs:66-96`) fabricates "NVIDIA GeForce RTX 3080" and "Intel UHD Graphics 770" devices — actively misleading in logs/reports. Behind feature `gpu`, enumerate via oxicuda-driver (count, name, total/free memory, compute capability, SM count); without it, return an empty list or a labeled `CpuFallback` entry, never fake hardware. Files: `src/gpu_computing.rs`
- [ ] (L) Back `GpuArrayOps` and kernel execution with oxicuda, keep CPU fallback. `matrix_multiply` (`src/gpu_computing.rs:370-394`) -> oxicuda-blas SGEMM under feature `gpu`; `add_arrays`/`multiply_arrays`/`apply_activation`/`reduce_sum`/`reduce_max` (344-423 ff.) -> oxicuda-primitives or oxicuda-launch kernels; replace `execute_kernel`'s `thread::sleep(1ms)` timing mock (188-216) and the optimized-executor variant (~907-1050) with real stream launch + event timing so `GpuKernelExecution.execution_time`/occupancy reflect actual work. Retain CPU implementations as the non-`gpu` path, dropping "Mock GPU computation" labels. File is 1850 lines — plan a splitrs split (device/, array_ops/, profiler/, cluster/) if oxicuda wiring grows it past 2000. Deferable past 0.2.0. Files: `src/gpu_computing.rs`
- [ ] (S) Record `distributed_computing` GPU fields as out of scope. `gpu_count`/`gpu_usage`/`min_gpu_count`/`gpu_time` in `src/distributed_computing/types.rs` (lines 301, 367, 574, 708, 740, 749, 1048, 1084, 1146, 1168) and `src/distributed_computing/functions.rs` (lines 25, 35, 53, 129, 419, 533) are pure scheduling/capacity metadata with no GPU API calls — noted here so future GPU audits do not re-flag them. Files: `src/distributed_computing/types.rs`, `src/distributed_computing/functions.rs`, `TODO.md`

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
