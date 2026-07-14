# TODO - v0.2.1

## Current Status
This crate is part of the sklears v0.2.1 release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: **done (2026-07-06)**. The `gpu_acceleration` module was rebuilt
on the oxicuda-backed `sklears_core::gpu` API and gated behind a real `gpu`
feature; the fake-eigenvalue correctness bug is fixed.

- [x] (S) Add a real `gpu` Cargo feature and gate the simulated module out of default builds — `gpu = ["sklears-core/gpu_support", "dep:oxicuda-driver"]` (`oxicuda-driver` only, pulled directly for the two `GpuDevice` fields `sklears_core::gpu::GpuUtils::device_properties()` doesn't surface; GEMM/array ops go through `sklears_core::gpu::GpuArray`/`GpuMatrixOps`, no other oxicuda-* crate is needed directly). `pub mod gpu_acceleration` (`src/lib.rs`) and the root re-exports (`GpuBackend`/`GpuConfig`/`GpuContext`/`GpuDevice`/`GpuNystroem`/`GpuRBFSampler`/`GpuProfiler`/`MemoryStrategy`/`Precision`) are now behind `#[cfg(feature = "gpu")]`, off by default. Module doc rewritten to describe what actually changed instead of the old false "CUDA and OpenCL backends" claim. Files: `Cargo.toml`, `src/lib.rs`, `src/gpu_acceleration.rs`
- [x] (M) Rebuild `GpuContext`/`GpuDevice` on `sklears_core::gpu` instead of local simulation — the local `GpuBackend {Cuda, OpenCL, Metal, Cpu}` enum is gone; `GpuBackend` is now a re-export of `sklears_core::gpu::GpuBackend`, and `GpuContext::initialize()` calls `GpuBackend::detect()` honestly (`Ok(None)` on this macOS dev host, no CUDA/OpenCL/Metal fallback pretending to run). `GpuDevice` fields (`compute_capability`, `total_memory`) come from `GpuUtils::device_properties()`; `multiprocessor_count`/`max_threads_per_block` come from a direct `oxicuda_driver::Device::info()` query (`GpuDevice::query`). block_size/grid_size heuristics kept, now driven by real device properties when present. Files: `src/gpu_acceleration.rs`
- [x] (L, partially — see note) `GpuRBFSampler`: the four duplicate `generate_features_cuda`/`_opencl`/`_metal`/`_cpu` RNG loops are consolidated into one `generate_random_features` (always host-side; there is no on-device RNG integration here, `cuRAND` would be a separate follow-up). `transform` now does a real on-device GEMM (`X · Wᵀ` via `GpuArray::matmul`) when a GPU is detected, downloads once, then applies the bias-add + cosine transform on the host (`apply_cosine_features`, shared with the CPU-only path) — no on-device cosine primitive is wired up in this crate. Files: `src/gpu_acceleration.rs`
- [x] (L, partially — see note) `GpuNystroem`: kernel-matrix computation (`compute_kernel_matrix`, shared by `fit`/`transform`) does a real on-device GEMM for the `O(n·m·d)` inner-product term of both `"linear"` and `"rbf"` kernels when a GPU is detected (norm expansion + `exp` for `"rbf"` stay host-side, `O(n·m)`). **The `eigendecomposition_cpu` correctness bug is fixed**: the power-iteration-plus-fabrication placeholder (which invented every eigenvalue past the first as a flat `0.1`) is replaced with a real symmetric eigendecomposition via `scirs2_linalg::compat::eigh`, with a regression test (`test_nystroem_eigendecomposition_is_not_fabricated`) asserting non-leading eigenvalues of a near-rank-1 matrix are not fabricated. No on-device eigensolver is used or claimed: `oxicuda-solver` 0.4.0's symmetric eigensolver is a documented exact-CPU host fallback as of this pass, so `eigh` runs on the CPU unconditionally, GPU feature or not — this is the honest choice, not a shortcut. Files: `src/gpu_acceleration.rs`

Note: the on-device GEMM work above covers the "Deferable past 0.2.0" (L) items
in a genuine-but-reduced form (real GEMM offload for the expensive inner-product
term; elementwise transforms stay host-side pending an on-device cos/exp
primitive). Nothing further is deferred: the mandatory
`eigendecomposition_cpu` correctness fix landed and is not gated behind any
feature.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
