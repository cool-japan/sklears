# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: simulated-gpu. The `gpu_acceleration` module currently ships a
simulated GPU layer (CPU loops behind CUDA/OpenCL/Metal-named methods) in default
builds. Phase 4 of the workspace plan wires it to the oxicuda-backed
`sklears_core::gpu` API (Wave A2) or honestly downscopes it.

- [ ] (S) Add a real `gpu` Cargo feature and gate the simulated module out of default builds — introduce `gpu = ["sklears-core/gpu_support", "dep:oxicuda-blas", "dep:oxicuda-solver"]` (both pinned 0.4.0 at workspace root); gate `pub mod gpu_acceleration` (`src/lib.rs:29`) and the root re-exports (`src/lib.rs:149-152`: `GpuBackend`/`GpuConfig`/`GpuContext`/`GpuDevice`/`GpuNystroem`/`GpuRBFSampler`/`GpuProfiler` etc.) behind `#[cfg(feature = "gpu")]` so GPU-named API stops shipping in default Pure-Rust builds; correct module doc lines 1-4 falsely advertising "CUDA and OpenCL backends". Files: `Cargo.toml`, `src/lib.rs`, `src/gpu_acceleration.rs`
- [ ] (M) Rebuild `GpuContext`/`GpuDevice` on `sklears_core::gpu` instead of local simulation — replace the local `GpuBackend {Cuda, OpenCL, Metal, Cpu}` and the `NotImplemented`-returning `initialize_cuda`/`opencl`/`metal` (`src/gpu_acceleration.rs:134-168`) with `sklears_core::gpu::GpuBackend::detect()`/`GpuContext`; drop OpenCL/Metal variants, keep the Cpu fallback; map `GpuDevice` fields (`compute_capability`, `total_memory`, `multiprocessor_count`, `max_threads_per_block`) to `GpuUtils::device_properties()`; move the block_size/grid_size heuristics (lines 170-182) to oxicuda launch-config helpers. Files: `src/gpu_acceleration.rs`
- [ ] (L) Implement `GpuRBFSampler` on oxicuda-blas GEMM — `generate_features_cuda`/`opencl`/`metal` (lines 229-281) are identical CPU RNG loops; consolidate to one host-side weight generation + `GpuArray` upload. `transform_cuda`/`opencl`/`metal` (lines 316-368, 463-506) all reduce to `X @ W^T + b` then `sqrt(2/D) * cos(.)`: GEMM via oxicuda-blas (`GpuArray::from_array2`/`BlasHandle`), elementwise cos-scale on downloaded results or an oxicuda-primitives/oxicuda-ptx kernel. Keep `transform_cpu` fallback; removes four near-duplicate triple-nested loops. Deferable past 0.2.0. Files: `src/gpu_acceleration.rs`
- [ ] (L) Implement `GpuNystroem` kernel matrix + eigendecomposition via oxicuda-blas/oxicuda-solver, fixing fake CPU eigenvalues — `compute_kernel_cuda`/`opencl`/`metal` (lines 596-623) call the O(n^2 d) CPU loop: implement linear (`X Y^T`) and RBF (norm expansion + exp) via oxicuda-blas GEMM. Replace `eigendecomposition_cuda`/`opencl`/`metal` (lines 671-694) AND the numerically bogus `eigendecomposition_cpu` power-iteration placeholder (lines 696-735, fabricates 0.1 eigenvalues for all non-leading components — a latent correctness bug distorting Nystroem scaling even on the default CPU path) with an oxicuda-solver symmetric eigensolver on GPU and scirs2-linalg `eigh` on CPU. `FittedGpuNystroem::transform` (lines 801-833) becomes two GEMMs plus `S^{-1/2}` scaling. The `eigendecomposition_cpu` correctness fix should NOT be deferred even if the GPU path is. File is 1022 lines — consider a splitrs split (sampler/nystroem/profiler) during migration. Files: `src/gpu_acceleration.rs`

Note: GPU-path items (L) are deferable without blocking 0.2.0, but the
`eigendecomposition_cpu` correctness fix is not.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
