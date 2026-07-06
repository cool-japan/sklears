# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: simulated-gpu. The GPU acceleration layer currently ships null-pointer
stubs and a disabled `gpu` feature; this section wires it onto the oxicuda-backed
`sklears_core::gpu` module (Phase 4 of the workspace migration plan — honesty pass,
not blocking the 0.2.0 scirs2 excision goal).

- [ ] (S) Fix gpu/cuda feature wiring and module gating — `Cargo.toml`, `src/lib.rs`
  - Change `gpu = []` ("temporarily disabled", `Cargo.toml:79`) to `gpu = ["sklears-core/gpu_support"]`; keep `cuda = ["gpu"]` (`Cargo.toml:80`) only as a back-compat alias or remove it.
  - Regate `src/gpu_acceleration.rs` in `src/lib.rs:454` from `cfg(feature = "cuda")` to `cfg(feature = "gpu")` so `full = [.., "gpu", ..]` (`Cargo.toml:89`) actually compiles the module (today `full` silently excludes it because it enables `gpu` but not `cuda`).
  - Defaults stay GPU-free per Pure Rust policy; verify `--features gpu` and `--features full` both compile.
- [ ] (L) Rewrite `src/gpu_acceleration.rs` onto sklears-core gpu_support / oxicuda
  - Replace null-pointer `CudaStream` (lines 104, 471-477), `GpuBuffer` (122, 679-683), hardcoded-false `is_cuda_available` (285-291), and the `GpuNotAvailable`-returning compute stubs (493-590) with `sklears_core::gpu::{GpuBackend, GpuContext, GpuArray}`: detection via `GpuBackend::detect()`; `allocate_and_copy_*` -> `GpuArray::from_slice`; `copy_*` -> `GpuArray::to_vec`.
  - Implement on device: MSE/MAE (elementwise diff + reduction via oxicuda-primitives), accuracy (equality + sum reduction), Euclidean/cosine (oxicuda-blas dot/nrm2), `compute_distance_matrix` via GEMM Gram expansion, `parallel_reduction` Sum/Mean/Max/Min via oxicuda reduction primitives.
  - Map oxicuda `CudaError`/`BlasError` into `GpuMetricsError::CudaError` instead of unconditional `GpuNotAvailable`; wire `supports_mixed_precision` (592-594) to the f32 path; make `get_available_memory` (596-618) report device memory via driver memory-info instead of `/proc/meminfo` host RAM.
  - Unimplemented `GpuMetricType` entries (confusion matrix, ROC-AUC) keep returning `UnsupportedMetric` with rustdoc saying so.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
