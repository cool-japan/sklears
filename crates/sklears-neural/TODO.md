# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Status: fully migrated to oxicuda-* backends. Remaining work is hardening the
GPU path — making declared dependencies and capabilities real (Phase 5 of the
workspace migration plan; optional for the 0.2.0 release).

- [x] (L) Implement `tensor_core_conv2d` via oxicuda-dnn (declared dep currently unused) — done 2026-07-06: `GpuContext::tensor_core_conv2d` now builds a `DnnHandle` from the bound `Context`, NCHW `TensorDesc`/`TensorDescMut` descriptors + a `ConvolutionDescriptor::conv2d`, and calls `oxicuda_dnn::conv::api::conv_forward`. On `DnnError::WorkspaceRequired(bytes)` it allocates exactly that many bytes and retries once. `oxicuda-dnn` is now a real, referenced dependency. Files: `src/gpu.rs`.
- [x] (M) Wire real tensor-core / compute-capability detection — done 2026-07-06: `compute_capability` queries `oxicuda_driver::Device::compute_capability` (COMPUTE_CAPABILITY_MAJOR/MINOR) via `self.inner.context().device()` (no duplicate driver init — reuses the `Context` already owned by `sklears_core::gpu::GpuBackend`); `has_tensor_cores` derives `major >= 7` from it. This makes `tensor_core_gemm_f16`/`mixed_precision_gemm`/`tensor_core_matrix_multiply`'s f16 path and `tensor_core_recommendations` runtime-real (they still honestly report "not available" on this no-GPU host / on GPUs older than Volta). Files: `src/gpu.rs`.
- [x] (S) Resolve unused oxicuda-ptx and oxicuda-driver direct deps in the `gpu` feature — done 2026-07-06: `oxicuda-driver` is now genuinely used (`compute_capability` above, via `oxicuda_driver::Device`). `oxicuda-ptx` was dropped from `Cargo.toml` and the `gpu` feature — this crate's activation needs (relu/sigmoid/tanh) are already covered by `oxicuda_blas::elementwise`, and the conv2d path resolves algorithms/PTX internally inside `oxicuda-dnn`'s own `DnnHandle`, so no direct oxicuda-ptx usage was needed here. Files: `Cargo.toml`, `src/gpu.rs`.
- [ ] (M) Honor `GpuConfig` `memory_pool_size`/`max_streams`/`mixed_precision`; implement `memory_pool_stats` — the `GpuConfig` fields (`src/gpu.rs:47-58`) are `#[allow(dead_code)]` (236-237) and `memory_pool_stats` returns `(0.0, 0.0)` (383-385), which `performance_stats` reports as real hit rates (788-791). Back with an oxicuda-memory pooled allocator sized by `memory_pool_size` feeding `DeviceBuffer` allocations, and oxicuda-driver streams for `max_streams`; prefer implementing over stripping per IMPLEMENT POLICY. Files: `src/gpu.rs`.
- [ ] (M) Eliminate host round trip in `mixed_precision_gemm` (upstream oxicuda-blas coordination) — `mixed_precision_gemm` (`src/gpu.rs:427-436`) widens fp16 -> fp32 on host because `oxicuda_blas::level3::gemm` requires one uniform `GpuFloat` type (documented at 424-426). Add an fp16-in/fp32-out GEMM entry point or an on-device widen kernel upstream in the oxicuda workspace, then switch. Current behavior is honest but suboptimal. Files: `src/gpu.rs`.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
