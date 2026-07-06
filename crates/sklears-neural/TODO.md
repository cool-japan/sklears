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

- [ ] (L) Implement `tensor_core_conv2d` via oxicuda-dnn (declared dep currently unused) — `GpuContext::tensor_core_conv2d` (`src/gpu.rs:451-463`) returns NotImplemented while oxicuda-dnn sits in `Cargo.toml` (lines 40, 57) with zero code references. Wire a real forward pass via `oxicuda_dnn::conv::api::conv_forward`: a `DnnHandle` (distinct from the `BlasHandle` in `sklears_core::gpu::GpuBackend`), NCHW `TensorDesc`/`TensorDescMut` + `ConvolutionDescriptor`, and the workspace-sizing retry loop (`DnnError::WorkspaceRequired(bytes)` from Im2colGemm/Winograd/FftConv). Satisfies IMPLEMENT POLICY and makes the declared dep real. Files: `src/gpu.rs`, `Cargo.toml`.
- [ ] (M) Wire real tensor-core / compute-capability detection — `has_tensor_cores` (`src/gpu.rs:388-390`) hardcodes `false` and `compute_capability` (`src/gpu.rs:393-395`) hardcodes `None`, making `tensor_core_matrix_multiply`'s f16 path (601-641) and `tensor_core_gemm_f16`/`mixed_precision_gemm` (406-436) dead at runtime, and `tensor_core_recommendations` (656-722) always report "Not available". Query `CUdevice` COMPUTE_CAPABILITY_MAJOR/MINOR via oxicuda-driver — preferably by adding an accessor on `sklears_core::gpu::GpuContext` (it owns the `Device`) rather than duplicating driver init; derive `has_tensor_cores` from `major >= 7`. Files: `src/gpu.rs`.
- [ ] (S) Resolve unused oxicuda-ptx and oxicuda-driver direct deps in the `gpu` feature — the `gpu` feature (`Cargo.toml:57`) enables `dep:oxicuda-ptx` (line 41) and `dep:oxicuda-driver` (line 38), but only `oxicuda_blas`/`oxicuda_memory` are imported (`src/gpu.rs:39,41`; driver access is indirect via `sklears_core::gpu`). Either use them (oxicuda-ptx custom tanh/softmax kernels; oxicuda-driver for the tensor-core attribute queries) or drop them per unused-deps hygiene — decide in the same pass as the conv2d/tensor-core items. Files: `Cargo.toml`, `src/gpu.rs`.
- [ ] (M) Honor `GpuConfig` `memory_pool_size`/`max_streams`/`mixed_precision`; implement `memory_pool_stats` — the `GpuConfig` fields (`src/gpu.rs:47-58`) are `#[allow(dead_code)]` (236-237) and `memory_pool_stats` returns `(0.0, 0.0)` (383-385), which `performance_stats` reports as real hit rates (788-791). Back with an oxicuda-memory pooled allocator sized by `memory_pool_size` feeding `DeviceBuffer` allocations, and oxicuda-driver streams for `max_streams`; prefer implementing over stripping per IMPLEMENT POLICY. Files: `src/gpu.rs`.
- [ ] (M) Eliminate host round trip in `mixed_precision_gemm` (upstream oxicuda-blas coordination) — `mixed_precision_gemm` (`src/gpu.rs:427-436`) widens fp16 -> fp32 on host because `oxicuda_blas::level3::gemm` requires one uniform `GpuFloat` type (documented at 424-426). Add an fp16-in/fp32-out GEMM entry point or an on-device widen kernel upstream in the oxicuda workspace, then switch. Current behavior is honest but suboptimal. Files: `src/gpu.rs`.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
