# sklears-ensemble TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod model_selection` — Phase C-3 stretch: GENERIC_FIT pattern (6 Fit bounds in types.rs ~1700L); concrete-Fit macro needed; splitrs guardrail applies
  - **Goal:** Uncomment `pub mod model_selection` and `pub use model_selection::{}` in lib.rs; introduce concrete-Fit macro to replace generic bounds; splitrs types.rs if it exceeds 2000L after changes
  - **Files:** `src/lib.rs`, `src/model_selection.rs`, `src/types.rs`
  - **Done:** Phase C-3 complete (concrete type aliases), Phase C-4 complete (zero clippy warnings), Phase C-5 complete (no unwrap in production code)

## Source-level TODOs

- [x] src/stacking/mod.rs:202 — Fix MultiLayerStackingClassifier matrix dimension issue

## OxiCUDA Migration (v0.2.0)

Migration status: stub-gpu — the GPU layer is simulated (NotImplemented stubs / hardcoded-false detection) and must be wired to oxicuda-* 0.4.x or honestly downscoped. Part of the workspace Phase 4 honesty pass (parallelizable per crate, not blocking the 0.2.0 scirs2-GPU-removal goal).

- [x] (S) Add oxicuda-backed `gpu` feature — `Cargo.toml` currently has no gpu feature and no oxicuda deps (features: default/std/parallel/serde/simd/nightly). Add `gpu = ["dep:oxicuda-backend", "dep:oxicuda-memory", "dep:oxicuda-blas", "dep:oxicuda-driver", "sklears-core/gpu_support"]` with optional workspace deps (root pins 0.4.0); default features stay Pure Rust CPU-fallback. Files: `Cargo.toml`
- [x] (M) Rewire GpuContext device detection and memory manager to oxicuda — `detect_cuda_device` now calls `sklears_core::gpu::GpuBackend::with_device_id` behind `cfg(feature = "gpu")` and populates `GpuDeviceInfo` from real `oxicuda_driver::Device` attributes (name, live `cuMemGetInfo` memory, multiprocessor count, max threads/block, compute-capability-derived tensor-core support); `GpuMemoryManager` now allocates real `oxicuda_memory::DeviceBuffer<u8>` blocks when bound to a detected CUDA backend (`GpuMemoryManager::new_with_backend`) and only falls back to the honest host-side bookkeeping simulation for `CpuFallback` (the only reachable case without the `gpu` feature); `detect_available_backends`/`is_cuda_available` now probe `sklears_core::gpu::GpuBackend::is_available()` instead of a hardcoded `false`. OpenCL/Metal/Vulkan are documented (not deleted) as permanently unsupported by oxicuda (CUDA-only), returning an honest error naming the reason. Files: `src/gpu_acceleration.rs`
- [x] (L) Implement GpuTensorOps and gradient-boosting kernels via oxicuda, or explicitly downscope — `GpuTensorOps::matmul`/`elementwise_add` now dispatch to real `oxicuda-blas` GEMM/elementwise-add via `sklears_core::gpu::{GpuArray, GpuMatrixOps}` when a CUDA backend is bound, falling back to CPU `ndarray` ops otherwise; `reduce_sum`/`softmax` stay CPU-only with an honest doc note (no matching on-device primitive in `sklears_core::gpu` as of oxicuda-blas 0.4.0). Downscoped rather than implemented: deleted the `GpuKernel` trait and its four implementors (`HistogramKernel`/`SplitFindingKernel`/`TreeUpdateKernel`/`PredictionKernel`) and `GpuEnsembleTrainer::train_gradient_boosting` entirely — every one of those `execute` bodies only ever returned `NotImplemented`, so there was no working behavior to preserve, and a real on-device histogram/split-finding/tree-update trainer needs custom PTX kernels (`oxicuda-ptx`/`oxicuda-launch`) that are a substantial new kernel-authoring project, not a rewire (deferred 2026-07-06: GPU-side gradient-boosting training; CPU training already exists via `crate::gradient_boosting`). `GpuEnsembleTrainer::predict_ensemble` was rewritten to route through a real `GpuTensorOps::matmul` GEMM instead of the old always-failing prediction kernel. No other crate imported `sklears_ensemble::gpu_acceleration`, so the API reshape was unconstrained. Files: `src/gpu_acceleration.rs`
- [x] (S) Align `TensorDevice::Gpu` descriptor with oxicuda device ordinals — `src/tensor_ops.rs` `TensorDevice::Gpu(usize)` doc now states explicitly that the `usize` is an `oxicuda-driver` device ordinal; added `DeviceManager::refresh_gpu_devices` (behind `cfg(feature = "gpu")`) which probes `sklears_core::gpu::GpuBackend::detect` and pushes the real ordinal into `available_devices`; documented as an honest no-op without the `gpu` feature. Files: `src/tensor_ops.rs`

---

See also: [Workspace roadmap](../../TODO.md)
