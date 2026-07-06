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

- [ ] (S) Add oxicuda-backed `gpu` feature — `Cargo.toml` currently has no gpu feature and no oxicuda deps (features: default/std/parallel/serde/simd/nightly). Add `gpu = ["dep:oxicuda-backend", "dep:oxicuda-memory", "dep:oxicuda-blas", "dep:oxicuda-driver", "sklears-core/gpu_support"]` with optional workspace deps (root pins 0.4.0); default features stay Pure Rust CPU-fallback. Files: `Cargo.toml`
- [ ] (M) Rewire GpuContext device detection and memory manager to oxicuda — replace NotImplemented stubs `detect_cuda/opencl/metal/vulkan_device` (`src/gpu_acceleration.rs:214-239`) with oxicuda-driver device enumeration behind `cfg(feature = "gpu")` populating `GpuDeviceInfo` from real attributes; replace the fake usize-pointer `GpuMemoryManager` pool (93-106, 302-310) with oxicuda-memory allocations; make `detect_available_backends` (657-699, currently hardcoded-false `is_cuda_available`/`is_opencl_available` placeholders at 690-699) report Cuda when oxicuda-driver initializes, else CpuFallback. Remove or document Metal/Vulkan/OpenCL variants as unsupported by oxicuda. Files: `src/gpu_acceleration.rs`
- [ ] (L) Implement GpuTensorOps and gradient-boosting kernels via oxicuda, or explicitly downscope — `GpuTensorOps::matmul` (527-531) should dispatch to oxicuda-blas GEMM under feature `gpu` (keep `a.dot(b)` CPU fallback); `elementwise_add`/`reduce_sum`/`softmax` to oxicuda-primitives/oxicuda-launch. The four NotImplemented `GpuKernel::execute` impls (HistogramKernel 455-461, SplitFindingKernel 472-477, TreeUpdateKernel 488-493, PredictionKernel 504-509) mean `GpuEnsembleTrainer::train_gradient_boosting` (562-654) always errors for any non-CpuFallback backend today — either implement via oxicuda-launch/oxicuda-ptx or delete the kernel trait plumbing and route training through CPU gradient boosting + oxicuda-blas prediction. No other crate imports `sklears_ensemble::gpu_acceleration`, so the API can be reshaped freely. Files: `src/gpu_acceleration.rs`
- [ ] (S) Align `TensorDevice::Gpu` descriptor with oxicuda device ordinals — `src/tensor_ops.rs:37-38` `TensorDevice::Gpu(usize)` is metadata-only. When the gpu feature lands, map the usize to an oxicuda-driver device ordinal in `DeviceManager` and document that selection is a no-op without the feature. Doc/consistency only. Files: `src/tensor_ops.rs`

---

See also: [Workspace roadmap](../../TODO.md)
