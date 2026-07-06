# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: stub-gpu. This crate currently ships ~2225 lines of unconditionally-compiled GPU stub code where every operation fails at runtime. GPU dispatch belongs to the oxicuda-backed `sklears-core/src/gpu.rs` (Wave A2); sklears-simd's charter is CPU SIMD.

- [ ] (S) Delete commented cudarc/opencl3 remnants from `Cargo.toml` — remove lines 36-37 (commented `cudarc 0.11` / `opencl3 0.9` optional deps) and lines 49-51 (commented `cuda`/`opencl`/`gpu` features marked "Disabled for macOS compatibility"). Do NOT resurrect these: cudarc/opencl3 are FFI-backed crates violating the Pure Rust policy; the sanctioned path is oxicuda-*. Pure dead-comment cleanup with no build impact — these are the only cudarc/opencl3 dependency remnants in the workspace.
- [ ] (M) Remove the three always-erroring stub GPU modules in favor of `sklears-core::gpu` — delete `src/gpu.rs` (732 lines), `src/gpu_memory.rs` (614 lines), and `src/multi_gpu.rs` (879 lines) plus their `src/lib.rs` declarations (lines 90, 91, 100) and re-exports. Every allocate/copy/launch in these modules returns `SimdError::UnsupportedOperation` ("CUDA not available"), `get_device_info` returns mocks (`src/gpu.rs:164-175`), and `GPU_MANAGER` always reports no devices (`src/gpu.rs:586-620`). Breaking public API change (`initialize_gpu` / `is_gpu_available` / `MultiGpuCoordinator`) is acceptable for 0.2.0; no other workspace crate imports `sklears_simd::gpu*` (rg: zero hits). Removal also eliminates the `unsafe Send`/`Sync` impls over raw `*mut T` (`src/gpu.rs:49-59`).
  - Fallback if removal is rejected: reimplement behind an off-by-default `gpu = [dep:oxicuda-driver, dep:oxicuda-memory, dep:oxicuda-launch, dep:oxicuda-ptx]` feature, compiling the embedded CUDA kernel strings (`src/gpu.rs:233` ff.) via oxicuda-ptx (effort L; must reconcile with no-std mode). At minimum, rewrite the "Reserved for cudarc/opencl3" comments (`src/gpu.rs:53-55, 72-73, 81-82, 142-145, 347-351`; `src/gpu_memory.rs:436`) to reference oxicuda types.

Context: part of Phase 4 (stub/simulated GPU crate wiring, honesty pass) of the workspace-wide scirs2-core GPU removal — see the main [workspace TODO](../../TODO.md). Not blocking the 0.2.0 scirs2 excision goal.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
