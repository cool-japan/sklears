# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## OxiCUDA Migration (v0.2.0)

Facade-level GPU feature cleanup for the workspace-wide scirs2-GPU removal. The
`backend-cuda`/`backend-wgpu` features here are empty stubs; the migration
direction is oxicuda-exclusive via the Wave A2 `sklears_core::gpu` module.

- [ ] (S) Replace empty `backend-cuda`/`backend-wgpu` stub features with a real oxicuda-backed `gpu` feature — delete `backend-cuda = []` and `backend-wgpu = []` (`Cargo.toml:139-140`, both enable nothing) and add `gpu = ["sklears-core/gpu_support"]`, optionally forwarding subcrate gpu features (e.g. `sklears-linear/gpu`, `sklears-clustering/gpu`). Do not keep a wgpu alias.
- [ ] (S) Fix `src/lib.rs` doc claims about GPU backends — `src/lib.rs:15` ("GPU acceleration with optional CUDA and WebGPU backends") and `src/lib.rs:64` ("`gpu` - GPU acceleration (CUDA/WebGPU)") describe features that do not exist in `Cargo.toml`; rewrite to describe the oxicuda-backed `gpu` feature (CUDA via OxiCUDA, Pure Rust default build).

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
