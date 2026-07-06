# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Status: fully migrated to oxicuda — remaining items are hygiene passes (Phase 5 of the workspace migration plan; optional for the 0.2.0 release).

- [x] (S) Prune unused direct deps from the `gpu` feature: `oxicuda-ptx`, `oxicuda-backend`, and `bytemuck` were never referenced in src/tests/benches/examples (`oxicuda-backend` arrived transitively via `sklears-core/gpu_support` anyway; `oxicuda-ptx` was only cited in a doc comment as the future home of a signed-integer-power PTX kernel for the Polynomial path — that doc note stays, only the unused dep is gone). Dropped `dep:oxicuda-ptx`/`dep:oxicuda-backend`/`bytemuck` from the `gpu` feature list and the optional dependency entries in `Cargo.toml`. `cargo check -p sklears-svm --features gpu` is warning-free. Files: `Cargo.toml`, `src/gpu_kernels.rs`
- [x] (S) Retire dead WGPU-era `GpuKernelError` variants: `ShaderCompilationFailed`, `BufferCreationFailed`, `InsufficientMemory` were never constructed — all oxicuda failures funnel through `gpu_err()` into `ComputationFailed`. Removed the three dead variants and their corresponding match arms in `From<GpuKernelError> for SVMError`. Files: `src/gpu_kernels.rs`, `src/errors.rs`

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
