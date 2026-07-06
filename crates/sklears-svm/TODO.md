# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Status: fully migrated to oxicuda — remaining items are hygiene passes (Phase 5 of the workspace migration plan; optional for the 0.2.0 release).

- [ ] (S) Prune unused direct deps from the `gpu` feature: `oxicuda-ptx`, `oxicuda-backend`, and `bytemuck` are never referenced in src/tests/benches/examples (`oxicuda-backend` already arrives transitively via `sklears-core/gpu_support`; `oxicuda-ptx` is only cited in a doc comment at `src/gpu_kernels.rs:99` as the future home of a signed-integer-power PTX kernel for the Polynomial path). Drop `dep:oxicuda-ptx`/`dep:oxicuda-backend`/`bytemuck` from the feature list (`Cargo.toml:52`) and the optional dependency entries (`Cargo.toml:30,33,34`) — or keep `dep:oxicuda-ptx` only if the on-device Polynomial pow kernel lands this release. Files: `Cargo.toml`, `src/gpu_kernels.rs`
- [ ] (S) Retire dead WGPU-era `GpuKernelError` variants: `ShaderCompilationFailed`, `BufferCreationFailed`, `InsufficientMemory` (`src/gpu_kernels.rs:42-49`) are never constructed — all oxicuda failures funnel through `gpu_err()` into `ComputationFailed` (`src/gpu_kernels.rs:62-64`); they survive only as match arms in `From<GpuKernelError>` (`src/errors.rs:624-643`). Remove them (updating `errors.rs`), or rename `ShaderCompilationFailed` to a PTX/module-load variant if the oxicuda-ptx Polynomial kernel needs it. Note: `GpuKernelError` is public (re-exported via `src/lib.rs:119`), so ride the 0.2.0 version change. Files: `src/gpu_kernels.rs`, `src/errors.rs`

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
