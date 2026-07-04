# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Completed This Session
- [x] GPU backend foundation — DONE: added `sklears_core::gpu::{GpuBackend, GpuArray, GpuMatrixOps}`, wired directly to `oxicuda-driver` / `oxicuda-blas`, replacing the previous CPU-only stubs; `GpuBackend::detect()` gracefully returns `Ok(None)` when no GPU is present instead of faking a backend.
- [x] Re-enable `trait_explorer::graph_visualization` — DONE: module previously never compiled due to a raw-string-literal bug; now provides real graph algorithms (hub/bridge/bottleneck node detection, Newman modularity calculation, small-world coefficient).
- [x] `api_reference_generator` module — DONE: new module added.
- [x] Fix `trait_explorer::security_analysis` doc examples — DONE: examples were using incorrect `crate::` paths, now correctly use `sklears_core::` external paths.
- [x] 855 tests passing in this crate.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
