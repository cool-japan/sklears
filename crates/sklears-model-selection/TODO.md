# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Status: simulated-gpu — no scirs2-GPU usage exists in this crate; the items below are honesty/polish work so GPU-related config metadata reflects real oxicuda-backed detection (Phase 4 of the workspace migration plan).

- [ ] (S) Optionally derive `has_gpu`/`use_gpu` config defaults from oxicuda device detection: `has_gpu` (src/automl_algorithm_selection.rs:138-139, default false at :365) and `use_gpu` (src/config_management.rs:132-134, default false at :452) are manual metadata booleans. When sklears-core is built with `gpu_support`, default `has_gpu` from the oxicuda-driver-backed availability check (`sklears_core::gpu::GpuBackend::detect()`) and validate `use_gpu` against actual availability. Files: src/automl_algorithm_selection.rs, src/config_management.rs

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
