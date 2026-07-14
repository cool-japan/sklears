# TODO - v0.2.0

## Current Status
This crate is part of the sklears v0.2.0 release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Status: simulated-gpu — no scirs2-GPU usage exists in this crate; the items below are honesty/polish work so GPU-related config metadata reflects real oxicuda-backed detection (Phase 4 of the workspace migration plan).

- [x] (S) Optionally derive `has_gpu`/`use_gpu` config defaults from oxicuda device detection: `has_gpu` (`ComputationalConstraints` in src/automl_algorithm_selection.rs) and `use_gpu` (`ResourceConfig` in src/config_management.rs) now default from a real oxicuda-backed availability check instead of a hardcoded literal, gated behind a new crate-level `gpu` feature (`gpu = ["sklears-core/gpu_support"]` in Cargo.toml). With `gpu` off (default), `sklears_core::gpu` isn't compiled in at all, so both fall back to a local `false`-returning helper -- identical to the pre-migration behavior. With `gpu` on, they call `sklears_core::gpu::GpuBackend::is_available()`. `ConfigManager::validate_config` now also rejects `resources.use_gpu = true` when no GPU is actually detected, instead of silently accepting it. Files: src/automl_algorithm_selection.rs, src/config_management.rs, Cargo.toml

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
