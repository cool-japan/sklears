# TODO - v0.2.1

## Current Status (updated 2026-07-14)
This crate is part of the sklears v0.2.1 release line (initially shipped in v0.1.0).

## Completed in 0.2.0 (2026-07-14)
- [x] Restored the `preprocessing` Cargo feature and its `sklears-preprocessing` optional
  dependency, previously commented out workspace-wide (`# Temporarily disabled`). This had
  cascaded into disabling 8 algorithm-showcase examples and a benchmark target that all require
  it; all are re-enabled now that `preprocessing` is back: `linear_models_showcase`,
  `lasso_regression`, `kmeans_clustering`, `dbscan_clustering`, `hierarchical_clustering`,
  `mean_shift_clustering`, `spectral_clustering`, `gmm_clustering`, plus
  `performance_comparison_comprehensive` and the `tree_ensemble_benchmarks` bench target.
- [x] Fixed `E0308` mismatched-type compile errors in `benches/comprehensive_benchmarks.rs`,
  `benches/continuous_benchmarks.rs`, and `benches/tree_ensemble_benchmarks.rs` — these benches
  converted integer classification labels to `f64` and wrapped `random_state` in `Some(..)` to
  match a tree API that actually takes integer labels and a bare `u64` seed directly; call sites
  now pass the integer label array and unwrapped seed so the bench targets compile again.

## OxiCUDA Migration (v0.2.0)

Facade-level GPU feature cleanup for the workspace-wide scirs2-GPU removal. The
`backend-cuda`/`backend-wgpu` features here are empty stubs; the migration
direction is oxicuda-exclusive via the Wave A2 `sklears_core::gpu` module.

- [x] (S) Replace empty `backend-cuda`/`backend-wgpu` stub features with a real oxicuda-backed `gpu` feature — deleted `backend-cuda = []` and `backend-wgpu = []` and added `gpu = ["sklears-core/gpu_support", ...]`, forwarding to every algorithm subcrate that currently exposes its own `gpu` feature (via the weak-dependency `subcrate?/gpu` syntax so an optional subcrate is only pulled in if already enabled through its own facade feature; `sklears-utils/gpu` is forwarded directly since `sklears-utils` is a mandatory, non-optional facade dependency). No wgpu alias was kept.
- [x] (S) Fixed `src/lib.rs` doc claims about GPU backends — `src/lib.rs:15` and `src/lib.rs:64` now describe the oxicuda-backed `gpu` feature (CUDA only via OxiCUDA, honest `GpuBackend::detect() -> Ok(None)` fallback to CPU, Pure Rust default build) instead of the fictitious CUDA/WebGPU claim.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
