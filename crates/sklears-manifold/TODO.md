# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)
Migration status: fully-oxicuda (Wave A2 landed). Remaining hardening items:

- [x] (S) Resolve dead `cfg(not(feature = "gpu"))` branches in `src/gpu_acceleration.rs` vs `src/lib.rs` module gating — un-gated `pub mod gpu_acceleration;` in `src/lib.rs` (option (a), preferred) so the CPU pairwise/kNN/matrix paths and `GpuTSNE::fit_transform`'s `MissingDependency` error are reachable and exercised in default builds. The `#[cfg(not(feature = "gpu"))]` branches inside `gpu_acceleration.rs` are now live code, not dead. Gated the `gpu`-only unit tests (`test_gpu_tsne`, `test_gpu_tsne_fit_transform_is_deterministic_and_separates_clusters`, and their shared `well_separated_two_cluster_data` helper) behind `#[cfg(feature = "gpu")]`, and added `test_gpu_tsne_without_gpu_feature_reports_missing_dependency` under `#[cfg(not(feature = "gpu"))]` to cover the non-gpu path. Verified `cargo check -p sklears-manifold` (with and without `--features gpu`) is warning-free, and `cargo nextest run -p sklears-manifold gpu_acceleration::` passes in both configurations (2026-07-06).
- [x] (S) Replace `.unwrap()` in `GpuAccelerator` doc example — doctest at `src/gpu_acceleration.rs` rewritten with the `fn main() -> Result<(), SklearsError>` pattern and `?`; now runs under default features (module gating fixed above) and passes both with and without `--features gpu` (2026-07-06).

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
