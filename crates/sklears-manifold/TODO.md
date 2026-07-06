# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)
Migration status: fully-oxicuda (Wave A2 landed). Remaining hardening items:

- [ ] (S) Resolve dead `cfg(not(feature = "gpu"))` branches in `src/gpu_acceleration.rs` vs `src/lib.rs` module gating — `src/lib.rs:592-593` only compiles the module under feature `gpu`, yet `src/gpu_acceleration.rs` carries never-compiled `cfg(not(gpu))` fallbacks: `GpuAccelerator::new` (:110-111), `is_gpu_available` (:119-122), `gemm_pairwise_euclidean` CPU shim (:190-193), `GpuTSNE::fit_transform` MissingDependency path (:608-614). Either (a) un-gate the module in `src/lib.rs` so CPU pairwise/kNN/matrix paths and the MissingDependency error are reachable in default builds (preferred — the CPU paths are self-contained), or (b) keep the gate and delete the dead branches. Re-run `cargo check` with and without `--features gpu`.
- [ ] (S) Replace `.unwrap()` in `GpuAccelerator` doc example — doctest at `src/gpu_acceleration.rs:86-87` calls `GpuAccelerator::new().unwrap()` and `pairwise_distances(...).unwrap()`, violating the no-unwrap policy. Rewrite with the `fn main() -> Result<(), _>` doctest pattern and `?`; note the doctest currently never runs under default features (module gated), so fix alongside the gating item above.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
