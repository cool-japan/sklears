# TODO - v0.2.1

## Current Status (updated 2026-07-14)
This crate is part of the sklears v0.2.1 release line (initially shipped in v0.1.0).

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Status: fully migrated to OxiCUDA via `sklears_core::gpu`; Phase 5 hardening/cleanup complete (2026-07-06).

- [x] (M) Prune or genuinely use the directly-declared oxicuda deps in the `gpu` feature — went with option (a): slimmed `gpu = ["sklears-core/gpu_support"]` and dropped the four unused optional deps (`oxicuda-backend`, `oxicuda-blas`, `oxicuda-primitives`, `bytemuck`) from `Cargo.toml`; all GPU work continues to route through `sklears_core::gpu` (`GpuArray`/`GpuContext`/`GpuMatrixOps`) in `src/gpu_distances.rs`. Manhattan/Cosine remain CPU-only (documented, not fabricated). Files: `Cargo.toml`, `src/gpu_distances.rs`
- [x] (S) Remove dead, non-compiling `cfg(not(feature = "gpu"))` stub module in `gpu_distances.rs` — deleted the unreachable `pub mod stub` and its `cfg(not)` re-export; `src/lib.rs` already gated `pub mod gpu_distances` and its re-exports on `feature = "gpu"` only, so no no-gpu API surface was lost. Files: `src/gpu_distances.rs`
- [x] (S) Move `pollster` to `[dev-dependencies]` — verified already located under `[dev-dependencies]` in `Cargo.toml` (used only via `pollster::block_on` in the `#[cfg(test)]` module wrapping synchronous oxicuda calls); no change needed beyond confirming placement.
- [x] (S) Fix stale WebGPU wording and phantom test reference in README — rewrote `README.md`'s GPU section to name OxiCUDA (Pure Rust CUDA) via `sklears_core::gpu`, described the honest no-GPU CPU fallback, and replaced the nonexistent `--ignored gpu_distances::gpu::tests::test_gpu_distance_computation` invocation with the two real, non-ignored tests (`test_euclidean_distances`, `test_manhattan_distances`). Files: `README.md`

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
