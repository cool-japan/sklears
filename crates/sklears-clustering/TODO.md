# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Status: fully migrated to OxiCUDA via `sklears_core::gpu` — remaining items are hardening/cleanup (Phase 5 of the workspace migration plan; optional for the 0.2.0 release).

- [ ] (M) Prune or genuinely use the directly-declared oxicuda deps in the `gpu` feature — `Cargo.toml` line 56 pulls `dep:oxicuda-backend`, `dep:oxicuda-blas`, `dep:oxicuda-primitives`, and `bytemuck` into `gpu`, but no direct use exists in `src/`; all GPU work goes through `sklears_core::gpu` (`from_array2`/`matmul`/`to_array2` at `src/gpu_distances.rs:230-247`, `memory_info` at `:193`). Either (a) slim to `gpu = ["sklears-core/gpu_support"]` and drop the four optional deps (minimal 0.2.0 move), or (b) justify them by implementing native Manhattan/Cosine GPU distance kernels with oxicuda-primitives/oxicuda-blas (currently CPU-only at `:144-145`, `:295-340`). Files: `Cargo.toml`, `src/gpu_distances.rs`
- [ ] (S) Remove dead, non-compiling `cfg(not(feature = "gpu"))` stub module in `gpu_distances.rs` — `src/gpu_distances.rs:401-438` defines a `#[cfg(not(feature = "gpu"))] pub mod stub`, but `src/lib.rs:43-44` only declares the module under `cfg(feature = "gpu")`, so the stub is unreachable and latently broken (uses `Array2` at lines 426-429 without importing it). Delete the stub and the `cfg(not)` re-export (lines 437-438), or move the gating out of `lib.rs` and fix the import if a no-gpu API surface is wanted. Files: `src/gpu_distances.rs`, `src/lib.rs`
- [ ] (S) Move `pollster` to `[dev-dependencies]` — it is a non-optional runtime dep (`Cargo.toml` line 48) used only via `pollster::block_on` in gpu-gated unit tests (`src/gpu_distances.rs:349`, `:377`) wrapping fully-synchronous oxicuda calls. Move to `[dev-dependencies]` (`workspace = true`); consider de-asyncing the wrapped fns as a follow-up so pollster can be dropped entirely. Files: `Cargo.toml`, `src/gpu_distances.rs`
- [ ] (S) Fix stale WebGPU wording and phantom test reference in README — `README.md:71` calls the `gpu` feature "WebGPU-powered distance kernels" (implementation is OxiCUDA via `sklears_core::gpu`); `README.md:74` cites a nonexistent ignored test `gpu_distances::gpu::tests::test_gpu_distance_computation` (real tests: `test_euclidean_distances`, `test_manhattan_distances`, not ignored). Rewrite naming OxiCUDA (Pure Rust CUDA) and the real test names. Files: `README.md`

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
