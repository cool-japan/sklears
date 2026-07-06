# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: done (2026-07-06). This crate previously shipped ~2225 lines of unconditionally-compiled GPU stub code where every operation failed at runtime. GPU dispatch belongs to the oxicuda-backed `sklears-core/src/gpu.rs` (Wave A2); sklears-simd's charter is CPU SIMD.

- [x] (S) Delete commented legacy GPU-FFI dependency remnants from `Cargo.toml` — removed the commented third-party CUDA-FFI (0.11) / OpenCL-FFI (0.9) optional deps and the commented `cuda`/`opencl`/`gpu` features marked "Disabled for macOS compatibility". Those are FFI-backed crates violating the Pure Rust policy; the sanctioned path is oxicuda-*.
- [x] (M) Removed the three always-erroring stub GPU modules in favor of `sklears-core::gpu` — deleted `src/gpu.rs` (732 lines), `src/gpu_memory.rs` (614 lines), and `src/multi_gpu.rs` (879 lines) plus their `src/lib.rs` declarations (former lines 90, 91, 100). No re-exports existed elsewhere. Confirmed via workspace-wide `rg` that no other crate imported `sklears_simd::gpu*` (zero hits) before deleting. `cargo check -p sklears-simd` (default features and `--features parallel`) passes warning-free after removal.

Context: part of Phase 4 (stub/simulated GPU crate wiring, honesty pass) of the workspace-wide scirs2-core GPU removal — see the main [workspace TODO](../../TODO.md). Not blocking the 0.2.0 scirs2 excision goal.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
