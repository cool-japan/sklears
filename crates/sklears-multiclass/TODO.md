# sklears-multiclass TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod advanced` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
- [x] Re-enable `pub mod calibration` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
- [x] Re-enable `pub mod core` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
- [x] Re-enable `pub mod ensemble` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
- [x] Re-enable `pub mod boosting` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
- [x] Re-enable `pub mod dynamic_ensemble` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0. (resolved: file-form deleted; tree-form re-enabled via pub mod ensemble)
- [x] Re-enable `pub mod ecoc` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0. (resolved: file-form deleted; tree-form re-enabled via pub mod core)
- [x] Re-enable `pub mod one_vs_one` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
- [x] Re-enable `pub mod one_vs_rest` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
- [x] Re-enable `pub mod rotation_forest` — KNOWN ISSUE (v0.1.0): Modules disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0. (resolved: file-form deleted; tree-form re-enabled via pub mod ensemble)

## Canonicalization (delete duplicate/inferior forms)

- [x] Delete `src/ecoc.rs` — tree form `src/core/ecoc/` is strictly richer (OptimizedCodeMatrix, GPUMode, Quantized/Compressed)
- [x] Delete `src/rotation_forest.rs` — tree form `src/ensemble/rotation_forest.rs` has `warm_start: bool`
- [x] Delete `src/dynamic_ensemble.rs` — tree form `src/ensemble/dynamic_ensemble.rs` is 934L vs 697L
- [x] Delete `src/core/one_vs_rest.rs` — file form `src/one_vs_rest.rs` is canonical per author intent
- [x] Delete `src/core/one_vs_one.rs` — file form `src/one_vs_one.rs` is canonical per author intent

## Orphan files (delete, do not repair)

- [x] Delete `src/adaptive_decomposition.rs` — references undefined `DynamicMulticlassTrainedData`
- [x] Delete `src/multi_level_decomposition.rs` — wrong import paths (`sklears_core::estimator::*`, `sklears_core::Float`)
- [x] Delete `src/lib_new.rs` — unreferenced alternate lib file
- [x] Delete `src/imbalance/` — `mod.rs` uses undefined `MulticlassError`; declares non-existent submodule files

## OxiCUDA Migration (v0.2.0)

Status: simulated GPU layer — `src/gpu/` is a hand-rolled stub (bool-flag context, PhantomData arrays) and ECOC `GPUMode` only toggles CPU parallelism. Rewire onto the oxicuda-backed `sklears_core::gpu` module (Wave A2) so GPU claims are honest; defaults stay Pure Rust with `src/gpu/fallback.rs` as the no-device path.

- [x] (M) Rewire `src/gpu` module onto `sklears-core` `gpu_support` (oxicuda) types — replaced the hand-rolled `GpuConfig`/`GpuContext` bool-flag and `PhantomData` `GpuArray2` with `sklears_core::gpu::{GpuBackend, GpuContext, GpuArray}` (re-exported behind the `gpu` feature; a local honest `Ok(None)`-only stand-in is used when `gpu` is off, since `sklears_core::gpu` is itself gated behind `gpu_support` at the module level, not just per-item); changed `Cargo.toml` from `gpu = []` to `gpu = ["sklears-core/gpu_support"]`; implemented `memory::to_device`/`from_device` via `GpuArray::from_array2`/`to_array2`; added `OxiCudaMatrixOps` implementing `matmul_gpu`/`batch_matmul_gpu` on sklears-core's oxicuda-blas GEMM (`GpuArray::matmul`) and `compute_distances_gpu` (ECOC squared-Euclidean) as the GEMM expansion `||p||^2 + ||c||^2 - 2 p.c^T` (cross term on-device, norms host-side); deleted the legacy "CUDA-or-OpenCL FFI crate" placeholder comment; kept `src/gpu/fallback.rs` CPU impls as the no-device path, gated the device path (`OxiCudaMatrixOps`) behind `gpu`. Side cleanups done: replaced `src/gpu/fallback.rs`'s `.expect()` in production code with `f64::total_cmp` (no-unwrap policy); unified the test `array!` import onto `scirs2_core::ndarray`; rewrote `fallback.rs`'s tests to build a `GpuContext` via a `test_context()` helper (unit-struct stand-in without `gpu`, real `detect_context()` with graceful skip-if-no-device under `gpu`) instead of the deleted `GpuConfig`/`GpuContext::new()`. Verified: `cargo check -p sklears-multiclass` and `--features gpu` both pass warning-free; `cargo test --lib` (289 tests) and `--features gpu --lib` both pass. Files: `Cargo.toml`, `src/gpu/mod.rs`, `src/gpu/fallback.rs`
- [x] (M) Connect ECOC `GPUMode` to the real oxicuda-backed GPU layer — `aggregate_votes_gpu` (`src/core/ecoc/functions.rs`) now tries a new `aggregate_votes_device` first for `GPUMode::DistanceOps`/`Full` under the `gpu` feature: uploads the whole prediction/code-matrix batch and calls the rewired `compute_distances_gpu` (squared-Euclidean ranks identically to Hamming distance for +/-1 codes since `||a-b||^2 == 4*hamming(a,b)`), returning `Ok(None)` (not an error) to fall through to the existing rayon/`batch_hamming_distances` CPU path whenever `GpuBackend::detect()` finds no device or the feature is off — the honest `Ok(None)` contract is preserved end-to-end. `generate_code_matrix_gpu` (`src/core/ecoc/types.rs`) intentionally kept as CPU-only RNG batching (deferred 2026-07-06: RNG is inherently sequential host-side state and this runs once per `fit()`, not per prediction, so a device path has no clear benefit here) — its doc comment was rewritten to stop claiming GPU acceleration it doesn't do. Fixed builder rustdoc on `gpu_matrix_ops`/`gpu_distance_ops`/`gpu_full` (`src/core/ecoc/types.rs`) to state the actual fallback behavior instead of an unconditional "GPU acceleration" claim. Files: `src/core/ecoc/functions.rs`, `src/core/ecoc/types.rs`, `src/gpu/mod.rs`

Context: this crate is Phase 4 of the workspace 0.2.0 plan (stub/simulated GPU wiring — parallelizable, not blocking the scirs2-core GPU excision goal). oxicuda-* pins stay at 0.4.0 until 0.4.1 is published on crates.io.

## Source-level TODOs

(none found — `grep -rn "// TODO" src/` returned no results)

---

See also: [Workspace roadmap](../../TODO.md)
