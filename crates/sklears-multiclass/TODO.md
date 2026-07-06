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

- [ ] (M) Rewire `src/gpu` module onto `sklears-core` `gpu_support` (oxicuda) types — replace hand-rolled `GpuConfig`/`GpuContext` bool-flag (`src/gpu/mod.rs:37-88`) and `PhantomData` `GpuArray2` (`src/gpu/mod.rs:188-201`) with `sklears_core::gpu::{GpuBackend, GpuContext, GpuArray, GpuMatrixOps}`; change `Cargo.toml:37` from `gpu = []` to `gpu = ["sklears-core/gpu_support"]`; implement `memory::to_device`/`from_device` via `GpuArray::from_slice`/`to_vec` (`from_device` currently returns "Not implemented yet", `src/gpu/mod.rs:150-186`); implement `GpuMatrixOps::matmul_gpu`/`batch_matmul_gpu` on sklears-core's oxicuda-blas GEMM; implement `compute_distances_gpu` (ECOC squared-Euclidean) as GEMM expansion `||p||^2 + ||c||^2 - 2 p.c^T` on device; delete the "cudarc or opencl3" placeholder comment (`src/gpu/mod.rs:58`) — it prescribes policy-violating FFI crates; keep `src/gpu/fallback.rs` CPU impls as the no-device path, gate the device path behind `gpu` so defaults stay Pure Rust. Side cleanups: `src/gpu/fallback.rs:39` `.expect()` (no-unwrap policy); unify `scirs2_autograd::ndarray` test import (`src/gpu/fallback.rs:204`) with `scirs2_core::ndarray`. Files: `Cargo.toml`, `src/gpu/mod.rs`, `src/gpu/fallback.rs`, `src/lib.rs`
- [ ] (M) Connect ECOC `GPUMode` to the real oxicuda-backed GPU layer — `GPUMode::MatrixOps`/`DistanceOps`/`Full` (`src/core/ecoc/types.rs:987-997`) only toggle sequential-vs-rayon CPU paths; route `batch_hamming_distances` (`src/core/ecoc/functions.rs:86-98`) and `aggregate_votes_gpu` (`src/core/ecoc/functions.rs:100-132`) through the rewired `compute_distances_gpu` when feature `gpu` is on and `GpuBackend::detect()` reports a device; `generate_code_matrix_gpu` (`src/core/ecoc/types.rs:714-738`, dispatched from `src/core/ecoc/ecocclassifier_traits.rs:73`) may keep CPU RNG with a one-shot upload; preserve rayon as documented fallback; fix builder rustdoc (`src/core/ecoc/types.rs:912-929`) so "GPU acceleration" claims are honest. Files: `src/core/ecoc/functions.rs`, `src/core/ecoc/types.rs`, `src/core/ecoc/ecocclassifier_traits.rs`, `src/gpu/mod.rs`

Context: this crate is Phase 4 of the workspace 0.2.0 plan (stub/simulated GPU wiring — parallelizable, not blocking the scirs2-core GPU excision goal). oxicuda-* pins stay at 0.4.0 until 0.4.1 is published on crates.io.

## Source-level TODOs

(none found — `grep -rn "// TODO" src/` returned no results)

---

See also: [Workspace roadmap](../../TODO.md)
