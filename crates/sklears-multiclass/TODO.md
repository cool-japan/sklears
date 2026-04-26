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

## Source-level TODOs

(none found — `grep -rn "// TODO" src/` returned no results)

---

See also: [Workspace roadmap](../../TODO.md)
