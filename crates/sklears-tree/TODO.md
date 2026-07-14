# sklears-tree TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod incremental` — ENABLED (fixed rand::rngs::StdRng → scirs2_core::random::rngs::StdRng)
- [x] Re-enable `pub mod shap` — ENABLED (stale disable banner; no actual errors)
- [x] Add `pub mod multi_output` to lib.rs — ENABLED (module was absent from lib.rs)

## Source-level TODOs

- [x] `src/multi_output.rs:576` — Implement shared tree structure
  — Implemented `SharedMultiOutputTree`: single tree where each leaf stores an
    `Array1<f64>` prediction vector (one entry per output).  Split criterion
    sums impurity reductions across all output dimensions.  6 new tests added.
- [x] `src/multi_output.rs:585` — Implement chained outputs
  — Implemented `ChainedMultiOutputTree`: tree k is trained on
    `[X | pred_0 | ... | pred_{k-1}]` using in-bag predictions; predict applies
    the chain sequentially.  6 new tests added (shared with the above block).
- [x] `src/isolation_forest.rs:333` — Proper seeded RNG support
  — Already implemented: both extended and standard trees branch on
    `config.random_state` and call `seeded_rng(seed)` vs `thread_rng()`.
    The `thread_rng()` calls at lines ~354/376 are the intentional unseed path.
- [x] `src/isolation_forest.rs:747` — Proper seeded RNG support (streaming)
  — Already implemented in `rebuild_trees()` at line ~777: branches on
    `config.random_state`, uses `seeded_rng(seed)` for reproducibility.
- [x] `src/builder.rs:98` — Implement proper surrogate splits for missing value handling
  — Replaced naive mean-imputation with **column-level surrogate imputation**:
    for each column with NaN values, find the best-correlated surrogate column
    (highest threshold-agreement with the primary median split) and impute using
    the side-conditional mean of the primary column.  Falls back to column mean
    if no surrogate exceeds 50% agreement.  4 new tests added.
  — **Node-level NaN-aware predict routing wired** (`node.rs`):
    * `TreeNode` now stores `surrogate_splits: Vec<SurrogateSplit>` (sorted by
      descending agreement) and `majority_direction: bool` (last-resort fallback).
    * `TreeNode::route_sample(&self, sample: &[f64]) -> Option<SplitDirection>`
      implements the CART surrogate cascade: primary → first finite surrogate →
      majority direction.
    * `CompactTree` (the real production predict structure exported from `node.rs`)
      extended with `node_surrogates: Vec<NodeSurrogates>` and
      `set_node_surrogates(node_idx, surrogates, majority_direction)`.
      `predict_single` is now NaN-aware: when the primary feature is NaN it
      iterates registered surrogates, then falls back to majority direction.
    * 8 new tests in `node::surrogate_routing_tests` covering all code paths.
  — **Scope caveat**: the production `DecisionTreeClassifier` / `DecisionTreeRegressor`
    delegate to `smartcore` internally and do not expose `TreeNode` traversal.
    Surrogate routing is fully wired for `CompactTree`-based consumers.  Wiring
    into the smartcore path requires replacing smartcore's traversal entirely and
    is tracked separately.

## Phase C-4

- [x] Phase C-4: Removed all 7 blanket #![allow(...)] suppressors; 71 tests pass, 0 warnings

## Known Gaps (found during 2026-07-11 README audit)

- [ ] (L) Several top-level `src/*.rs` files are **not** declared as `mod`/`pub mod` anywhere in `lib.rs` (verified via `git log --all -p -- src/lib.rs`, which shows they have never been wired in, even back to v0.1.0) and are therefore dead code excluded from the compiled crate: `bart.rs`, `soft_tree.rs`, `gradient_boosting.rs` (+ its 15 internal submodules: GOSS/EFB/histogram training/etc.), `fairness.rs`, `spatial.rs`, `temporal.rs`, `adaboost.rs`, `categorical.rs`, `tree_visualization.rs`, `numa.rs`, `mmap_storage.rs`, `feature_importance.rs`, `sparse.rs`, `distributed.rs`, `extra_trees.rs`, `building.rs`, `compact.rs`, `shared.rs`, `loss_functions.rs`, and the whole `tree_interpretation/` directory (LIME, decision-path, anchor explainers). README.md previously advertised BART, Soft Decision Trees, LightGBM-style GOSS/EFB gradient boosting, Fairness-Aware Trees, Temporal/Spatial trees, Memory-Mapped storage, distributed training, and LIME as implemented (✓) — none of that is reachable from the public API. README.md has been corrected to describe only genuinely wired modules (Decision Tree/Random Forest/Isolation Forest/Model Tree/Multi-Output/Multi-Label trees, oblique+CHAID splits, TreeSHAP, Hoeffding/incremental/online-gradient-boosting streaming, parallel training, compact/bit-packed tree storage).
  - **Action needed:** either (a) wire the salvageable files in as real `pub mod` declarations after auditing each for correctness/completeness (largest lift: `gradient_boosting.rs` and its submodule tree), or (b) delete the orphaned files entirely if they're abandoned drafts, to stop future confusion. Until one of these happens, do not re-add these as README features.

---

See also: [Workspace roadmap](../../TODO.md)
