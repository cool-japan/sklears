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

---

See also: [Workspace roadmap](../../TODO.md)
