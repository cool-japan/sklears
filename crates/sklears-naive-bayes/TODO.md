# sklears-naive-bayes TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod model_selection` — Phase C-2 Part C: DONE. Used concrete Mat2f64/Vec1i32 type aliases to resolve HRTB; re-enabled in lib.rs.
  - **Files:** `src/lib.rs`, `src/model_selection.rs`

- [x] Re-enable `pub mod finance` — Phase C-2 Part B: DONE. DMatrix/DVector are ndarray type aliases; verified and re-enabled in lib.rs.
  - **Files:** `src/lib.rs`, `src/finance/mod.rs`

## Source-level TODOs

- [x] `src/lib.rs:32` — Phase C-2 Part A: confirmed 0 nalgebra refs; delete banner line (`mod attention_naive_bayes`)
- [x] `src/lib.rs:43` — Phase C-2 Part A: confirmed 0 nalgebra refs; delete banner line (`mod continual_learning`)
- [x] `src/lib.rs:50` — Phase C-2 Part A: one test-only DVector hit at L1858; banner stale; delete banner line (`mod federated`)
- [x] `src/lib.rs:52` — stale: handled by the Re-enable finance item above; this line-specific item is superseded (`//mod finance` — currently disabled)
- [x] `src/lib.rs:72` — Confirmed STALE: quantum.rs declares `type DMatrix<T> = Array2<T>` / `type DVector<T> = Array1<T>` over `scirs2_core::ndarray` (lines 13-14). All 4 call-site hits (L476, L1147: ndarray tuple-shape zeros; L1424: from_iter; L1608 test: from_vec) use ndarray API. No nalgebra in file. No banner present in lib.rs. Same pattern as finance/federated. No action needed.
- [x] `src/lib.rs:97` — Phase C-2 Part A: delete stale banner line (`pub use attention_naive_bayes`)
- [x] `src/lib.rs:126` — Phase C-2 Part A: delete stale banner line (`pub use continual_learning`)
- [x] `src/lib.rs:155` — Phase C-2 Part A: delete stale banner line (`pub use federated`)
- [x] `src/lib.rs:161` — stale: handled by Re-enable finance item (`// pub use finance` — currently disabled)
- [x] `src/lib.rs:235` — Confirmed STALE: `pub use quantum::{...}` already active in lib.rs; DMatrix/DVector are ndarray type aliases (not nalgebra); no banner present in lib.rs; superseded by the lib.rs:72 item above.
- [x] `src/continual_learning.rs:16` — Migrated: added `random_state: Option<u64>` to `ContinualLearningConfig`; replaced `thread_rng()` with `seeded_rng(effective_seed)` for reproducibility. scirs2_core::random API is stable at 0.4.2.
- [x] `src/temporal.rs:860` — Migrated: stale TODO comment removed; existing `CoreRandom::seed_from_u64` pattern was already correct. scirs2_core::random API is stable at 0.4.2.

---

See also: [Workspace roadmap](../../TODO.md)
