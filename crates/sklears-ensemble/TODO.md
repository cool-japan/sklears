# sklears-ensemble TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod model_selection` — Phase C-3 stretch: GENERIC_FIT pattern (6 Fit bounds in types.rs ~1700L); concrete-Fit macro needed; splitrs guardrail applies
  - **Goal:** Uncomment `pub mod model_selection` and `pub use model_selection::{}` in lib.rs; introduce concrete-Fit macro to replace generic bounds; splitrs types.rs if it exceeds 2000L after changes
  - **Files:** `src/lib.rs`, `src/model_selection.rs`, `src/types.rs`
  - **Done:** Phase C-3 complete (concrete type aliases), Phase C-4 complete (zero clippy warnings), Phase C-5 complete (no unwrap in production code)

## Source-level TODOs

- [x] src/stacking/mod.rs:202 — Fix MultiLayerStackingClassifier matrix dimension issue

---

See also: [Workspace roadmap](../../TODO.md)
