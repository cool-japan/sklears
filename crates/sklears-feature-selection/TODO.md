# sklears-feature-selection TODO

## Disabled modules (re-enable per empirical protocol)

All 7 modules that were blocked in v0.1.1 have been successfully re-enabled in Phase B-1/B-2.

**Root cause (resolved):** ndarray 0.17 added a third type parameter `A = <S as RawData>::Elem` to
`ArrayBase`. Generic trait bounds were restructured to use concrete types; blanket `#![allow(clippy::all)]`
removed; zero clippy warnings; 227 tests pass.

- [x] Re-enable `pub mod wrapper` — ENABLED (Phase B-1/B-2: resolved ndarray 0.17 `A` type param normalization in generic bounds)
- [x] Re-enable `pub mod validation` — ENABLED (Phase B-1/B-2: resolved same ndarray 0.17 generic bound issues)
- [x] Re-enable `pub mod ensemble_selectors` — ENABLED (Phase B-1/B-2: wrapper traits successfully re-exported)
- [x] Re-enable `pub mod tree_based_selectors` — ENABLED (Phase B-1/B-2: ensemble_selectors dependency resolved)
- [x] Re-enable `pub mod genetic_optimization` — ENABLED (Phase B-1/B-2: wrapper and ensemble_selectors dependencies resolved)
- [x] Re-enable `pub mod embedded` — ENABLED (Phase B-1/B-2: all upstream dependencies resolved)
- [x] Re-enable `pub mod comparison_tests` — ENABLED (Phase B-1/B-2: embedded::LassoSelector dependency resolved)

## Source-level TODOs

- [x] src/lib.rs:28 — TODO: ndarray 0.17 - uses disabled embedded module (resolved by Phase B-1/B-2: `pub mod embedded` enabled, TODO comment removed)
- [x] src/wrapper.rs:807 — TODO: Disabled due to ndarray 0.17 HRTB trait bound issues with generic helper methods (restored `SequentialFeatureSelector` via concrete-Fit macro pattern matching RFECV; 5 new tests pass; zero clippy warnings)
- [x] src/domain_specific/finance/algorithms/portfolio_optimization.rs:37 — Implemented Markowitz tangency portfolio QP via Cholesky-based analytical solution (`Σ z = μ`, non-negative simplex projection) with active-set refinement; 5 new tests pass; zero clippy warnings

---

See also: [Workspace roadmap](../../TODO.md)
