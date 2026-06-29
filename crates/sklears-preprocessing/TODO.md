# sklears-preprocessing TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod gpu_acceleration` — DONE (2026-06-21): `scirs2_core::memory::BufferPool` exists in scirs2-core 0.5.0; re-enabled.
- [x] Re-enable `pub mod lazy_evaluation` — DONE (2026-06-21): BufferPool available; re-enabled.
- [x] Re-enable `pub mod memory_management` — DONE (2026-06-21): BufferPool available; re-enabled.

## Source-level TODOs

- [x] src/robust_preprocessing.rs:366 — Implement fit/transform for OutlierAwareImputer (already implemented; fixed NaN propagation in RobustScaler::fit)
- [x] src/robust_preprocessing.rs:457 — Implement proper RobustScaler with these methods (already implemented; root fix was filtering non-finite values before quantile computation)
- [x] src/robust_preprocessing.rs:459 — Implement fit for RobustScaler (resolved as part of NaN fix)
- [x] src/pipeline.rs:1038 — Fix this test - requires properly fitted transformers (added is_fitted() to ParallelBranches; uncommented test)
- [x] src/pipeline.rs:1058 — Fix this test - requires properly fitted transformers (added len()/is_empty() to AdvancedPipeline; uncommented test)
- [x] src/pipeline.rs:1071 — Fix this test - requires properly fitted transformers (DynamicPipeline already had len()/is_empty(); uncommented test)

## Phase C-4

- [x] Phase C-4: Removed all 20 blanket #![allow(...)] suppressors; 300 tests pass, 0 warnings

---

See also: [Workspace roadmap](../../TODO.md)
