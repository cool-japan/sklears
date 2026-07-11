# sklears-discriminant-analysis TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod async_optimization` — ENABLED (fixed SklearsError::ComputationError → ProcessingError, array view add, RwLockGuard across await)
- [x] Re-enable `pub mod boundary_adjustment` — ENABLED (fixed associated type bound: T::Fitted → explicit F generic with Fitted=F + Send+Sync)
- [x] Re-enable `pub mod gpu_acceleration` — DONE (2026-07-03): retargeted from dead `scirs2_core::gpu::*` to the real `sklears_core::gpu::{GpuBackend, GpuArray, GpuMatrixOps}` foundation; real GEMM-based class means/covariance; oxicuda-solver wiring for QDA inverse/log-determinant and an LDA generalized eigensolve via Cholesky reduction (no sygvd in oxicuda-solver 0.4.0); found and fixed a real bug in `NumericalStability::stable_inverse` (was a stub returning NotImplemented for non-symmetric matrices, now uses scirs2_linalg::inv); 322 tests passing (16 new gpu_acceleration tests, all GPU-path tests gracefully skip on this no-CUDA machine rather than silently no-op-passing).
- [x] Re-enable `pub mod phantom_types` — ENABLED (const _: () = assert!() in generic fn → runtime assert!(), const fn → fn in ConfigurationValidator, added ldlt_solver() builder method)

## Source-level TODOs

- [x] src/parallel_eigen.rs:133 — DONE (2026-06-21): wired to `scirs2_linalg::eigh` (accurate sequential kernel; upstream multi-worker kernel returns inf in 0.5.0, documented); parallelism retained across independent blocks.
- [x] src/tests/canonical_tests.rs:5 — Extract from original tests.rs (7 tests moved from canonical_discriminant.rs inline tests)
- [x] src/tests/bayesian_tests.rs:5 — Extract from original tests.rs (8 new smoke tests written — no inline tests in source)
- [x] src/tests/mixture_tests.rs:5 — Extract from original tests.rs (7 new smoke tests written — no inline tests in source)
- [x] src/tests/hierarchical_tests.rs:5 — Extract from original tests.rs (8 tests moved from hierarchical.rs inline tests)
- [x] src/tests/locality_preserving_tests.rs:6 — Extract from original tests.rs (7 new smoke tests written — no inline tests in source)
- [x] src/tests/locality_preserving_tests.rs:12 — Extract from original tests.rs (covered by same 7 test set above)
- [x] src/tests/discriminant_locality_alignment_tests.rs:5 — Extract from original tests.rs (7 new smoke tests written — no inline tests in source)
- [x] src/tests/error_correcting_tests.rs:5 — Extract from original tests.rs (6 tests moved from error_correcting_output_codes.rs inline tests; generate_code_matrix made pub)
- [x] src/tests/kernel_methods_tests.rs:5 — Extract from original tests.rs (12 tests moved from kernel_discriminant.rs inline tests)
- [x] src/tests/stochastic_tests.rs:5 — Extract from original tests.rs (7 tests moved from stochastic_discriminant.rs inline tests)
- [x] src/tests/multi_task_tests.rs:5 — Extract from original tests.rs (7 tests moved from multi_task.rs inline tests)
- [x] src/tests/marginal_fisher_tests.rs:5 — Extract from original tests.rs (7 new smoke tests written — no inline tests in source)
- [x] src/multi_view_discriminant/untrained.rs:27 — Extract from original multi_view_discriminant.rs (11 builder methods implemented)

## Phase C-4

- [x] Phase C-4: Removed all 7 blanket #![allow(...)] suppressors; 300 tests pass, 0 warnings

## OxiCUDA Migration (v0.2.0)

Status: fully migrated — `gpu_acceleration` already runs on the oxicuda-backed `sklears_core::gpu` foundation (see 2026-07-03 entry above) and serves as the reference pattern for sklears-preprocessing. Remaining item is a forward-looking enhancement only.

- [ ] Adopt oxicuda-solver `sygvd` for the LDA generalized eigensolve when it ships (S) — `solve_generalized_eigen_gpu` (src/gpu_acceleration.rs:554-598) hand-reduces `S_b w = lambda S_w w` via `dense::cholesky` + `dense::inverse` + two GEMMs + `dense::syevd` because oxicuda-solver 0.4.x lacks a symmetric generalized eigensolver. Track oxicuda-solver releases; when `sygvd` (or on-device `syevd`) lands, replace the 4-step reduction with a single call. (blocked upstream 2026-07-06: oxicuda-solver 0.4.0 lacks sygvd — verified by inspecting both the crates.io 0.4.0 source cache and the ~/work/oxicuda dev tree (workspace version 0.4.1): `rg -rni "sygvd|generalized"` across `oxicuda-solver/src` finds only `dense/qz.rs`'s non-symmetric QZ solver (`qz_host`, an explicit CPU host fallback), no symmetric-definite generalized eigensolver and no on-device kernel. Re-check when oxicuda-solver cuts a release past 0.4.1.)

## Known Gaps (found during 2026-07-11 README audit)

- [x] README previously advertised "Probability Calibration: Built-in support for Platt scaling and isotonic calibration for multiclass scenarios" — there is no Platt/isotonic calibration logic anywhere in this crate (`rg -i "platt|isotonic"` finds zero hits). The only nearby field, `CostSensitiveDiscriminantAnalysis`'s `calibration_method: String` (defaults to `"platt"`), is set in the config but never read/applied anywhere in `cost_sensitive_discriminant_analysis.rs` — it is a dead config knob, not a working feature. Removed the fabricated bullet from README.md; kept the accurate `predict_proba` claim.

---

See also: [Workspace roadmap](../../TODO.md)
