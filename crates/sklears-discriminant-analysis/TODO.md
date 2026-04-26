# sklears-discriminant-analysis TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod async_optimization` — ENABLED (fixed SklearsError::ComputationError → ProcessingError, array view add, RwLockGuard across await)
- [x] Re-enable `pub mod boundary_adjustment` — ENABLED (fixed associated type bound: T::Fitted → explicit F generic with Fitted=F + Send+Sync)
- [~] Re-enable `pub mod gpu_acceleration` — DEFERRED: scirs2_core::gpu::kernels::reduction::ReductionKernel and scirs2_core::gpu::memory path do not exist in scirs2-core 0.4.2; requires gpu feature dep not in Cargo.toml
- [x] Re-enable `pub mod phantom_types` — ENABLED (const _: () = assert!() in generic fn → runtime assert!(), const fn → fn in ConfigurationValidator, added ldlt_solver() builder method)

## Source-level TODOs

- [ ] src/parallel_eigen.rs:133 — Implement when SciRS2 parallel eigenvalue solver is available
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

---

See also: [Workspace roadmap](../../TODO.md)
