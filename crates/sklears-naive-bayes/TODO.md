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

## OxiCUDA Migration (v0.2.0)

Phase 4 of the workspace 0.2.0 GPU-honesty pass: this crate currently ships a stub GPU path — `GpuOptimizer` hardcodes `gpu_available = false` and always falls back to a naive CPU loop, so `OptimizationStrategy::Gpu` is unreachable. Wire it through the oxicuda-backed `sklears_core::gpu` module (Wave A2) or delete the placeholder.

- [x] (M) Back `GpuOptimizer` with oxicuda-blas gemm behind a `gpu` feature, or delete the placeholder. DONE: added `gpu = ["sklears-core/gpu_support"]` to `Cargo.toml`; `GpuOptimizer::new` now calls `sklears_core::gpu::GpuBackend::detect()` once and caches the `Option<GpuBackend>` (honestly `None` on this GPU-less host, no fabricated availability); `gpu_matrix_multiply` dispatches `f32`/`f64` to a real on-device GEMM (`GpuArray::from_array2` + `GpuMatrixOps::matmul` + `to_array2`, via a `TypeId`-checked-then-`transmute` monomorphization dispatch) when a backend is present, and falls back to the existing CPU triple loop for every other element type / GPU-disabled builds / GPU-less hosts. `OptimizationStrategy::Gpu` is now backed by real logic when reached via the feature.
  - **Files:** `Cargo.toml`, `src/feature_engineering/performance_optimization.rs`
- [x] (S) Trim the module doc at `src/feature_engineering/performance_optimization.rs:5`, which pairs a false "GPU acceleration" claim with "SciRS2 Policy"; reword to reflect the actual (oxicuda or CPU-only) implementation. DONE: module doc now states the GPU path is real (oxicuda-blas GEMM) behind the `gpu` feature for f32/f64, with an honest CPU fallback everywhere else.
  - **Files:** `src/feature_engineering/performance_optimization.rs`

---

See also: [Workspace roadmap](../../TODO.md)
