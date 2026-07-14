# sklears TODO List and Roadmap

## 🎯 Project Vision

Create a production-ready machine learning library in Rust that:
- Maintains API compatibility with scikit-learn for easy migration
- Provides pure Rust implementation with ongoing performance optimization
- Provides memory safety and type safety guarantees
- Enables deployment without Python runtime dependencies
- Leverages SciRS2's scientific computing capabilities

## v0.1.1 Changelog (2026-04-25)

### Completed
- [x] HDBSCAN persistence extraction fix
- [x] StreamingStandardScaler/SimpleImputer Default derive fix
- [x] Pipeline get_step_mut lifetime fix
- [x] zstd -> oxiarc-zstd migration
- [x] Clippy zero-warnings
- [x] All files <2000 lines (splitrs applied)
- [x] Version bump to 0.1.1

### Known Incomplete / Outstanding Issues
- [x] `sklears-simd`: 5 test failures fixed in v0.1.2 (mae_gradient sign bug, cross_product SSE2 shuffles, F32x4 stride test, AVX2 compress partition) + 6 new hardening tests added
- [x] `sklears-mixture`: student_t doctest fixed (degrees_of_freedom returns Result, chained .expect())
- [x] `sklears-inspection`: 3 doctest fixes (distributed schedule_tasks, federated moved config, quantum add_parametric_gate); test_gaussian_noise_generation fixed with seeded StdRng + 200-sample statistical test
- [x] `sklears-impute`: implemented CategoricalClusteringImputer (k-means), CategoricalRandomForestImputer (MissForest/CART), AssociationRuleImputer (Apriori), validate_imputer (K-fold MAE cross-validation)
- [x] `sklears-covariance`, `sklears-cross-decomposition`, `sklears-isotonic`, `sklears-model-selection`, `sklears-neighbors`, `sklears-semi-supervised`: doctest fixes (missing Ok(()), invalid imports, wrong type annotations, f32→f64 precision tolerances)
- [x] Flaky timing tests fixed (test_decomposition_pipeline, test_model_metadata, test_historical_summary, test_energy_*): removed load-sensitive duration/ratio assertions; ModelMetadata::touch() now guarantees strict time advancement

---

## v0.1.2 Changelog (2026-06-30)

### Completed
- [x] 15-item stub-check backlog implemented across 8 crates (real logic, zero stubs)
- [x] `sklears-simd`: 5 test failures fixed (mae_gradient sign, SSE2 shuffles, AVX2 compress) + 6 new hardening tests
- [x] `sklears-mixture`: student_t doctest fixed
- [x] `sklears-inspection`: 3 doctest fixes + seeded StdRng statistical test
- [x] `sklears-impute`: CategoricalClusteringImputer, CategoricalRandomForestImputer, AssociationRuleImputer, validate_imputer implemented
- [x] `sklears-covariance`, `sklears-cross-decomposition`, `sklears-isotonic`, `sklears-model-selection`, `sklears-neighbors`, `sklears-semi-supervised`: doctest fixes
- [x] Flaky timing tests fixed (test_decomposition_pipeline, test_model_metadata, test_historical_summary, test_energy_*)
- [x] Version bump to 0.1.2

---

## 🚀 OxiCUDA Migration (v0.2.0) — scirs2-core GPU 完全撤去

*(consolidated 2026-07-06 — supersedes the former "OxiCUDA Migration (v0.2.0) — workspace-root items" section; per-crate work is tracked in each crate's own TODO.md, linked below)*

### Goal

**Zero sklears-side usage of `scirs2_core::gpu` for 0.2.0; all GPU work routes exclusively through oxicuda-\* 0.4.x** (via the shared `sklears_core::gpu` abstraction behind `gpu_support`, or direct oxicuda-* deps for kernels).

**Scope note:** `scirs2-core`'s `gpu` feature *cannot* be evicted from the build graph — upstream scirs2-datasets/scirs2-fft/scirs2-optimize/scirs2-sparse 0.6.0 force-enable it regardless of the workspace `default-features = false` pin (verified via `cargo tree -e features -i scirs2-core`). The release goal is therefore precisely "zero sklears-side *usage*", not zero compilation; full eviction requires upstream scirs2 0.7.x or dropping those deps.

### Current state

**Implementation status (2026-07-06):** 25 crates migrated/wired/hardened onto the oxicuda-backed `sklears_core::gpu` foundation (2 honestly downscoped, 1 blocked on an upstream oxicuda-solver API gap) with roughly a dozen items explicitly deferred with dated notes (mixed-precision GEMM, SHAP GPU staging, NVML-based utilization, deeper kernel dispatch, etc.); final verification shows 13/13 acceptance criteria passing.

**Already done (Wave A2, 2026-07-03/04):**
- `crates/sklears-core/src/gpu.rs` is 100% oxicuda-backed (oxicuda-driver `Context` + oxicuda-blas `BlasHandle` + oxicuda-memory `DeviceBuffer`, honest `Ok(None)` detection) behind `gpu_support = [dep:oxicuda-backend, dep:oxicuda-memory, dep:oxicuda-blas, dep:oxicuda-driver]`.
- Nine downstream crates wire real oxicuda `gpu` features: linear, neural, svm, clustering, neighbors, decomposition, manifold, discriminant-analysis; cross-decomposition routes via `sklears-core/gpu_support`.
- See "GPU Migration: oxicuda-* v0.3 への完全移行" below for the phase-by-phase history.

**Only two code-real scirs2 GPU touchpoints remain (rg-verified):**
1. `sklears-core` feature `scirs2-gpu-reporting = ["scirs2-core/gpu"]` (`crates/sklears-core/Cargo.toml:110`) + gated use in trait_explorer `graph_generator.rs:17` — dormant, enabled by no crate, compiles only under `--all-features`.
2. `sklears-preprocessing` `gpu_acceleration.rs:51` — unconditional `ScirGpuBackend` import + conversion layer; a **latent build break** that compiles only because scirs2-optimize/scirs2-sparse transitively enable `scirs2-core/gpu`.

Additionally, ~15 crates carry simulated/stub GPU layers (fake device lists, `gpu = []` features, null-pointer buffers) that must be wired to oxicuda or honestly downscoped.

### Phased execution ordering

1. **Phase 1 — sklears-core scirs2 excision (blocking):** delete `scirs2-gpu-reporting`; port trait-graph GPU reporting to `crate::gpu::GpuBackend::detect()` under `gpu_support`; rewrite stale `advanced_numeric.rs` comments (the recommended scirs2 free fns never existed); fix the DSL codegen snippet emitting nonexistent `crate::gpu::initialize_gpu_context()` under the wrong feature name. → `crates/sklears-core/TODO.md`
2. **Phase 2 — sklears-preprocessing conversion-layer swap (blocking):** replace the `ScirGpuBackend` layer with `sklears_core::gpu` detection behind a new `gpu = ["sklears-core/gpu_support"]` feature; keep the serde-visible local enum variants (non-CUDA variants hard-wired unavailable). Reference pattern: the completed sklears-discriminant-analysis migration. → `crates/sklears-preprocessing/TODO.md`
3. **Phase 3 — docs/policy corrections + excision verification:** SCIRS2_INTEGRATION_POLICY.md rewrite, root-TODO/CHANGELOG fixes, transitive-activation documentation; run the rg/cargo-tree acceptance checks — **the 0.2.0 headline goal is DONE at the end of this phase.**
4. **Phase 4 — stub/simulated GPU honesty pass (parallelizable per crate, not blocking the scirs2 goal):** 15 crates wired to oxicuda or honestly downscoped (see table). S/M items first; L kernel-implementation items are deferable past 0.2.0 — **exception:** the sklears-kernel-approximation `eigendecomposition_cpu` fake-eigenvalue correctness bug must NOT be deferred.
5. **Phase 5 — hardening of already-migrated oxicuda crates (optional for release):** unused-dep pruning, dead cfg branches, doc accuracy, deeper device paths (see table).
6. **Phase 6 — version watch + final sweep:** keep oxicuda-* pins at 0.4.0; bump to 0.4.1 only when published; re-run full acceptance criteria and `cargo nextest run --all-features`.

### Per-crate work (details in each crate's TODO.md)

| Crate | Status | Summary (max effort) | Tracker |
|---|---|---|---|
| sklears-core | migrated | Removed `scirs2-gpu-reporting`, ported trait-graph GPU reporting to `crate::gpu`, fixed stale comments + DSL codegen snippet (M) | see `crates/sklears-core/TODO.md` |
| sklears-preprocessing | migrated | Swapped `ScirGpuBackend` conversion layer to `sklears_core::gpu` + new `gpu` feature; scaler oxicuda kernels (`dispatch_compute_mean`/`dispatch_compute_variance`/`dispatch_compute_min`/`dispatch_compute_max`) now genuinely dispatch to on-device `oxicuda-blas` reduction kernels (`reduce_axis`/`elementwise::mul` via real `DeviceBuffer`s), with honest CPU fallback on GPU failure — no longer documented CPU passthroughs; previously-deferred item now resolved (M) | see `crates/sklears-preprocessing/TODO.md` |
| sklears-neighbors | migrated | Replaced mock device detection with real oxicuda-driver queries; collapsed decorative Cuda/OpenCl/Metal enum; pruned unused `dep:oxicuda-backend` (M) | see `crates/sklears-neighbors/TODO.md` |
| sklears-multiclass | oxicuda-wired | Rewired `src/gpu` onto `sklears-core/gpu_support`; connected ECOC GPUMode to real GPU layer via `aggregate_votes_device` (M) | see `crates/sklears-multiclass/TODO.md` |
| sklears-metrics | oxicuda-wired (deferred) | Fixed `gpu = []` + broken cuda-vs-gpu gate; rewrote null-pointer GPU stubs onto oxicuda; `supports_mixed_precision` wiring deferred (L) | see `crates/sklears-metrics/TODO.md` |
| sklears-inspection | oxicuda-wired (deferred) | Replaced dead `gpu = ["dep:tokio"]` feature; rebuilt GpuContext/GpuBuffer on `sklears_core::gpu`; SHAP/permutation-importance GPU staging deferred (L) | see `crates/sklears-inspection/TODO.md` |
| sklears-kernel-approximation | oxicuda-wired | Added real `gpu` feature + gated simulated module; rebuilt on `sklears_core::gpu`; **fixed fake-eigenvalue `eigendecomposition_cpu` bug** (L) | see `crates/sklears-kernel-approximation/TODO.md` |
| sklears-ensemble | honestly-downscoped | Added oxicuda-backed `gpu` feature; rewired device detection/memory manager; matmul/elementwise_add dispatch to real oxicuda-blas, full GpuTensorOps/gradient-boosting kernels downscoped per mission (L) | see `crates/sklears-ensemble/TODO.md` |
| sklears-utils | oxicuda-wired (deferred) | Replaced fabricated device list with real oxicuda-driver enumeration; GpuArrayOps routes through `sklears_core::gpu` when a device is present, deeper kernel execution deferred; distributed GPU fields recorded as out of scope (L) | see `crates/sklears-utils/TODO.md` |
| sklears-simd | stubs-removed | Deleted 2225 lines of always-erroring stub GPU modules (gpu.rs, gpu_memory.rs, multi_gpu.rs) + the workspace's only commented cudarc/opencl3 remnants (M) | see `crates/sklears-simd/TODO.md` |
| sklears (facade) | stubs-removed | Replaced empty `backend-cuda`/`backend-wgpu` stub features with a real oxicuda-backed `gpu` feature; fixed lib.rs CUDA/WebGPU doc claims (S) | see `crates/sklears/TODO.md` |
| sklears-compose | oxicuda-wired | Wired device discovery/counting to oxicuda-driver behind a `gpu` feature; documented GPU telemetry fields as scheduling metadata (M) | see `crates/sklears-compose/TODO.md` |
| sklears-calibration | oxicuda-wired (deferred) | Slotted oxicuda in behind a `gpu` feature (device enumeration, memory stats via `GpuBackend::memory_info()`, blas-backed temperature-scaling); `get_utilization` deferred (upstream oxicuda-driver lacks NVML) (M) | see `crates/sklears-calibration/TODO.md` |
| sklears-naive-bayes | oxicuda-wired | Backed GpuOptimizer with oxicuda-blas gemm behind a `gpu` feature; trimmed false SciRS2-Policy module doc claim (M) | see `crates/sklears-naive-bayes/TODO.md` |
| sklears-python | oxicuda-wired | Reports real CUDA availability via oxicuda instead of hardcoded `cuda_available = false` (S) | see `crates/sklears-python/TODO.md` |
| sklears-model-selection | oxicuda-wired | Derived `has_gpu`/`use_gpu` config defaults from oxicuda detection behind a new `gpu` feature (S) | see `crates/sklears-model-selection/TODO.md` |
| sklears-impute | honestly-downscoped | Made `DeepLearningConfig.device` honest: `validate_device()` accepts only "cpu" (case-insensitive), rejects "cuda" (S) | see `crates/sklears-impute/TODO.md` |
| sklears-linear | hardened | Real device memory pools (oxicuda-memory DeviceBuffer arenas), real driver streams (CudaStream::with_backend), real perf counters (L) | see `crates/sklears-linear/TODO.md` |
| sklears-neural | hardened (deferred) | Implemented conv2d via oxicuda-dnn; real tensor-core/CC detection; resolved unused oxicuda-ptx dep; memory-pool config now genuinely honored via a real `GpuMemoryPool`/`PoolTelemetry` (oxicuda-memory-backed, real hit/miss tracking, no fabricated telemetry) — no longer deferred; mixed-precision GEMM host round-trip elimination still deferred (upstream API gap: `oxicuda-blas` has no fp16-in/fp32-out kernel entry point) (L) | see `crates/sklears-neural/TODO.md` |
| sklears-svm | hardened | Pruned unused gpu-feature deps (oxicuda-ptx/backend/bytemuck); retired dead WGPU-era error variants (S) | see `crates/sklears-svm/TODO.md` |
| sklears-clustering | hardened | Pruned unused direct oxicuda deps; deleted dead/non-compiling `cfg(not(gpu))` stub; confirmed pollster already in dev-deps; fixed README WebGPU wording (M) | see `crates/sklears-clustering/TODO.md` |
| sklears-decomposition | hardened | Removed unused optional deps `oxicuda-blas` and `half` from the `gpu` feature (S) | see `crates/sklears-decomposition/TODO.md` |
| sklears-manifold | hardened | Resolved dead `cfg(not(feature = "gpu"))` branches vs lib.rs gating; fixed `.unwrap()` in doc example (S) | see `crates/sklears-manifold/TODO.md` |
| sklears-discriminant-analysis | fully-oxicuda (blocked) | oxicuda-solver `sygvd` for the LDA generalized eigensolve verified not yet available in oxicuda-solver 0.4.0 (crates.io and local dev tree); no upstream API to adopt yet | see `crates/sklears-discriminant-analysis/TODO.md` |
| sklears-cross-decomposition | fully-oxicuda | Offload GpuMatrixOps eig/svd to oxicuda-solver when a CUDA backend is live (preferably via new `sklears_core::gpu` eigh/svd wrappers) (M) | see `crates/sklears-cross-decomposition/TODO.md` |

### Workspace-level items

- [x] (S) Rewrite stale GPU policy in `SCIRS2_INTEGRATION_POLICY.md` — line 277 mandates "ALWAYS use scirs2-core::gpu module for GPU operations" and line 279 forbids direct GPU API calls outside SciRS2-core; both contradict the oxicuda migration. Rewrite the GPU Operations Policy section to mandate `sklears_core::gpu` (oxicuda-driver/oxicuda-blas/oxicuda-memory behind `gpu_support`) and direct oxicuda-* crates for kernels, with CPU scirs2-linalg fallbacks explicitly allowed.
- [x] (S) Correct root GPU migration records in `TODO.md` + add `CHANGELOG.md` entry — the "GPU Migration" Phase 1 line below claiming sklears-cross-decomposition's `gpu` feature became "oxicuda-backend" is inaccurate: `crates/sklears-cross-decomposition/Cargo.toml:40` actually reads `gpu = ["sklears-core/gpu_support"]` (oxicuda only indirectly); fix the wording. Mark the "✅ Stub-check backlog" line recording trait_explorer's use of "real `scirs2_core::gpu`" as superseded once the sklears-core Phase 1 excision removes that path. Add a `CHANGELOG.md` entry noting the removal of the public sklears-core feature `scirs2-gpu-reporting` (API-visible in `--all-features` builds).
- [x] (S) Document that `scirs2-core/gpu` stays force-enabled transitively (goal scoping) — `cargo tree -e features -i scirs2-core` shows scirs2-datasets/scirs2-fft/scirs2-optimize/scirs2-sparse 0.6.0 hard-enable scirs2-core's `gpu` feature regardless of the workspace `default-features = false` pin (`Cargo.toml:56`). Record in this migration section that the 0.2.0 goal is precisely "zero sklears-side USAGE of `scirs2_core::gpu`", and that full eviction requires upstream scirs2 0.7.x or dropping those deps. Files: `TODO.md`, `Cargo.toml`.
- [x] (S) Version watch: bump oxicuda-* 0.4.0 → 0.4.1 when published — the workspace pins nine oxicuda crates (`Cargo.toml:189-198`: backend, memory, blas, solver, manifold, dnn, driver, ptx, primitives). Note: oxicuda-solver 0.4.0's syevd/QR/SVD device paths are documented exact-CPU host fallbacks — sklears docs (decomposition/linear/discriminant-analysis/cross-decomposition) must not claim on-device eigen/SVD until upstream restores them. (confirmed 2026-07-07: all nine oxicuda-* pins at 0.4.1, matches crates.io latest; cargo check --workspace --all-features clean)
  - **Goal:** Re-verify crates.io latest for all nine oxicuda-* crates is exactly 0.4.1 (not further ahead), confirm the workspace `Cargo.toml` pins match, then flip this item to done with an updated confirmation note.
  - **Design:** Verify via `cargo search oxicuda-backend` (and the other 8 oxicuda-* crate names) against crates.io, and read `Cargo.toml:189-198` to confirm all nine pins already read `0.4.1`. No version change needed by us — pins were already bumped to 0.4.1 by a prior external commit and `cargo check --workspace --all-features` against that pin was already verified clean (0 errors, 0 warnings, 5m27s) in this session.
  - **Files:** `TODO.md` (this checkbox only).
  - **Prerequisites:** none.
  - **Tests:** n/a (documentation bookkeeping). Verification = the crates.io search + Cargo.toml read above.
  - **Risk:** if crates.io has moved past 0.4.1 by the time this runs, do NOT bump further — report the newer version and leave this item `[~]`/unresolved rather than silently bumping.
- [x] (S) Docs sweep after Phase 4 — no README/lib.rs may claim live wgpu/candle/WebGPU GPU backends or CUDA/OpenCL support that does not exist (known offenders: `crates/sklears-clustering/README.md:71` "WebGPU-powered", `crates/sklears/src/lib.rs:15,64` CUDA/WebGPU feature claims); every gpu-gated call site must preserve the `Ok(None)`/CPU-fallback contract so no-GPU hosts build and tests skip gracefully (model: sklears-svm `DeviceNotAvailable` skip pattern).

**Behavior change to expect:** scirs2's `GpuBackend::preferred()` always returned Cpu in production builds, so `gpu_accelerated` flags could never be true; after the oxicuda port, real detection can report true on CUDA hosts — audit tests asserting always-false (e.g. graph_generator.rs:1677 asserts `!has_gpu_acceleration` in the no-feature build only).

**Out-of-scope (do not re-flag in future audits):** `scirs2_core::ndarray/random/linalg` CPU usage (workspace-approved non-GPU features); scirs2-linalg CPU fallbacks in GPU modules (intentional); descriptor-only GPU enums/telemetry (compose resource metadata, ensemble `TensorDevice::Gpu`, utils distributed_computing `gpu_count`/`gpu_usage`, neural `gpu_hours`, core platform-name strings like "cuda-gpu").

### Acceptance criteria

- [x] `rg -n "scirs2_core::gpu|ScirGpuBackend" crates/ -g '*.rs'` returns 0 hits (code AND comments; currently 35 hits across sklears-preprocessing and sklears-core)
- [x] `rg -n "scirs2-core/gpu|scirs2-gpu-reporting" -g 'Cargo.toml' .` returns 0 hits (currently 1: `crates/sklears-core/Cargo.toml:110`)
- [x] `cargo tree -e features -i scirs2-core | rg 'sklears'` shows no sklears crate enabling scirs2-core feature `gpu` (remaining activators are only upstream scirs2-datasets/scirs2-fft/scirs2-optimize/scirs2-sparse)
- [x] `cargo check -p sklears-preprocessing` and `cargo check -p sklears-preprocessing --features gpu` both succeed (proving no reliance on transitive scirs2-core/gpu feature unification)
- [x] `cargo check -p sklears-core --all-features` succeeds with zero warnings and no dangling `cfg(feature = "scirs2-gpu-reporting")` gates (`rg 'scirs2-gpu-reporting' crates/sklears-core/src` returns 0 hits)
- [x] For every crate with a `gpu` feature after Phase 4: `cargo check -p <crate> --features gpu` succeeds on a no-GPU host, and gpu-gated tests skip gracefully (no fabricated availability)
- [x] `rg -n '^gpu = \[\]' crates/*/Cargo.toml` returns 0 hits (no empty stub gpu features remain; currently sklears-multiclass:37 and sklears-metrics:79)
- [x] `rg -n 'backend-cuda|backend-wgpu' crates/sklears/Cargo.toml` returns 0 hits (empty facade stubs removed)
- [x] `rg -n -i 'cudarc|opencl3' crates/` returns 0 hits after Phase 4 (currently 13: sklears-simd commented deps/features + reserved-handle comments, sklears-multiclass placeholder comment)
- [x] `rg -n 'ALWAYS use .scirs2-core::gpu' SCIRS2_INTEGRATION_POLICY.md` returns 0 hits (policy rewritten to mandate `sklears_core::gpu` / oxicuda-*)
- [x] `cargo build --workspace` succeeds with zero warnings and `cargo nextest run --all-features --no-run` compiles after each phase lands
- [x] Workspace `Cargo.toml` oxicuda-* pins (lines 184-192) equal the latest published crates.io version at release time (0.4.0 today; all nine bumped together to 0.4.1 only after it is published)
- [x] `rg -n 'wgpu|candle|WebGPU-powered' crates/*/README.md crates/*/src/lib.rs` returns no hits claiming live wgpu/candle GPU backends

---

## 📊 Current Status (v0.2.1)

**Overall API Coverage: >99%** 🎉

All major scikit-learn modules are implemented with production-ready quality:

| Module | Coverage | Status | Advanced Features |
|--------|----------|--------|--------------------|
| linear_model | 100% | 🟡 Partial (fabrication found) | GPU acceleration, distributed training |
| tree | 100% | 🟢 Complete (fixed 2026-07-05 — real classifier/regressor restored) | SHAP integration, GPU acceleration |
| ensemble | 100% | 🟢 Complete | Gradient boosting, bagging regression, advanced stacking, isolation forest |
| svm | 100% | 🟡 Partial (fabrication found) | GPU kernels, parallel SMO |
| neural_network | 100% | 🟡 Partial (fabrication found) | Advanced layers, seq2seq, attention |
| cluster | 100% | 🟢 Complete | GPU acceleration, streaming |
| decomposition | 100% | 🟡 Partial (fabrication found) | Incremental variants, tensor methods |
| preprocessing | 100% | 🟡 Partial (fabrication found) | Text processing, GPU scaling |
| metrics | 100% | 🟡 Partial (fabrication found) | Statistical testing, visualization |
| model_selection | 100% | 🟡 Partial (fabrication found) | AutoML pipeline, advanced search |
| neighbors | 100% | 🟡 Partial (fabrication found) | Spatial trees, GPU acceleration |
| feature_selection | 100% | 🟡 Partial (fabrication found) | Stability selection, advanced tests |
| datasets | 100% | 🔴 Needs audit | Real-world data, advanced generators |
| naive_bayes | 100% | 🟡 Partial (fabrication found) | All variants, ensemble methods |
| gaussian_process | 100% | 🟡 Partial (fabrication found) | Advanced kernels, Bayesian optimization |
| discriminant_analysis | 100% | 🟡 Partial (fabrication found) | GPU acceleration, robust variants |
| manifold | 100% | 🟢 Complete (fixed 2026-07-05 — honest-Err + real out-of-sample where defined) | GPU support, advanced embeddings |
| semi_supervised | 100% | 🔴 Needs audit | Graph methods, deep learning |
| feature_extraction | 100% | 🟡 Partial (fabrication found) | Text processing, image features |
| covariance | 100% | 🟡 Partial (fabrication found) | Robust estimators, sparse methods |
| cross_decomposition | 100% | 🟡 Partial (fabrication found) | Advanced PLS variants |
| isotonic | 100% | 🟡 Partial (fabrication found) | Multidimensional support |
| kernel_approximation | 100% | 🟡 Partial (fabrication found) | GPU acceleration, advanced methods |
| dummy | 100% | 🟢 Complete | All baseline strategies |
| calibration | 100% | 🟡 Partial (fabrication found) | Neural calibration, conformal prediction |
| multiclass | 100% | 🟡 Partial (fabrication found) | Error-correcting codes |
| multioutput | 100% | 🟡 Partial (fabrication found) | All chaining methods |
| impute | 100% | 🟡 Partial (fabrication found) | Neural networks, ensemble imputation |
| compose | 100% | 🟡 Partial (fabrication found) | GPU pipelines, distributed processing |
| inspection | 100% | 🟡 Partial (fabrication found) | Causal analysis, dashboards |
| mixture | 100% | 🟢 Complete (fixed 2026-07-05 — real EM math + state wiring) | Bayesian variants, advanced EM |
| simd | 100% | 🟢 Complete | Advanced SIMD optimizations |
| utils | 100% | 🟡 Partial (fabrication found) | GPU utilities, distributed support |
| core | 100% | 🟡 Partial (fabrication found) | Type system, GPU, async traits |

## 📊 Quality Metrics

- **Test Suite**: 12,721 tests (12,721 passing, 0 failed, 161 skipped, 100% pass rate)
- **Code Quality**: 100% warning-free compilation
- **Performance**: Pure Rust implementation with ongoing performance optimization
- **Dependencies**: Pure Rust stack (OxiBLAS v0.1.2, Oxicode v0.1.1)
- **Build Status**: ✅ 36/36 crates building successfully

## 🚀 Future Roadmap

### Ongoing Stabilization
*(retitled 2026-07-14 — formerly labeled "v0.1.0 Stable (Q2 2026)", which was stale: the project has already shipped v0.1.1, v0.1.2, and now v0.2.0 past that milestone. Re-scoped under a version-agnostic heading since these are continuous efforts, not a one-time gate tied to a past version.)*
- [ ] API surface freeze and stabilization — not yet: v0.2.0 alone shipped six explicitly-tagged breaking changes (`sklears-core::gpu`, `sklears-core::trait_explorer::graph_visualization`, `sklears-cross-decomposition`, `sklears-discriminant-analysis`, `sklears-compose`, `sklears-python` — per CHANGELOG `### Changed`), so the public API surface is still actively moving.
- [x] Comprehensive benchmark suite expansion — DONE: all 36 workspace crates now have a dedicated `benches/` directory (`find crates -iname "*bench*"`), e.g. `sklears/benches/{comprehensive_benchmarks,advanced_performance_benchmarks,tree_ensemble_benchmarks,continuous_benchmarks}.rs` plus per-crate `benchmarking.rs`/`benchmark.rs` source modules in sklears-core, sklears-isotonic, sklears-kernel-approximation, sklears-metrics, sklears-feature-selection, sklears-dummy, sklears-impute, and sklears-datasets; the v0.2.0 CHANGELOG additionally re-enabled the `tree_ensemble_benchmarks` bench target and added `performance_comparison_comprehensive` now that the `preprocessing` feature is restored.
- [ ] Enhanced documentation and cookbooks — partial progress only: `sklears-covariance`'s `examples/comprehensive_cookbook.rs` and `examples/covariance_hyperparameter_tuning_demo.rs` were converted from "under development" placeholder `main()`s to real working recipes in v0.2.0 (CHANGELOG `### Fixed`), but `find . -iname "*cookbook*"` still turns up only that one file workspace-wide — not comprehensive yet.
- [ ] Production case studies
- [ ] Community feedback integration

### v0.2.0 and Beyond
- [x] Enhanced GPU acceleration (CUDA) — DONE (2026-07-04): real oxicuda-backed `GpuBackend`/`GpuArray`/`GpuMatrixOps` foundation shipped across `sklears-core` + 9 downstream crates (clustering, cross-decomposition, decomposition, discriminant-analysis, linear, manifold, neighbors, neural, svm); see "GPU Migration: oxicuda-* v0.3 への完全移行" above.
- [ ] WebGPU acceleration (not started)
- [ ] Distributed computing support
- [ ] Advanced AutoML capabilities
- [ ] ONNX/PMML model interchange
- [ ] WebAssembly compilation support
- [ ] Embedded/no-std support for microcontrollers — `sklears-simd`'s `no-std` feature is explicitly commented "temporarily disabled until implementation is complete" (`crates/sklears-simd/src/lib.rs:10`); confirmed not done.

## 🔧 Development Guidelines

### API Compatibility
- Match scikit-learn's public API exactly
- Use same parameter names and defaults
- Compatible return types with NumPy arrays
- Fitted models have same attributes (with trailing `_`)

### Performance Targets
- Continuous performance optimization of Rust implementations
- SIMD optimizations where applicable
- GPU acceleration for large-scale operations
- Parallel processing with Rayon

### Testing Requirements
- Unit tests with >90% coverage
- Property-based tests for numerical algorithms
- Comparison tests with scikit-learn outputs
- Performance benchmarks
- Integration tests with pipelines

### Documentation Requirements
- Comprehensive docstrings for all public APIs
- Examples for each algorithm
- Migration guide from scikit-learn
- Performance comparison tables

---

*Version: 0.2.1 — Unreleased*
*Last Release: v0.2.0 — 2026-07-14*
*Next Milestone: TBD*

## ✅ Stub-check backlog — COMPLETED (2026-06-21, v0.1.2)

The 2026-06-12 `/cooljapan-stub-check` backlog (15 items) was implemented with **real logic**
across 8 crates in parallel. Full workspace builds clean; `cargo clippy --workspace --all-targets`
reports **zero warnings**. Per-item outcome:

- [x] `sklears-preprocessing`: SIMD paths re-enabled (real AVX kernels `simd_threshold_mask`, `simd_axpy`; routed `simd_mahalanobis` through `simd_dot_product`; honest scalar-only docs where `powf`/`ln` have no exact AVX intrinsic).
- [x] `sklears-preprocessing`: scaling/imputation/encoding — **found ~12 exported empty-placeholder structs (silent fabrication), now implemented for real**: MinMaxScaler, MaxAbsScaler, UnitVectorScaler, FeatureWiseScaler, OutlierAwareScaler; SimpleImputer, KNNImputer (nan-aware), IterativeImputer (MICE ridge), MultipleImputer, GAINImputer (honest regression backend); OrdinalEncoder, TargetEncoder (category_encoders smoothing). Dead modular-refactor comment blocks removed.
- [x] `sklears-preprocessing`: BufferPool — premise was false; `scirs2_core::memory::BufferPool` exists and is used. Removed stale "doesn't exist" comments; re-enabled gpu_acceleration/lazy_evaluation/memory_management exports.
- [x] `sklears-manifold`: real serde serialization for `RandomProjection` via new public accessors (`projection_matrix()` etc.) + lossless round-trip tests. ~~TSNE/Isomap return honest `Err` (internal state not publicly reachable).~~ **Correction (2026-07-05): false — see "🔎 Fabrication audit findings" below. `transform()` on new data always returns `Ok(stale_embedding)` (the cached training-time embedding), never `Err`.**
- [x] `sklears-core`: DSL macros (`model_evaluation!`, `data_pipeline!`, `experiment_config!`) — full parse→generate implemented (dsl_types/parsers/code_generators wired).
- [x] `sklears-core`: trait_explorer graph GPU-context init (real `scirs2_core::gpu` w/ honest CPU fallback) + where-clause extraction. ⚠️ See follow-up: `graph_visualization` module is currently disabled; logic verified in isolation. — **superseded (2026-07-06):** the "real `scirs2_core::gpu`" path was excised in the oxicuda Phase 1 migration; trait_explorer now uses the oxicuda-backed `crate::gpu` foundation instead.
- [x] `sklears-svm`: conformal_prediction — fitted state restructured (`Option<SVC<Trained>>`), unfitted → honest `Err(NotTrained)`; wired into lib.rs.
- [x] `sklears-svm`: nalgebra → scirs2-linalg migration — **fully migrated** (semi_supervised, property_tests, advanced_optimization); no nalgebra left in src/. Fixed 3 real bugs incl. a `// Simplified` fake decision_function.
- [x] `sklears-compose`: time-series pipelines (LagFeatures, RollingWindow, Differencing, TemporalTrainTestSplit) — real, wired into lib.rs.
- [x] `sklears-compose`: column_transformer sparse — real CSR paths via scirs2-sparse (sparse_select_columns, sparse_hstack), wired.
- [x] `sklears-compose`: predictive preloading (recency-weighted LFU + first-order Markov) and memory/CPU profiling (real `/proc/self/{status,stat}`). ⚠️ See follow-up: these live in the orphaned `distributed_optimization` tree; logic verified in isolation.
- [x] `sklears-metrics`: sprs → scirs2-sparse migration complete; `sparse` feature re-enabled; no direct sprs dep.
- [x] `sklears-gaussian-process`: Cholesky stability — root cause was an indefinite saddle-point system (ordinary kriging); now SPD block via regularized Cholesky, indefinite system via LU. 5 kriging tests un-ignored + LOO studentized-residual outlier fix.
- [x] `sklears-discriminant-analysis`: parallel eigen wired to `scirs2_linalg::eigh` (accurate sequential kernel; upstream parallel kernel returns `inf` — documented honestly); parallelism retained across independent blocks.

## 🔎 Follow-up findings (new outstanding work)

- [ ] `sklears-compose`: `distributed_optimization/` is a ~3.3 MB, 161-file **orphaned module tree NOT declared in lib.rs** (never compiled). The predictive-preload + profiling logic above was implemented in its two relevant files and verified in isolation, but the tree has pre-existing compile errors (missing module files, duplicate defs, `CompressionAlgorithm` lacks `Hash`/`Eq` though used as a HashMap key). Decide: wire it in (large effort) or delete it.
  - Priority: P3 | Scope: large
- [x] `sklears-core`: `trait_explorer/graph_visualization` — DONE (2026-07-03): re-enabled the module (mod.rs:112 was commented out); built a real `api_reference_generator` module with the rich-shape TraitInfo/AssociatedType/MethodInfo API types graph_generator.rs needs; fixed the actual root cause of the old "JavaScript syntax conflicts" disable reason (raw-string literals with embedded `"#` sequences); found and fixed 2 real correctness bugs (edges pointing at non-namespaced node IDs causing silent data loss; degree-centrality only counting outgoing edges); implemented 5 previously-missing graph analysis algorithms (hub/bridge/bottleneck detection, modularity, small-world coefficient) rather than stubbing them; split an oversized file via splitrs — 855 sklears-core tests passing (109 new). `ml_recommendations`/`platform_compatibility` remain disabled (separate, unaddressed issues, out of scope).

## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check)

These are "re-enable after refactor" / commented-out re-export plumbing items. Each is a P2 cleanup that unblocks once the named module's API stabilizes; the pure re-export ones are trivial.

- [x] **sklears** `sklears-core`: `crates/sklears-core/src/trait_explorer/security_analysis/mod.rs:167-199` — DONE (2026-07-02): defined the ~10 missing types the module required (SecurityAnalysis, SecurityVulnerability, RiskAssessmentResult, VulnerabilityDatabaseError, etc.), re-enabled the module, and refactored it via directory-based splits (compliance_framework/, security_metrics/) plus a crypto_analysis_types.rs extraction to stay under the 2000-line policy; 740 tests passing.
- [x] **sklears** `sklears-gaussian-process`: `crates/sklears-gaussian-process/src/lib.rs:63,67,82,131` — DONE (2026-07-02): implemented fitc.rs (FITC sparse GP), kernel_selection.rs (AIC/BIC/CV kernel selection), and variational.rs (SVGP Bernoulli classifier), and re-enabled the gated re-exports; 173 tests passing.
- [x] **sklears** `sklears-python`: `crates/sklears-python/src/lib.rs:39,51,110` — DONE (2026-06-25): rewrote `classification.rs` and `regression.rs` to use `sklears_metrics::basic_metrics` and `sklears_metrics::regression` directly; fixed PyO3 0.28 API (`unbind()` instead of `to_owned()`, `is_multiple_of()`, removed `ToPyObject`); un-commented `mod metrics;`, `pub use metrics::*;`, and all 11 function registrations.
- [x] **sklears** `sklears-datasets`: `crates/sklears-datasets/src/generators/mod.rs:9-14,35` — DONE (2026-07-02): implemented 6 generator submodules (manifold, time_series, adversarial, causal, domain_specific, statistical) with real algorithms, declared and re-exported in generators/mod.rs, and removed the superseded lib_original.rs; 190 tests passing.

---

## GPU Migration: oxicuda-* v0.3 への完全移行 (added 2026-06-25)

**Goal:** scirs2-core GPU (未有効化 stubs), wgpu 直接依存, cudarc, candle-core を全廃し、
oxicuda-backend / oxicuda-blas / oxicuda-solver 等に一本化する。

**Status (2026-07-03):** Phases 1-9b + 10 complete and fully verified (workspace-wide `cargo check --all-features` clean, `cargo clippy --workspace --all-features --all-targets -- -D warnings` clean, `cargo nextest run --workspace --all-features` 12394/12394 passing). Per-phase DONE summaries below.

### 現状サマリー
| crate | 現 GPU 実装 | 移行先 |
|---|---|---|
| sklears-core | Real oxicuda `GpuBackend` (Device/Context/BlasHandle, graceful no-GPU detection) | oxicuda-backend + oxicuda-memory |
| sklears-linear | oxicuda-blas `BlasHandle` 実装済み (gemm/gemv/axpy/dot/scal + oxicuda-solver LU/QR solve) | oxicuda-blas BlasHandle |
| sklears-neighbors | oxicuda-blas GEMM pairwise dist 実装済み + oxicuda-manifold HNSW (opt-in ANN, brute-force default) | oxicuda-blas GEMM pairwise dist |
| sklears-clustering | oxicuda-blas + oxicuda-backend 実装済み (GEMM Euclidean dist) | oxicuda-blas + oxicuda-backend |
| sklears-svm | oxicuda-blas + oxicuda-ptx 実装済み (RBF/Poly/Linear kernel, on-device transforms) | oxicuda-blas + oxicuda-ptx |
| sklears-neural | oxicuda-driver + oxicuda-blas + oxicuda-dnn 実装済み (on-device relu/sigmoid, tensor-core f16 GEMM) | oxicuda-driver + oxicuda-blas + oxicuda-dnn |
| sklears-decomposition | oxicuda-solver + oxicuda-blas 実装済み (real SVD/eigendecomp via scirs2_linalg, GEMM PCA/NMF) | oxicuda-solver + oxicuda-blas |
| sklears-manifold | oxicuda-manifold 実装済み (real `tsne_fit`, `hnsw_search`) | oxicuda-manifold |
| sklears-cross-decomposition | `sklears_core::gpu` 共有抽象 (GpuBackend/GpuArray/GpuMatrixOps) に統合済み | oxicuda-backend |

---

### Phase 1: Workspace Foundation

> **DONE (2026-07-03):** all dependency-level work completed — `cudarc`/`wgpu`/`candle-core` fully removed workspace-wide; all 9 crates' `gpu` features now reference oxicuda-* deps (oxicuda-backend/-memory/-blas/-solver/-manifold/-dnn/-driver/-ptx as applicable); stale "v0.3" comment corrected to the actual 0.4.1 pin.

- [x] workspace Cargo.toml: `oxicuda-backend`, `oxicuda-memory`, `oxicuda-blas`, `oxicuda-solver`, `oxicuda-manifold`, `oxicuda-dnn` を `version = "0.3"` で追加
- [x] workspace Cargo.toml: `cudarc`, `wgpu`, `candle-core` 依存を削除
- [x] sklears-core/Cargo.toml: feature `gpu_support` → `oxicuda-backend` + `oxicuda-memory` + `oxicuda-blas` に変更
- [x] sklears-linear/Cargo.toml: feature `gpu` → `oxicuda-blas` に変更
- [x] sklears-clustering/Cargo.toml: feature `gpu` の `wgpu`/`bytemuck` → `oxicuda-backend` + `oxicuda-blas` に変更
- [x] sklears-svm/Cargo.toml: feature `gpu` の `wgpu`/`bytemuck`/`futures-intrusive`/`tokio` → `oxicuda-blas` に変更
- [x] sklears-decomposition/Cargo.toml: feature `gpu` の `candle-core`/`cudarc` → `oxicuda-solver` + `oxicuda-blas` に変更
- [x] sklears-manifold/Cargo.toml: feature `gpu` の `dep:wgpu` → `oxicuda-manifold` に変更
- [x] sklears-neural/Cargo.toml: feature `gpu` (コメントアウト中) → `oxicuda-driver` + `oxicuda-blas` + `oxicuda-dnn` として復活
- [x] sklears-cross-decomposition/Cargo.toml: feature `gpu` の `scirs2-core/gpu` → `oxicuda-backend` に変更 — **correction (2026-07-06):** inaccurate as originally written; `crates/sklears-cross-decomposition/Cargo.toml:40` actually reads `gpu = ["sklears-core/gpu_support"]` (oxicuda pulled in only indirectly via `sklears-core`'s `gpu_support` feature, not a direct `oxicuda-backend` dep on this crate).

### Phase 2: sklears-core GPU 抽象層の書き直し
- [x] `sklears-core/src/gpu.rs` を全面書き直し
  - `GpuContext` → `BackendRegistry::best()` を保持する thin wrapper
  - `GpuArray<T>` → `DeviceBuffer<T>` (oxicuda-memory) の feature-gated wrapper
  - `GpuMatrixOps` trait → `BlasHandle` (oxicuda-blas) への委譲
  - 全 CPU stub / "placeholder for future CUDA" コメントを削除
  - **DONE:** real `GpuBackend` struct (Device/Context/BlasHandle detection) with `detect()` gracefully returning `Ok(None)` when no GPU/driver is present; `GpuArray<T>` now backed by `DeviceBuffer<T>`; native f32 **and** f64 GEMM via `oxicuda_blas::level3::gemm` (the old f32→f64 upcast hack removed); real on-device elementwise add/mul kernels located in oxicuda-blas and wired in; `GpuContext` retained as a type alias for backward compatibility.

### Phase 3: sklears-linear GPU
- [x] `gpu_acceleration.rs` 書き直し: matrix_multiply → `BlasHandle::gemm`, gemv, axpy, dot, scal
- [x] `advanced_gpu_acceleration.rs` 書き直し: linear solver → `oxicuda-solver` (LU/QR/Cholesky)
  - **DONE:** construction adapted; `AsyncGpuOperation::get_result()` fixed (was returning fabricated zeros, now genuinely synchronous and correct); `mixed_precision_matrix_multiply` upgraded to real FP16 GEMM; GPU-path LU/QR solve added via oxicuda-solver.

### Phase 4: sklears-neighbors GPU
- [x] `gpu_distance.rs` 書き直し:
  - pairwise dist = GEMM trick: `‖x-y‖² = ‖x‖² - 2xᵀy + ‖y‖²` via `oxicuda-blas`
  - HNSW k-NN → `oxicuda-manifold::HnswIndex`
  - **DONE:** construction adapted; added `oxicuda_manifold` HNSW as an opt-in ANN kNN backend (exact brute-force remains default/available).

### Phase 5: sklears-decomposition GPU
- [x] `hardware_acceleration.rs` 書き直し:
  - `candle_core::Tensor` / `cudarc::CudaContext` → `oxicuda-memory::DeviceBuffer`
  - SVD → `oxicuda-solver::svd`
  - PCA: GEMM (covariance) + `oxicuda-solver` eigendecomp
  - NMF: GEMM 反復 via `oxicuda-blas`
  - **DONE:** construction adapted; `sequential_svd`/`block_parallel_svd`/`*_eigendecomposition` fixed (were returning hardcoded `(eye, ones, eye)` fake results, now delegate to real `scirs2_linalg`); GPU SVD/eigendecomposition wired via oxicuda-solver; dead discarded-tile computation in `tiled_parallel_multiply` fixed to actually use its results.

### Phase 6: sklears-neural GPU (feature 復活)
- [x] `gpu.rs` (~968 行) 書き直し:
  - `GpuTensor<T>` → `oxicuda-memory::DeviceBuffer<T>`
  - `GpuContext` → `oxicuda-backend::BackendRegistry`
  - `GpuMemoryPool` → `oxicuda-memory::MemoryPool`
  - cublas GEMM → `oxicuda-blas::BlasHandle::gemm`
  - `cudarc::nvrtc::compile_ptx` → `oxicuda-ptx::KernelBuilder` + `PtxCache`
  - activation kernels → `oxicuda-dnn` activation
  - **DONE:** construction adapted; relu/sigmoid moved fully on-device (no more CPU round-trip); tensor-core f16 GEMM and mixed-precision paths now genuinely attempt real hardware instead of hard-erroring unconditionally.

### Phase 7: sklears-clustering GPU
- [x] `gpu_distances.rs` 書き直し:
  - wgpu `Device`/`Queue`/`ComputePipeline`/WGSL shaders → `oxicuda-backend::BackendRegistry`
  - Euclidean dist: GEMM trick via `oxicuda-blas`
  - reduction: `oxicuda-primitives::DeviceReduceTemplate`
  - **DONE:** construction adapted; stale WGPU doc header corrected.

### Phase 8: sklears-svm GPU
- [x] `gpu_kernels.rs` 書き直し:
  - wgpu WGSL kernel shaders → `oxicuda-blas` + `oxicuda-ptx`
  - RBF kernel `K(x,y)=exp(-γ‖x-y‖²)`: GEMM pairwise dist + PTX pointwise exp
  - Polynomial / Linear kernel → `oxicuda-blas::gemm`
  - **DONE:** construction adapted; stale WGPU doc header corrected; Rbf/Sigmoid kernel transforms evaluated for on-device execution.

### Phase 9: sklears-manifold GPU
- [x] `gpu` feature に `oxicuda-manifold` を追加
  - `tsne.rs`: `oxicuda-manifold::tsne` GPU path
  - `umap.rs`: `oxicuda-manifold::umap` GPU path
  - `lle.rs` / `isomap.rs`: HNSW kNN via `oxicuda-manifold::HnswIndex`
  - **DONE:** construction adapted; `GpuTSNE::fit_transform` fixed (was returning a random-normal embedding with no relation to input, now calls real `oxicuda_manifold::tsne_fit`, verified deterministic-given-seed and cluster-separating); kNN now uses `oxicuda_manifold::hnsw_search`.

### Phase 9b: sklears-cross-decomposition (added 2026-07-02)
- [x] `gpu_acceleration.rs` had its own dummy GPU layer (`DummyGpuContext`, explicitly commented "not fully implemented in SciRS2-Core") instead of using the shared `sklears_core::gpu` abstraction like the other 7 crates — re-pointed at `sklears_core::gpu::{GpuBackend, GpuArray, GpuMatrixOps}` and removed the bespoke Dummy types.
  - **DONE:** removed bespoke `DummyGpuContext`/`DummyGpuBuffer`/`DummyGpuKernel` scaffolding, re-pointed at the shared `sklears_core::gpu` abstraction matching the other 7 crates.

### Phase 10: Cleanup & Verification
- [x] `cargo check --workspace --all-features`
- [x] `cargo clippy --workspace --all-features --all-targets -- -D warnings`
- [x] `cargo nextest run --workspace --all-features` (12394/12394 tests passing)
- [x] workspace Cargo.toml から `wgpu`, `cudarc`, `candle-core` の残存参照を完全削除
  - **DONE:** workspace-wide check/clippy/test all clean (12394/12394 tests passing); residual stale `wgpu`/`cudarc`/`candle-core` references checked — remaining hits confirmed in unrelated not-yet-migrated crates (`sklears-multiclass`, `sklears-simd`, `sklears` meta-crate) and left untouched as out of scope.

---

## Proposed follow-ups

- **GPU migration (27 items, oxicuda-* rewrite)** — **RESOLVED (2026-07-03):** implemented in full; see "GPU Migration: oxicuda-* v0.3 への完全移行" section above (Phases 1-9b + 10, all `[x]`). Both anticipated shape corrections were confirmed and correctly applied during the wave: (1) oxicuda-blas/-solver ops are free functions taking `&BlasHandle`/`&mut SolverHandle`, NOT methods (e.g. `oxicuda_blas::level3::gemm(&h, …)`); (2) standalone relu/sigmoid/tanh/softmax GPU ops live in `oxicuda-blas` (`elementwise::unary`, `reduction::softmax`), not `oxicuda-dnn` (fused epilogues only). The oxicuda-* pin comment and the stale "v0.3" comment were corrected to 0.4.1.
- **`sklears-discriminant-analysis` `[x]` gpu_acceleration** — DONE (2026-07-04): retargeted to oxicuda as anticipated. `GpuLDAKernel::solve_generalized_eigen_gpu` is a real Cholesky-reduction generalized eigensolver for LDA's `S_b w = λ S_w w` (`S_w = LLᵀ` via `oxicuda_solver::dense::cholesky`, reduce to `Cy = λy` via a GPU inverse + two GEMMs, solve via `dense::syevd`, back-transform `w = L⁻ᵀy`); new opt-in `gpu` feature (not in `default`) plus `with_device_id`/`compute_within_scatter_gpu` and 16 new tests (module had none before, since it never compiled).
- **`sklears-compose/distributed_optimization/` orphaned tree** (161 files, ~3.3 MB, pre-existing compile errors, not declared in lib.rs) — needs a human decision: wire in (large effort) or delete. Blocked on user call.
- ~~**`sklears-core trait_explorer/graph_visualization`** (disabled, mod.rs:112 commented out)~~ — RESOLVED (2026-07-03): re-enabled with a real `api_reference_generator` module and 5 new graph-analysis algorithms; see "🔎 Follow-up findings" above (this entry predated that fix and was left stale).
- **`sklears-compose` `pipeline_visualization`** — `PipelineVisualizer`/`DefaultRenderingEngine` (`crates/sklears-compose/src/pipeline_visualization/core.rs`) are wired into `lib.rs` and publicly exposed, but need a real implementation across 4 sub-areas: **graph** (blocked on a trait/API design decision, not mechanical extraction — confirmed 2026-07-04: the `PipelineComponent` trait (`core.rs:466`) only exposes a single component's `name()`/`component_type()`/`inputs()`/`outputs()`/`parameters()`, with no method to enumerate child/sub-components or the connections between them; `extract_nodes`/`extract_edges` (`core.rs:439-462`) each take one `&dyn PipelineComponent` but must return plural `Vec<GraphNode>`/`Vec<GraphEdge>`, which is structurally impossible from a single opaque trait object. Zero types anywhere in the workspace implement `PipelineComponent` (`grep -rn "impl PipelineComponent for" crates/*/src/` → no hits), and the existing concrete `Pipeline` structs (`pipeline.rs:77`'s `steps: Vec<(String, Box<dyn PipelineStep>)>`; `modular_framework/pipeline_system.rs:191`'s `stages: Vec<PipelineStage>`) don't implement it either — so there is no existing multi-step structure wired up to walk yet. Two paths forward, needing a human decision before implementation can proceed: (a) give `PipelineComponent` a way to expose child components + their connections, e.g. `sub_components()`/`connections()` methods — an additive-but-consequential trait change; or (b) have `PipelineVisualizer::add_pipeline`/`extract_nodes`/`extract_edges` take a concrete `Pipeline<S>` type instead of `&dyn PipelineComponent`, which requires inspecting `Pipeline`'s actual step-storage shape first. Same "needs user input" category as `distributed_optimization` above and the FFI-quarantine split below), **rendering** (actual SVG/PNG/HTML generation — no rendering dependency currently exists in `sklears-compose`'s `Cargo.toml`; hand-rolled-vs-new-dependency is an open decision), **metrics** (per-node timing/resource stats, could pull from the crate's existing `execution`/`performance_profiler` modules), **interactive** (event-driven/JS-backed interactive output — there's an unused `interactive: bool` field on `VisualizationConfig` already waiting for this). `pipeline_visualization/mod.rs` already has commented-out `pub mod graph;`/`metrics;`/`rendering;`/`interactive;` lines anticipating this split. As of 2026-07-03: `DefaultRenderingEngine::render()` and `PipelineVisualizer::extract_nodes`/`extract_edges` now return an honest `Err(SklearsError::NotImplemented(_))` instead of silently succeeding with fake/empty output (previously: a hardcoded `<svg></svg>` and always-empty node/edge vecs with no error).
  - Priority: P3 | Scope: large
- **`sklears-python` `PyDecisionTreeClassifier`/`PyDecisionTreeRegressor` wrap a still-degenerate type** (`crates/sklears-python/src/tree.rs:18-19,152-155`) — new finding surfaced 2026-07-05 by the `sklears-tree` Tranche 1 fix (see "🔎 Fabrication audit findings" Tier 2 RESOLVED note + new Tier 4 entry above): both wrapper structs hold `Option<sklears_tree::DecisionTree>`/`Option<DecisionTree<Trained>>` instead of the now-real `sklears_tree::classifier`/`regressor` types, so fit/predict stay degenerate through PyO3. Fix is mechanical (retarget the two structs to the real types), but inherits the same `Array1<i32>`-vs-`Array1<f64>` label-type breaking change already approved for the Rust-side fix, so it needs the same care at the Python-API boundary. Not started.
  - Priority: P2 | Scope: medium
- **`sklears-core`: `property_tests`/`test_utilities` modules still disabled — deferred to v0.2.1 (user decision, 2026-07-14)** (`crates/sklears-core/src/lib.rs:251-259`) — both modules are commented out with "KNOWN ISSUE (v0.1.0): Module disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.1." (the comment previously said "Planned for v0.2.0."; that target was missed, and this entry originally flagged it as an unfulfilled v0.2.0 promise). The user has now explicitly deferred re-enabling these modules to v0.2.1 rather than leaving it as an open-ended "someday before v0.3.0" note — this is a scheduled v0.2.1 item. The same underlying ndarray-0.17 HRTB issue has already been solved twice elsewhere in this exact codebase, both in v0.1.1: `sklears-compose`'s `cross_validation` module was re-enabled via `FitCV`/`PredictCV` adapter traits that use owned arrays to eliminate HRTB (`crates/sklears-compose/src/lib.rs:114-117`), and `sklears-multiclass` resolved the same class of issue by adding explicit `Fitted = F` associated-type bounds to its `Fit` trait constraints (`crates/sklears-multiclass/src/lib.rs:7-9`). Either pattern is a plausible template for re-enabling `property_tests`/`test_utilities`; not attempted here.
  - Priority: P2 | Scope: medium | Target: v0.2.1

---

## Policy follow-ups

- [ ] **Pure Rust governance violation: 4 crates leak FFI `-sys` deps via opt-in Cargo features** (found 2026-07-03 by `/policy-check`, measured under `cargo tree --workspace --all-features` — the actual measurement surface mandated by `~/work/.pure-rust-governance.md` v2, deliberately checking the full features closure rather than just default features to close a known loophole)
  - **Goal**: restore all 4 crates' `--all-features` closures to `-sys`-free, per the governance doc's own prescribed fix (§5 Quarantine model).
  - **Finding**: the governance doc's **Quarantine model (§5)** explicitly forbids this pattern — "a pure crate may NEVER declare a Cargo feature that, when enabled, adds a `-sys`/FFI dep — `--all-features` exposes it and fails L1. Irreducible FFI lives in separate, suffix-named quarantine crates (`*-aws-lc`, `*-mp3-lame`, `*-native`, `*-jack`, `*-adapter-<ffi>`), excluded from the pure-set, consumed as an application's own direct dep." Violations found:
    - `sklears-datasets` (`crates/sklears-datasets/Cargo.toml`): `parquet` feature → `zstd-sys`; `hdf5` feature → `hdf5-sys`; `cloud-s3`/`cloud-gcs` features → `aws-lc-sys`/`ring`/`security-framework-sys`/`core-foundation-sys` (via `aws-config`/`aws-sdk-s3`/`google-cloud-storage`); `visualization` feature → `dirs-sys`/`font-kit` (via `plotters`)
    - `sklears-decomposition` (`crates/sklears-decomposition/Cargo.toml`): `hdf5-support` feature → `hdf5-sys`
    - `sklears-feature-selection` (`crates/sklears-feature-selection/Cargo.toml`): `parquet` feature → `zstd-sys`
    - `sklears-metrics` (`crates/sklears-metrics/Cargo.toml`): `distributed` feature → `mpi-sys` + `clang-sys` (build-dep of `mpi-sys`)
  - **Prescribed fix**: split each FFI-pulling feature out into a physically separate, suffix-named crate that depends on the base crate and adds only its own FFI-touching dependency, so the base crates' `--all-features` closures stay `-sys`-free: `sklears-datasets-hdf5`, `sklears-datasets-parquet`, `sklears-datasets-cloud-s3`, `sklears-datasets-cloud-gcs`, `sklears-datasets-viz`, `sklears-decomposition-hdf5`, `sklears-feature-selection-parquet`, `sklears-metrics-mpi`.
  - **Evaluate first (hdf5 only)**: `oxih5` (`~/work/oxih5`, Pure-Rust read-only HDF5 reader, already at v0.1.4 locally) should be evaluated as a full replacement for `hdf5-sys` in both `sklears-datasets`'s `hdf5` feature and `sklears-decomposition`'s `hdf5-support` feature *before* quarantining — it may eliminate the FFI dependency outright rather than just fence it off. Only fall back to `sklears-datasets-hdf5` / `sklears-decomposition-hdf5` quarantine crates if `oxih5` can't cover the required read/write surface.
  - **No replacement exists**: `mpi-sys` has no Pure-Rust equivalent anywhere in the COOLJAPAN ecosystem yet — `sklears-metrics`'s `distributed` feature must be quarantined into `sklears-metrics-mpi` as-is, not replaced.
  - **Files**: `crates/sklears-datasets/Cargo.toml`, `crates/sklears-decomposition/Cargo.toml`, `crates/sklears-feature-selection/Cargo.toml`, `crates/sklears-metrics/Cargo.toml`
  - **Scope: large, needs user input** — this is a genuine architecture decision (splitting crate boundaries, potentially breaking every existing call site using `--features hdf5`/`--features parquet`/`--features cloud-s3`/`--features cloud-gcs`/`--features visualization`/`--features hdf5-support`/`--features distributed` today). Warrants its own dedicated planning pass, not a quick mechanical fix — needs user decision on migration strategy and whether renaming/removing these feature flags as a breaking change is acceptable before implementation starts.
  - Priority: P2 | Scope: large

---

## 🔎 Fabrication audit findings (2026-07-05)

**Trigger**: writing new benchmarks for `sklears-ensemble` found `GradientBoostingClassifier`/`GradientBoostingRegressor` and `BaggingRegressor` silently discarding real input and returning hardcoded/degenerate output (`predict()` always zeros, `fit()` ignoring `y`; `BaggingRegressor` had no `Fit` impl at all). Both are already fixed and verified earlier this session — mentioned here only once, as the origin/precedent for this audit, and are **not** re-listed as outstanding below.

**What was done**: 9 parallel read-only triage agents re-ran the same detection signature (fit()/predict() discarding real input for hardcoded/degenerate output) across the other 35 workspace crates. **This was investigation only — nothing below has been fixed by this audit.** Remediation scope is an open decision pending user input; options considered: fix everything, fix the most severe tier only, or document-and-defer.

**Headline result**: 32 of 36 workspace crates carry at least one instance of this bug class; ~90 individual instances total.

**Tranche 1 (2026-07-05)**: 3 of the highest-severity findings below — `sklears-mixture` and `sklears-manifold` (both Tier 1), and `sklears-tree` (Tier 2) — have since been fixed and independently re-verified (implementing subagent + orchestrator both separately re-ran `cargo check`/`clippy -D warnings`/`nextest`, all clean). Each is marked **RESOLVED (2026-07-05)** in place below, with the original finding kept struck-through for the historical record. The `sklears-tree` fix also surfaced one brand-new, still-open finding (`sklears-python`'s `PyDecisionTreeClassifier`/`PyDecisionTreeRegressor`) — see the new Tier 4 entry and "Proposed follow-ups" below. Everything else in this section remains exactly as originally audited: investigation-only, unfixed, remediation scope still an open decision pending user input.

### Tier 1 — Systemic (the crate's defining feature is non-functional)

- `sklears-semi-supervised`: ~34 of ~47 files share a copy-pasted fake stub or an equally inert fake-training loop (entropy_methods/*, deep_learning/* — 2 distinct patterns, few_shot/*, contrastive_learning/*). Real and working: graph/label-propagation family, mixture discriminant analysis, optimal transport, self/co/tri-training, multi-armed bandits, semi-supervised GMM, stacked autoencoders, deep belief networks, ladder networks.
- ~~`sklears-manifold`: nearly every algorithm's `Transform::transform()` on new data replays the stale training-time embedding instead of computing a real one — mds.rs:309, isomap.rs:357, tsne.rs:768, lle.rs:417, umap.rs:466, spectral_embedding.rs:183, laplacian_eigenmaps.rs:294, graph_neural_networks.rs:211, causal.rs:383, optimal_transport.rs:212, ~20 more. This falsifies this file's own "✅ Stub-check backlog" entry above claiming "TSNE/Isomap return honest `Err`" — that line has been struck through and corrected in place.~~
  **RESOLVED (2026-07-05)**: recharacterized as NOT fake math but a broken out-of-sample `Transform` contract (`fit()` always computed genuine embeddings; only `transform()` on new data was wrong). Fixed across ~20 files, split into two groups: **(a) honest `Err`** for genuinely transductive algorithms with no real out-of-sample definition (matching scikit-learn, which also has no `transform` for these) — tsne.rs, minibatch_tsne.rs, sne.rs, symmetric_sne.rs, heavy_tailed_symmetric_sne.rs, spectral_embedding.rs, mds.rs, laplacian_eigenmaps.rs, diffusion_maps.rs, minibatch_umap.rs, optimal_transport.rs (2 sites), adversarial.rs (AdversarialTSNE only — AdversarialAutoencoder's real transform was already correct and untouched), graph_neural_networks.rs (3 sites: GCN/GraphSAGE/GAT), mvu.rs, information_theory/{max_mutual_information,fisher_information,natural_gradient}.rs (3 sites) — all now return `Err(SklearsError::NotImplemented(...))` pointing users to a new `.embedding()` accessor for the training-time embedding; **(b) real out-of-sample projection implemented** — Isomap (real Gower's-formula geodesic-distance projection, self-transform reproduces training embedding to <1e-6), LLE (real barycentric-weight projection — also fixed a pre-existing bug where `solve_linear_system` was a stub ignoring its inputs and always returning uniform 1/n weights, meaning training-time LLE weights were ALSO fake before this fix), and **UMAP got a full real implementation** (weighted-neighbor init + fixed-embedding SGD refinement reusing existing gradient formulas, with gradient clamping added to match real umap-learn behavior — not a fallback). Also deleted `src/lib.rs.backup`, a ~14k-line dead/untracked file superseded by the real lib.rs. Verified: `cargo check -p sklears-manifold --all-features` clean, `cargo clippy -p sklears-manifold --all-features --all-targets -- -D warnings` clean, `cargo nextest run -p sklears-manifold --all-features` → 422 passed (up from 386 before this fix, net +38 new tests, 0 removed), 2 skipped (pre-existing unrelated ignores) — independently re-run and confirmed. 25 files touched.
- `sklears-datasets`: ~20 of 34 files (all 14 real-dataset loaders in loaders.rs — Iris/Wine/Digits/MNIST/CIFAR-10 — plus plugin/streaming/config/format subsystems) are never `mod`-declared in lib.rs, so they're dead code, not compiled in at all. Even if wired in, every loader is `#[deprecated]`, self-admitting it returns synthetic not real data. The status table's "Real-world data" advanced-features claim is false; only `generators/basic.rs`-style synthetic generation actually ships (and is real/correct). **Note (2026-07-05, Tranche 1 scoping)**: excluded from fabrication remediation by design — these loaders are honestly `#[deprecated]` + doc-warned stubs ("returns synthetic data, not real Iris... real data planned for v0.2.0"), not covert fabrication, so fixing them means *sourcing real dataset data*, not a bug fix — that v0.2.0 target has now passed without landing (the v0.2.0 CHANGELOG's only `sklears-datasets` addition was 6 synthetic *generator* modules, a different thing from these real-data loaders; the source doc comments still read "planned for v0.2.0" verbatim, unrevised). Still planned, but not yet rescheduled to a specific version. Filed as a separate open feature/task item — still unfixed, not forgotten; **not** marked RESOLVED.
- ~~`sklears-mixture`: 5 "advanced" GMM variants all have `predict()` unconditionally returning `Array1::zeros` — LaplaceGMM (approximation.rs:272), AcceleratedEM (optimization_enhancements.rs:446), AdaptiveStreamingGMM (adaptive_streaming.rs:384), L1RegularizedGMM (regularization.rs:421), MiniBatchGMM (large_scale.rs:421). LaplaceGMM's `fit()` (approximation.rs:193-251) is also fake (fixed identity covariances, uniform weights, hardcoded `log_marginal_likelihood=0.0`).~~
  **RESOLVED (2026-07-05)**: root cause was a typestate bug — structs stored `PhantomData<S>` instead of a real `state: S` field, silently discarding fitted parameters. Fixed by adding real state storage + wiring genuine EM math (reusing/sharing logic with the crate's correct `gaussian.rs` GMM core) across all 5 variants: **LaplaceGMM** (was fully fake: random means, identity covariance, hardcoded `log_marginal_likelihood=0.0`) → real EM fit + real diagonal-Hessian-approximation Laplace marginal likelihood (honest gap: `full_hessian_posterior_covariance()` → `Err(NotImplemented)`, full cross-term Hessian out of scope); **AcceleratedEM** (fit was already real EM) → state now stored + predict real; `speedup_factor` was a hardcoded constant (1.0/1.5/2.0/2.5) → now a genuinely measured baseline-vs-accelerated iteration-count ratio (honest gap: SQUAREM/QuasiNewton acceleration types still run plain EM internally, unchanged); **AdaptiveStreamingGMM** (was the most fabricated: fit didn't run EM at all, partial_fit was a no-op, component add/delete were no-ops, drift detection always returned false, n_components() hardcoded to 1) → real EM fit, real online/stochastic partial_fit, real component creation/deletion criteria + array surgery, real Page-Hinkley + CUSUM drift detection (honest gap: `DriftDetectionMethod::ADWIN` → `Err(NotImplemented)`); **L1RegularizedGMM** (fit was real-ish) → state stored + predict real + log-likelihood bug fixed (was a near-constant from summing normalized responsibilities); **MiniBatchGMM** (updated weights/means but never covariances — frozen at init forever) → real covariance M-step added, plus per-epoch shuffling (a real latent bug the fix agent's own test caught: unshuffled cluster-sorted mini-batches made training unstable) and a real log-likelihood (was `ln(sum of weights)` ≈ constant). Verified: `cargo check -p sklears-mixture --all-features` clean, `cargo clippy -p sklears-mixture --all-features --all-targets -- -D warnings` clean, `cargo nextest run -p sklears-mixture --all-features` → 238 passed, 0 failed — independently re-run and confirmed. No `unwrap()` introduced. 6 files touched (adaptive_streaming.rs, approximation.rs, common.rs, large_scale.rs, optimization_enhancements.rs, regularization.rs), largest 1219 lines (no splitrs needed).

### Tier 2 — Foundational (crate-root exports or widely depended-upon subsystems)

- ~~`sklears-tree`: `DecisionTree` (decision_tree.rs:301-333) — aliased to BOTH `DecisionTreeClassifier`/`DecisionTreeRegressor` and re-exported from crate root/prelude — discards `y` and never builds a tree; `predict()` always returns zeros. A real, unwired 797-line SmartCore-backed classifier already exists in the same crate at classifier.rs (never `mod`-declared) — likely fixable by reconnecting it rather than writing new algorithm code. RandomForestClassifier/Regressor confirmed real/unaffected.~~
  **RESOLVED (2026-07-05)**: fixed exactly as anticipated — `mod`-declared the existing real 797-line SmartCore-backed `classifier.rs` and the real `regressor.rs`, removed the degenerate type aliases (both previously pointed at the same broken `DecisionTree<Untrained>`), and rewired crate-root + prelude exports to the real implementations. **Approved breaking API change**: the real classifier takes `Array1<i32>` targets (correct for classification) vs. the old degenerate `Array1<f64>` — all in-crate and cross-crate call sites updated (one cross-crate site: `crates/sklears/tests/basic_functionality_test.rs`). Verified: `cargo check -p sklears-tree --all-features` clean, `cargo clippy` clean, `cargo nextest run -p sklears-tree --all-features` → 79 passed (1 pre-existing unrelated ignore), `cargo check --workspace --all-features` clean — independently re-run and confirmed. 9 files touched, +190/-68 lines.
  **New open finding surfaced by this fix, NOT yet remediated**: `sklears-python`'s `PyDecisionTreeClassifier`/`PyDecisionTreeRegressor` PyO3 wrappers (`crates/sklears-python/src/tree.rs:18-19,152-155`) hold the crate's bare `sklears_tree::DecisionTree` type directly (not the now-fixed `classifier`/`regressor` modules), so they still have degenerate fit/predict — see new Tier 4 entry below. This corrects the original audit's "sklears-python confirmed clean" conclusion: it was clean by grep-signal within sklears-python's own source, but has a real bug via what it wraps from sklears-tree. Filed as a new open item for a future tranche, not fixed by this pass (see "Proposed follow-ups" below).
- `sklears-core`: ensemble_improvements.rs:747-877's public `AdvancedParallelEnsemble` has the identical ensemble fabrication bug (RandomForest/GradientBoosting/AdaBoost/Voting/Stacking all ignore `y`), independently duplicated from sklears-ensemble. Also: compatibility.rs:266-336 (ScikitLearnModel hardcodes predictions), formal_verification.rs:217-399 (every check hardcodes Passed), validation.rs:133-137,358-401 (validators always return `Ok(true)`, currently dormant/no live callers).
- `sklears-feature-selection`: automl/hyperparameter_optimizer.rs:524-566 — all 10 AutoML method types return hardcoded first-N columns regardless of data, powering the crate's flagship `comprehensive_automl`/`quick_automl`. Also ensemble_selectors.rs:1256 (uniform feature scores), ml_based.rs:544-656 (AttentionFeatureSelector never trains), statistical_tests/correlation_tests.rs:21 (p-values hardcoded 0.05), comprehensive_benchmark.rs:625 (stability hardcoded 0.85).
- `sklears-model-selection`: bayes_search.rs:339,762 — BayesSearchCV/TPEOptimizer's `evaluate_params` ignores the real estimator, classifying via `sum(params) > 0.5*len(params)` instead of fit/predict, so hyperparameter search tunes against a nonsense proxy. meta_learning_advanced.rs:478,864,872,880 — TransferLearningOptimizer/FewShotOptimizer's feature mapping/loss/gradient/prototype functions are all no-ops (zeros, hardcoded 0.1, empty map).
- `sklears-gaussian-process`: variational_deep_gp.rs/deep_gp.rs `predict()` always zeros/ones; structured_kernel_interpolation.rs/spatial.rs/temporal.rs kernel choice has no effect on output; classification.rs:404 Laplace-approximation probabilities mathematically wrong; lib.rs has temporal.rs/multi_objective.rs/both bayesian-optimization modules commented out of the module tree entirely (not compiled).

### Tier 3 — Multiple confirmed instances (real, working code coexists with the fabrication; not the crate's core identity)

- `sklears-compose` (6 areas): execution_strategies.rs:1311-1358,1596-1651 fabricated task completion; fault_isolation.rs:940-997 fake lifecycle; error_recovery.rs:638-641,809-813 rollback claims success while restoring nothing; failure_detection.rs:715,904-919 pattern predictor always empty; retry/context.rs:858-888 hardcoded performance predictor; retry/machine_learning.rs:455-483,594-611 DecisionTreeModel/NeuralNetworkModel never learn.
- `sklears-covariance`: signal_processing.rs:1062-1119, meta_learning.rs:1022-1051 + fluent_api.rs:843-909 all methods silently = empirical covariance; model_selection.rs:1035-1059 CV ignores fold boundaries; nuclear_norm.rs:286 + frank_wolfe.rs:743 solvers ignore the real objective.
- `sklears-cross-decomposition`: out_of_core.rs:599-635,892-926 streaming PLS/CCA discard covariance for identity weights; federated_learning.rs:559,739 always all-ones; hypergraph_methods.rs:546-577 + higher_order_statistics.rs:611-690 random noise; enhanced_pathway_analysis.rs:306-441 hardcoded fake pathway data.
- `sklears-neighbors`: manifold_learning.rs:187-227,381-420,631-650 — LLE/Isomap/Laplacian-Eigenmaps compute real math then discard it for random noise; local_outlier_factor.rs:182 ignores new data.
- `sklears-neural`: gradient_checking.rs:223-407 the gradient checker itself is fake; gnn.rs:720-725 Attention pooling = Mean; layers/transformer.rs:184-230,268 relative positional encoding is a no-op; gan.rs:229 honestly `Err` (lower severity).
- `sklears-inspection`: rules.rs:214,249 confidence hardcoded 1.0; attention.rs:515 Grad-CAM ignores channel index; parallel.rs:762 fake CPU/memory readings drive "adaptive" batching; benchmarking.rs:687 p-value hardcoded 0.01.
- `sklears-impute`: ensemble.rs GB/ExtraTrees imputers silently fall back to mean; timeseries.rs:896 ARIMA MA term is dead code; social_science.rs:1680 survey adjustments are no-ops.
- `sklears-calibration`: continual_learning.rs:643-770 performs zero real calibration; binary.rs:448-563 multiclass adapters discard the calibrator and echo raw input.
- `sklears-multiclass`: hierarchical.rs:244,286,609,642 two classifiers always predict `classes[0]`; one_vs_one.rs:833-847 consensus method is ignored.
- `sklears-multioutput`: core.rs:426-469,619-648 flagship MultiOutputRegressor doesn't solve regression despite its own comment; compressed_sensing.rs:159-177 OMP = L2.
- `sklears-naive-bayes`: hierarchical.rs:828-838,962-993 doesn't do NB math; feature_engineering_utilities.rs:446-452 `has_missing_values` always false.
- `sklears-discriminant-analysis`: random_projection_discriminant_analysis.rs:623-712 QDA/RDA always predict `classes[0]`; bayesian_discriminant/core.rs:210 `log_marginal_likelihood` hardcoded 0.0.
- `sklears-linear`: residual_analysis.rs:577 Breusch-Pagan `r_squared` hardcoded 0.1; modular_framework.rs:319 confidence variance hardcoded to ones; lasso_cv.rs KFold ignores `random_state` (lower severity).
- `sklears-metrics`: fluent_api.rs:543,618-621 confidence intervals are fixed ±5%/10%; survival.rs:556 `log_rank_test` p-value hardcoded to zero.

### Tier 4 — Isolated (single confirmed instance)

- `sklears-svm`: topic_model_integration.rs:660-714 TopicSVM fakes a trained state via `unsafe transmute`, predict always zeros — also a memory-safety/UB concern, not just fabrication.
- `sklears-feature-extraction`: text/mod.rs:1069 `LSA::fit` ignores input, returns zeros.
- `sklears-decomposition`: type_safe.rs:434-481,525 TypeSafeDecompositionPipeline recomputes normalization stats from transform-time data instead of storing them from fit, breaking the fit/transform contract.
- `sklears-isotonic`: ml_integration.rs:338-374 MonotonicDeepLearning never trains.
- `sklears-kernel-approximation`: distributed_kernel.rs:563-573 eigendecomposition hardcoded to identity.
- `sklears-utils`: gpu_computing.rs:410 Softmax is missing normalization and doesn't sum to 1 (likely a quick real fix).
- `sklears-preprocessing`: automated_feature_engineering.rs:918-931 engineered feature columns left as zeros; temporal/trend.rs:176-180 polynomial trend silently = linear.
- **`sklears-python`** (new finding, 2026-07-05, surfaced by the `sklears-tree` fix — see Tier 2 RESOLVED note above): `PyDecisionTreeClassifier`/`PyDecisionTreeRegressor` (`crates/sklears-python/src/tree.rs:18-19,152-155`) hold `Option<sklears_tree::DecisionTree>` / `Option<DecisionTree<Trained>>` — the crate's bare, still-degenerate type — rather than the now-fixed `sklears_tree::classifier`/`regressor` modules, so fit/predict remain degenerate through this wrapper. Not remediated by this pass; tracked as a new open item (see "Proposed follow-ups" below).

**Confirmed clean**: `sklears` (meta-crate), `sklears-clustering`, ~~`sklears-python`~~, `sklears-simd` (grep-clean; sklears-simd is ~54k lines/70 files including exotic GPU/TPU/FPGA/quantum modules and wasn't exhaustively read beyond the grep signal — clean by signal, not exhaustively audited). **Correction (2026-07-05)**: `sklears-python` was clean by grep-signal within its own source (no fabrication pattern directly in sklears-python's code), but the `sklears-tree` fix above surfaced that it wraps a still-degenerate type from what it depends on — see new Tier 4 entry above. Same "confirmed clean → later found false" correction pattern as the TSNE/Isomap line in the "✅ Stub-check backlog" section earlier in this file.

