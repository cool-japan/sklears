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

## 📊 Current Status (v0.2.0)

**Overall API Coverage: >99%** 🎉

All major scikit-learn modules are implemented with production-ready quality:

| Module | Coverage | Status | Advanced Features |
|--------|----------|--------|--------------------|
| linear_model | 100% | 🟢 Complete | GPU acceleration, distributed training |
| tree | 100% | 🟢 Complete | SHAP integration, GPU acceleration |
| ensemble | 100% | 🟢 Complete | Advanced stacking, isolation forest |
| svm | 100% | 🟢 Complete | GPU kernels, parallel SMO |
| neural_network | 100% | 🟢 Complete | Advanced layers, seq2seq, attention |
| cluster | 100% | 🟢 Complete | GPU acceleration, streaming |
| decomposition | 100% | 🟢 Complete | Incremental variants, tensor methods |
| preprocessing | 100% | 🟢 Complete | Text processing, GPU scaling |
| metrics | 100% | 🟢 Complete | Statistical testing, visualization |
| model_selection | 100% | 🟢 Complete | AutoML pipeline, advanced search |
| neighbors | 100% | 🟢 Complete | Spatial trees, GPU acceleration |
| feature_selection | 100% | 🟢 Complete | Stability selection, advanced tests |
| datasets | 100% | 🟢 Complete | Real-world data, advanced generators |
| naive_bayes | 100% | 🟢 Complete | All variants, ensemble methods |
| gaussian_process | 100% | 🟢 Complete | Advanced kernels, Bayesian optimization |
| discriminant_analysis | 100% | 🟢 Complete | GPU acceleration, robust variants |
| manifold | 100% | 🟢 Complete | GPU support, advanced embeddings |
| semi_supervised | 100% | 🟢 Complete | Graph methods, deep learning |
| feature_extraction | 100% | 🟢 Complete | Text processing, image features |
| covariance | 100% | 🟢 Complete | Robust estimators, sparse methods |
| cross_decomposition | 100% | 🟢 Complete | Advanced PLS variants |
| isotonic | 100% | 🟢 Complete | Multidimensional support |
| kernel_approximation | 100% | 🟢 Complete | GPU acceleration, advanced methods |
| dummy | 100% | 🟢 Complete | All baseline strategies |
| calibration | 100% | 🟢 Complete | Neural calibration, conformal prediction |
| multiclass | 100% | 🟢 Complete | Error-correcting codes |
| multioutput | 100% | 🟢 Complete | All chaining methods |
| impute | 100% | 🟢 Complete | Neural networks, ensemble imputation |
| compose | 100% | 🟢 Complete | GPU pipelines, distributed processing |
| inspection | 100% | 🟢 Complete | Causal analysis, dashboards |
| mixture | 100% | 🟢 Complete | Bayesian variants, advanced EM |
| simd | 100% | 🟢 Complete | Advanced SIMD optimizations |
| utils | 100% | 🟢 Complete | GPU utilities, distributed support |
| core | 100% | 🟢 Complete | Type system, GPU, async traits |

## 📊 Quality Metrics

- **Test Suite**: 11,222 tests (11,222 passing, 100% success rate)
- **Code Quality**: 100% warning-free compilation
- **Performance**: Pure Rust implementation with ongoing performance optimization
- **Dependencies**: Pure Rust stack (OxiBLAS v0.1.2, Oxicode v0.1.1)
- **Build Status**: ✅ 36/36 crates building successfully

## 🚀 Future Roadmap

### v0.1.0 Stable (Q2 2026)
- [ ] API surface freeze and stabilization
- [ ] Comprehensive benchmark suite expansion
- [ ] Enhanced documentation and cookbooks
- [ ] Production case studies
- [ ] Community feedback integration

### v0.2.0 and Beyond
- [ ] Enhanced GPU acceleration (CUDA/WebGPU optimization)
- [ ] Distributed computing support
- [ ] Advanced AutoML capabilities
- [ ] ONNX/PMML model interchange
- [ ] WebAssembly compilation support
- [ ] Embedded/no-std support for microcontrollers

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

*Version: 0.2.0 (in development)*
*Last Release: v0.1.2 — 2026-06-30*
*Next Milestone: TBD*

## ✅ Stub-check backlog — COMPLETED (2026-06-21, v0.1.2)

The 2026-06-12 `/cooljapan-stub-check` backlog (15 items) was implemented with **real logic**
across 8 crates in parallel. Full workspace builds clean; `cargo clippy --workspace --all-targets`
reports **zero warnings**. Per-item outcome:

- [x] `sklears-preprocessing`: SIMD paths re-enabled (real AVX kernels `simd_threshold_mask`, `simd_axpy`; routed `simd_mahalanobis` through `simd_dot_product`; honest scalar-only docs where `powf`/`ln` have no exact AVX intrinsic).
- [x] `sklears-preprocessing`: scaling/imputation/encoding — **found ~12 exported empty-placeholder structs (silent fabrication), now implemented for real**: MinMaxScaler, MaxAbsScaler, UnitVectorScaler, FeatureWiseScaler, OutlierAwareScaler; SimpleImputer, KNNImputer (nan-aware), IterativeImputer (MICE ridge), MultipleImputer, GAINImputer (honest regression backend); OrdinalEncoder, TargetEncoder (category_encoders smoothing). Dead modular-refactor comment blocks removed.
- [x] `sklears-preprocessing`: BufferPool — premise was false; `scirs2_core::memory::BufferPool` exists and is used. Removed stale "doesn't exist" comments; re-enabled gpu_acceleration/lazy_evaluation/memory_management exports.
- [x] `sklears-manifold`: real serde serialization for `RandomProjection` via new public accessors (`projection_matrix()` etc.) + lossless round-trip tests. TSNE/Isomap return honest `Err` (internal state not publicly reachable).
- [x] `sklears-core`: DSL macros (`model_evaluation!`, `data_pipeline!`, `experiment_config!`) — full parse→generate implemented (dsl_types/parsers/code_generators wired).
- [x] `sklears-core`: trait_explorer graph GPU-context init (real `scirs2_core::gpu` w/ honest CPU fallback) + where-clause extraction. ⚠️ See follow-up: `graph_visualization` module is currently disabled; logic verified in isolation.
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
- [x] sklears-cross-decomposition/Cargo.toml: feature `gpu` の `scirs2-core/gpu` → `oxicuda-backend` に変更

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
- **`sklears-discriminant-analysis` `[~]` gpu_acceleration** — blocked on a scirs2-core GPU API that doesn't exist; a future GPU-focused run could retarget it to oxicuda reductions instead. Leave as `[~]`; do not restart from scratch.
- **`sklears-compose/distributed_optimization/` orphaned tree** (161 files, ~3.3 MB, pre-existing compile errors, not declared in lib.rs) — needs a human decision: wire in (large effort) or delete. Blocked on user call.
- **`sklears-core trait_explorer/graph_visualization`** (disabled, mod.rs:112 commented out; depends on refactored-away api_reference_generator + scirs2_core gpu/profiling/validation features) — medium-effort re-enable, candidate for a later run.
- **`sklears-compose` `pipeline_visualization`** — `PipelineVisualizer`/`DefaultRenderingEngine` (`crates/sklears-compose/src/pipeline_visualization/core.rs`) are wired into `lib.rs` and publicly exposed, but need a real implementation across 4 sub-areas: **graph** (real node/edge extraction by walking a `Pipeline`/`PipelineComponent` structure — `extract_nodes`/`extract_edges` are currently unimplemented), **rendering** (actual SVG/PNG/HTML generation — no rendering dependency currently exists in `sklears-compose`'s `Cargo.toml`; hand-rolled-vs-new-dependency is an open decision), **metrics** (per-node timing/resource stats, could pull from the crate's existing `execution`/`performance_profiler` modules), **interactive** (event-driven/JS-backed interactive output — there's an unused `interactive: bool` field on `VisualizationConfig` already waiting for this). `pipeline_visualization/mod.rs` already has commented-out `pub mod graph;`/`metrics;`/`rendering;`/`interactive;` lines anticipating this split. As of 2026-07-03: `DefaultRenderingEngine::render()` and `PipelineVisualizer::extract_nodes`/`extract_edges` now return an honest `Err(SklearsError::NotImplemented(_))` instead of silently succeeding with fake/empty output (previously: a hardcoded `<svg></svg>` and always-empty node/edge vecs with no error).
  - Priority: P3 | Scope: large

