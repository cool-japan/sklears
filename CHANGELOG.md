# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

### Changed

### Fixed

## [0.1.2] - 2026-06-30

### Added
- `sklears-core`: New `system_info` module — `SystemMemory` struct with `system_memory()` and `process_rss_bytes()` reading real OS stats (Linux `/proc/meminfo`; macOS `host_statistics64` + `sysctlbyname`; sysconf on other Unix; `GlobalMemoryStatusEx` on Windows)
- `sklears-core`: DSL macros fully implemented — `model_evaluation!`, `data_pipeline!`, `experiment_config!` wired through parse → generate pipeline via `dsl_types`/`parsers`/`code_generators`
- `sklears-core`: `trait_explorer` GPU-context init wired to `scirs2_core::gpu` with honest CPU fallback; where-clause extraction implemented
- `sklears-compose`: New `comprehensive_benchmarking` module with full regression-detection subsystem (15 trait modules: `AdaptiveThresholds`, `AlertSuppression`, `BaselineComparisons`, `BusinessImpactAssessment`, `EffectSizeAnalysis`, `PatternRecognition`, `RegressionAlertSystem`, `RegressionCache`, `RegressionDetector`, `RegressionDetectorConfig`, `RegressionMetadata`, `SeverityAssessment`, `SignificanceTesting`, `SmartSuppression`, `ThresholdManagement`)
- `sklears-compose`: `time_series_pipelines` — `LagFeatures`, `RollingWindow`, `Differencing`, `TemporalTrainTestSplit` implemented and re-exported
- `sklears-compose`: `enhanced_wasm_integration` re-exported in `lib.rs`; new `utils` module
- `sklears-compose`: Column transformer sparse paths via real CSR operations from `scirs2-sparse` (`sparse_select_columns`, `sparse_hstack`)
- `sklears-preprocessing`: 12 previously-stub implementations now real — scalers: `MinMaxScaler`, `MaxAbsScaler`, `UnitVectorScaler`, `FeatureWiseScaler`, `OutlierAwareScaler`; imputers: `SimpleImputer`, `KNNImputer` (NaN-aware), `IterativeImputer` (MICE ridge), `MultipleImputer`, `GAINImputer`; encoders: `OrdinalEncoder`, `TargetEncoder` (category smoothing)
- `sklears-preprocessing`: SIMD paths re-enabled — real AVX kernels `simd_threshold_mask`, `simd_axpy`; `simd_mahalanobis` routed through `simd_dot_product`
- `sklears-simd`: AVX2 quicksort implementation — `quicksort_avx2_impl`, `partition_avx2_buffered`, `build_compress_lut`; 6 new hardening tests (already_sorted, reverse_sorted, all_equal, heavy_duplicates, non_multiple_of_8, large)
- `sklears-impute`: `CategoricalClusteringImputer` (k-means), `CategoricalRandomForestImputer` (MissForest/CART), `AssociationRuleImputer` (Apriori), and `validate_imputer` (K-fold MAE cross-validation)
- `sklears-manifold`: Real serde serialization for `RandomProjection` via new public accessors (`projection_matrix()`, etc.) with lossless round-trip tests
- `sklears-svm`: `SVC` conformal prediction restructured to `Option<SVC<Trained>>`; unfitted state returns honest `Err(NotTrained)`
- Workspace: `oxicuda-backend`, `oxicuda-memory`, `oxicuda-blas`, `oxicuda-solver`, `oxicuda-manifold`, `oxicuda-dnn`, `oxicuda-driver`, `oxicuda-ptx`, `oxicuda-primitives` v0.3 added (replacing direct `wgpu`/`cudarc`/`candle-core` dependencies)

### Changed
- All SciRS2 workspace dependencies updated 0.4.2 → 0.5.1: `scirs2-core`, `scirs2-autograd`, `scirs2-optimize`, `scirs2-linalg`, `scirs2-stats`, `scirs2-cluster`, `scirs2-metrics`, `scirs2-datasets`, `scirs2-sparse`, `scirs2-neural`, `scirs2-special`, `scirs2-spatial`, `scirs2-signal`, `scirs2-series`, `scirs2-text`, `scirs2-fft`, `scirs2-graph`
- `oxicode` updated 0.2 → 0.2.4; `oxifft` updated 0.3.0 → 0.3.2; `oxiarc-deflate` updated 0.2.6 → 0.3.3; `oxiarc-zstd` updated 0.2.7 → 0.3.3
- `sklears-svm`: Fully migrated from `nalgebra` → `scirs2-linalg` across `semi_supervised`, `property_tests`, and `advanced_optimization`; no `nalgebra` remaining in `src/`
- `sklears-metrics`: Fully migrated from `sprs` → `scirs2-sparse`; `sparse` feature re-enabled
- `sklears-discriminant-analysis`: Parallel eigen computation wired to `scirs2_linalg::eigh`; upstream parallel kernel limitation documented

### Fixed
- `sklears-simd`: 5 test failures — MAE gradient sign bug, cross-product SSE2 shuffles, F32x4 stride test, AVX2 compress partition
- `sklears-gaussian-process`: Cholesky stability for indefinite saddle-point system (ordinary kriging) — SPD via regularized Cholesky, indefinite via LU; 5 previously-ignored kriging tests re-enabled; LOO studentized-residual outlier fix
- `sklears-core`: `system_info` macOS compatibility — `_SC_AVPHYS_PAGES` replaced with `host_statistics64` (the constant is absent from the macOS libc ABI)
- `sklears-mixture`: `student_t` doctest — `degrees_of_freedom` returns `Result`, chained `.expect()` added
- `sklears-inspection`: 3 doctest fixes (`schedule_tasks` distributed, federated moved config, quantum `add_parametric_gate`); `test_gaussian_noise_generation` fixed with seeded `StdRng` + 200-sample statistical test
- `sklears-svm`: 3 real bugs fixed during `nalgebra` migration, including a fabricated `decision_function` implementation replaced with a correct one
- Doctest fixes across `sklears-covariance`, `sklears-cross-decomposition`, `sklears-isotonic`, `sklears-model-selection`, `sklears-neighbors`, `sklears-semi-supervised` (missing `Ok(())`, invalid imports, wrong type annotations, f32→f64 precision tolerances)
- Flaky timing tests fixed (`test_decomposition_pipeline`, `test_model_metadata`, `test_historical_summary`, `test_energy_*`) — removed load-sensitive duration/ratio assertions; `ModelMetadata::touch()` now guarantees strict time advancement

## [0.1.1] - 2026-04-25

### Fixed
- HDBSCAN persistence extraction: corrected root node detection and propagation order
- StreamingStandardScaler / StreamingSimpleImputer: replaced manual Default impls with derive
- Pipeline get_step_mut: fixed lifetime elision for dyn PipelineStep
- GpuAcceleration struct field name mismatch in hardware_acceleration.rs
- Arrow StringArray collection from Option<&str> iterator in serialization
- SpectralGraphConfig missing random_seed field in graph_clustering tests
- StreamingSimpleImputer: use ? operator for Option early return

### Changed
- Version bump to 0.1.1

## [0.1.0] - 2026-03-20

### Added

- **36 crates** covering >99% of scikit-learn's API surface
- **Algorithm coverage**:
  - Linear models (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, GLMs)
  - Tree-based models (DecisionTree, RandomForest, ExtraTrees)
  - Support Vector Machines (SVC, SVR with multiple kernels)
  - Neural networks (MLP, RBM, Autoencoders)
  - Clustering (KMeans, DBSCAN, Hierarchical, MeanShift, SpectralClustering)
  - Decomposition (PCA, IncrementalPCA, KernelPCA, ICA, NMF, FactorAnalysis)
  - Ensemble methods (Voting, Stacking, AdaBoost, GradientBoosting)
  - Gaussian processes, Naive Bayes, Nearest Neighbors, Discriminant Analysis
  - Preprocessing (Scalers, Encoders, Transformers, Imputers)
  - Model selection (Cross-validation, GridSearchCV, RandomizedSearchCV, BayesSearchCV)
  - Feature extraction, feature selection, manifold learning, isotonic regression
  - Calibration, multi-class, multi-output, semi-supervised learning
  - Gaussian Mixture Models, covariance estimation, dummy estimators
- **Pure Rust implementation** — zero C/Fortran system dependencies
  - OxiBLAS for BLAS/LAPACK operations
  - Oxicode for SIMD-optimized serialization
  - SciRS2 ecosystem for scientific computing
- **Type-safe state machines** enforcing compile-time model state validation (Untrained → Trained)
- **Builder pattern** for ergonomic algorithm configuration
- **SIMD optimizations** via `std::simd` for vectorized operations
- **Parallel processing** with Rayon work-stealing scheduler
- **Python bindings** via PyO3 (`sklears-python`)
- **Polars DataFrame integration** for data manipulation
- **AutoML capabilities** with hyperparameter search
- **Memory-mapped dataset support** and CSV/Parquet data loaders
- **Comprehensive test suite** (4,400+ tests, >99% pass rate)
- **Benchmarking suite** using Criterion

### Dependencies
- SciRS2 v0.1.3 (scientific computing ecosystem)
- OxiBLAS v0.1.2 (Pure Rust BLAS/LAPACK)
- Oxicode v0.1.1 (SIMD-optimized serialization)
- Polars 0.52 (DataFrame operations)
- Rayon 1.11 (parallelism)
- Minimum Rust version: 1.70+

---

[Unreleased]: https://github.com/cool-japan/sklears/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cool-japan/sklears/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/cool-japan/sklears/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/sklears/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/sklears/releases/tag/v0.1.0
