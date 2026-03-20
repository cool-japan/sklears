# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Performance parity with scikit-learn on large datasets
- Enhanced GPU acceleration (CUDA/WebGPU)
- Distributed computing support
- ONNX/PMML model interchange
- WebAssembly compilation support

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

[Unreleased]: https://github.com/cool-japan/sklears/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cool-japan/sklears/releases/tag/v0.1.0
