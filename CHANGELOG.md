# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.2] - 2025-12-22

### Added
- 1,279 new tests across the workspace, bringing total to 11,292 tests passing
- Enhanced property-based testing coverage for edge cases
- Improved integration test scenarios for cross-crate reliability
- Updated examples and quick-start guides across all crates

### Changed
- Refined public API interfaces based on alpha.1 feedback
- Improved error messages and diagnostics across all modules
- Enhanced cross-crate integration reliability
- Updated all crate README files with current version references
- Aligned with latest stable versions of core dependencies

### Fixed
- Edge cases discovered through expanded testing
- Various stability improvements from alpha.1

### Documentation
- Improved API documentation consistency across 35+ crates
- Refreshed module documentation
- Updated all version references to 0.1.0-alpha.2

### Testing
- Test suite: 11,292 passing, 170 skipped, 0 failed
- All Clippy and rustfmt checks pass
- Integration tests validate cross-crate behavior

## [0.1.0-alpha.1] - 2025-10-13

### Added
- **Initial Public Alpha Release**
- >99% scikit-learn API coverage across 35+ crates
- Comprehensive algorithm implementations:
  - Linear models (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, GLMs)
  - Tree-based models (DecisionTree, RandomForest, ExtraTrees)
  - Support Vector Machines (SVC, SVR with multiple kernels)
  - Neural Networks (MLP, RBM, Autoencoders)
  - Clustering (KMeans, DBSCAN, Hierarchical, MeanShift, SpectralClustering, GaussianMixture)
  - Decomposition (PCA, IncrementalPCA, KernelPCA, ICA, NMF, FactorAnalysis)
  - Ensemble methods (Voting, Stacking, AdaBoost, GradientBoosting)
  - Preprocessing (Scalers, Encoders, Transformers, Imputers)
  - Model selection (Cross-validation, GridSearchCV, RandomizedSearchCV, BayesSearchCV)
- Performance optimizations: 3-100× speedup over Python implementations
- SIMD optimizations using std::simd
- GPU backend support (CUDA and WebGPU)
- Type-safe state machines for compile-time model state validation
- Zero-cost trait abstractions for polymorphic ML algorithms
- Comprehensive Polars DataFrame integration
- Python bindings via PyO3 (sklears-python)
- AutoML capabilities with hyperparameter search
- Memory-mapped dataset support
- CSV/Parquet data loaders
- Extensive benchmarking suite using Criterion
- Property-based tests using proptest

### Documentation
- Comprehensive README with examples and migration guide
- API documentation across all crates
- SciRS2 integration policy documentation
- Benchmark results and performance comparisons
- Quick-start guides for all major algorithms

### Testing
- Test suite: 10,013 passing, 69 skipped, 0 failed
- Integration tests for cross-crate behavior
- GPU offload testing
- Benchmarks validated against June 2025 baselines

### Dependencies
- Built on SciRS2 0.1.0-rc.1 ecosystem
- Polars 0.52 for DataFrame operations
- Rayon 1.11 for parallelism
- Minimum Rust version: 1.70+

### Performance Features
- Parallel processing with Rayon work-stealing
- Cache-friendly memory layouts
- Lock-free algorithms for concurrent operations
- Profile-guided optimization support
- In-place operations and zero-copy views

### Known Limitations
- BLAS/OpenMP linking issues on macOS ARM64
- Some integration tests disabled due to dependency constraints
- Minor API changes expected before 0.1.0-beta

## [Unreleased] - Future Plans

### Planned for 0.1.0-beta
- Freeze public API surface
- Finalize RFC process for new features
- Expand cookbook and migration guide coverage
- Complete automated release pipelines (crates.io + PyPI)
- Community outreach with sample applications

### Long-term Vision
- 100% scikit-learn compatibility
- Enhanced GPU acceleration
- Distributed computing support
- Advanced AutoML capabilities
- ONNX/PMML model interchange
- Production deployment tools
- WebAssembly compilation support
- Embedded/no-std support for microcontrollers

---

[Unreleased]: https://github.com/cool-japan/sklears/compare/v0.1.0-alpha.2...HEAD
[0.1.0-alpha.2]: https://github.com/cool-japan/sklears/compare/v0.1.0-alpha.1...v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/cool-japan/sklears/releases/tag/v0.1.0-alpha.1
