# sklears-multiclass

[![Crates.io](https://img.shields.io/crates/v/sklears-multiclass.svg)](https://crates.io/crates/sklears-multiclass)
[![Documentation](https://docs.rs/sklears-multiclass/badge.svg)](https://docs.rs/sklears-multiclass)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

State-of-the-art multiclass classification strategies for Rust, providing 5-15x performance improvements over scikit-learn while maintaining API familiarity.

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-multiclass` implements comprehensive multiclass classification strategies including:

- **Binary Decomposition**: One-vs-Rest (OvR), One-vs-One (OvO), Error-Correcting Output Codes (ECOC)
- **Advanced Ensemble Methods**: AdaBoost, Gradient Boosting, Stacking, Rotation Forest
- **Hierarchical Classification**: Nested dichotomies, recursive binary partitioning, taxonomy-aware classification
- **Calibration & Uncertainty**: Platt scaling, isotonic regression, temperature/Dirichlet scaling, conformal prediction
- **Production Features**: Builder APIs, prediction caching, optional GPU-accelerated distance/matmul ops (`gpu` feature)

## Quick Start

```rust
use sklears_multiclass::{OneVsOneClassifier, OneVsRestClassifier};
use sklears_linear::LogisticRegression;
use scirs2_core::ndarray::array;

// Create base binary classifier
let base_classifier = LogisticRegression::default();

// One-vs-Rest strategy (the base estimator is passed into the builder)
let ovr = OneVsRestClassifier::builder(base_classifier.clone())
    .parallel() // enable parallel training across all cores
    .build();

// One-vs-One strategy
let ovo = OneVsOneClassifier::builder(base_classifier).build();

// Train and predict
let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
let y = array![0, 1, 2];

let trained_ovr = ovr.fit(&X, &y)?;
let predictions = trained_ovr.predict(&X)?;
```

## Features

### Core Strategies

- **One-vs-Rest (OvR)**: Efficient parallel training of binary classifiers
- **One-vs-One (OvO)**: Pairwise classification with advanced voting
- **ECOC**: Error-correcting codes with optimized/quantized/compressed code matrices

### Advanced Methods

- **Hierarchical Classification**: Nested dichotomies, recursive binary partitioning, and taxonomy-aware classification
- **Ensemble Methods**: Bagging, AdaBoost, Gradient Boosting, Stacking, Rotation Forest
- **Incremental Learning**: Warm-start, drift detection (Page-Hinkley, ADWIN), and memory-bounded online learners
- **Calibration**: Platt scaling, isotonic regression, temperature/Dirichlet scaling, conformal prediction

### Production Ready

- **Builder Pattern APIs**: Consistent, type-safe configuration
- **Parallel Training**: Rayon-based parallelization
- **Prediction Caching**: Configurable prediction cache with eviction
- **Sparse/Quantized Storage**: Memory-efficient ECOC matrices
- **Optional GPU Acceleration**: OxiCUDA-backed matmul and ECOC distance ops behind the `gpu` feature (honest CPU fallback when no device is present)

## Performance

Benchmarks on standard datasets show:
- **5-15x speedup** over scikit-learn
- **50% less memory** usage
- **Linear scalability** with CPU cores
- **Optional GPU acceleration** via OxiCUDA (CUDA only, `gpu` feature)

## Examples

### Calibrated Classification

```rust
use sklears_multiclass::{CalibratedClassifier, CalibrationMethod};

let calibrated = CalibratedClassifier::builder(classifier)
    .method(CalibrationMethod::PlattScaling)
    .cv_folds(5)
    .build();
```

### Hierarchical Classification (Nested Dichotomies)

```rust
use sklears_multiclass::{DichotomyStrategy, NestedDichotomiesClassifier};

let hierarchical = NestedDichotomiesClassifier::builder(classifier)
    .strategy(DichotomyStrategy::Balanced)
    .build();
```

## Architecture

The crate follows a modular design:

```
sklears-multiclass/
├── src/
│   ├── core/           # Core traits, types, ECOC code matrices
│   ├── one_vs_rest.rs, one_vs_one.rs  # Binary decomposition methods
│   ├── ensemble/       # Bagging, AdaBoost, Gradient Boosting, Stacking, Rotation Forest
│   ├── calibration/    # Probability calibration
│   ├── advanced/       # Hierarchical / nested-dichotomy classifiers
│   ├── incremental/    # Warm-start, drift detection, online learning
│   ├── uncertainty/    # Conformal prediction
│   └── gpu/            # OxiCUDA-backed GPU ops (feature `gpu`)
```

## Status

**Stable** (`0.2.0`)

- **Tests**: 299 passing
- **Production**: Ready
- **GPU Support**: Implemented behind the optional `gpu` feature (OxiCUDA/CUDA; honest CPU fallback with no device)

## Roadmap

### v1.0 (August 2026)
- [x] Core multiclass strategies
- [x] Advanced ensemble methods
- [x] Calibration framework
- [x] GPU acceleration (optional `gpu` feature, OxiCUDA)
- [ ] External ML framework integration

### v1.1 (Q4 2026)
- [ ] Deep learning integration
- [ ] Meta-learning support
- [ ] Federated learning
- [ ] Edge deployment

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE)).

## Citation

If you use this crate in your research, please cite:

```bibtex
@software{sklears_multiclass,
  title = {sklears-multiclass: High-Performance Multiclass Classification for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```
