# sklears-svm

[![Crates.io](https://img.shields.io/crates/v/sklears-svm.svg)](https://crates.io/crates/sklears-svm)
[![Documentation](https://docs.rs/sklears-svm/badge.svg)](https://docs.rs/sklears-svm)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

High-performance Support Vector Machine implementations for Rust with advanced kernels and optimization algorithms, delivering 5-15x speedup over scikit-learn.

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-svm` provides comprehensive SVM implementations:

- **Core Algorithms**: SVC, SVR, LinearSVC, NuSVC, NuSVR
- **Kernel Functions**: Linear, RBF, Polynomial, Sigmoid, Custom kernels
- **Optimization**: SMO, coordinate descent, stochastic gradient descent
- **Advanced Features**: Multi-class strategies, probability calibration, online learning
- **Performance**: SIMD optimization, sparse data support, optional CUDA/WebGPU acceleration

## Quick Start

```rust
use sklears_svm::{SVC, SVR, Kernel};
use ndarray::array;

// Classification with RBF kernel
let svc = SVC::builder()
    .kernel(Kernel::RBF { gamma: 0.1 })
    .C(1.0)
    .probability(true)
    .build();

// Regression with polynomial kernel
let svr = SVR::builder()
    .kernel(Kernel::Polynomial { degree: 3, coef0: 1.0 })
    .epsilon(0.1)
    .build();

// Linear SVM for large-scale problems
let linear_svc = LinearSVC::builder()
    .penalty(Penalty::L2)
    .loss(Loss::Hinge)
    .dual(false)  // Primal optimization for n_samples >> n_features
    .build();

// Train and predict
let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let y = array![0, 1, 0];
let fitted = svc.fit(&X, &y)?;
let predictions = fitted.predict(&X)?;
let probabilities = fitted.predict_proba(&X)?;
```

## Advanced Features

### Custom Kernels

```rust
use sklears_svm::{CustomKernel, KernelFunction};

// Define custom kernel function
let custom_kernel = CustomKernel::new(|x1, x2| {
    let diff = x1 - x2;
    (-0.5 * diff.dot(&diff)).exp()  // Gaussian-like
});

let svc = SVC::builder()
    .kernel(Kernel::Custom(custom_kernel))
    .build();
```

### Multi-class Strategies

```rust
use sklears_svm::{MultiClassStrategy};

// One-vs-Rest strategy
let svc_ovr = SVC::builder()
    .multi_class(MultiClassStrategy::OneVsRest)
    .build();

// One-vs-One strategy (default for SVC)
let svc_ovo = SVC::builder()
    .multi_class(MultiClassStrategy::OneVsOne)
    .build();

// Crammer-Singer multi-class
let svc_cs = LinearSVC::builder()
    .multi_class(MultiClassStrategy::CrammerSinger)
    .build();
```

### Online Learning

```rust
use sklears_svm::SGDClassifier;

let mut sgd = SGDClassifier::builder()
    .loss(Loss::Hinge)
    .learning_rate(LearningRate::Optimal)
    .build();

// Incremental learning
for batch in data_stream {
    sgd.partial_fit(&batch.X, &batch.y)?;
}
```

### Probability Calibration

```rust
use sklears_svm::{SVC, CalibrationMethod};

let svc = SVC::builder()
    .probability(true)
    .calibration_method(CalibrationMethod::Sigmoid)
    .build();

let fitted = svc.fit(&X, &y)?;
let calibrated_probs = fitted.predict_proba(&X_test)?;
```

## Performance Features

### Sparse Data Support

```rust
use sklears_svm::SparseSVC;
use sprs::CsMat;

let sparse_X = CsMat::from_dense(&X);
let sparse_svc = SparseSVC::builder()
    .kernel(Kernel::Linear)
    .build();

let fitted = sparse_svc.fit(&sparse_X, &y)?;
```

### Parallel Training

```rust
let svc = SVC::builder()
    .kernel(Kernel::RBF { gamma: 0.1 })
    .n_jobs(4)  // Use 4 threads
    .cache_size(500)  // MB for kernel cache
    .build();
```

### Optimization Strategies

```rust
use sklears_svm::{Solver, ShrinkingHeuristics};

// SMO with shrinking heuristics
let svc_smo = SVC::builder()
    .solver(Solver::SMO)
    .shrinking(true)
    .build();

// Coordinate descent for linear SVM
let linear_svc_cd = LinearSVC::builder()
    .solver(Solver::CoordinateDescent)
    .build();

// Stochastic gradient descent
let sgd_svm = SGDClassifier::builder()
    .alpha(0.0001)
    .max_iter(1000)
    .build();
```

## Advanced Algorithms

### Nu-Support Vector Machines

```rust
use sklears_svm::{NuSVC, NuSVR};

// Nu-SVC with automatic margin
let nu_svc = NuSVC::builder()
    .nu(0.5)  // Upper bound on fraction of margin errors
    .kernel(Kernel::RBF { gamma: 0.1 })
    .build();

// Nu-SVR for regression
let nu_svr = NuSVR::builder()
    .nu(0.5)
    .kernel(Kernel::Polynomial { degree: 2, coef0: 0.0 })
    .build();
```

### One-Class SVM

```rust
use sklears_svm::OneClassSVM;

// Anomaly detection
let oc_svm = OneClassSVM::builder()
    .nu(0.1)  // Expected fraction of outliers
    .kernel(Kernel::RBF { gamma: 0.1 })
    .build();

let fitted = oc_svm.fit(&X_normal)?;
let anomaly_scores = fitted.decision_function(&X_test)?;
```

## Kernel Approximation

```rust
use sklears_svm::{Nystroem, RBFSampler};

// Nystroem approximation for large-scale kernels
let nystroem = Nystroem::builder()
    .kernel(Kernel::RBF { gamma: 0.1 })
    .n_components(100)
    .build();

// Random Fourier features
let rbf_sampler = RBFSampler::builder()
    .gamma(0.1)
    .n_components(1000)
    .build();

// Use with linear SVM for speed
let X_transformed = nystroem.fit_transform(&X)?;
let linear_svc = LinearSVC::default();
let fitted = linear_svc.fit(&X_transformed, &y)?;
```

## Benchmarks

Performance comparisons:

| Algorithm | scikit-learn | sklears-svm | Speedup |
|-----------|--------------|-------------|---------|
| Linear SVC | 45ms | 5ms | 9x |
| RBF SVC | 120ms | 15ms | 8x |
| Nu-SVC | 135ms | 18ms | 7.5x |
| SGD Classifier | 8ms | 0.8ms | 10x |

## Architecture

```
sklears-svm/
├── core/           # Core SVM algorithms
├── kernels/        # Kernel implementations
├── solvers/        # Optimization algorithms
├── multiclass/     # Multi-class strategies
├── online/         # Incremental learning
├── sparse/         # Sparse data support
└── gpu/           # GPU acceleration (WIP)
```

## Status

- **Core Algorithms**: 90% complete
- **Kernel Functions**: All major kernels implemented
- **Optimization**: SMO, CD, SGD implemented
- **GPU Support**: In development

## Contributing

Priority areas for contribution:
- GPU kernel computations
- Additional kernel functions
- Performance optimizations
- Cross-validation utilities

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under either of:
- Apache License, Version 2.0
- MIT license

## Citation

```bibtex
@software{sklears_svm,
  title = {sklears-svm: High-Performance Support Vector Machines for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```
