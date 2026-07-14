# sklears-svm

[![Crates.io](https://img.shields.io/crates/v/sklears-svm.svg)](https://crates.io/crates/sklears-svm)
[![Documentation](https://docs.rs/sklears-svm/badge.svg)](https://docs.rs/sklears-svm)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

High-performance Support Vector Machine implementations for Rust with advanced kernels and optimization algorithms, delivering 5-15x speedup over scikit-learn.

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-svm` provides comprehensive SVM implementations:

- **Core Algorithms**: SVC, SVR, LinearSVC, NuSVC, NuSVR, SparseSVM
- **Kernel Functions**: Linear, RBF, Polynomial, Sigmoid, Cosine, Chi-Squared, Intersection, custom `Kernel` trait implementations
- **Optimization**: SMO, coordinate descent, stochastic gradient descent
- **Advanced Features**: Multi-class strategies (One-vs-Rest, One-vs-One, ECOC, hierarchical), probability calibration (Platt scaling, isotonic), online/incremental learning
- **Performance**: SIMD optimization, sparse data support, optional CUDA acceleration via `oxicuda-blas` (feature-gated)

## Quick Start

```rust
use sklears_svm::{SVC, SVR, LinearSVC};
use sklears_core::traits::{Fit, Predict};
use scirs2_core::ndarray::array;

// Classification with RBF kernel
let svc = SVC::new()
    .rbf(Some(0.1))
    .c(1.0)
    .probability(true);

// Regression with polynomial kernel
let svr = SVR::new()
    .poly(3, None, 1.0)
    .epsilon(0.1);

// Linear SVM for large-scale problems
let linear_svc = LinearSVC::new()
    .with_penalty("l2")
    .with_loss("hinge")
    .with_dual(false); // Primal optimization for n_samples >> n_features

// Train and predict
let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let y = array![0.0, 1.0, 0.0];
let fitted = svc.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
let probabilities = fitted.predict_proba(&x)?;
```

## Advanced Features

### Custom Kernels

Implement the `Kernel` trait for a custom kernel function, or select any of the
built-in kernels through `KernelType`:

```rust
use sklears_svm::{Kernel, KernelType, SVC};
use scirs2_core::ndarray::ArrayView1;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct WaveKernel { period: f64, sigma: f64 }

impl Kernel for WaveKernel {
    fn compute(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        let d = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        (std::f64::consts::PI * d / self.period).cos()
            * (-d * d / (2.0 * self.sigma * self.sigma)).exp()
    }
    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::from([("period".to_string(), self.period), ("sigma".to_string(), self.sigma)])
    }
}

// Built-in kernels are selected through `KernelType`
let svc = SVC::new().kernel(KernelType::Cosine).c(1.0);
```

### Multi-class Strategies

```rust
use sklears_svm::{MultiClassSVC, MultiClassStrategy};

// One-vs-Rest strategy (default)
let svc_ovr = MultiClassSVC::new().strategy(MultiClassStrategy::OneVsRest);

// One-vs-One strategy with majority voting
let svc_ovo = MultiClassSVC::new().strategy(MultiClassStrategy::OneVsOne);

// Error-Correcting Output Codes
let svc_ecoc = MultiClassSVC::new().strategy(MultiClassStrategy::Ecoc);
```

### Online Learning

```rust
use sklears_svm::{OnlineSvm, kernels::LinearKernel};

let mut online_svm = OnlineSvm::new(LinearKernel::new(), Default::default());

// Incremental learning, one sample at a time
for (x, y) in data_stream {
    online_svm.partial_fit(&x, y)?;
}
```

### Probability Calibration

```rust
use sklears_svm::SVC;

let svc = SVC::new().rbf(Some(0.1)).probability(true); // fits Platt scaling internally

let fitted = svc.fit(&x, &y)?;
let calibrated_probs = fitted.predict_proba(&x_test)?;
```

## Performance Features

### Sparse Data Support

```rust
use sklears_svm::SparseSVM;

let sparse_svc = SparseSVM::new()
    .with_c(1.0)
    .with_loss("hinge");

let fitted = sparse_svc.fit(&x, &y)?; // y: Array1<i32> class labels
```

### Parallel Training

Enable the `parallel` feature (on by default) to parallelize kernel-matrix and
gradient computations internally via `rayon`. Kernel-cache size is tunable per
estimator:

```rust
let svc = SVC::new()
    .rbf(Some(0.1))
    .cache_size(500); // MB for kernel cache
```

### Optimization Strategies

```rust
use sklears_svm::{SVC, LinearSVC, SGDClassifier};

// SMO with shrinking heuristics (SVC's built-in solver)
let svc_smo = SVC::new().rbf(Some(0.1)).shrinking(true);

// Coordinate descent for linear SVM ('dual_cd', 'primal_cd', 'enhanced_cd')
let linear_svc_cd = LinearSVC::new().with_solver("primal_cd");

// Stochastic gradient descent
let sgd_svm = SGDClassifier::new().with_alpha(0.0001).with_max_iter(1000);
```

## Advanced Algorithms

### Nu-Support Vector Machines

```rust
use sklears_svm::{KernelType, NuSVC, NuSVR};

// Nu-SVC with automatic margin
let nu_svc = NuSVC::new()
    .nu(0.5)? // Upper bound on fraction of margin errors
    .kernel(KernelType::Rbf { gamma: 0.1 });

// Nu-SVR for regression
let nu_svr = NuSVR::new()
    .nu(0.5)?
    .kernel(KernelType::Polynomial { gamma: 1.0, coef0: 0.0, degree: 2.0 });
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
└── gpu/           # GPU acceleration (feature-gated, oxicuda-blas)
```

## Status

- **Tests**: 302 passing crate tests for `0.2.0` (38 skipped)
- **Core Algorithms**: 90% complete
- **Kernel Functions**: All major kernels implemented
- **Optimization**: SMO, CD, SGD implemented
- **GPU Support**: Implemented behind the `gpu` feature — `oxicuda-blas` GEMM drives the inner-product term for Linear/RBF/Polynomial/Sigmoid kernels, with the RBF/Sigmoid non-linear transform also running on-device

## Contributing

Priority areas for contribution:
- Additional kernel functions
- Performance optimizations
- Cross-validation utilities

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0.

## Citation

```bibtex
@software{sklears_svm,
  title = {sklears-svm: High-Performance Support Vector Machines for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```
