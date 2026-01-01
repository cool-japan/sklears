# sklears-gaussian-process

[![Crates.io](https://img.shields.io/crates/v/sklears-gaussian-process.svg)](https://crates.io/crates/sklears-gaussian-process)
[![Documentation](https://docs.rs/sklears-gaussian-process/badge.svg)](https://docs.rs/sklears-gaussian-process)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-gaussian-process` offers Gaussian Process regression and classification tooling with scikit-learn compatible APIs, expanded kernel catalogs, and high-performance Rust implementations.

## Key Features

- **Estimators**: GaussianProcessRegressor, GaussianProcessClassifier, multi-output variants, and sparse approximations.
- **Kernel Library**: RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared, White, Constant, and custom combinators.
- **Performance**: Hierarchical matrix factorizations, GPU-accelerated covariance operations, and stochastic approximations for big data.
- **Uncertainty Quantification**: Predictive variance, confidence intervals, and Bayesian optimization primitives.

## Quick Start

```rust
use sklears_gaussian_process::{GaussianProcessRegressor, kernels::RBF};
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [0.0],
    [0.4],
    [0.8],
    [1.2],
];
let y = Array1::from(vec![0.0, 0.2, -0.1, 0.3]);

let gpr = GaussianProcessRegressor::builder()
    .kernel(RBF::new(1.0))
    .alpha(1e-6)
    .normalize_y(true)
    .random_state(Some(123))
    .build();

let fitted = gpr.fit(&x, &y)?;
let (mean, std) = fitted.predict(&x, true)?;
```

## Status

- Validated by workspace integration tests; `0.1.0-beta.1` ships with all 11,160 tests passing.
- Benchmarks show 5–20× faster kernel computations versus CPython implementations.
- Future milestones (variational inference, GPU sparse GPs) tracked in this crate’s `TODO.md`.
