# sklears-gaussian-process

[![Crates.io](https://img.shields.io/crates/v/sklears-gaussian-process.svg)](https://crates.io/crates/sklears-gaussian-process)
[![Documentation](https://docs.rs/sklears-gaussian-process/badge.svg)](https://docs.rs/sklears-gaussian-process)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-gaussian-process` offers Gaussian Process regression and classification tooling with scikit-learn compatible APIs, expanded kernel catalogs, and high-performance Rust implementations.

## Key Features

- **Estimators**: GaussianProcessRegressor, GaussianProcessClassifier, multi-output variants, and sparse approximations.
- **Kernel Library**: RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared, White, Constant, and custom combinators.
- **Performance**: Hierarchical GP composition (`HierarchicalGaussianProcessRegressor`) and sparse/stochastic approximations for big data (Nystrom, FITC, sparse-spectrum, random Fourier features). CPU-only in this release — no GPU/CUDA backend.
- **Uncertainty Quantification**: Predictive variance, confidence intervals, and Bayesian optimization primitives.
- **Deep & Multi-Output GPs**: `deep_gp` (composable Deep Gaussian Process layers) and `convolution_processes` (an Álvarez & Lawrence-style Convolution Process / dependent multi-output GP) that provably collapses to a standard single-output GP in the degenerate case and demonstrably shares information across correlated outputs.

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

let gpr = GaussianProcessRegressor::new()
    .kernel(Box::new(RBF::new(1.0)))
    .alpha(1e-6)
    .normalize_y(true)
    .random_state(Some(123));

let fitted = gpr.fit(&x, &y)?;
let (mean, std) = fitted.predict_with_std(&x)?;
```

## Status

- Validated by 182 passing tests in `0.2.0` (Partial — actively evolving).
- Benchmarks show 5–20× faster kernel computations versus CPython implementations.
- `deep_gp` (deep Gaussian Process layers) and `convolution_processes` (dependent multi-output GP) are now enabled; `convolution_processes` is verified to collapse exactly to a standard single-output GP in the degenerate case and to demonstrably share information across correlated outputs.
- Future milestones (variational inference, GPU sparse GPs) tracked in this crate’s `TODO.md`.
