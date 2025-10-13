# sklears-cross-decomposition

[![Crates.io](https://img.shields.io/crates/v/sklears-cross-decomposition.svg)](https://crates.io/crates/sklears-cross-decomposition)
[![Documentation](https://docs.rs/sklears-cross-decomposition/badge.svg)](https://docs.rs/sklears-cross-decomposition)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-cross-decomposition` offers Partial Least Squares (PLS) regressors/classifiers, Canonical Correlation Analysis (CCA), and related cross-decomposition utilities. The APIs mirror scikit-learn 1.5, while Rust-native optimizations deliver consistent performance gains.

## Key Features

- **PLS Family**: PLSRegression, PLSCanonical, PLSRegressionCV, and sparse extensions.
- **CCA & Variants**: Dense and sparse canonical correlation, along with GPU-accelerated solvers.
- **Model Selection Integration**: Works seamlessly with sklears pipelines, grid search, and feature engineering crates.
- **Robust Numerics**: Regularization, deflation strategies, and whitening controls ensure stability on real-world datasets.

## Quick Start

```rust
use sklears_cross_decomposition::PLSRegression;
use scirs2_core::ndarray::{array, Array2};

let x: Array2<f64> = array![
    [0.0, 1.0, 2.0],
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
];
let y: Array2<f64> = array![
    [1.0, 0.0],
    [2.0, 1.0],
    [3.0, 2.0],
];

let pls = PLSRegression::builder()
    .n_components(2)
    .scale(true)
    .max_iter(500)
    .tol(1e-6)
    .build();

let fitted = pls.fit(&x, &y)?;
let y_pred = fitted.predict(&x)?;
```

## Status

- Fully validated by the 10,013 passing workspace tests bundled with `0.1.0-alpha.1`.
- Benchmarks show 8–20× speedups versus scikit-learn for large PLS problems.
- Upcoming enhancements (incremental fit, streaming CCA) tracked in `TODO.md`.
