# sklears-discriminant-analysis

[![Crates.io](https://img.shields.io/crates/v/sklears-discriminant-analysis.svg)](https://crates.io/crates/sklears-discriminant-analysis)
[![Documentation](https://docs.rs/sklears-discriminant-analysis/badge.svg)](https://docs.rs/sklears-discriminant-analysis)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-discriminant-analysis` implements Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and related subspace methods with scikit-learn compatible APIs. The crate emphasizes numerical robustness, GPU acceleration, and seamless integration with the broader sklears ecosystem.

## Key Features

- **Comprehensive Algorithms**: LDA, QDA, shrinkage estimators, regularized discriminant analysis, and Bayesian variants.
- **Performance Optimizations**: SIMD-enabled linear algebra, batched matrix factorizations, and optional GPU backends.
- **Pipeline Support**: Works with sklears pipelines, calibration, and model selection utilities.
- **Probability Calibration**: Built-in support for Platt scaling and isotonic calibration for multiclass scenarios.

## Quick Start

```rust
use sklears_discriminant_analysis::LinearDiscriminantAnalysis;
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [6.0, 9.0],
];
let y = Array1::from(vec![0, 0, 1, 1]);

let lda = LinearDiscriminantAnalysis::builder()
    .solver("svd")
    .shrinkage(None)
    .n_components(Some(1))
    .build();

let fitted = lda.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
```

## Status

- Covered by the shared 11,292 passing workspace tests in the `0.1.0-beta.1` release.
- Numerical stability validated on high-dimensional datasets using SciRS2 linear algebra backends.
- Future enhancements (incremental LDA, GPU QDA) tracked within this crate's `TODO.md`.
