# sklears-discriminant-analysis

[![Crates.io](https://img.shields.io/crates/v/sklears-discriminant-analysis.svg)](https://crates.io/crates/sklears-discriminant-analysis)
[![Documentation](https://docs.rs/sklears-discriminant-analysis/badge.svg)](https://docs.rs/sklears-discriminant-analysis)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

`sklears-discriminant-analysis` implements Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and related subspace methods with scikit-learn compatible APIs. The crate emphasizes numerical robustness, GPU acceleration, and seamless integration with the broader sklears ecosystem.

## Key Features

- **Comprehensive Algorithms**: LDA, QDA, shrinkage estimators, regularized discriminant analysis, and Bayesian variants.
- **Performance Optimizations**: SIMD-enabled linear algebra, batched matrix factorizations, and (behind the opt-in `gpu` feature, not enabled by default) a real `sklears_core::gpu`-backed GPU path (GEMM-based class-statistics, an LDA generalized-eigenvalue solve via Cholesky reduction verified against SciPy to ~1e-15, and QDA via `oxicuda-solver`).
- **Pipeline Support**: Works with sklears pipelines and model selection utilities; `predict_proba` is available on the fitted estimators.

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

let lda = LinearDiscriminantAnalysis::new()
    .solver("svd")
    .shrinkage(None)
    .n_components(Some(1));

let fitted = lda.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
```

## Status

- Covered by 322 passing tests in `0.2.0` (Partial — actively evolving).
- Numerical stability validated on high-dimensional datasets using SciRS2 linear algebra backends; `NumericalStability::stable_inverse` now correctly handles non-symmetric matrices via `scirs2_linalg::inv` (previously a stub that returned `NotImplemented`).
- GPU acceleration now targets the `sklears_core::gpu` foundation (`GpuBackend`/`GpuArray`/`GpuMatrixOps`), replacing the previous dead `scirs2_core::gpu::*` path.
- Future enhancements (incremental LDA) tracked within this crate's `TODO.md`.
