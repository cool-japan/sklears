# sklears-covariance

[![Crates.io](https://img.shields.io/crates/v/sklears-covariance.svg)](https://crates.io/crates/sklears-covariance)
[![Documentation](https://docs.rs/sklears-covariance/badge.svg)](https://docs.rs/sklears-covariance)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-covariance` offers covariance estimation, robust covariance models, and shrinkage estimators with scikit-learn compatibility and Rust performance.

## Key Features

- **Estimators**: EmpiricalCovariance, LedoitWolf, OAS, MinCovDet, GraphicalLasso, EllipticEnvelope.
- **Robust Statistics**: Outlier detection, Mahalanobis distances, and anomaly scoring APIs.
- **Performance**: SIMD-enabled linear algebra, batch processing, and optional GPU acceleration.
- **Integration**: Used by Gaussian processes, anomaly detection, and manifold learning crates.

## Quick Start

```rust
use sklears_covariance::{EmpiricalCovariance, LedoitWolf};
use scirs2_core::ndarray::Array2;

let x: Array2<f64> = // load your dataset
    Array2::zeros((500, 20));

let empirical = EmpiricalCovariance::default().fit(&x)?;
let cov = empirical.covariance();

let shrunk = LedoitWolf::default().fit(&x)?;
let precision = shrunk.precision();
```

## Status

- Exercised via the workspace’s 10,013 passing tests for `0.1.0-alpha.1`.
- Numerical accuracy matches scikit-learn baselines within tolerance on standard benchmarks.
- Future items (GPU graphical lasso, streaming covariance) tracked in this crate’s `TODO.md`.
