# sklears-impute

[![Crates.io](https://img.shields.io/crates/v/sklears-impute.svg)](https://crates.io/crates/sklears-impute)
[![Documentation](https://docs.rs/sklears-impute/badge.svg)](https://docs.rs/sklears-impute)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-impute` provides data imputation algorithms and utilities that match scikit-learn’s impute module, with Rust-first performance improvements and extended functionality.

## Key Features

- **Imputers**: SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator, and multivariate extensions.
- **Advanced Strategies**: Matrix completion, expectation-maximization, GPU-accelerated KNN imputation.
- **Pipelines**: Drop-in compatibility with sklears pipelines and preprocessing workflows.
- **Diagnostics**: Missingness profiling, confidence intervals, and imputation quality metrics.

## Quick Start

```rust
use sklears_impute::SimpleImputer;
use scirs2_core::ndarray::array;

let x = array![
    [1.0, f64::NAN, 2.0],
    [3.0, 4.0, f64::NAN],
    [f64::NAN, 6.0, 1.0],
];

let imputer = SimpleImputer::builder()
    .strategy("mean")
    .add_missing_value(f64::NAN)
    .build();

let imputed = imputer.fit_transform(&x)?;
```

## Status

- Included in 118 passing tests in `0.2.0` (Stable).
- Supports dense and sparse matrices with deterministic output.
- Future tasks (streaming imputers, categorical encoders) tracked in this crate’s `TODO.md`.
