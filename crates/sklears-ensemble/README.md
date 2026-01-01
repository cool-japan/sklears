# sklears-ensemble

[![Crates.io](https://img.shields.io/crates/v/sklears-ensemble.svg)](https://crates.io/crates/sklears-ensemble)
[![Documentation](https://docs.rs/sklears-ensemble/badge.svg)](https://docs.rs/sklears-ensemble)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-ensemble` delivers bagging, boosting, stacking, voting, and random forest implementations with scikit-learn parity and Rust-first performance.

## Key Features

- **Tree Ensembles**: RandomForest, ExtraTrees, GradientBoosting, Histogram-based boosting, IsolationForest.
- **Linear/Stochastic Ensembles**: Bagging, AdaBoost, Stacking, Voting, Snapshot ensembles, warm-starting.
- **GPU + SIMD**: Accelerated split finding, batched inference, and quantized histograms.
- **AutoML Integration**: Works with feature selection, model selection, and inspection crates for end-to-end workflows.

## Quick Start

```rust
use sklears_ensemble::RandomForestClassifier;
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [0.0, 1.0, 2.0],
    [1.0, 0.5, 2.1],
    [0.5, 2.0, 1.5],
];
let y = Array1::from(vec![0, 1, 0]);

let forest = RandomForestClassifier::builder()
    .n_estimators(500)
    .max_depth(Some(10))
    .n_jobs(-1)
    .bootstrap(true)
    .build();

let fitted = forest.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
```

## Status

- Included in the 11,292 passing workspace tests for `0.1.0-beta.1`.
- Benchmarks demonstrate 5–30× faster training versus scikit-learn on medium to large datasets.
- Roadmap items (GPU GradientBoosting, federated ensembles) live in this crate’s `TODO.md`.
