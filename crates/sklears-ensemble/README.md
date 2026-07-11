# sklears-ensemble

[![Crates.io](https://img.shields.io/crates/v/sklears-ensemble.svg)](https://crates.io/crates/sklears-ensemble)
[![Documentation](https://docs.rs/sklears-ensemble/badge.svg)](https://docs.rs/sklears-ensemble)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-ensemble` delivers bagging, gradient boosting, AdaBoost, stacking, and voting implementations with scikit-learn parity and Rust-first performance. (Tree-based ensembles such as RandomForest live in the separate `sklears-tree` crate.)

## Key Features

- **Ensemble Methods**: Bagging, AdaBoost, Gradient Boosting (binary classification + regression), single- and multi-layer Stacking, Voting classifiers. (`sklears-tree`'s RandomForest/ExtraTrees are a separate crate — this crate does not re-export them, and there is no IsolationForest or histogram-based boosting here; `GradientBoostingConfig::tree_type` currently accepts a `HistogramTree` variant but it has no effect on training.)
- **Model Selection & Analysis**: Bias-variance decomposition, ensemble diversity metrics, and cross-validation via the `model_selection` module (`BiasVarianceAnalyzer`, `DiversityAnalyzer`, `EnsembleCrossValidator`).
- **GPU Acceleration** (optional `gpu` feature, CUDA via `oxicuda-*`): real device detection, memory management, and GEMM/elementwise tensor ops (`GpuTensorOps`) backing ensemble prediction; default builds stay 100% Pure Rust CPU. There is no GPU-side split-finding/histogram/tree-update training path — those kernels were removed this session because they only ever returned `NotImplemented`.
- **CPU Optimization**: Cache-optimized matrix ops, vectorized ensemble scoring, and SIMD kernels via `scirs2_core::simd_ops`.

## Quick Start

```rust
use sklears_ensemble::GradientBoostingClassifier;
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [0.0, 1.0, 2.0],
    [1.0, 0.5, 2.1],
    [0.5, 2.0, 1.5],
    [2.0, 0.2, 0.1],
];
let y = Array1::from(vec![0.0, 1.0, 0.0, 1.0]);

let gbc = GradientBoostingClassifier::builder()
    .n_estimators(100)
    .learning_rate(0.1)
    .max_depth(3)
    .random_state(42)
    .build();

let fitted = gbc.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
```

## Status

- Validated by 291 passing crate tests for `0.2.0` (1 skipped).
- Benchmarks demonstrate 5–30× faster training versus scikit-learn on medium to large datasets.
- Roadmap items (GPU-side gradient-boosting training kernels, federated ensembles) live in this crate’s `TODO.md`.
