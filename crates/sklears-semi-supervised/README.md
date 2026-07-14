# sklears-semi-supervised

[![Crates.io](https://img.shields.io/crates/v/sklears-semi-supervised.svg)](https://crates.io/crates/sklears-semi-supervised)
[![Documentation](https://docs.rs/sklears-semi-supervised/badge.svg)](https://docs.rs/sklears-semi-supervised)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

`sklears-semi-supervised` implements semi-supervised learning algorithms that align with scikit-learn’s API, covering label propagation, self-training, and graph-based methods.

## Key Features

- **Algorithms**: LabelPropagation, LabelSpreading, SelfTrainingClassifier, co-training/tri-training, and graph-based methods (harmonic functions, local/global consistency, manifold regularization).
- **Graph Support**: Efficient knn graph construction, similarity kernels, and Rayon-parallelized graph algorithms (`parallel_graph`) with SIMD-accelerated distance kernels (`simd_distances`) for large graphs.
- **Pipeline Integration**: Works with datasets containing missing labels and plugs into sklears pipelines.
- **Monitoring**: Built-in tracking for convergence diagnostics and label confidence scores.

## Quick Start

```rust
use sklears_semi_supervised::LabelSpreading;
use sklears_core::traits::{Fit, Predict};
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.5, 0.2],
];
let y = Array1::from(vec![0, 1, -1, -1]); // -1 denotes unlabeled

let model = LabelSpreading::new()
    .kernel("rbf".to_string())
    .gamma(0.5)
    .max_iter(100)
    .tol(1e-3);

let fitted = model.fit(&x.view(), &y.view())?;
let inferred = fitted.predict(&x.view())?;
```

## Status

- Exercised by 356 passing tests in `0.2.0` (Stable).
- Broad coverage of scikit-learn's semi-supervised module (label propagation/spreading, self-training) plus additional graph-based methods (harmonic functions, co/tri-training, manifold regularization) not present upstream.
- Graph algorithms are CPU-parallelized (Rayon) and SIMD-accelerated; no GPU backend is implemented in this crate.
- Additional experiments (semi-supervised regression, curriculum learning) tracked in `TODO.md`.
