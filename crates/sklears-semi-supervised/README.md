# sklears-semi-supervised

[![Crates.io](https://img.shields.io/crates/v/sklears-semi-supervised.svg)](https://crates.io/crates/sklears-semi-supervised)
[![Documentation](https://docs.rs/sklears-semi-supervised/badge.svg)](https://docs.rs/sklears-semi-supervised)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-semi-supervised` implements semi-supervised learning algorithms that align with scikit-learn’s API, covering label propagation, self-training, and graph-based methods.

## Key Features

- **Algorithms**: LabelPropagation, LabelSpreading, SelfTrainingClassifier, CoTraining prototypes, and graph-based methods.
- **Graph Support**: Efficient knn graph construction, similarity kernels, and CUDA/WebGPU backends for large graphs.
- **Pipeline Integration**: Works with datasets containing missing labels and plugs into sklears pipelines.
- **Monitoring**: Built-in tracking for convergence diagnostics and label confidence scores.

## Quick Start

```rust
use sklears_semi_supervised::LabelSpreading;
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.5, 0.2],
];
let y = Array1::from(vec![0, 1, -1, -1]); // -1 denotes unlabeled

let model = LabelSpreading::builder()
    .kernel("rbf")
    .gamma(0.5)
    .max_iter(100)
    .tol(1e-3)
    .build();

let fitted = model.fit(&x, &y)?;
let inferred = fitted.transduced_labels();
```

## Status

- Exercised by the shared 11,292 passing workspace tests for `0.1.0-beta.1`.
- Delivers >99% parity with scikit-learn’s semi-supervised module, plus GPU graph acceleration.
- Additional experiments (semi-supervised regression, curriculum learning) tracked in `TODO.md`.
