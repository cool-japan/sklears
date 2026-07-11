# sklears-inspection

[![Crates.io](https://img.shields.io/crates/v/sklears-inspection.svg)](https://crates.io/crates/sklears-inspection)
[![Documentation](https://docs.rs/sklears-inspection/badge.svg)](https://docs.rs/sklears-inspection)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-inspection` provides model interpretation tools, mirroring scikit-learn’s inspection module with additional Rust-first performance and visualization hooks.

## Key Features

- **Permutation Importance**: CPU/GPU (`gpu` feature) implementations with configurable scoring functions.
- **Partial Dependence**: Fast vectorized PDP and ICE computations.
- **Feature Influence**: SHAP-style approximations and feature-interaction strength metrics.
- **Visualization Hooks**: Data structures aligned with `sklears-python` plotting adapters.

## Quick Start

```rust
use sklears_inspection::{permutation_importance, ScoreFunction};
use scirs2_core::ndarray::array;

let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
let y = array![6.0, 15.0, 24.0];
let predict_fn = |x: &scirs2_core::ndarray::ArrayView2<f64>| {
    x.rows().into_iter().map(|row| row.sum()).collect::<Vec<f64>>()
};

let result = permutation_importance(
    &predict_fn,
    &x.view(),
    &y.view(),
    ScoreFunction::R2,
    5,
    Some(42),
)?;

println!("Mean importance for feature 0: {}", result.importances_mean[0]);
```

## Status

- Extensively covered by workspace integration tests; 640 crate tests pass for `0.2.0`.
- Cross-crate sanity checks ensure compatibility with pipelines, model selection, and visualization crates.
- Further enhancements (GPU ICE surfaces, streaming importance) tracked in `TODO.md`.
