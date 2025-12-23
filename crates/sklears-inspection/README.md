# sklears-inspection

[![Crates.io](https://img.shields.io/crates/v/sklears-inspection.svg)](https://crates.io/crates/sklears-inspection)
[![Documentation](https://docs.rs/sklears-inspection/badge.svg)](https://docs.rs/sklears-inspection)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.2` (December 22, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.2.md) for highlights and upgrade guidance.

## Overview

`sklears-inspection` provides model interpretation tools, mirroring scikit-learn’s inspection module with additional Rust-first performance and visualization hooks.

## Key Features

- **Permutation Importance**: CPU/GPU implementations with grouped feature support.
- **Partial Dependence**: Fast vectorized PDP and ICE computations for dense and sparse models.
- **Feature Influence**: SHAP-style approximations, ALE plots, and interaction strength metrics.
- **Visualization Hooks**: Data structures aligned with `sklears-python` plotting adapters.

## Quick Start

```rust
use sklears_inspection::permutation_importance;
use sklears_ensemble::RandomForestClassifier;

let model = RandomForestClassifier::builder()
    .n_estimators(500)
    .n_jobs(-1)
    .build()
    .fit(&x_train, &y_train)?;

let importance = permutation_importance(
    &model,
    &x_val,
    &y_val,
    None,
    10,
)?;

println!("Mean importance for feature 0: {}", importance.importances_mean[0]);
```

## Status

- Extensively covered by workspace integration tests; all 11,292 tests passed for `0.1.0-alpha.2`.
- Cross-crate sanity checks ensure compatibility with pipelines, model selection, and visualization crates.
- Further enhancements (GPU ICE surfaces, streaming importance) tracked in `TODO.md`.
