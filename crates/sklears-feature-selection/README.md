# sklears-feature-selection

[![Crates.io](https://img.shields.io/crates/v/sklears-feature-selection.svg)](https://crates.io/crates/sklears-feature-selection)
[![Documentation](https://docs.rs/sklears-feature-selection/badge.svg)](https://docs.rs/sklears-feature-selection)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-feature-selection` brings the complete scikit-learn feature selection toolbox to Rust, including filter, wrapper, and embedded methods. The crate underpins AutoML workflows, feature pipelines, and inspection utilities across the sklears project.

## Key Features

- **Filter Methods**: VarianceThreshold, mutual information, ANOVA F-tests, chi-square tests, and more.
- **Wrapper Methods**: RFE/RFECV, SequentialFeatureSelector, model-based selectors with parallel evaluation.
- **Embedded Techniques**: L1-based selection, tree-based importance, stability selection, and feature importance scoring.
- **Streaming & GPU Support**: Optional streaming evaluators and CUDA/WebGPU acceleration for heavy scoring tasks.

## Quick Start

```rust
use sklears_feature_selection::{SequentialFeatureSelector, Strategy};
use sklears_linear::LogisticRegression;

let estimator = LogisticRegression::builder()
    .max_iter(200)
    .multi_class("auto")
    .build();

let selector = SequentialFeatureSelector::builder()
    .estimator(estimator)
    .strategy(Strategy::Forward)
    .n_features_to_select(5)
    .n_jobs(4)
    .build();

let fitted = selector.fit(&x_train, &y_train)?;
let x_selected = fitted.transform(&x_train)?;
```

## Status

- Covered by the 11,292 passing workspace tests executed for `0.1.0-beta.1`.
- Supports >99% of scikit-learn’s feature selection API surface.
- Additional milestones (distributed scoring, SHAP-guided selection) tracked in this crate’s `TODO.md`.
