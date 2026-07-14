# sklears-feature-selection

[![Crates.io](https://img.shields.io/crates/v/sklears-feature-selection.svg)](https://crates.io/crates/sklears-feature-selection)
[![Documentation](https://docs.rs/sklears-feature-selection/badge.svg)](https://docs.rs/sklears-feature-selection)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

`sklears-feature-selection` brings the complete scikit-learn feature selection toolbox to Rust, including filter, wrapper, and embedded methods. The crate underpins AutoML workflows, feature pipelines, and inspection utilities across the sklears project.

## Key Features

- **Filter Methods**: VarianceThreshold, mutual information, ANOVA F-tests, chi-square tests, and more.
- **Wrapper Methods**: RFE/RFECV, SequentialFeatureSelector, model-based selectors with parallel evaluation.
- **Embedded Techniques**: L1-based selection (Lasso/ElasticNet/Ridge selectors), tree-based importance, stability selection, and feature importance scoring.
- **Streaming Support**: Online and concept-drift-aware selectors (`streaming` module: `OnlineFeatureSelector`, `ConceptDriftAwareSelector`) for evolving data; heavy scoring workloads parallelize via the `parallel` module. There is no CUDA/WebGPU backend in this crate.

## Quick Start

```rust
use sklears_feature_selection::SequentialFeatureSelector;
use sklears_linear::LogisticRegression;

let estimator = LogisticRegression::new().max_iter(200);

let selector = SequentialFeatureSelector::new(estimator)
    .n_features_to_select(5)
    .direction("forward")?
    .cv(5);

let fitted = selector.fit(&x_train, &y_train)?;
let x_selected = fitted.transform(&x_train)?;
```

## Status

- Validated by 290 passing crate tests in `0.2.0`.
- Supports >99% of scikit-learn’s feature selection API surface.
- Additional milestones (distributed scoring, SHAP-guided selection) tracked in this crate’s `TODO.md`.
