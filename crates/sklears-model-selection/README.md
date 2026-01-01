# sklears-model-selection

[![Crates.io](https://img.shields.io/crates/v/sklears-model-selection.svg)](https://crates.io/crates/sklears-model-selection)
[![Documentation](https://docs.rs/sklears-model-selection/badge.svg)](https://docs.rs/sklears-model-selection)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-model-selection` implements the full suite of scikit-learn model selection utilities—grid search, random search, halving strategies, cross-validation splits, and scoring helpers—optimized for Rust performance and concurrency.

## Key Features

- **Search Strategies**: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearch`, `HalvingRandomSearch`, Bayesian/Adaptive search prototypes.
- **Cross-Validation**: K-fold, stratified, grouped, time-series splits, repeated strategies, and custom splitter APIs.
- **Scoring & Metrics**: `make_scorer`, scorer registry, multi-metric evaluation, and custom scorer plugins.
- **Parallel Execution**: Rayon-powered evaluators with cancellation hooks and result caching.

## Quick Start

```rust
use sklears_model_selection::{GridSearchCV, ParamGrid};
use sklears_linear::LogisticRegression;

let estimator = LogisticRegression::builder()
    .max_iter(200)
    .multi_class("auto")
    .build();

let param_grid = ParamGrid::builder()
    .add("C", vec![0.1, 1.0, 10.0])
    .add("penalty", vec!["l2".into()])
    .build();

let grid_search = GridSearchCV::builder()
    .estimator(estimator)
    .param_grid(param_grid)
    .cv(5)
    .n_jobs(8)
    .scoring("f1_macro")
    .build();

let fitted = grid_search.fit(&x_train, &y_train)?;
let best_params = fitted.best_params();
```

## Status

- Validated by the 11,292 passing workspace tests bundled with `0.1.0-beta.1`.
- Supports >99% of scikit-learn’s model selection API (including paired scoring functions and CV splitters).
- Upcoming improvements (asynchronous evaluators, distributed tuning) documented in `TODO.md`.
