# sklears-model-selection

[![Crates.io](https://img.shields.io/crates/v/sklears-model-selection.svg)](https://crates.io/crates/sklears-model-selection)
[![Documentation](https://docs.rs/sklears-model-selection/badge.svg)](https://docs.rs/sklears-model-selection)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

`sklears-model-selection` implements the full suite of scikit-learn model selection utilities—grid search, random search, halving strategies, cross-validation splits, and scoring helpers—optimized for Rust performance and concurrency.

## Key Features

- **Search Strategies**: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearch`, `HalvingRandomSearch`, Bayesian/Adaptive search prototypes.
- **Cross-Validation**: K-fold, stratified, grouped, time-series splits, repeated strategies, and custom splitter APIs.
- **Scoring & Metrics**: `make_scorer`, scorer registry, multi-metric evaluation, and custom scorer plugins.
- **Parallel Execution**: Rayon-powered parallel hyperparameter evaluation (`n_jobs`) for grid/randomized/halving search.

## Quick Start

```rust
use sklears_linear::{LogisticRegression, Penalty};
use sklears_model_selection::{GridSearchCV, KFold, ParameterGrid, ParameterValue, Scoring};
use std::collections::HashMap;

let estimator = LogisticRegression::new().max_iter(200);

let mut param_grid: ParameterGrid = HashMap::new();
param_grid.insert(
    "C".to_string(),
    vec![
        ParameterValue::Float(0.1),
        ParameterValue::Float(1.0),
        ParameterValue::Float(10.0),
    ],
);

// `config_fn` applies one parameter combination to a fresh estimator clone
let grid_search = GridSearchCV::new(estimator, param_grid, |est, params| {
    if let Some(ParameterValue::Float(c)) = params.get("C") {
        Ok(est.penalty(Penalty::L2(*c)))
    } else {
        Ok(est)
    }
})
.cv(KFold::new(5))
.n_jobs(Some(8))
.scoring(Scoring::EstimatorScore);

let fitted = grid_search.fit(&x_train, &y_train)?;
let best_params = fitted.best_params();
```

## Status

- Validated by 351 passing crate tests for `0.2.0` (6 skipped).
- Supports >99% of scikit-learn’s model selection API (including paired scoring functions and CV splitters).
- Upcoming improvements (asynchronous evaluators, distributed tuning) documented in `TODO.md`.
