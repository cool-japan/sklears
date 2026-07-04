# sklears-compose

[![Crates.io](https://img.shields.io/crates/v/sklears-compose.svg)](https://crates.io/crates/sklears-compose)
[![Documentation](https://docs.rs/sklears-compose/badge.svg)](https://docs.rs/sklears-compose)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-compose` implements pipelines, column transformers, target transformers, and composite estimator utilities matching scikit-learn’s compose module.

## Key Features

- **Pipelines**: Type-safe, state-aware `Pipeline` and `FeatureUnion` implementations with parallel execution support.
- **Column Transforms**: ColumnTransformer, make_column_transformer, and feature selection by dtype or name.
- **Target Transformations**: TransformedTargetRegressor, inverse-transform aware scorers, and custom adapters.
- **Serialization**: Friendly with serde-powered persistence and Python interoperability via `sklears-python`.
- **Stacking**: `stacking::StackingEnsemble` performs genuine out-of-fold k-fold stacked generalization (leakage-safe meta-features) over a pluggable set of base learners plus a meta-learner (defaults to OLS).
- **Hierarchical Composition**: `hierarchical_composition::HierarchicalComposition` trains every level for real; its `Stacked` strategy builds meta-features via genuine out-of-fold cross-validation instead of a placeholder.
- **Model Fusion**: `model_fusion::ModelFusion` trains its base models via real `fit()` calls; the `WeightedLinear` strategy solves a real OLS/ridge-regularized least-squares problem for fusion weights, and `NeuralNetwork` fusion trains a small MLP via real gradient descent. Other fusion strategies honestly return `NotImplemented` rather than fabricating a result.
- **Pipeline Visualization (not yet functional)**: `pipeline_visualization` types are wired into the public API, but rendering and graph/node/edge extraction now honestly return `NotImplemented` errors instead of fake or empty output — there is no working visualization output yet (see `TODO.md`).

## Quick Start

```rust
use sklears_compose::{Pipeline, make_column_transformer};
use sklears_preprocessing::{StandardScaler, OneHotEncoder};
use sklears_linear::LinearRegression;

let preprocessor = make_column_transformer()
    .with_transformer("numeric", StandardScaler::default(), vec![0, 1, 2])
    .with_transformer("categorical", OneHotEncoder::default(), vec![3])
    .build();

let pipeline = Pipeline::builder()
    .add_step("preprocess", preprocessor)
    .add_step("model", LinearRegression::default())
    .build();

let fitted = pipeline.fit(&x_train, &y_train)?;
let predictions = fitted.predict(&x_test)?;
```

## Status

- Verified through workspace integration tests; `0.2.0` records 782 passing tests.
- Supports all major scikit-learn compose APIs plus Rust-centric ergonomic improvements.
- This session replaced several silently-faked training paths with real implementations: ensemble `model_fusion` (real fit calls, real OLS/ridge weight solve, real gradient-descent neural fusion), `hierarchical_composition` (every level genuinely trained, real out-of-fold stacking), and a new `stacking` module. `pipeline_visualization` now honestly reports `NotImplemented` instead of returning fake output, but rendering itself is still not implemented.
- Future enhancements (async pipelines, streaming feature unions, full pipeline visualization) tracked in `TODO.md`.
