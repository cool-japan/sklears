# sklears-compose

[![Crates.io](https://img.shields.io/crates/v/sklears-compose.svg)](https://crates.io/crates/sklears-compose)
[![Documentation](https://docs.rs/sklears-compose/badge.svg)](https://docs.rs/sklears-compose)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-compose` implements pipelines, column transformers, target transformers, and composite estimator utilities matching scikit-learn’s compose module.

## Key Features

- **Pipelines**: Type-safe, state-aware `Pipeline` and `FeatureUnion` implementations with parallel execution support.
- **Column Transforms**: ColumnTransformer, make_column_transformer, and feature selection by dtype or name.
- **Target Transformations**: TransformedTargetRegressor, inverse-transform aware scorers, and custom adapters.
- **Serialization**: Friendly with serde-powered persistence and Python interoperability via `sklears-python`.

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

- Verified through workspace integration tests; `0.1.0-alpha.1` recorded 10,013 passes with zero failures.
- Supports all major scikit-learn compose APIs plus Rust-centric ergonomic improvements.
- Future enhancements (async pipelines, streaming feature unions) tracked in `TODO.md`.
