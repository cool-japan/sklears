# sklears-preprocessing

[![Crates.io](https://img.shields.io/crates/v/sklears-preprocessing.svg)](https://crates.io/crates/sklears-preprocessing)
[![Documentation](https://docs.rs/sklears-preprocessing/badge.svg)](https://docs.rs/sklears-preprocessing)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-preprocessing` contains scalers, encoders, transformers, and feature engineering utilities that mirror scikit-learn’s preprocessing module while leveraging Rust performance.

## Key Features

- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer.
- **Encoders**: OneHotEncoder, OrdinalEncoder, TargetEncoder, PolynomialFeatures, Binarizer.
- **Feature Utilities**: Normalizer, FunctionTransformer, MissingIndicator, discretizers, and outlier filters.
- **Hardware Acceleration**: SIMD, multi-threading, and optional GPU support for large tabular datasets.

## Quick Start

```rust
use sklears_preprocessing::{StandardScaler, PolynomialFeatures};

let scaler = StandardScaler::default().fit(&x_train)?;
let x_scaled = scaler.transform(&x_train)?;

let poly = PolynomialFeatures::builder()
    .degree(3)
    .include_bias(false)
    .interaction_only(false)
    .build();

let x_poly = poly.fit_transform(&x_scaled)?;
```

## Status

- Extensively covered by the 10,013 passing workspace tests in `0.1.0-alpha.1`.
- Provides >99% parity with scikit-learn preprocessing APIs, including sparse support.
- Future enhancements (GPU categorical encoders, streaming scalers) tracked in `TODO.md`.
