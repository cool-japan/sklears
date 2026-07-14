# sklears-preprocessing

[![Crates.io](https://img.shields.io/crates/v/sklears-preprocessing.svg)](https://crates.io/crates/sklears-preprocessing)
[![Documentation](https://docs.rs/sklears-preprocessing/badge.svg)](https://docs.rs/sklears-preprocessing)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

`sklears-preprocessing` contains scalers, encoders, transformers, and feature engineering utilities that mirror scikit-learn’s preprocessing module while leveraging Rust performance.

## Key Features

- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer.
- **Encoders**: OneHotEncoder, OrdinalEncoder, TargetEncoder, PolynomialFeatures, Binarizer.
- **Feature Utilities**: Normalizer, FunctionTransformer, SimpleImputer/KNNImputer/IterativeImputer, discretizers, and outlier filters.
- **Hardware Acceleration**: SIMD, multi-threading, and an optional OxiCUDA `gpu` feature with real on-device kernels for `StandardScaler`/`MinMaxScaler` (mean/variance/min/max reductions plus a multi-kernel elementwise `transform` pipeline via `oxicuda-blas`), with an honest CPU fallback when no device is present.

## Quick Start

```rust
use sklears_preprocessing::{StandardScaler, PolynomialFeatures};

let scaler = StandardScaler::default().fit(&x_train, &())?;
let x_scaled = scaler.transform(&x_train)?;

let poly = PolynomialFeatures::new()
    .degree(3)
    .include_bias(false)
    .interaction_only(false);

let fitted_poly = poly.fit(&x_scaled, &())?;
let x_poly = fitted_poly.transform(&x_scaled)?;
```

## Status

- Extensively covered; 377 passing crate tests in `0.2.0` (0 skipped).
- Provides >99% parity with scikit-learn preprocessing APIs, including sparse support.
- Streaming scalers (`StreamingStandardScaler`, `StreamingMinMaxScaler`, `StreamingRobustScaler`, and adaptive variants) already ship in `src/streaming.rs`; OxiCUDA-accelerated categorical encoders remain a future enhancement tracked in `TODO.md`.
