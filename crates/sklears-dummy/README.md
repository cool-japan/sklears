# sklears-dummy

[![Crates.io](https://img.shields.io/crates/v/sklears-dummy.svg)](https://crates.io/crates/sklears-dummy)
[![Documentation](https://docs.rs/sklears-dummy/badge.svg)](https://docs.rs/sklears-dummy)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-dummy` implements baseline estimators for regression and classification, mirroring scikit-learn’s dummy module. These models provide sanity checks, benchmarking baselines, and quick data diagnostics.

## Key Features

- **Strategies**: Mean, median, constant, stratified, most frequent, uniform, and custom priors.
- **Compatibility**: Works with classification, regression, multi-output, and probabilistic evaluation.
- **Pipelines**: Seamless integration with sklears pipelines, metrics, and inspection tooling.
- **Diagnostics**: Utilities for baseline comparisons and sanity checks during experimentation.

## Quick Start

```rust
use sklears_dummy::DummyClassifier;
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
];
let y = Array1::from(vec![0, 1, 1]);

let dummy = DummyClassifier::builder()
    .strategy("most_frequent")
    .random_state(Some(42))
    .build();

let fitted = dummy.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
```

## Status

- Included in the 11,292 passing workspace tests for `0.1.0-beta.1`.
- Perfect for establishing baselines before deploying advanced models.
- Future enhancements (time-series baselines, streaming priors) logged in `TODO.md`.
