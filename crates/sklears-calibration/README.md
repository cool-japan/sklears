# sklears-calibration

[![Crates.io](https://img.shields.io/crates/v/sklears-calibration.svg)](https://crates.io/crates/sklears-calibration)
[![Documentation](https://docs.rs/sklears-calibration/badge.svg)](https://docs.rs/sklears-calibration)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-calibration` provides probability calibration tools, matching scikit-learn’s calibration module with additional Rust-centric performance improvements.

## Key Features

- **CalibratedClassifierCV**: Platt scaling, isotonic regression, and temperature scaling strategies.
- **Probability Tools**: Reliability diagrams, Brier score decomposition, and calibration curve generation.
- **Integration**: Works with sklears pipelines, model selection, and inspection modules.
- **GPU Support**: Optional CUDA/WebGPU acceleration for large-scale calibration workloads.

## Quick Start

```rust
use sklears_calibration::CalibratedClassifierCV;
use sklears_ensemble::RandomForestClassifier;

let base = RandomForestClassifier::builder()
    .n_estimators(200)
    .n_jobs(-1)
    .build();

let calibrated = CalibratedClassifierCV::builder()
    .base_estimator(base)
    .method("sigmoid")
    .cv(5)
    .build();

let fitted = calibrated.fit(&x_train, &y_train)?;
let probas = fitted.predict_proba(&x_test)?;
```

## Status

- Covered by the 11,292 passing workspace tests in `0.1.0-beta.1`.
- API parity with scikit-learn 1.5, including multi-class calibration.
- Future work (Bayesian calibration, streaming reliability) tracked in this crate’s `TODO.md`.
