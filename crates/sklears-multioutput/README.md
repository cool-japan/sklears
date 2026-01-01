# sklears-multioutput

[![Crates.io](https://img.shields.io/crates/v/sklears-multioutput.svg)](https://crates.io/crates/sklears-multioutput)
[![Documentation](https://docs.rs/sklears-multioutput/badge.svg)](https://docs.rs/sklears-multioutput)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-multioutput` implements multi-output regression and classification wrappers that allow any estimator to generalize to multi-label and multi-output settings, mirroring scikit-learn’s multioutput module.

## Key Features

- **Wrappers**: MultiOutputRegressor, MultiOutputClassifier, ClassifierChain, RegressorChain.
- **Parallelism**: Multi-threaded fitting of per-target estimators with shared caching.
- **Integration**: Works with pipelines, model selection, and calibration components out of the box.
- **Advanced Modes**: Supports probabilistic chaining, custom meta-estimators, and GPU-enabled base learners.

## Quick Start

```rust
use sklears_multioutput::MultiOutputRegressor;
use sklears_linear::Ridge;

let base_estimator = Ridge::builder()
    .alpha(1.0)
    .fit_intercept(true)
    .build();

let wrapper = MultiOutputRegressor::new(base_estimator);
let fitted = wrapper.fit(&x_train, &y_train)?;
let predictions = fitted.predict(&x_test)?;
```

## Status

- Validated by the overall 11,292 passing workspace tests for `0.1.0-beta.1`.
- Ensures full parity with scikit-learn’s multi-output utilities while leveraging Rust’s performance.
- Future enhancements (asynchronous chaining, probabilistic calibration) tracked in `TODO.md`.
