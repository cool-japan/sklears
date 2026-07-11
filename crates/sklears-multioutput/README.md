# sklears-multioutput

[![Crates.io](https://img.shields.io/crates/v/sklears-multioutput.svg)](https://crates.io/crates/sklears-multioutput)
[![Documentation](https://docs.rs/sklears-multioutput/badge.svg)](https://docs.rs/sklears-multioutput)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-multioutput` implements meta-estimators for multi-target regression and classification — independent per-target prediction (`MultiOutputRegressor`/`MultiOutputClassifier`), classifier/regressor chains, and multi-label utilities — mirroring scikit-learn's multioutput module.

## Key Features

- **Estimators**: MultiOutputRegressor, MultiOutputClassifier, ClassifierChain, RegressorChain, and `EnsembleOfChains`.
- **Parallelism**: Multi-threaded fitting of per-target models via `n_jobs`.
- **Integration**: Works with pipelines, model selection, and calibration components out of the box.
- **Advanced Modes**: Probabilistic (Bayesian-inference) classifier chains and configurable chain ordering/cross-validation.

## Quick Start

```rust
use sklears_multioutput::MultiOutputRegressor;
use sklears_core::traits::{Fit, Predict};
use scirs2_core::ndarray::array;

let x_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 5.0]];
let y_train = array![[1.5, 2.5], [2.5, 3.5], [2.0, 1.5], [3.0, 4.0]];

let wrapper = MultiOutputRegressor::new().n_jobs(Some(4));
let fitted = wrapper.fit(&x_train.view(), &y_train)?;
let predictions = fitted.predict(&x_train.view())?;
```

## Status

- Validated by 246 passing tests in `0.2.0` (Stable).
- Ensures full parity with scikit-learn’s multi-output utilities while leveraging Rust’s performance.
- Future enhancements (asynchronous chaining, probabilistic calibration) tracked in `TODO.md`.
