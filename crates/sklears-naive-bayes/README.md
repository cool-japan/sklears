# sklears-naive-bayes

[![Crates.io](https://img.shields.io/crates/v/sklears-naive-bayes.svg)](https://crates.io/crates/sklears-naive-bayes)
[![Documentation](https://docs.rs/sklears-naive-bayes/badge.svg)](https://docs.rs/sklears-naive-bayes)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-naive-bayes` implements the full family of Naive Bayes estimators with scikit-learn compatible APIs and Rust-powered performance.

## Key Features

- **Supported Models**: GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB, Passive-Aggressive hybrids.
- **Performance**: SIMD-accelerated likelihood computation, streaming updates, and GPU-backed batch scoring.
- **Calibration**: Compatibility with sklears calibration and inspection crates for probability post-processing.
- **Pipeline Integration**: Works seamlessly with preprocessing, feature selection, and model selection modules.

## Quick Start

```rust
use sklears_naive_bayes::MultinomialNB;
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [1u32, 0, 2],
    [0, 1, 1],
    [3, 0, 0],
];
let y = Array1::from(vec![0, 1, 0]);

let model = MultinomialNB::builder()
    .alpha(1.0)
    .fit_prior(true)
    .build();

let fitted = model.fit(&x, &y)?;
let log_probs = fitted.predict_log_proba(&x)?;
```

## Status

- Covered by workspace tests; all 10,013 suites passed for `0.1.0-alpha.1`.
- Extensive unit and property tests guarantee numerical stability on sparse matrices.
- Planned enhancements (GPU multinomial smoothing, mixed-type inputs) live in this crate’s `TODO.md`.
