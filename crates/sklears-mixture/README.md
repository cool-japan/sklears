# sklears-mixture

[![Crates.io](https://img.shields.io/crates/v/sklears-mixture.svg)](https://crates.io/crates/sklears-mixture)
[![Documentation](https://docs.rs/sklears-mixture/badge.svg)](https://docs.rs/sklears-mixture)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-mixture` implements Gaussian Mixture Models, Bayesian mixtures, Dirichlet process mixtures, and clustering utilities consistent with scikit-learn’s mixture module.

## Key Features

- **Algorithms**: GaussianMixture, BayesianGaussianMixture (variational), Dirichlet-process / nonparametric mixtures, and full/diagonal/tied/spherical covariance options.
- **Inference**: Expectation-Maximization, variational inference (mean-field, stochastic, structured), NUTS/ADVI, and online/streaming updates.
- **Time Series**: Hidden Markov Models, regime-switching models, switching state-space models, and temporal/dynamic Gaussian mixtures.
- **Integration**: Compatible with preprocessing, model selection, and inspection crates for pipeline workflows.

## Quick Start

```rust
use sklears_mixture::{GaussianMixture, CovarianceType};
use sklears_core::traits::{Fit, Predict};
use scirs2_core::ndarray::Array2;

let x: Array2<f64> = Array2::zeros((1000, 5));

let gmm = GaussianMixture::builder()
    .n_components(4)
    .covariance_type(CovarianceType::Full)
    .max_iter(200)
    .tol(1e-3)
    .random_state(42);

let fitted = gmm.fit(&x.view(), &())?;
let labels = fitted.predict(&x.view())?;
```

## Status

- 238 crate tests pass for `0.2.0`, 0 skipped.
- Planned features (GPU-accelerated inference, streaming DPGMM) tracked in `TODO.md`.
