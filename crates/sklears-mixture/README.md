# sklears-mixture

[![Crates.io](https://img.shields.io/crates/v/sklears-mixture.svg)](https://crates.io/crates/sklears-mixture)
[![Documentation](https://docs.rs/sklears-mixture/badge.svg)](https://docs.rs/sklears-mixture)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-mixture` implements Gaussian Mixture Models, Bayesian mixtures, Dirichlet process mixtures, and clustering utilities consistent with scikit-learn’s mixture module.

## Key Features

- **Algorithms**: GaussianMixture, BayesianGaussianMixture, DirichletProcessGaussianMixture, and spherical/covariance options.
- **Inference**: Expectation-Maximization, variational inference, and online updates for streaming data.
- **Accelerated Kernels**: SIMD and GPU-accelerated responsibilities, log-likelihood evaluation, and sampling.
- **Integration**: Compatible with preprocessing, model selection, and inspection crates for pipeline workflows.

## Quick Start

```rust
use sklears_mixture::GaussianMixture;
use scirs2_core::ndarray::Array2;

let x: Array2<f64> = // load or generate data
    Array2::zeros((1000, 5));

let gmm = GaussianMixture::builder()
    .n_components(4)
    .covariance_type("full")
    .max_iter(200)
    .tol(1e-3)
    .random_state(Some(42))
    .build();

let fitted = gmm.fit(&x)?;
let labels = fitted.predict(&x)?;
```

## Status

- Fully covered by the 10,013 passing workspace tests for `0.1.0-alpha.1`.
- Achieves 5–15× speedups over scikit-learn on medium-sized datasets.
- Planned features (GPU variational inference, streaming DPGMM) tracked in `TODO.md`.
