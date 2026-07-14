# sklears-isotonic

[![Crates.io](https://img.shields.io/crates/v/sklears-isotonic.svg)](https://crates.io/crates/sklears-isotonic)
[![Documentation](https://docs.rs/sklears-isotonic/badge.svg)](https://docs.rs/sklears-isotonic)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

`sklears-isotonic` delivers isotonic regression utilities that mirror scikit-learn while taking advantage of Rust performance. The crate powers monotonic calibration, pairwise ranking, and constrained curve fitting across the wider sklears ecosystem.

## Key Features

- **Isotonic Regression**: Fit monotonic functions for regression, probability calibration, and ranking tasks.
- **Sparse Solution Detection**: `SparseIsotonicRegression` and structured-sparsity solvers (group, hierarchical, total-variation) alongside the dense `ndarray`-based core algorithms.
- **Extensive Algorithm Suite**: Convex-optimization solvers (ADMM, interior point, active set, dual decomposition, QP), differential-equation-constrained fitting, Bayesian/GP-based isotonic models, and graph-constrained regression.
- **Pipeline Integration**: Seamlessly composes with `sklears` preprocessing, model selection, and calibration APIs.

## Quick Start

```rust
use sklears_isotonic::IsotonicRegression;
use sklears_core::traits::{Fit, Predict};
use scirs2_core::ndarray::{array, Array1};

let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
let y = Array1::from(vec![0.1, 0.4, 0.3, 0.8, 0.9]);

let model = IsotonicRegression::new()
    .increasing(true)
    .y_min(0.0)
    .y_max(1.0);

let fitted = model.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
```

## Status

- Validated via 345 passing tests in `0.2.0` (Stable).
- API surface aligns with scikit-learn 1.5 isotonic regression modules.
- Detailed roadmap items live in `TODO.md` within this crate.
