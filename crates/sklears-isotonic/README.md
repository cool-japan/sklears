# sklears-isotonic

[![Crates.io](https://img.shields.io/crates/v/sklears-isotonic.svg)](https://crates.io/crates/sklears-isotonic)
[![Documentation](https://docs.rs/sklears-isotonic/badge.svg)](https://docs.rs/sklears-isotonic)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-isotonic` delivers isotonic regression utilities that mirror scikit-learn while taking advantage of Rust performance. The crate powers monotonic calibration, pairwise ranking, and constrained curve fitting across the wider sklears ecosystem.

## Key Features

- **Isotonic Regression**: Fit monotonic functions for regression, probability calibration, and ranking tasks.
- **Sparse + Dense Support**: Optimized for both dense `ndarray` inputs and sparse CSR matrices.
- **GPU-Ready Kernels**: Optional CUDA/WebGPU acceleration for large calibration workloads.
- **Pipeline Integration**: Seamlessly composes with `sklears` preprocessing, model selection, and calibration APIs.

## Quick Start

```rust
use sklears_isotonic::IsotonicRegression;
use scirs2_core::ndarray::{array, Array1};

let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
let y = Array1::from(vec![0.1, 0.4, 0.3, 0.8, 0.9]);

let model = IsotonicRegression::builder()
    .increasing(true)
    .y_min(0.0)
    .y_max(1.0)
    .build();

let fitted = model.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;
```

## Status

- Validated via the 10,013-passing workspace test suite included with `0.1.0-beta.1`.
- API surface aligns with scikit-learn 1.5 isotonic regression modules.
- Detailed roadmap items live in `TODO.md` within this crate.
