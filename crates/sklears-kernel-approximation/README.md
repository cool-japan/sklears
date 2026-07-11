# sklears-kernel-approximation

[![Crates.io](https://img.shields.io/crates/v/sklears-kernel-approximation.svg)](https://crates.io/crates/sklears-kernel-approximation)
[![Documentation](https://docs.rs/sklears-kernel-approximation/badge.svg)](https://docs.rs/sklears-kernel-approximation)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-kernel-approximation` houses fast kernel feature map transformers, enabling scalable kernel methods for large datasets. The implementations track the scikit-learn 1.5 API while exploiting Rust's parallelism and SIMD acceleration.

## Key Features

- **Random Feature Maps**: RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler, and more.
- **GPU Acceleration**: Optional CUDA backend (`gpu` feature, via `sklears-core`'s oxicuda-backed `gpu` module) for on-device GEMM in `GpuNystroem`/`GpuRBFSampler`.
- **Pipeline Ready**: Estimators integrate with `sklears` pipelines, grid search, and calibration stages.
- **Deterministic Testing**: Extensive property-based and integration tests ensure reproducible embeddings.

## Quick Start

```rust
use sklears_kernel_approximation::RBFSampler;
use sklears_core::traits::{Fit, Transform};
use scirs2_core::ndarray::Array2;

let features: Array2<f64> = Array2::zeros((1024, 32));

let rbf = RBFSampler::new(4096)
    .gamma(0.5)
    .random_state(42);

let fitted = rbf.fit(&features, &())?;
let mapped = fitted.transform(&features)?;
```

## Status

- Verified by 532 passing tests in `0.2.0` (Stable).
- Benchmarked against scikit-learn to provide 10–30× faster random feature generation.
- Further roadmap tasks (e.g., online updates, streaming sampling) tracked in `TODO.md`.
