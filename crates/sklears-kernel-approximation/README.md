# sklears-kernel-approximation

[![Crates.io](https://img.shields.io/crates/v/sklears-kernel-approximation.svg)](https://crates.io/crates/sklears-kernel-approximation)
[![Documentation](https://docs.rs/sklears-kernel-approximation/badge.svg)](https://docs.rs/sklears-kernel-approximation)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.2` (December 22, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.2.md) for highlights and upgrade guidance.

## Overview

`sklears-kernel-approximation` houses fast kernel feature map transformers, enabling scalable kernel methods for large datasets. The implementations track the scikit-learn 1.5 API while exploiting Rust's parallelism and SIMD acceleration.

## Key Features

- **Random Feature Maps**: RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler, and more.
- **GPU Acceleration**: Optional CUDA/WebGPU backends for massive random feature expansions.
- **Pipeline Ready**: Builders integrate with `sklears` pipelines, grid search, and calibration stages.
- **Deterministic Testing**: Extensive property-based and integration tests ensure reproducible embeddings.

## Quick Start

```rust
use sklears_kernel_approximation::RBFSampler;
use scirs2_core::ndarray::Array2;

let features: Array2<f64> = // load your data
    Array2::zeros((1024, 32));

let transformer = RBFSampler::builder()
    .gamma(0.5)
    .n_components(4096)
    .random_state(Some(42))
    .build();

let mapped = transformer.fit_transform(&features)?;
```

## Status

- Verified by the workspace-wide 11,292 passing tests in `0.1.0-alpha.2`.
- Benchmarked against scikit-learn to provide 10–30× faster random feature generation.
- Further roadmap tasks (e.g., online updates, streaming sampling) tracked in `TODO.md`.
