# sklears-neural

[![Crates.io](https://img.shields.io/crates/v/sklears-neural.svg)](https://crates.io/crates/sklears-neural)
[![Documentation](https://docs.rs/sklears-neural/badge.svg)](https://docs.rs/sklears-neural)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.2` (December 22, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.2.md) for highlights and upgrade guidance.

## Overview

`sklears-neural` delivers multilayer perceptrons and neural utility blocks that align with scikit-learn’s neural-network module while embracing Rust’s performance story.

## Key Features

- **Models**: MLPClassifier, MLPRegressor, RBMs, autoencoders, and incremental learning variants.
- **Optimizers**: SGD, Adam, LBFGS, RMSProp, and adaptive learning-rate schedules.
- **Hardware Acceleration**: SIMD kernels, CUDA/WebGPU execution, and mixed-precision training.
- **Integration**: Works with sklears pipelines, calibration, inspection, and export utilities.

## Quick Start

```rust
use sklears_neural::MLPClassifier;
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
];
let y = Array1::from(vec![0, 1, 1, 0]);

let mlp = MLPClassifier::builder()
    .hidden_layer_sizes(vec![16, 16])
    .activation("relu")
    .solver("adam")
    .max_iter(1000)
    .random_state(Some(42))
    .build();

let fitted = mlp.fit(&x, &y)?;
let probs = fitted.predict_proba(&x)?;
```

## Status

- Exercised via the 11,292 passing workspace tests in `0.1.0-alpha.2`.
- Verified against scikit-learn parity tests for convergence and scoring APIs.
- Roadmap items (ONNX export, distillation helpers) documented in this crate’s `TODO.md`.
