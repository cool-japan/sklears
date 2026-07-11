# sklears-neural

[![Crates.io](https://img.shields.io/crates/v/sklears-neural.svg)](https://crates.io/crates/sklears-neural)
[![Documentation](https://docs.rs/sklears-neural/badge.svg)](https://docs.rs/sklears-neural)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-neural` delivers multilayer perceptrons and neural utility blocks that align with scikit-learn’s neural-network module while embracing Rust’s performance story.

## Key Features

- **Models**: MLPClassifier, MLPRegressor, RBMs, autoencoders (including self-supervised/contrastive variants).
- **Optimizers**: SGD, Adam, AdamW, Nadam, RMSprop, L-BFGS, and adaptive learning-rate schedules.
- **Hardware Acceleration**: SIMD kernels and CUDA execution (oxicuda-backed `gpu` feature), with mixed-precision training support.
- **Integration**: Works with sklears pipelines, calibration, inspection, and export utilities.

## Quick Start

```rust
use sklears_neural::{Activation, MLPClassifier, Solver};
use scirs2_core::ndarray::array;

let x = array![
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
];
let y: Vec<usize> = vec![0, 1, 1, 0];

let mlp = MLPClassifier::new()
    .hidden_layer_sizes(&[16, 16])
    .activation(Activation::Relu)
    .solver(Solver::Adam)
    .max_iter(1000)
    .random_state(42);

let fitted = mlp.fit(&x, &y)?;
let probs = fitted.predict_proba(&x)?;
```

## Status

- Exercised via 449 passing crate tests in `0.2.0` (86 skipped).
- Verified against scikit-learn parity tests for convergence and scoring APIs.
- Roadmap items (ONNX export, distillation helpers) documented in this crate’s `TODO.md`.
