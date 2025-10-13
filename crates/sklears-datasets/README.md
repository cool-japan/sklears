# sklears-datasets

[![Crates.io](https://img.shields.io/crates/v/sklears-datasets.svg)](https://crates.io/crates/sklears-datasets)
[![Documentation](https://docs.rs/sklears-datasets/badge.svg)](https://docs.rs/sklears-datasets)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-datasets` centralizes dataset loaders, synthetic generators, and data utilities used throughout the sklears ecosystem. It mirrors scikit-learn’s dataset module while adding Rust-first performance and IO enhancements.

## Key Features

- **Classic Loaders**: Diabetes, Iris, Digits, Wine, Breast Cancer, 20 Newsgroups, and more.
- **Synthetic Generators**: `make_blobs`, `make_moons`, `make_circles`, Gaussian quantiles, regression surfaces, and streaming generators.
- **File IO**: CSV, Parquet, Arrow IPC, and memory-mapped dataset support with Polars integration.
- **Benchmark Utilities**: Deterministic dataset splits and sampling strategies for reproducible experiments.

## Quick Start

```rust
use sklears_datasets::{load_iris, make_blobs};

// Built-in dataset
let iris = load_iris()?;
println!("{} samples, {} features", iris.data.nrows(), iris.data.ncols());

// Synthetic data
let blobs = make_blobs(1000)
    .n_features(10)
    .centers(4)
    .cluster_std(2.5)
    .random_state(Some(42))
    .build()?;
```

## Status

- All loaders/generators validated through the 10,013 passing workspace tests for `0.1.0-alpha.1`.
- Supports lazy loading and streaming for large-scale workflows.
- Future work (federated dataset shards, synthetic time series) tracked in this crate’s `TODO.md`.
