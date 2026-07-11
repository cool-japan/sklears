# sklears-utils

[![Crates.io](https://img.shields.io/crates/v/sklears-utils.svg)](https://crates.io/crates/sklears-utils)
[![Documentation](https://docs.rs/sklears-utils/badge.svg)](https://docs.rs/sklears-utils)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-utils` hosts shared utilities used across the sklears workspace—configuration helpers, logging, dataset helpers, validation routines, and developer ergonomics.

## Key Features

- **Configuration**: Environment-aware configuration loading (`Config::load_environment`), hot-reloadable config files, and validation.
- **Validation**: Data shape checks, feature metadata, error context propagation, and safe casting utilities.
- **Parallel Helpers**: Custom `ThreadPool`/`WorkStealingQueue` primitives plus chunked `ParallelIterator`/`ParallelReducer` helpers.
- **Testing Support**: Synthetic dataset generators (`make_classification`, `make_regression`, `make_blobs`, ...) and deterministic RNG wrappers (seeded `get_rng`) for reproducible tests.

## Usage

This crate is primarily consumed internally, but developers extending sklears can depend on it for consistent utilities.

```rust
use sklears_utils::validation::check_array_2d;
use scirs2_core::ndarray::Array2;

let x: Array2<f64> = // load data
    Array2::zeros((64, 10));

check_array_2d(&x)?;
```

## Status

- Validated by 494 passing tests in `0.2.0` (Stable).
- Acts as shared infrastructure for dozens of crates.
- Additional helpers and refactors are tracked in this crate’s `TODO.md`.
