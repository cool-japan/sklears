# sklears-utils

[![Crates.io](https://img.shields.io/crates/v/sklears-utils.svg)](https://crates.io/crates/sklears-utils)
[![Documentation](https://docs.rs/sklears-utils/badge.svg)](https://docs.rs/sklears-utils)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-utils` hosts shared utilities used across the sklears workspace—configuration helpers, logging, dataset helpers, validation routines, and developer ergonomics.

## Key Features

- **Configuration**: Environment-aware configuration loaders, feature flag combinators, and CLI helpers.
- **Validation**: Data shape checks, feature metadata, error context propagation, and safe casting utilities.
- **Parallel Helpers**: Rayon thread-pool orchestration, chunking strategies, and work-stealing helpers.
- **Testing Support**: Fixtures, golden data loaders, and deterministic RNG wrappers for reproducible tests.

## Usage

This crate is primarily consumed internally, but developers extending sklears can depend on it for consistent utilities.

```rust
use sklears_utils::validation::check_array;
use scirs2_core::ndarray::Array2;

let x: Array2<f64> = // load data
    Array2::zeros((64, 10));

check_array(&x)?.ensure_finite()?;
```

## Status

- Validated indirectly by the entire workspace test suite (10,013 passing tests) in `0.1.0-alpha.1`.
- Acts as shared infrastructure for dozens of crates.
- Additional helpers and refactors are tracked in this crate’s `TODO.md`.
