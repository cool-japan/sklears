# sklears-datasets

[![Crates.io](https://img.shields.io/crates/v/sklears-datasets.svg)](https://crates.io/crates/sklears-datasets)
[![Documentation](https://docs.rs/sklears-datasets/badge.svg)](https://docs.rs/sklears-datasets)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-datasets` centralizes synthetic dataset generators and data-quality utilities used throughout the sklears ecosystem. The crate's currently-compiled public API (`src/lib.rs`) is a focused generator/validation surface, not the full scikit-learn-style loader catalogue described in older drafts of this document — see [Status](#status) for what is actually reachable today.

## Key Features

- **Synthetic Generators**: Core scikit-learn-style generators (`make_blobs`, `make_circles`, `make_moons`, `make_classification`, `make_regression`) plus a much larger set of specialized generators under `sklears_datasets::generators::*`: manifold shapes (Swiss roll, S-curve, severed sphere, helix), time series (AR/MA/ARMA processes, seasonal trend, random walk), spatial statistics (point processes, geostatistical and geographic data), causal inference (treatment effects, instrumental variables, confounded regression), adversarial robustness (label noise, outlier contamination, covariate shift), domain-specific data (financial returns, sensor streams, survival data), and multimodal/multi-agent simulations.
- **Type-Safe Generation**: `generators::type_safe` provides a const-generic `TypeSafeDataset<T, N_SAMPLES, N_FEATURES>` API for compile-time-checked dataset shapes.
- **Streaming & Parallel Generation**: `DatasetStream`, `parallel_generate`, and `stream_classification`/`stream_regression`/`stream_blobs` helpers (`generators::performance`) for large-scale or out-of-core generation; `parallel_rng::ParallelRng` and `make_*_parallel` helpers for multi-threaded generation.
- **Validation & Quality**: `validation` module covers statistical validation (normality, correlation structure, outlier detection), distribution goodness-of-fit tests (KS, chi-square), dataset quality metrics, and anomaly/drift detection.
- **Versioning & Visualization**: `DatasetVersion`/`ProvenanceInfo` (`versioning`) for dataset provenance and checksums; optional `visualization`-feature-gated 2D plotting (`viz::plot_2d_classification`, etc.) via `plotters`.
- **Note**: a large set of additional modules exist in `src/` (classic-dataset loaders such as `load_iris`/`load_wine`, CSV/Parquet/Arrow IO, benchmarking utilities, streaming file formats) but are **not currently wired into `lib.rs`** and are not part of the compiled public API — see `TODO.md`.

## Quick Start

```rust
use sklears_datasets::{make_blobs, make_classification, make_regression};

// make_blobs(n_samples, n_features, centers, cluster_std, random_state)
let (features, targets) = make_blobs(1000, 10, 4, 2.5, Some(42))?;

// make_classification(n_samples, n_features, n_informative, n_redundant, n_classes, random_state)
let (x_class, y_class) = make_classification(150, 4, 2, 1, 3, Some(42))?;

// make_regression(n_samples, n_features, n_informative, noise, random_state)
let (x_reg, y_reg) = make_regression(200, 5, 3, 0.1, Some(42))?;
```

## Status

- All generators/validation utilities validated through 190 passing tests in `0.2.0` (`cargo nextest run -p sklears-datasets --all-features`) (Partial — see the note above about disabled loader/IO modules).
- Supports streaming and parallel generation for large-scale workflows.
- Future work (re-wiring the disabled classic-loader/file-IO modules, federated dataset shards, synthetic time series) tracked in this crate’s `TODO.md`.
