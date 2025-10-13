# sklears-manifold

[![Crates.io](https://img.shields.io/crates/v/sklears-manifold.svg)](https://crates.io/crates/sklears-manifold)
[![Documentation](https://docs.rs/sklears-manifold/badge.svg)](https://docs.rs/sklears-manifold)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Overview

`sklears-manifold` implements manifold learning, nonlinear dimensionality reduction, and embedding algorithms mirroring scikit-learn’s manifold module.

## Key Features

- **Algorithms**: t-SNE, UMAP-compatible neighbors, Isomap, Locally Linear Embedding, Spectral Embedding, MDS.
- **Performance**: Barnes-Hut and FFT-based t-SNE, GPU nearest neighbors, and multithreaded eigen solvers.
- **Visualization**: Embedding utilities that integrate with `sklears-inspection` and Python plotting stacks.
- **Pipeline Support**: Works seamlessly with preprocessing, decomposition, and clustering crates.

## Quick Start

```rust
use sklears_manifold::TSNE;
use scirs2_core::ndarray::Array2;

let x: Array2<f32> = // load dataset
    Array2::zeros((2000, 128));

let tsne = TSNE::builder()
    .n_components(2)
    .perplexity(30.0)
    .learning_rate(200.0)
    .n_iter(1000)
    .build();

let embedding = tsne.fit_transform(&x)?;
```

## Status

- Validated by the workspace’s 10,013 passing tests for `0.1.0-alpha.1`.
- Performance parity (and in many cases superiority) compared with scikit-learn’s manifold implementations.
- Upcoming tasks (GPU UMAP, streaming embeddings) tracked in `TODO.md`.
