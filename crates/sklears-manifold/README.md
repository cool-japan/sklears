# sklears-manifold

[![Crates.io](https://img.shields.io/crates/v/sklears-manifold.svg)](https://crates.io/crates/sklears-manifold)
[![Documentation](https://docs.rs/sklears-manifold/badge.svg)](https://docs.rs/sklears-manifold)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-manifold` implements manifold learning, nonlinear dimensionality reduction, and embedding algorithms mirroring scikit-learn’s manifold module.

## Key Features

- **Algorithms**: t-SNE, UMAP, Isomap, Locally Linear Embedding, Spectral Embedding, MDS.
- **Performance**: Barnes-Hut t-SNE, GPU-accelerated pairwise distances and kNN search (`gpu` feature, via `oxicuda-manifold`), and parallel kNN search (rayon).
- **Visualization**: Embedding utilities that integrate with `sklears-inspection` and Python plotting stacks.
- **Pipeline Support**: Works seamlessly with preprocessing, decomposition, and clustering crates.

## Quick Start

```rust
use sklears_manifold::TSNE;
use sklears_core::traits::{Fit, Transform};
use scirs2_core::ndarray::array;

let x = array![
    [0.0, 0.0], [1.0, 1.0], [2.0, 2.0],
    [10.0, 10.0], [11.0, 11.0], [12.0, 12.0],
];

let tsne = TSNE::new()
    .n_components(2)
    .perplexity(2.0)
    .n_iter(100);

let fitted = tsne.fit(&x.view(), &())?;
let embedding = fitted.embedding();
```

## Status

- Validated by 422 passing crate tests for `0.2.0` (2 skipped).
- Performance parity (and in many cases superiority) compared with scikit-learn’s manifold implementations.
- Upcoming tasks (GPU UMAP, streaming embeddings) tracked in `TODO.md`.
