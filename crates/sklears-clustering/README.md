# sklears-clustering

[![Crates.io](https://img.shields.io/crates/v/sklears-clustering.svg)](https://crates.io/crates/sklears-clustering)
[![Documentation](https://docs.rs/sklears-clustering/badge.svg)](https://docs.rs/sklears-clustering)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Clustering algorithms for the sklears machine learning library.

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

This crate provides implementations of clustering algorithms including:

- **K-Means**: Classic centroid-based clustering with k-means++ initialization
- **Mini-Batch K-Means**: Scalable variant for large datasets
- **DBSCAN**: Density-based clustering for arbitrary shaped clusters
- **Hierarchical Clustering**: Agglomerative clustering with various linkage methods
- **Mean Shift**: Mode-seeking clustering algorithm

## Usage

```toml
[dependencies]
sklears-clustering = "0.2.1"
```

## Examples

### K-Means Clustering

```rust
use sklears_clustering::{KMeans, KMeansConfig, KMeansInit};
use sklears_core::traits::{Fit, Predict};
use scirs2_core::ndarray::Array1;

let config = KMeansConfig {
    n_clusters: 3,
    init: KMeansInit::KMeansPlusPlus,
    max_iter: 300,
    random_seed: Some(42),
    ..Default::default()
};
let model = KMeans::new(config);

// The `y` argument is currently unused by KMeans::fit (unsupervised algorithm)
let y_dummy = Array1::zeros(data.nrows());
let fitted = model.fit(&data, &y_dummy)?;

let labels = &fitted.labels;      // training-set cluster assignments
let centers = &fitted.centroids;  // Array2<f64> of cluster centroids
let new_labels = fitted.predict(&new_data)?;
```

### DBSCAN

```rust
use sklears_clustering::DBSCAN;
use sklears_core::traits::Fit;

let model = DBSCAN::new().eps(0.5).min_samples(5);

let fitted = model.fit(&data, &())?;
let labels = fitted.labels();
// -1 indicates noise points
```

## Performance Features

- **SIMD Distance Calculations**: Vectorized distance computations
- **Parallel Assignment**: Multi-threaded cluster assignment
- **Efficient K-D Trees**: For neighbor searches in DBSCAN
- **Memory-Efficient Updates**: In-place operations where possible

### GPU-Accelerated Distances

Enable the optional `gpu` feature to accelerate distance kernels with OxiCUDA (Pure Rust CUDA) via `sklears_core::gpu`. On hosts without a CUDA-capable GPU, device detection honestly returns "no GPU" and every operation transparently falls back to the CPU implementation:

```bash
cargo test -p sklears-clustering --features gpu gpu_distances::gpu::tests::test_euclidean_distances
cargo test -p sklears-clustering --features gpu gpu_distances::gpu::tests::test_manhattan_distances
```

## Metrics

The crate includes clustering metrics:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Inertia

## Status

- **Tests**: 417 passing crate tests for `0.2.0` (7 skipped)

## License

Licensed under the Apache License, Version 2.0.