# sklears-clustering

[![Crates.io](https://img.shields.io/crates/v/sklears-clustering.svg)](https://crates.io/crates/sklears-clustering)
[![Documentation](https://docs.rs/sklears-clustering/badge.svg)](https://docs.rs/sklears-clustering)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Clustering algorithms for the sklears machine learning library.

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

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
sklears = { version = "0.2.0", features = ["clustering"] }
```

## Examples

### K-Means Clustering

```rust
use sklears::cluster::KMeans;
use sklears::cluster::InitMethod;

let model = KMeans::new(3)
    .init_method(InitMethod::KMeansPlusPlus)
    .max_iter(300)
    .n_init(10)
    .random_state(42);

let fitted = model.fit(&data)?;
let labels = fitted.predict(&new_data)?;
let centers = fitted.cluster_centers();
```

### DBSCAN

```rust
use sklears::cluster::DBSCAN;

let model = DBSCAN::new()
    .eps(0.5)
    .min_samples(5)
    .metric(Distance::Euclidean);

let labels = model.fit_predict(&data)?;
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

## License

Licensed under the Apache License, Version 2.0.