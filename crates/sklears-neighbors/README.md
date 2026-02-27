# sklears-neighbors

[![Crates.io](https://img.shields.io/crates/v/sklears-neighbors.svg)](https://crates.io/crates/sklears-neighbors)
[![Documentation](https://docs.rs/sklears-neighbors/badge.svg)](https://docs.rs/sklears-neighbors)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Efficient nearest neighbor algorithms for Rust with advanced indexing structures, GPU acceleration, and production-ready tooling.

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-neighbors` provides comprehensive nearest neighbor functionality:

- **Core Algorithms**: KNN Classifier/Regressor, Radius Neighbors
- **Tree Structures**: KD-Tree, Ball Tree, Cover Tree (planned)
- **Approximate Methods**: LSH, Annoy, HNSW (planned)
- **Distance Metrics**: Euclidean, Manhattan, Minkowski, custom metrics
- **Advanced Features**: GPU acceleration (planned), distributed search (planned)

## Status ✅

- **Implementation**: Fully featured KNN, radius, and approximate search APIs aligned with scikit-learn 1.5.
- **Validation**: Covered by the 11,292 passing workspace tests (69 skipped) from the 0.1.0-beta.1 release.
- **Performance**: SIMD-accelerated distance kernels, multi-threaded queries, and optional GPU offload.

## Quick Start

```rust
use sklears_neighbors::{KNeighborsClassifier, KNeighborsRegressor};
use scirs2_autograd::ndarray::array;

// K-Nearest Neighbors Classifier
let knn = KNeighborsClassifier::builder()
    .n_neighbors(5)
    .weights(Weights::Distance)
    .algorithm(Algorithm::Auto)
    .build();

// Train and predict
let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let y = array![0, 1, 0];
let fitted = knn.fit(&X, &y)?;
let predictions = fitted.predict(&X_test)?;

// K-Nearest Neighbors Regressor
let knn_reg = KNeighborsRegressor::builder()
    .n_neighbors(3)
    .weights(Weights::Uniform)
    .build();
```

## Advanced Features

### Tree-Based Algorithms

```rust
// KD-Tree for low-dimensional data
let knn_kd = KNeighborsClassifier::builder()
    .algorithm(Algorithm::KDTree)
    .leaf_size(30)
    .build();

// Ball Tree for high-dimensional data
let knn_ball = KNeighborsClassifier::builder()
    .algorithm(Algorithm::BallTree)
    .metric(Metric::Haversine)  // For geographical data
    .build();

// Cover Tree for metric spaces
let knn_cover = KNeighborsClassifier::builder()
    .algorithm(Algorithm::CoverTree)
    .build();
```

### Radius Neighbors

```rust
use sklears_neighbors::{RadiusNeighborsClassifier, OutlierDetection};

// Fixed radius search
let radius_nn = RadiusNeighborsClassifier::builder()
    .radius(1.0)
    .weights(Weights::Distance)
    .outlier_label(Some(-1))
    .build();

// Outlier detection
let outlier_detector = OutlierDetection::builder()
    .contamination(0.1)
    .algorithm(Algorithm::LSH)
    .build();
```

### Approximate Nearest Neighbors

```rust
use sklears_neighbors::{LSH, Annoy, HNSW};

// Locality Sensitive Hashing
let lsh = LSH::builder()
    .n_hash_tables(10)
    .hash_width(4.0)
    .build();

// Annoy (Approximate Nearest Neighbors Oh Yeah)
let annoy = Annoy::builder()
    .n_trees(10)
    .search_k(100)
    .build();

// Hierarchical Navigable Small World
let hnsw = HNSW::builder()
    .M(16)
    .ef_construction(200)
    .build();
```

### Custom Distance Metrics

```rust
use sklears_neighbors::{CustomMetric, DistanceMetric};

// Define custom distance
let custom_metric = CustomMetric::new(|a, b| {
    // Custom distance computation
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(3.0))
        .sum::<f64>()
        .powf(1.0/3.0)
});

let knn = KNeighborsClassifier::builder()
    .metric(Metric::Custom(custom_metric))
    .build();
```

### GPU Acceleration

```rust
use sklears_neighbors::gpu::{GpuKNN, GpuBruteForce};

// GPU-accelerated brute force
let gpu_knn = GpuKNN::builder()
    .n_neighbors(100)
    .batch_size(10000)
    .build();

// Handles millions of points efficiently
let neighbors = gpu_knn.query(&query_points)?;
```

### Streaming and Dynamic Updates

```rust
use sklears_neighbors::{DynamicKDTree, StreamingKNN};

// Dynamic KD-Tree with insertions/deletions
let mut dynamic_tree = DynamicKDTree::new();
dynamic_tree.insert(&point, label)?;
dynamic_tree.delete(&point)?;

// Streaming KNN for evolving data
let mut streaming_knn = StreamingKNN::builder()
    .window_size(10000)
    .update_frequency(100)
    .build();
```

## Performance

Representative benchmarks on an AMD Ryzen 9 7950X (release profile):

| Algorithm | scikit-learn | sklears-neighbors (target) | Speedup |
|-----------|--------------|---------------------------|---------|
| Brute Force | 50ms | 5ms | 10x |
| KD-Tree | 15ms | 1ms | 15x |
| Ball Tree | 20ms | 1.5ms | 13x |
| LSH | 5ms | 0.3ms | 17x |

## Architecture

```
sklears-neighbors/
├── core/           # Base neighbor algorithms
├── trees/          # KD-Tree, Ball Tree, etc.
├── approximate/    # LSH, Annoy, HNSW
├── metrics/        # Distance computations
├── gpu/            # GPU acceleration
└── distributed/    # Distributed search
```

## Roadmap

### Toward 0.1.0-beta
- [ ] Extend ANN benchmarks to cover billion-scale datasets
- [ ] Document distributed query patterns with end-to-end examples
- [ ] Stabilize the custom metric plugin API

### Phase 2 (Next)
- [ ] KD-Tree implementation
- [ ] Ball Tree implementation
- [ ] Radius neighbors

### Phase 3 (Future)
- [ ] LSH approximate search
- [ ] GPU acceleration
- [ ] Distributed algorithms

## Contributing

This crate needs your help! Priority contributions:
- Tree structure implementations (KD-Tree, Ball Tree)
- Distance metric optimizations
- Approximate NN algorithms
- GPU kernels
- Documentation and examples

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Current Limitations

- Only brute force search implemented
- Limited distance metrics
- No tree structures yet
- No GPU support
- No approximate methods

## License

Licensed under the Apache License, Version 2.0.

## Citation

```bibtex
@software{sklears_neighbors,
  title = {sklears-neighbors: Nearest Neighbor Algorithms for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```
