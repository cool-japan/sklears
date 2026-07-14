# sklears-neighbors

[![Crates.io](https://img.shields.io/crates/v/sklears-neighbors.svg)](https://crates.io/crates/sklears-neighbors)
[![Documentation](https://docs.rs/sklears-neighbors/badge.svg)](https://docs.rs/sklears-neighbors)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Efficient nearest neighbor algorithms for Rust with advanced indexing structures, GPU acceleration, and production-ready tooling.

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-neighbors` provides comprehensive nearest neighbor functionality:

- **Core Algorithms**: KNN Classifier/Regressor, Radius Neighbors, Nearest Centroid
- **Tree Structures**: KD-Tree, Ball Tree, Cover Tree, VP-Tree (all implemented, selectable via `Algorithm`)
- **Approximate Methods**: LSH (`LshIndex`/`LshKNeighborsClassifier`), plus GPU-backed HNSW search behind the optional `gpu` feature (via `oxicuda-manifold`)
- **Distance Metrics**: Euclidean, Manhattan, Chebyshev, Minkowski, Cosine, Hamming, Mahalanobis, RBF/Polynomial/Sigmoid/Laplacian kernels, and custom closures
- **Advanced Features**: OxiCUDA-backed GPU distance/ANN search (optional `gpu` feature, honest CPU fallback), distributed and MapReduce-style search, incremental/streaming indices

## Status ✅

- **Implementation**: KNN, radius-neighbors, tree-based (KD/Ball/Cover/VP), LSH, and GPU-accelerated search, alongside a broad set of specialized modules (outlier detection, metric learning, bioinformatics, computer vision, NLP, time-series, graph methods).
- **Validation**: Covered by 405 passing crate tests (2 skipped) from the 0.2.0 release.
- **Performance**: SIMD-accelerated distance kernels (`simd_distance`), multi-threaded queries (`parallel` feature via Rayon), and optional OxiCUDA GPU offload (`gpu` feature).

## Quick Start

```rust
use sklears_neighbors::{Algorithm, KNeighborsClassifier, KNeighborsRegressor, WeightStrategy};
use sklears_core::traits::{Fit, Predict};
use scirs2_core::ndarray::array;

// K-Nearest Neighbors Classifier
let knn = KNeighborsClassifier::new(5)
    .with_weights(WeightStrategy::Distance)
    .with_algorithm(Algorithm::Brute);

// Train and predict
let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let y = array![0, 1, 0];
let fitted = knn.fit(&X, &y)?;
let predictions = fitted.predict(&X_test)?;

// K-Nearest Neighbors Regressor
let knn_reg = KNeighborsRegressor::new(3)
    .with_weights(WeightStrategy::Uniform);
```

## Advanced Features

### Tree-Based Algorithms

```rust
use sklears_neighbors::{Algorithm, KNeighborsClassifier};

// KD-Tree for low-dimensional data
let knn_kd = KNeighborsClassifier::new(5).with_algorithm(Algorithm::KdTree);

// Ball Tree for high-dimensional data
let knn_ball = KNeighborsClassifier::new(5).with_algorithm(Algorithm::BallTree);

// Cover Tree with theoretical search guarantees
let knn_cover = KNeighborsClassifier::new(5).with_algorithm(Algorithm::CoverTree);

// Vantage-Point Tree for general metric spaces
let knn_vp = KNeighborsClassifier::new(5).with_algorithm(Algorithm::VpTree);
```

### Radius Neighbors

```rust
use sklears_neighbors::{RadiusNeighborsClassifier, WeightStrategy};

// Fixed radius search, with a fallback label for points with no neighbors
let radius_nn = RadiusNeighborsClassifier::new(1.0)
    .with_weights(WeightStrategy::Distance)
    .with_outlier_label(-1);
```

### Approximate Nearest Neighbors

```rust
use sklears_neighbors::LshKNeighborsClassifier;

// LSH with random projection (cosine similarity): k, num_tables, num_hashes, dimension
let lsh = LshKNeighborsClassifier::new_random_projection(5, 10, 8, 128);

// LSH with MinHash (Jaccard similarity): k, num_tables, num_hashes
let lsh_minhash = LshKNeighborsClassifier::new_minhash(5, 10, 8);
```

### Custom Distance Metrics

```rust
use sklears_neighbors::{Distance, KNeighborsClassifier};

// Define a custom distance via the `Distance::Custom` closure variant
let custom_metric = Distance::Custom(Box::new(|a, b| {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(3.0))
        .sum::<f64>()
        .powf(1.0 / 3.0)
}));

let knn = KNeighborsClassifier::new(5).with_metric(custom_metric);
```

### GPU Acceleration (`gpu` feature)

```rust
use sklears_neighbors::{GpuConfig, GpuDistanceCalculator, GpuKNeighborsSearch};

// OxiCUDA-backed pairwise distances; honestly falls back to CPU compute
// when the `gpu` feature is off or no device is detected.
let calculator = GpuDistanceCalculator::with_config(GpuConfig::default());

// GPU k-nearest neighbors: `kneighbors()` prefers an HNSW approximate index
// when `with_ann(true)` and the metric is Euclidean/Cosine, transparently
// falling back to the exact brute-force path otherwise (or without the `gpu`
// feature). `kneighbors_exact()` always takes the exact path directly.
let gpu_knn = GpuKNeighborsSearch::new(10, GpuConfig::default()).with_ann(true);
```

### Streaming and Dynamic Updates

```rust
use sklears_neighbors::{IncrementalKNeighborsClassifier, MemoryStrategy};

// Incremental KNN classifier: k neighbors, capped at max_samples in memory
let mut streaming_knn = IncrementalKNeighborsClassifier::new(5, 10_000)
    .with_memory_strategy(MemoryStrategy::KeepRecent);
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
├── knn.rs, radius_neighbors.rs    # Core KNN / radius-neighbor estimators
├── tree/                          # KD-Tree, Ball Tree, Cover Tree, VP-Tree
├── lsh.rs                         # LSH approximate search
├── gpu_distance.rs                # OxiCUDA GPU distances + HNSW ANN (feature `gpu`)
├── distance.rs, simd_distance.rs  # Distance metrics + SIMD kernels
├── distributed_neighbors.rs, mapreduce_neighbors.rs, federated_neighbors.rs
└── ...                            # outlier detection, metric learning, bioinformatics,
                                    # computer vision, NLP, time-series, streaming, validation
```

## Roadmap

### Delivered in 0.2.0
- [x] KD-Tree, Ball Tree, Cover Tree, and VP-Tree implementations
- [x] Radius neighbors (fixed and adaptive)
- [x] LSH approximate search
- [x] OxiCUDA-backed GPU distance computation and HNSW-based ANN search (`gpu` feature)

### Future
- [ ] Extend ANN benchmarks to cover billion-scale datasets
- [ ] Document distributed query patterns with end-to-end examples
- [ ] Stabilize the custom metric plugin API
- [ ] Standalone Annoy-style random-projection-forest ANN index

## Contributing

This crate needs your help! Priority contributions:
- Additional distance metric optimizations
- Billion-scale ANN benchmarking
- Distributed/federated search end-to-end examples
- Documentation and examples

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Current Limitations

- Approximate search covers LSH plus GPU-backed HNSW (behind the `gpu` feature); there is no standalone Annoy-style index yet.
- Distributed and federated search modules are implemented but not yet benchmarked at very large scale.
- GPU support (`gpu` feature) targets OxiCUDA (CUDA) only; there is no OpenCL/Metal backend.

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
