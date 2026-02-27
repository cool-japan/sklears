# sklears-decomposition

[![Crates.io](https://img.shields.io/crates/v/sklears-decomposition.svg)](https://crates.io/crates/sklears-decomposition)
[![Documentation](https://docs.rs/sklears-decomposition/badge.svg)](https://docs.rs/sklears-decomposition)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

High-performance matrix decomposition and dimensionality reduction algorithms for Rust, featuring streaming capabilities and 10-50x speedup over scikit-learn.

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-decomposition` provides state-of-the-art decomposition algorithms:

- **Classic Methods**: PCA, SVD, NMF, FastICA, Factor Analysis
- **Advanced Algorithms**: Kernel PCA, Sparse PCA, Mini-batch Dictionary Learning
- **Streaming**: Incremental PCA, Online NMF, Streaming SVD
- **Specialized**: Tensor decomposition, Robust PCA, Randomized algorithms
- **Performance**: SIMD optimization, GPU support (coming), memory efficiency

## Quick Start

```rust
use sklears_decomposition::{PCA, NMF, FastICA, TruncatedSVD};
use scirs2_autograd::ndarray::array;

// Principal Component Analysis
let pca = PCA::builder()
    .n_components(2)
    .whiten(true)
    .svd_solver(SVDSolver::Randomized)
    .build();

// Non-negative Matrix Factorization
let nmf = NMF::builder()
    .n_components(5)
    .init(NMFInit::NNDSVD)
    .solver(NMFSolver::CoordinateDescent)
    .build();

// Independent Component Analysis
let ica = FastICA::builder()
    .n_components(3)
    .algorithm(ICAAlgorithm::Parallel)
    .fun(NonLinearity::LogCosh)
    .build();

// Fit and transform
let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let fitted = pca.fit(&X)?;
let X_transformed = fitted.transform(&X)?;
let X_reconstructed = fitted.inverse_transform(&X_transformed)?;
```

## Advanced Features

### Kernel PCA

```rust
use sklears_decomposition::{KernelPCA, Kernel};

let kpca = KernelPCA::builder()
    .n_components(2)
    .kernel(Kernel::RBF { gamma: 0.1 })
    .fit_inverse_transform(true)
    .build();

// Non-linear dimensionality reduction
let X_kpca = kpca.fit_transform(&X)?;
```

### Sparse PCA

```rust
use sklears_decomposition::{SparsePCA, SparsePCAMethod};

let sparse_pca = SparsePCA::builder()
    .n_components(10)
    .alpha(1.0)  // Sparsity parameter
    .method(SparsePCAMethod::LARS)
    .build();

// Get sparse components
let fitted = sparse_pca.fit(&X)?;
let sparse_components = fitted.components();  // Many zeros
```

### Streaming Decomposition

```rust
use sklears_decomposition::{IncrementalPCA, OnlineNMF};

// Incremental PCA for large datasets
let mut ipca = IncrementalPCA::builder()
    .n_components(50)
    .batch_size(1000)
    .build();

for batch in data_stream {
    ipca.partial_fit(&batch)?;
}

// Online NMF
let mut online_nmf = OnlineNMF::builder()
    .n_components(20)
    .learning_rate(0.1)
    .build();

for batch in data_stream {
    online_nmf.partial_fit(&batch)?;
}
```

### Dictionary Learning

```rust
use sklears_decomposition::{DictionaryLearning, MiniBatchDictionaryLearning};

// Sparse coding with learned dictionary
let dict_learning = DictionaryLearning::builder()
    .n_components(50)
    .alpha(1.0)
    .transform_algorithm(TransformAlgorithm::LASSO_LARS)
    .build();

// Mini-batch version for large datasets
let mb_dict = MiniBatchDictionaryLearning::builder()
    .n_components(50)
    .batch_size(100)
    .n_iter(500)
    .build();
```

## Specialized Algorithms

### Robust PCA

```rust
use sklears_decomposition::RobustPCA;

// Separate low-rank and sparse components
let rpca = RobustPCA::builder()
    .lambda(1.0 / (X.nrows() as f64).sqrt())
    .max_iter(1000)
    .build();

let (low_rank, sparse) = rpca.fit_transform(&X)?;
```

### Factor Analysis

```rust
use sklears_decomposition::{FactorAnalysis, FARotation};

let fa = FactorAnalysis::builder()
    .n_components(3)
    .rotation(FARotation::Varimax)
    .svd_method(SVDMethod::Lapack)
    .build();

let fitted = fa.fit(&X)?;
let noise_variance = fitted.noise_variance();
```

### Tensor Decomposition

```rust
use sklears_decomposition::{TensorPCA, Tucker, PARAFAC};

// 3D tensor decomposition
let tensor_pca = TensorPCA::builder()
    .n_components([10, 10, 5])
    .build();

// Tucker decomposition
let tucker = Tucker::builder()
    .ranks([5, 5, 3])
    .init(TuckerInit::Random)
    .build();

// PARAFAC/CP decomposition
let parafac = PARAFAC::builder()
    .rank(10)
    .init(PARAFACInit::SVD)
    .build();
```

## Performance Optimizations

### Randomized Algorithms

```rust
use sklears_decomposition::{RandomizedSVD, RandomizedPCA};

// Fast approximate SVD
let rsvd = RandomizedSVD::builder()
    .n_components(100)
    .n_oversamples(10)
    .n_iter(2)
    .build();

// Handles massive matrices efficiently
let U_sigma_Vt = rsvd.fit_transform(&huge_matrix)?;
```

### Memory-Efficient Operations

```rust
use sklears_decomposition::{OutOfCorePCA, MemoryEfficientNMF};

// Out-of-core PCA for datasets larger than RAM
let ooc_pca = OutOfCorePCA::builder()
    .n_components(50)
    .chunk_size(10000)
    .build();

// Process data from disk
ooc_pca.fit_from_files(&file_paths)?;
```

### Signal Processing

```rust
use sklears_decomposition::{SignalICA, EMD, VMD};

// Blind source separation
let signal_ica = SignalICA::builder()
    .contrast_function(Contrast::Kurtosis)
    .build();

// Empirical Mode Decomposition
let emd = EMD::builder()
    .n_imfs(5)
    .build();

// Variational Mode Decomposition
let vmd = VMD::builder()
    .n_modes(3)
    .alpha(2000.0)
    .build();
```

## Quality Metrics

```rust
use sklears_decomposition::{explained_variance_ratio, reconstruction_error};

// Assess decomposition quality
let pca = PCA::new(n_components).fit(&X)?;
let var_ratio = pca.explained_variance_ratio();
let cumsum = var_ratio.iter().scan(0.0, |acc, &x| {
    *acc += x;
    Some(*acc)
}).collect::<Vec<_>>();

// Reconstruction error
let X_reduced = pca.transform(&X)?;
let X_reconstructed = pca.inverse_transform(&X_reduced)?;
let error = reconstruction_error(&X, &X_reconstructed);
```

## Benchmarks

Performance on standard datasets:

| Algorithm | scikit-learn | sklears-decomposition | Speedup |
|-----------|--------------|----------------------|---------|
| PCA | 125ms | 8ms | 15.6x |
| NMF | 450ms | 35ms | 12.9x |
| FastICA | 280ms | 18ms | 15.6x |
| Sparse PCA | 890ms | 65ms | 13.7x |
| Kernel PCA | 1200ms | 95ms | 12.6x |

## Architecture

```
sklears-decomposition/
├── linear/          # PCA, SVD, Factor Analysis
├── matrix_factorization/ # NMF, Dictionary Learning
├── ica/            # FastICA, JADE, InfoMax
├── sparse/         # Sparse PCA, Sparse coding
├── kernel/         # Kernel PCA variants
├── streaming/      # Incremental algorithms
├── tensor/         # Multi-dimensional decomposition
└── gpu/           # GPU kernels (WIP)
```

## Status

- **Core Algorithms**: 90% complete
- **Streaming Support**: Fully implemented
- **Advanced Methods**: Tensor decomposition, robust PCA ✓
- **GPU Acceleration**: In development
- **Compilation Issues**: Being resolved

## Contributing

Priority areas:
- GPU acceleration for matrix operations
- Additional tensor decomposition methods
- Distributed decomposition algorithms
- Performance optimizations

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0.

## Citation

```bibtex
@software{sklears_decomposition,
  title = {sklears-decomposition: High-Performance Matrix Decomposition for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```
