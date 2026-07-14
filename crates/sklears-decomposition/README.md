# sklears-decomposition

[![Crates.io](https://img.shields.io/crates/v/sklears-decomposition.svg)](https://crates.io/crates/sklears-decomposition)
[![Documentation](https://docs.rs/sklears-decomposition/badge.svg)](https://docs.rs/sklears-decomposition)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

High-performance matrix decomposition and dimensionality reduction algorithms for Rust, featuring streaming capabilities and SIMD/GPU-accelerated kernels.

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-decomposition` provides state-of-the-art decomposition algorithms:

- **Classic Methods**: `PCA` (truncated-SVD based), `NMF`, `FastICA`/`JADE`/`InfoMax`, `FactorAnalysis`
- **Advanced Algorithms**: `KernelPCA`, `DictionaryLearning`, `MiniBatchDictionaryLearning`
- **Streaming**: `IncrementalPCA`, `OnlineNMF`, `StreamingPCA`, `StreamingICA`
- **Specialized**: Tensor decomposition (`CPDecomposition`, `TuckerDecomposition`), robust low-rank recovery (`LowRankMatrixRecovery`, `MEstimatorDecomposition`), `CanonicalCorrelationAnalysis`, `PartialLeastSquares`
- **Performance**: SIMD-accelerated signal processing kernels, an oxicuda-backed `gpu` feature (device discovery via `sklears_core::gpu`), and dedicated memory-efficiency utilities

> **Note**: a handful of names in older drafts of this document (`TruncatedSVD`, `RandomizedSVD`, `RandomizedPCA`, `OutOfCorePCA`, `TensorPCA`, `Tucker`, `PARAFAC`, `SignalICA`, `EMD`, `VMD`, `MemoryEfficientNMF`) do not exist as public types in this crate; the sections below use the real, verified type and method names. `RobustPCA`, `SparsePCA`, and `ProbabilisticPCA` also exist as public names, but only as empty placeholder marker structs with no fields or methods — see [Status](#status).

## Quick Start

```rust
use sklears_decomposition::{PCA, NMF, FastICA};
use sklears_decomposition::{NMFInit, NMFSolver};
use sklears_core::traits::{Fit, Transform};
use scirs2_core::ndarray::array;

// Principal Component Analysis
let pca = PCA::builder()
    .n_components(Some(2))
    .whiten(true)
    .build();

// Non-negative Matrix Factorization
let nmf = NMF::new(5)
    .init(NMFInit::Nndsvd)
    .solver(NMFSolver::CoordinateDescent);

// Independent Component Analysis (FastICA)
let ica = FastICA::new().n_components(3);

// Fit and transform
let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 1.0, 4.0], [5.0, 3.0, 2.0]];
let fitted = pca.fit(&x, &())?;
let x_transformed = fitted.transform(&x)?;
```

## Advanced Features

### Kernel PCA

```rust
use sklears_decomposition::{KernelPCA, KernelFunction};

let kpca = KernelPCA::new()
    .n_components(2)
    .kernel(KernelFunction::Rbf { gamma: 0.1 });

// Non-linear dimensionality reduction
let fitted = kpca.fit(&x, &())?;
let x_kpca = fitted.transform(&x)?;
```

### Streaming Decomposition

```rust
use sklears_decomposition::{IncrementalPCA, OnlineNMF};

// Incremental PCA for large datasets
let mut ipca = IncrementalPCA::new()
    .n_components(50)
    .batch_size(1000);

for batch in data_stream {
    ipca = ipca.partial_fit(&batch, &())?;
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
use sklears_decomposition::{DictionaryLearning, DictionaryTransformAlgorithm, MiniBatchDictionaryLearning, MiniBatchConfig};

// Sparse coding with learned dictionary
let dict_learning = DictionaryLearning::builder()
    .n_components(50)
    .alpha(1.0)
    .transform_algorithm(DictionaryTransformAlgorithm::LARS)
    .build();

// Mini-batch version for large datasets
let mb_dict = MiniBatchDictionaryLearning::new(MiniBatchConfig {
    n_components: 50,
    batch_size: 100,
    max_iter: 500,
});
```

## Specialized Algorithms

### Robust Low-Rank Recovery

```rust
use sklears_decomposition::{LowRankMatrixRecovery, RecoveryAlgorithm};

// Separate a low-rank component from sparse corruption (Robust PCA / PCP-style recovery)
let rpca = LowRankMatrixRecovery::new()
    .algorithm(RecoveryAlgorithm::RPCA)
    .lambda(1.0 / (x.nrows() as f64).sqrt())
    .max_iter(1000);

let fitted = rpca.fit(&x, &())?;
let low_rank = fitted.low_rank_component();
let sparse = fitted.sparse_component();
```

### Factor Analysis

```rust
use sklears_decomposition::FactorAnalysis;

let fa = FactorAnalysis::new(3).random_state(42);

let fitted = fa.fit(&x, &())?;
let noise_variance = fitted.noise_variance();
```

### Tensor Decomposition

```rust
use sklears_decomposition::{CPDecomposition, TuckerDecomposition, TuckerAlgorithm};
use scirs2_core::ndarray::Array3;

// CP/PARAFAC-style decomposition
let cp = CPDecomposition::new(10 /* rank */);

// Tucker decomposition
let tucker = TuckerDecomposition::new(vec![5, 5, 3])
    .algorithm(TuckerAlgorithm::HOSVD);

let tensor: Array3<f64> = Array3::zeros((10, 10, 5));
let fitted_cp = cp.fit(&tensor, &())?;
```

## Performance Optimizations

### Blind Source Separation (Signal Processing)

```rust
use sklears_decomposition::{FastICA, NonLinearityType};

// Blind source separation, returning sources/mixing/unmixing matrices directly
let signal_ica = FastICA::new().fun(NonLinearityType::Cube);
let bss_result = signal_ica.fit_transform(&mixed_signals)?;
let sources = &bss_result.sources;
```

### Empirical Mode Decomposition

```rust
use sklears_decomposition::EmpiricalModeDecomposition;

let emd = EmpiricalModeDecomposition::default();
let result = emd.decompose(&signal)?;
```

### Memory-Efficient Operations

The `memory_efficiency` and `hardware_acceleration` modules provide SIMD-accelerated
matrix ops, aligned buffers, and (behind the `gpu` feature) oxicuda-backed device
discovery/acceleration (`GpuAcceleration`, `GpuDecomposition`) — see `TODO.md` for the
current migration status.

## Quality Metrics

```rust
use sklears_decomposition::PCA;

// Assess decomposition quality
let pca = PCA::builder().n_components(Some(2)).build();
let fitted = pca.fit(&x, &())?;
let var_ratio = &fitted.explained_variance_ratio; // public field, not a method
let cumsum: Vec<f64> = var_ratio.iter().scan(0.0, |acc, &v| {
    *acc += v;
    Some(*acc)
}).collect();

let x_reduced = fitted.transform(&x)?;
// Note: `PCA`/`PcaTrained` does not yet implement `inverse_transform` (see `TODO.md`);
// use the `quality_metrics` module's `QualityAssessment` for reconstruction diagnostics instead.
```

The `quality_metrics` module also provides a `QualityAssessment` type with
`reconstruction_quality()`, `goodness_of_fit()`, `model_comparison()`, and
`overall_quality_score()` methods for more comprehensive evaluation.

## Architecture

Top-level modules actually exported from `src/lib.rs`:

```
sklears-decomposition/
├── pca.rs, kernel_pca.rs, incremental_pca.rs   # PCA family
├── nmf.rs, online_nmf.rs                       # NMF family
├── ica.rs, signal_processing/                  # ICA, FastICA/JADE/InfoMax, EMD, wavelets, STFT
├── dictionary_learning/                         # Dictionary learning, mini-batch, OMP/LARS/K-SVD
├── factor_analysis.rs, cca.rs, pls.rs           # Factor analysis, CCA, PLS
├── tensor_decomposition.rs                      # CP / Tucker decomposition
├── matrix_completion.rs, robust_methods.rs      # Low-rank recovery, M-estimator robust methods
├── streaming.rs, time_series.rs                 # StreamingPCA/ICA, SSA, seasonal decomposition
├── hardware_acceleration.rs, memory_efficiency.rs, distributed.rs  # SIMD/GPU/distributed
├── modular_framework.rs, type_safe.rs, fluent_api.rs                # Pipeline/composition APIs
├── quality_metrics.rs, validation.rs, visualization.rs              # Quality & diagnostics
└── sklearn_compat.rs, format_support.rs, integration.rs             # Interop
```

## Status

- **Tests**: 380 passing crate tests (`cargo nextest run -p sklears-decomposition --all-features`, verified 2026-07-14).
- **Core Algorithms**: PCA, NMF, FastICA/JADE/InfoMax, Kernel PCA, Factor Analysis, Dictionary Learning (+ mini-batch), Incremental PCA, Online NMF, CP/Tucker tensor decomposition, CCA, PLS, low-rank matrix recovery (PCP/RPCA-style) are real, tested implementations.
- **Known gaps**: `RobustPCA`, `SparsePCA`, and `ProbabilisticPCA` in `pca.rs` are currently empty placeholder marker structs (no fields, no methods) kept only for name compatibility — use `LowRankMatrixRecovery` for robust/sparse-plus-low-rank recovery instead. `PcaConfig::svd_solver` is a `String` field that is not yet read anywhere in the fit path (no working randomized-SVD path).
- **Streaming Support**: Fully implemented (`IncrementalPCA`, `OnlineNMF`, `StreamingPCA`, `StreamingICA`).
- **GPU Acceleration**: oxicuda-backed device discovery/acceleration behind the `gpu` feature; see `TODO.md` for migration details.

## Contributing

Priority areas:
- Real implementations behind the `RobustPCA`/`SparsePCA`/`ProbabilisticPCA` placeholder names (or removing them)
- Wiring `PcaConfig::svd_solver` into the actual fit path
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
