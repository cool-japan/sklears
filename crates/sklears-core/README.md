# sklears-core

[![Crates.io](https://img.shields.io/crates/v/sklears-core.svg)](https://crates.io/crates/sklears-core)
[![Documentation](https://docs.rs/sklears-core/badge.svg)](https://docs.rs/sklears-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

The foundational crate for sklears, providing core traits, types, and utilities that power the entire machine learning ecosystem. Actively evolving (Partial) — core traits and error handling are stable, while some advanced modules are still maturing; see [Status](#status).

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-core` provides the fundamental building blocks for all sklears algorithms:

- **Core Traits**: Comprehensive ML abstractions with type-safe state management
- **Advanced Type System**: Compile-time validation, phantom types, const generics
- **Performance Infrastructure**: SIMD, an oxicuda-backed GPU backend (`gpu::{GpuBackend, GpuArray, GpuMatrixOps}`) that gracefully reports "no GPU" instead of faking a CPU fallback, memory pooling, parallel processing
- **Error Handling**: Rich error types with context propagation and recovery
- **Integration**: scikit-learn compatibility, format I/O, cross-framework support
- **Trait Explorer Tooling**: Graph-based analysis of the crate's own trait relationships (hub/bridge/bottleneck node detection, Newman modularity, small-world coefficient), plus API reference generation. Also includes `trait_explorer::security_analysis`, an internal dev-tooling module (not part of the public ML API) for compliance/security-metrics assessment of trait usage, with data-backed constructors for common regulatory frameworks (GDPR, HIPAA, CCPA, SOX, FERPA, ISO 27001, NIST CSF, COBIT, ITIL, CIS Controls).

## Status

- **Implementation**: 0.2.0 ships with >99% of the planned v0.1 APIs implemented (141 stubs remaining). Status: **Partial** — actively evolving, not yet claiming full stability.
- **Validation**: Covered by 863 passing crate tests (`cargo nextest run -p sklears-core --all-features`).
- **Performance**: Pure Rust implementation with ongoing performance optimization via SIMD, threading, and cache-friendly layouts. An oxicuda-backed GPU backend (`gpu::GpuBackend` / `GpuArray` / `GpuMatrixOps`) is available behind the `gpu_support` feature, wired directly to `oxicuda-driver` / `oxicuda-blas`; `GpuBackend::detect()` gracefully returns `Ok(None)` on machines without a usable GPU rather than silently substituting a fake backend.
- **API Stability**: Minor breaking changes possible in pre-1.0 releases; stabilization roadmap tracked in the root `TODO.md`.

## Core Trait System

### Base Traits

#### `Estimator<State>`
The foundational trait for all ML models with compile-time state tracking:

```rust
pub trait Estimator<State = Untrained> {
    type Config;
    type Error: std::error::Error;
}
```

### Learning Traits

```rust
// Supervised learning
pub trait Fit<X, Y, State = Untrained> {
    type Fitted;
    fn fit(self, x: &X, y: &Y) -> Result<Self::Fitted>;
}

// Incremental/online learning
pub trait PartialFit<X, Y> {
    fn partial_fit(&mut self, x: &X, y: &Y) -> Result<()>;
}

// Unsupervised learning
pub trait FitTransform<X, Y = (), Output = X> {
    fn fit_transform(self, x: &X, y: Option<&Y>) -> Result<Output>;
}
```

### Prediction Traits

```rust
// Standard predictions
pub trait Predict<X, Output> {
    fn predict(&self, x: &X) -> Result<Output>;
}

// Probabilistic predictions
pub trait PredictProba<X, Output> {
    fn predict_proba(&self, x: &X) -> Result<Output>;
}

// Decision scores
pub trait DecisionFunction<X, Output> {
    fn decision_function(&self, x: &X) -> Result<Output>;
}
```

### Advanced Features

#### Async Trait Support
```rust
pub trait AsyncFit<X, Y> {
    async fn fit_async(self, x: &X, y: &Y) -> Result<Self::Fitted>;
}

pub trait AsyncPredict<X, Output> {
    async fn predict_async(&self, x: &X) -> Result<Output>;
}
```

#### GPU Acceleration
Behind the `gpu_support` feature, backed by real `oxicuda-driver` / `oxicuda-blas` calls (no CPU-fallback stub):

```rust
use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps};

// `detect()` gracefully returns `Ok(None)` when no GPU/driver is present,
// instead of silently substituting a fake backend.
if let Some(backend) = GpuBackend::detect()? {
    let a = GpuArray::from_array2(&backend, &matrix_a)?;
    let b = GpuArray::from_array2(&backend, &matrix_b)?;
    let result = a.matmul(&b)?.to_array2()?;
}
```

## Type-Safe State Management

Prevent common ML errors at compile time:

```rust
use sklears_core::{Untrained, Trained};

// Model starts untrained
struct Model<State = Untrained> {
    config: Config,
    state: PhantomData<State>,
    weights_: Option<Weights>,
}

// Only untrained models can be fitted
impl Fit<X, Y> for Model<Untrained> {
    type Fitted = Model<Trained>;
    
    fn fit(self, x: &X, y: &Y) -> Result<Self::Fitted> {
        // Training logic...
        Ok(Model {
            config: self.config,
            state: PhantomData,
            weights_: Some(trained_weights),
        })
    }
}

// Only trained models can predict
impl Predict<X, Y> for Model<Trained> {
    fn predict(&self, x: &X) -> Result<Y> {
        let weights = self.weights_.as_ref().unwrap(); // Safe!
        // Prediction logic...
    }
}
```

This prevents:
- Calling `predict()` on untrained models
- Accessing parameters before fitting
- Double-fitting models
- All caught at compile time!

## Advanced Type System

### Compile-Time Validation
```rust
use sklears_core::compile_time_validation::{ValidatedConfig, RangeValidator};

// `ValidatedConfig<T, S>` tracks validated/unvalidated state via a phantom type parameter;
// `.validate()` moves an `Unvalidated` config into a `ValidatedState` one, or returns an error.
let config = ValidatedConfig::new(my_config);
let validated = config.validate()?;

// Const-generic range validators, e.g. RangeValidator::<0, 1>, implement `ParameterValidator`
// for `i32`/`f64` parameters.
```

### Phantom Types for Safety

The `Untrained`/`Trained`-style phantom-type pattern shown above for `Estimator<State>` is used
throughout the crate (and downstream estimator crates) to encode task/state distinctions at
compile time. There is no separate `sklears_core::phantom` module — the pattern is applied
directly via each type's own state parameter (as in the `Model<State>` example above), not via a
shared `Classification`/`Regression` marker-type module.

## Performance Features

### SIMD Optimizations
```rust
use sklears_core::simd::SimdOps;

// Automatic SIMD acceleration
let distances = SimdOps::euclidean_distances_simd(&points_x, &points_y)?;
```

### Memory Efficiency
```rust
use sklears_core::types::memory_pool::MemoryPool;

// Reusable-buffer pool (max_buffers, buffer_size)
let pool: MemoryPool<f64> = MemoryPool::new(16, 1000);
let mut buffer = pool.get_buffer();
```

## Error Handling

Rich error types with context:

```rust
use sklears_core::{Result, SklearsError, validate};

fn train_model(x: &Array2<f64>, y: &Array1<f64>) -> Result<Model> {
    // Comprehensive validation
    validate::check_consistent_length(x, y)?;
    validate::check_finite(learning_rate, "learning_rate")?;
    validate::check_no_missing(x)?;
    
    // Error context propagation
    let model = complex_training(x, y)
        .context("Failed during gradient computation")?;
    
    Ok(model)
}
```

## Macro System

Powerful macros for boilerplate reduction:

```rust
use sklears_core::quick_dataset;
use scirs2_core::ndarray::{arr1, arr2};

// Quick dataset creation (field names are `data`/`target`, not `features`/`feature_names`)
let dataset = quick_dataset!(
    data: arr2(&[[1.0, 2.0], [3.0, 4.0]]),
    target: arr1(&[0.0, 1.0])
);

// ML-specific trait-bound alias (takes a single trait name; the bound list itself is fixed)
sklears_core::define_ml_float_bounds!(MLFloat);
fn process<T: MLFloat>(x: T) -> T { x }

// Automatic test generation (takes only the estimator type name)
sklears_core::estimator_test_suite!(MyEstimator);
```

## Integration & Compatibility

`sklears_core::compatibility` provides metadata-level interop helpers (not full zero-copy tensor
exchange):

```rust
use sklears_core::compatibility::serialization::CrossPlatformModel;
use sklears_core::compatibility::numpy::NumpyArray;
use sklears_core::compatibility::pytorch::ndarray_to_pytorch_tensor;
use sklears_core::compatibility::pandas::DataFrame;

// scikit-learn metadata round-trip (parameters/weights/version, not a live estimator)
let model = CrossPlatformModel::from_sklearn_metadata(metadata_map)?;

// NumPy-compatible array wrapper (shape/strides/dtype), built from an owned ndarray
let np_array = NumpyArray::from_ndarray(&owned_array)?;

// PyTorch-compatible tensor bytes + metadata (shape/dtype/device)
let (tensor_bytes, tensor_meta) = ndarray_to_pytorch_tensor(&array2, false)?;

// Pandas-compatible DataFrame built from an ndarray
let df = DataFrame::from_ndarray(&array2, None)?;
```

Note: there is no `SklearnEstimator::from_sklearn(...)` drop-in estimator conversion, and no
`array.to_numpy()`/`array.to_torch_tensor()`/`Dataset::from_polars()` methods — the real surface is
the module-level metadata/tensor-descriptor helpers shown above.

### Format I/O
`sklears_core::format_io::DataFormat` covers:
- CSV, JSON, Parquet
- HDF5, NPY/NPZ
- Arrow, Feather, Binary, MessagePack

(ONNX/PMML/MLflow are mentioned in module docs as aspirational targets but are not yet implemented formats in `DataFormat`.)

## Builder Pattern

Consistent API across all estimators:

```rust
let model = LinearRegression::builder()
    .learning_rate(0.01)
    .max_iter(1000)
    .early_stopping(true)
    .validation_fraction(0.2)
    .n_jobs(4)
    .random_state(42)
    .build()?;
```

## Testing Infrastructure

### Contract Testing
`sklears_core::contract_testing` provides infrastructure for verifying estimator contracts
(shape/state invariants) hold across implementations.

### Mock Objects
```rust
use sklears_core::mock_objects::{MockEstimator, MockBehavior};

// Simulates fit/predict timing and failure behavior for testing error-handling code,
// rather than returning a caller-supplied canned prediction.
let mock = MockEstimator::builder()
    .with_behavior(MockBehavior::FeatureSum)
    .with_fit_failure_probability(0.1)
    .build();
```

There is no `sklears_core::testing` module or `proptest`-based `properties::assert_*` helper
module — property-based testing in this crate is done ad hoc per-module with `proptest!` directly,
not through a shared assertion-helper API.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../../CONTRIBUTING.md).

## License

Licensed under the Apache License, Version 2.0.

## Citation

```bibtex
@software{sklears_core,
  title = {sklears-core: Type-Safe ML Foundation for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```