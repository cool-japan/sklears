# Sklears Python Bindings

[![Crates.io](https://img.shields.io/crates/v/sklears-python.svg)](https://crates.io/crates/sklears-python)
[![Documentation](https://docs.rs/sklears-python/badge.svg)](https://docs.rs/sklears-python)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Python bindings for the sklears machine learning library, providing a high-performance, scikit-learn compatible interface through PyO3.

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Features

- **Drop-in replacement** for scikit-learn's most common algorithms
- **Pure Rust implementation** with ongoing performance optimization
- **Full NumPy array compatibility** with zero-copy operations where possible
- **Comprehensive error handling** with Python exceptions
- **Memory-safe operations** with automatic reference counting
- **Scikit-learn compatible API** for easy migration

## Status

- **Partial**: covers linear models, ensembles, MLP, naive Bayes, clustering (KMeans/DBSCAN), core preprocessing, `train_test_split`/`KFold`, and the common classification/regression metrics — all genuinely wired to their underlying `sklears-*` implementations (no stub bindings).
- Covered by 55 passing crate tests for `0.2.0`.
- Tree-based models (`RandomForestClassifier`, `DecisionTreeClassifier`) are implemented in `src/tree.rs` but still commented out of the Python module registration; `StratifiedKFold` and `cross_val_score` are not implemented yet. See `TODO.md` for details.

## Supported Algorithms

### Linear Models
- `LinearRegression` - Ordinary least squares linear regression
- `Ridge` - Ridge regression with L2 regularization
- `Lasso` - Lasso regression with L1 regularization
- `ElasticNet` - Elastic-net regularization
- `BayesianRidge` - Bayesian ridge regression
- `ARDRegression` - Automatic Relevance Determination regression
- `LogisticRegression` - Logistic regression for classification

### Ensemble Methods
- `GradientBoostingClassifier` - Gradient boosting for classification
- `GradientBoostingRegressor` - Gradient boosting for regression
- `AdaBoostClassifier` - Adaptive boosting classifier
- `VotingClassifier` - Voting ensemble classifier
- `BaggingClassifier` - Bagging ensemble classifier

### Neural Networks
- `MLPClassifier` - Multi-layer perceptron classifier
- `MLPRegressor` - Multi-layer perceptron regressor

### Naive Bayes
- `GaussianNB` - Gaussian Naive Bayes
- `MultinomialNB` - Multinomial Naive Bayes
- `BernoulliNB` - Bernoulli Naive Bayes
- `ComplementNB` - Complement Naive Bayes

### Clustering
- `KMeans` - K-Means clustering algorithm (K-means++ init; exposes `fit`/`predict`/`fit_predict` plus `labels_`, `cluster_centers_`, `inertia_`, `n_iter_`)
- `DBSCAN` - Density-based spatial clustering (transductive like scikit-learn's implementation: exposes `fit_predict()` + `labels_`, no `predict()` on new data since the underlying algorithm has none)

### Preprocessing
- `StandardScaler` - Standardize features by removing mean and scaling to unit variance
- `MinMaxScaler` - Scale features to a given range
- `LabelEncoder` - Encode target labels with value between 0 and n_classes-1 (accepts a list of strings, e.g. `le.fit(["a", "b", "c"])`)

### Tree Models _(coming soon)_
- `RandomForestClassifier` - Random forest for classification
- `DecisionTreeClassifier` - Decision tree for classification

### Model Selection
- `train_test_split` - Split arrays into random train and test subsets (`test_size`, `random_state`; always shuffles — `train_size`/`stratify` not yet supported)
- `KFold` - K-Fold cross-validator (`n_splits`, `shuffle`, `random_state`)
- `StratifiedKFold` _(coming soon)_ - Stratified K-Fold cross-validator
- `cross_val_score` _(coming soon)_ - Evaluate metric(s) by cross-validation

### Metrics
- `accuracy_score` - Classification accuracy
- `mean_squared_error` - Mean squared error for regression
- `mean_absolute_error` - Mean absolute error for regression
- `mean_squared_log_error` - Mean squared logarithmic error for regression
- `median_absolute_error` - Median absolute error for regression
- `r2_score` - R² (coefficient of determination) score
- `precision_score` - Precision for classification
- `recall_score` - Recall for classification
- `f1_score` - F1 score for classification
- `confusion_matrix` - Confusion matrix for classification
- `classification_report` - Text report of classification metrics

## Installation

### Prerequisites

- Python 3.9 or later
- NumPy
- Rust 1.70 or later
- PyO3 and Maturin for building

### Building from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cool-japan/sklears.git
   cd sklears/crates/sklears-python
   ```

2. **Install Maturin:**
   ```bash
   pip install maturin
   ```

3. **Build and install the package:**
   ```bash
   maturin develop --release
   ```

4. **Or build a wheel:**
   ```bash
   maturin build --release
   pip install target/wheels/sklears-*.whl
   ```

## Quick Start

```python
import numpy as np
import sklears as skl

# Generate sample data
X = np.random.randn(100, 4)
y = np.random.randn(100)

# Train a linear regression model
model = skl.LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Calculate R² score
score = model.score(X, y)
print(f"R² score: {score:.3f}")
```

### Clustering, Preprocessing, and Model Selection

```python
import numpy as np
import sklears as skl

X = np.random.randn(100, 4)
y = np.random.randn(100)

# train_test_split requires y as float64 (PyReadonlyArray1<f64>) -- cast
# integer label arrays with .astype(np.float64) first.
X_train, X_test, y_train, y_test = skl.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# StandardScaler: fit/transform (or fit_transform) both genuinely compute
# per-feature mean/variance now.
scaler = skl.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KMeans: fit()/predict()/fit_predict() are wired to the real
# sklears-clustering implementation (K-means++ initialization).
kmeans = skl.KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
cluster_labels = kmeans.predict(X_test_scaled)
print(f"Inertia: {kmeans.inertia_:.3f}, iterations: {kmeans.n_iter_}")

# DBSCAN has no predict() on new data (matches scikit-learn's transductive
# behavior) -- use fit_predict() and the labels_ attribute instead.
dbscan = skl.DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_train_scaled)

# KFold: n_splits/shuffle/random_state are all real now.
kfold = skl.KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kfold.split(X_train_scaled):
    pass  # train_idx / test_idx are lists of row indices
```

## Performance Comparison

Here's a typical performance comparison with scikit-learn:

```python
import time
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import sklears as skl
from sklearn.linear_model import LinearRegression as SklearnLR

# Generate data
X, y = make_regression(n_samples=10000, n_features=100, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Sklears
start = time.time()
model = skl.LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
sklears_time = time.time() - start

# Scikit-learn
start = time.time()
sklearn_model = SklearnLR()
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)
sklearn_time = time.time() - start

print(f"Sklears time: {sklears_time:.4f}s")
print(f"Sklearn time: {sklearn_time:.4f}s")
print(f"Speedup: {sklearn_time / sklears_time:.2f}x")
```

## API Compatibility

The sklears Python bindings are designed to be API-compatible with scikit-learn. Most existing scikit-learn code should work with minimal changes:

### Before (scikit-learn):
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

### After (sklears):
```python
import sklears as skl

# Available classes and functions
model = skl.LinearRegression()
X_train, X_test, y_train, y_test = skl.train_test_split(X, y.astype("float64"))

scaler = skl.StandardScaler()
X_train = scaler.fit_transform(X_train)

# mean_squared_error, r2_score, accuracy_score, etc. are all available too.

# Note: StratifiedKFold, cross_val_score - coming soon
```

## Memory Management

The bindings are designed to be memory-efficient:

- **Zero-copy operations** where possible using NumPy's C API
- **Automatic memory management** through PyO3's reference counting
- **Efficient data structures** using ndarray and sprs for sparse matrices
- **Streaming support** for large datasets that don't fit in memory

## Error Handling

All Rust errors are properly converted to Python exceptions:

```python
import sklears as skl
import numpy as np

try:
    # This will raise a ValueError if arrays have incompatible shapes
    model = skl.LinearRegression()
    model.fit(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]))  # Shape mismatch
except ValueError as e:
    print(f"Error: {e}")
```

## System Information

Get information about your sklears installation:

```python
import sklears as skl

# Version information (tracks the crate's Cargo.toml version automatically)
print(f"Version: {skl.get_version()}")
print(f"Version: {skl.__version__}")

# Build information
build_info = skl.get_build_info()
for key, value in build_info.items():
    print(f"{key}: {value}")

# Hardware capability flags (avx2/fma/neon/...) plus the real CPU core
# count under "num_cpus" (previously miscoded as a bool).
hardware_info = skl.get_hardware_info()
for key, value in hardware_info.items():
    print(f"{key}: {value}")

# Basic timing benchmarks (matrix multiply, dot product, allocation), in ms.
timings = skl.benchmark_basic_operations()
for key, value in timings.items():
    print(f"{key}: {value:.3f}")
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `python_demo.py` - Complete demonstration of all features
- Performance comparison scripts
- Real-world use cases

## Contributing

Contributions are welcome! Please see the main sklears repository for contribution guidelines.

## License

This project is licensed under the Apache-2.0 license.

## Acknowledgments

- Built with [PyO3](https://pyo3.rs/) for Rust-Python interoperability
- Compatible with [NumPy](https://numpy.org/) arrays
- API inspired by [scikit-learn](https://scikit-learn.org/)
