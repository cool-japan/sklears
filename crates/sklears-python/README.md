# Sklears Python Bindings

[![Crates.io](https://img.shields.io/crates/v/sklears-python.svg)](https://crates.io/crates/sklears-python)
[![Documentation](https://docs.rs/sklears-python/badge.svg)](https://docs.rs/sklears-python)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Python bindings for the sklears machine learning library, providing a high-performance, scikit-learn compatible interface through PyO3.

> **Latest release:** `0.1.0-alpha.1` (October 13, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.1.md) for highlights and upgrade guidance.

## Features

- **Drop-in replacement** for scikit-learn's most common algorithms
- **3-100x performance improvements** over scikit-learn
- **Full NumPy array compatibility** with zero-copy operations where possible
- **Comprehensive error handling** with Python exceptions
- **Memory-safe operations** with automatic reference counting
- **Scikit-learn compatible API** for easy migration

## Supported Algorithms

### Linear Models
- `LinearRegression` - Ordinary least squares linear regression
- `Ridge` - Ridge regression with L2 regularization
- `Lasso` - Lasso regression with L1 regularization
- `LogisticRegression` - Logistic regression for classification

### Clustering
- `KMeans` - K-Means clustering algorithm
- `DBSCAN` - Density-based spatial clustering

### Preprocessing
- `StandardScaler` - Standardize features by removing mean and scaling to unit variance
- `MinMaxScaler` - Scale features to a given range
- `LabelEncoder` - Encode target labels with value between 0 and n_classes-1

### Model Selection
- `train_test_split` - Split arrays into random train and test subsets
- `KFold` - K-Fold cross-validator
- `StratifiedKFold` - Stratified K-Fold cross-validator
- `cross_val_score` - Evaluate metric(s) by cross-validation
- `cross_val_predict` - Generate cross-validated estimates

### Metrics
- `accuracy_score` - Classification accuracy
- `mean_squared_error` - Mean squared error for regression
- `mean_absolute_error` - Mean absolute error for regression
- `r2_score` - R² (coefficient of determination) score
- `precision_score` - Precision for classification
- `recall_score` - Recall for classification
- `f1_score` - F1 score for classification
- `confusion_matrix` - Confusion matrix for classification
- `classification_report` - Text report of classification metrics

## Installation

### Prerequisites

- Python 3.8 or later
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
   pip install target/wheels/sklears_python-*.whl
   ```

## Quick Start

```python
import numpy as np
import sklears_python as skl

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

## Performance Comparison

Here's a typical performance comparison with scikit-learn:

```python
import time
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import sklears_python as skl
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
import sklears_python as skl

# All functions and classes are available in the main module
model = skl.LinearRegression()
scaler = skl.StandardScaler()
X_train, X_test, y_train, y_test = skl.train_test_split(X, y)
mse = skl.mean_squared_error(y_true, y_pred)
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
import sklears_python as skl
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
import sklears_python as skl

# Version information
print(f"Version: {skl.get_version()}")

# Build information
build_info = skl.get_build_info()
for key, value in build_info.items():
    print(f"{key}: {value}")

# Hardware capabilities
hardware_info = skl.get_hardware_info()
print("Hardware support:")
for feature, supported in hardware_info.items():
    print(f"  {feature}: {supported}")

# Performance benchmarks
benchmarks = skl.benchmark_basic_operations()
print("Performance benchmarks:")
for operation, time_ms in benchmarks.items():
    print(f"  {operation}: {time_ms:.2f} ms")
```

## Configuration

Set global configuration options:

```python
import sklears_python as skl

# Set number of threads for parallel operations
skl.set_config("n_jobs", "4")

# Get current configuration
config = skl.get_config()
print(config)
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `python_demo.py` - Complete demonstration of all features
- Performance comparison scripts
- Real-world use cases

## Contributing

Contributions are welcome! Please see the main sklears repository for contribution guidelines.

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Acknowledgments

- Built with [PyO3](https://pyo3.rs/) for Rust-Python interoperability
- Compatible with [NumPy](https://numpy.org/) arrays
- API inspired by [scikit-learn](https://scikit-learn.org/)
