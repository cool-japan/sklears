# Sklears Python Bindings

[![Crates.io](https://img.shields.io/crates/v/sklears-python.svg)](https://crates.io/crates/sklears-python)
[![Documentation](https://docs.rs/sklears-python/badge.svg)](https://docs.rs/sklears-python)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Python bindings for the sklears machine learning library, providing a high-performance, scikit-learn compatible interface through PyO3.

> **Latest release:** `0.1.0-rc.1` (February 2026). See the [workspace release notes](../../docs/releases/0.1.0-rc.1.md) for highlights and upgrade guidance.

## Features

- **Drop-in replacement** for scikit-learn's most common algorithms
- **Pure Rust implementation** with ongoing performance optimization
- **Full NumPy array compatibility** with zero-copy operations where possible
- **Comprehensive error handling** with Python exceptions
- **Memory-safe operations** with automatic reference counting
- **Scikit-learn compatible API** for easy migration

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
- `KMeans` - K-Means clustering algorithm
- `DBSCAN` - Density-based spatial clustering

### Preprocessing _(coming soon)_
- `StandardScaler` - Standardize features by removing mean and scaling to unit variance
- `MinMaxScaler` - Scale features to a given range
- `LabelEncoder` - Encode target labels with value between 0 and n_classes-1

### Tree Models _(coming soon)_
- `RandomForestClassifier` - Random forest for classification
- `DecisionTreeClassifier` - Decision tree for classification

### Model Selection
- `train_test_split` - Split arrays into random train and test subsets
- `KFold` - K-Fold cross-validator
- `StratifiedKFold` _(coming soon)_ - Stratified K-Fold cross-validator
- `cross_val_score` _(coming soon)_ - Evaluate metric(s) by cross-validation

### Metrics _(coming soon)_
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
X_train, X_test, y_train, y_test = skl.train_test_split(X, y)

# Note: StandardScaler, MinMaxScaler, LabelEncoder - coming soon
# Note: mean_squared_error, r2_score, accuracy_score, etc. - coming soon
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

# Version information
print(f"Version: {skl.get_version()}")

# Build information
build_info = skl.get_build_info()
for key, value in build_info.items():
    print(f"{key}: {value}")

# Note: get_hardware_info() and benchmark_basic_operations() - coming soon
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
