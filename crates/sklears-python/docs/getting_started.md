# Getting Started with Sklears Python Bindings

Welcome to Sklears! This guide will help you get up and running with the high-performance machine learning library that provides a pure Rust implementation with ongoing performance optimization while maintaining full API compatibility.

## Table of Contents

1. [Installation](#installation)
2. [First Steps](#first-steps)
3. [Basic Usage Examples](#basic-usage-examples)
4. [Performance Comparison](#performance-comparison)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

## Installation

### Prerequisites

Before installing Sklears, ensure you have:

- **Python 3.9 or later**
- **NumPy** (automatically installed as dependency)
- **Rust 1.70 or later** (for building from source)

### Method 1: Install from Pre-built Wheels (Recommended)

```bash
pip install sklears
```

### Method 2: Build from Source

If pre-built wheels are not available for your platform:

1. **Install Rust and Maturin:**
   ```bash
   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   
   # Install Maturin
   pip install maturin
   ```

2. **Clone and build:**
   ```bash
   git clone https://github.com/cool-japan/sklears.git
   cd sklears/crates/sklears-python
   maturin develop --release
   ```

### Verify Installation

```python
import sklears as skl
print(f"Sklears version: {skl.get_version()}")
print("Installation successful!")
```

## First Steps

### Import Convention

We recommend importing sklears as follows:

```python
import sklears as skl
import numpy as np
```

This gives you access to all algorithms and utilities in a single namespace, similar to how you might use `sklearn`.

### Check System Capabilities

Before diving into machine learning, let's check what your system supports:

```python
import sklears as skl

# Check version and build information
print(f"Version: {skl.get_version()}")

# Check build details
build_info = skl.get_build_info()
print(f"Build info: {build_info}")
```

> **Note:** `get_hardware_info()` and `benchmark_basic_operations()` are not available in this release. Use `get_version()` and `get_build_info()` instead.

## Basic Usage Examples

### Linear Regression

Let's start with a simple linear regression example:

```python
import numpy as np
import sklears as skl
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = skl.train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = skl.LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance using the model's built-in score method (R²)
r2_score = model.score(X_test, y_test)

# Compute MSE manually (skl.mean_squared_error is Coming Soon)
mse = float(np.mean((y_test - predictions) ** 2))

print(f"R² Score: {r2_score:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
```

### Classification with Logistic Regression

```python
import numpy as np
import sklears as skl
from sklearn.datasets import make_classification

# Generate classification data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = skl.train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = skl.LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Compute accuracy manually (skl.accuracy_score is Coming Soon)
accuracy = float(np.mean(np.array(predictions) == np.array(y_test)))
print(f"Accuracy: {accuracy:.4f}")
```

### Clustering with K-Means

```python
import numpy as np
import sklears as skl
from sklearn.datasets import make_blobs

# Generate clustering data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Perform clustering
kmeans = skl.KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Number of clusters found: {len(np.unique(labels))}")
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
```

### Data Preprocessing

> **Coming Soon:** `StandardScaler` and `MinMaxScaler` are not yet exposed in this release. Use scikit-learn's preprocessing utilities as an interim solution:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Generate sample data
X = np.random.randn(100, 5) * 10 + 5

# Standard scaling (via scikit-learn until sklears exposes these)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Original data - Mean: {X.mean():.2f}, Std: {X.std():.2f}")
print(f"Scaled data - Mean: {X_scaled.mean():.2f}, Std: {X_scaled.std():.2f}")

# Min-Max scaling
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
X_minmax = minmax_scaler.fit_transform(X)

print(f"MinMax scaled - Min: {X_minmax.min():.2f}, Max: {X_minmax.max():.2f}")
```

### Cross-Validation

```python
import numpy as np
import sklears as skl
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=200, n_features=10, random_state=42)

# Create model
model = skl.LinearRegression()

# Perform cross-validation
kfold = skl.KFold(n_splits=5, shuffle=True, random_state=42)
splits = kfold.split(X, y)

scores = []
for train_idx, test_idx in splits:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
```

## Performance Comparison

One of Sklears' main advantages is performance. Here's how to measure it:

```python
import time
import numpy as np
import sklears as skl
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLR

# Generate large dataset
X, y = make_regression(n_samples=10000, n_features=100, noise=0.1, random_state=42)

# Benchmark Sklears
start_time = time.time()
model = skl.LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
sklears_time = time.time() - start_time

# Benchmark Scikit-learn
start_time = time.time()
sklearn_model = SklearnLR()
sklearn_model.fit(X, y)
sklearn_predictions = sklearn_model.predict(X)
sklearn_time = time.time() - start_time

print(f"Sklears time: {sklears_time:.4f} seconds")
print(f"Sklearn time: {sklearn_time:.4f} seconds")
print(f"Speedup: {sklearn_time / sklears_time:.2f}x")

# Verify results are similar
mse_diff = np.mean((predictions - sklearn_predictions) ** 2)
print(f"Prediction difference (MSE): {mse_diff:.8f}")
```

## Common Patterns

### Pipeline-like Operations

While Sklears doesn't have Pipeline class yet, you can chain operations manually:

```python
import numpy as np
import sklears as skl
from sklearn.preprocessing import StandardScaler  # Coming Soon in sklears

# Generate data
X = np.random.randn(100, 5) * 10 + 5
y = np.random.randn(100)

# Chain preprocessing and modeling
# Note: StandardScaler is Coming Soon in sklears; use sklearn.preprocessing as interim
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = skl.LinearRegression()
model.fit(X_scaled, y)

# For new data, apply same preprocessing
X_new = np.random.randn(10, 5) * 10 + 5
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

### Model Persistence

While built-in serialization is in development, you can save model parameters:

```python
import pickle
import sklears as skl
import numpy as np

# Train model
X, y = np.random.randn(100, 5), np.random.randn(100)
model = skl.LinearRegression()
model.fit(X, y)

# Save model parameters (placeholder - actual implementation may vary)
model_data = {
    'coefficients': model.coef_,
    'intercept': model.intercept_,
    'type': 'LinearRegression'
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully")
```

### Error Handling

Sklears provides clear error messages:

```python
import sklears as skl
import numpy as np

try:
    # This will raise an error due to shape mismatch
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3])  # Wrong shape!
    
    model = skl.LinearRegression()
    model.fit(X, y)
except ValueError as e:
    print(f"Error caught: {e}")
```

## Troubleshooting

### Common Issues

1. **Import Error**: If you get import errors, ensure you've built/installed the package correctly:
   ```bash
   python -c "import sklears; print('Success!')"
   ```

2. **Performance not as expected**: Check that you're using the release build:
   ```bash
   maturin develop --release
   ```

3. **Memory issues with large datasets**: Use data chunking or consider increasing system memory.

### Getting Help

1. **Check build information**:
   ```python
   import sklears as skl
   print(skl.get_build_info())
   ```

2. **Check version**:
   ```python
   import sklears as skl
   print(skl.get_version())
   ```

> **Note:** `show_versions()`, `benchmark_basic_operations()`, and `get_hardware_info()` are not available in this release.

## Next Steps

Now that you're familiar with the basics, explore these advanced topics:

1. **[Migration Guide](migration_guide.md)** - Moving from scikit-learn to sklears
2. **[Performance Optimization](performance_guide.md)** - Getting maximum performance
3. **[API Reference](api_reference.md)** - Complete function and class documentation
4. **[Advanced Examples](advanced_examples.md)** - Complex use cases and patterns

### Example Projects

Try these complete examples to deepen your understanding:

1. **Regression on Boston Housing Data**
2. **Classification with Cross-Validation**
3. **Clustering Analysis**
4. **Time Series Preprocessing**
5. **Hyperparameter Optimization**

### Contributing

Interested in contributing? Check out:

- **[Development Guide](development_guide.md)**
- **[Testing Framework](testing_guide.md)**
- **[Benchmarking Tools](benchmarking_guide.md)**

---

**Ready to see dramatic performance improvements in your machine learning workflows? Start experimenting with Sklears today!**