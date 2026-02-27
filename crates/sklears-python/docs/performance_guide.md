# Performance Optimization Guide

This guide helps you get maximum performance from Sklears by understanding how to optimize your machine learning workflows.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Hardware Optimization](#hardware-optimization)
3. [Data Optimization](#data-optimization)
4. [Algorithm-Specific Tips](#algorithm-specific-tips)
5. [Memory Management](#memory-management)
6. [Benchmarking and Profiling](#benchmarking-and-profiling)
7. [Common Performance Pitfalls](#common-performance-pitfalls)

## Performance Overview

Sklears provides significant performance improvements over scikit-learn through several key optimizations:

- **Rust Implementation**: Native code with zero-cost abstractions
- **SIMD Vectorization**: Automatic use of AVX2/NEON instructions
- **Memory Efficiency**: Optimized memory layouts and minimal allocations
- **Parallel Processing**: Automatic parallelization where beneficial
- **Streaming Algorithms**: Handle datasets larger than memory

### Typical Performance Gains

| Algorithm | Small Data (< 1K samples) | Medium Data (1K-100K) | Large Data (> 100K) |
|-----------|---------------------------|----------------------|---------------------|
| Linear Regression | 3-5x | 5-15x | 10-50x |
| K-Means | 2-4x | 4-10x | 8-30x |
| Preprocessing | 5-10x | 10-25x | 15-100x |

## Hardware Optimization

### Check Your Hardware Capabilities

First, understand what your system supports:

```python
import sklears as skl

# NOTE: get_hardware_info() - Coming Soon (not yet available in this release)
# NOTE: benchmark_basic_operations() - Coming Soon (not yet available)

# Use available introspection instead:
print(f"Sklears version: {skl.get_version()}")
print(f"Build info: {skl.get_build_info()}")

# Check CPU features via Python's platform module in the meantime:
import platform
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
```

### CPU Optimization

#### SIMD Instructions

Sklears automatically uses SIMD instructions when available:

```python
# Enable SIMD-optimized operations
import os
os.environ['RUST_LOG'] = 'debug'  # To see SIMD usage in logs

import sklears as skl
import numpy as np

# Large matrices benefit most from SIMD
X = np.random.randn(10000, 100)
# NOTE: StandardScaler - Coming Soon (not yet exposed)
# Large model operations already use SIMD internally:
model = skl.LinearRegression()
y = np.random.randn(10000)
model.fit(X, y)  # Uses AVX2/NEON internally if available
```

#### Multi-threading

Configure parallelism for your system:

```python
import sklears as skl

# NOTE: set_config() and get_config() - Coming Soon (not yet available)
# Parallelism is managed automatically by the Rust backend.
# Thread count can be set at the OS level with the RAYON_NUM_THREADS env var:
import os
os.environ['RAYON_NUM_THREADS'] = '4'  # Use 4 threads before importing sklears
```

### Memory Hierarchy Optimization

#### Cache-Friendly Data Access

```python
import numpy as np
import sklears as skl

# Use C-contiguous arrays for best performance
X = np.random.randn(1000, 50)
X_contiguous = np.ascontiguousarray(X)  # Ensure C-contiguous

# This will be faster than non-contiguous data
model = skl.LinearRegression()
model.fit(X_contiguous, y)
```

## Data Optimization

### Data Types

Choose appropriate data types for your use case:

```python
import numpy as np
import sklears as skl
import time

# Compare float32 vs float64 performance
X_f64 = np.random.randn(5000, 100).astype(np.float64)
X_f32 = X_f64.astype(np.float32)
y = np.random.randn(5000)

# Benchmark float64
start = time.time()
model64 = skl.LinearRegression()
model64.fit(X_f64, y)
time_f64 = time.time() - start

# Benchmark float32 (often faster, slightly less precision)
start = time.time()
model32 = skl.LinearRegression()
model32.fit(X_f32, y.astype(np.float32))
time_f32 = time.time() - start

print(f"Float64 time: {time_f64:.4f}s")
print(f"Float32 time: {time_f32:.4f}s")
print(f"Speedup: {time_f64 / time_f32:.2f}x")
```

### Data Layout

#### Row-major vs Column-major

```python
import numpy as np
import sklears as skl

# C-order (row-major) is preferred for most operations
X_c = np.random.randn(1000, 50)  # C-order by default
X_f = np.asfortranarray(X_c)     # Fortran-order (column-major)

print(f"C-order contiguous: {X_c.flags.c_contiguous}")
print(f"F-order contiguous: {X_f.flags.f_contiguous}")

# C-order is generally faster for most ML operations
```

#### Memory Alignment

```python
import numpy as np

# Ensure proper memory alignment for SIMD
def aligned_array(shape, dtype=np.float64, align=32):
    """Create aligned array for optimal SIMD performance"""
    size = np.prod(shape)
    buf = np.empty(size + align // np.dtype(dtype).itemsize, dtype=dtype)
    offset = (-buf.ctypes.data % align) // np.dtype(dtype).itemsize
    return buf[offset:offset+size].reshape(shape)

# Use aligned arrays for large computations
X_aligned = aligned_array((10000, 100))
X_aligned[:] = np.random.randn(10000, 100)
```

### Preprocessing for Performance

#### Batch Processing

```python
import numpy as np
import sklears as skl

# Process data in optimal batch sizes
def process_in_batches(X, y, batch_size=10000):
    """Process large datasets in memory-efficient batches"""
    n_samples = X.shape[0]
    results = []
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]
        y_batch = y[start:end]
        
        model = skl.LinearRegression()
        model.fit(X_batch, y_batch)
        predictions = model.predict(X_batch)
        results.append(predictions)
    
    return np.concatenate(results)
```

## Algorithm-Specific Tips

### Linear Models

#### Choose the Right Solver

```python
import sklears as skl
import numpy as np

# For large datasets, consider different algorithms
X_large = np.random.randn(50000, 1000)
y_large = np.random.randn(50000)

# Linear regression is fastest for well-conditioned problems
model = skl.LinearRegression()
model.fit(X_large, y_large)

# Ridge regression for ill-conditioned problems
ridge = skl.Ridge(alpha=1.0)
ridge.fit(X_large, y_large)

# Lasso for feature selection
lasso = skl.Lasso(alpha=0.1, max_iter=1000)
lasso.fit(X_large, y_large)
```

#### Regularization Path

```python
# For hyperparameter tuning, compute regularization paths efficiently
alphas = np.logspace(-4, 1, 50)
scores = []

X_train, X_test, y_train, y_test = skl.train_test_split(X, y, test_size=0.2)

for alpha in alphas:
    model = skl.Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

best_alpha = alphas[np.argmax(scores)]
```

### Clustering

#### K-Means Optimization

```python
import sklears as skl
import numpy as np

# Optimize K-Means parameters for performance
X = np.random.randn(10000, 50)

# Use fewer initializations for large datasets
kmeans_fast = skl.KMeans(
    n_clusters=8,
    n_init=3,        # Fewer initializations
    max_iter=100,    # Lower iteration limit
    tol=1e-3,        # Relaxed tolerance
    random_state=42
)

labels = kmeans_fast.fit_predict(X)
```

#### Scaling for Large Datasets

```python
# For very large datasets, consider mini-batch approach
def chunked_kmeans(X, n_clusters, chunk_size=10000):
    """Apply K-Means to large datasets in chunks"""
    n_samples = X.shape[0]
    all_labels = np.empty(n_samples, dtype=int)
    
    # Fit on first chunk to get initial centers
    X_init = X[:chunk_size]
    kmeans = skl.KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_init)
    all_labels[:chunk_size] = kmeans.labels_
    
    # Predict on remaining chunks
    for start in range(chunk_size, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk = X[start:end]
        chunk_labels = kmeans.predict(X_chunk)
        all_labels[start:end] = chunk_labels
    
    return all_labels
```

### Preprocessing

#### Pipeline Optimization

```python
import numpy as np
import sklears as skl

# Combine preprocessing steps for efficiency
X = np.random.randn(10000, 100) * 10 + 5

# NOTE: StandardScaler - Coming Soon (not yet exposed in Sklears)
# Use manual standardization for now:
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_scaled = (X - X_mean) / X_std
# ... more preprocessing steps

# Combined preprocessing (more efficient)
def preprocess_combined(X):
    """Combine multiple preprocessing steps"""
    # Standardization
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std
    
    # Additional steps can be combined here
    return X_scaled, X_mean, X_std

X_processed, mean, std = preprocess_combined(X)
```

## Memory Management

### Memory-Efficient Operations

#### In-place Operations

```python
import numpy as np
import sklears as skl

# NOTE: StandardScaler with copy=False - Coming Soon (not yet exposed)
# Use numpy in-place operations as a workaround:
X = np.random.randn(10000, 100)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8

# In-place standardization (avoids extra allocation)
X -= X_mean
X /= X_std
# X is now standardized in-place
X_scaled_inplace = X
```

#### Memory Monitoring

```python
import psutil
import os

def monitor_memory():
    """Monitor memory usage during computation"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

# Monitor memory during computation
print(f"Initial memory: {monitor_memory():.1f} MB")

X = np.random.randn(50000, 200)
print(f"After data creation: {monitor_memory():.1f} MB")

model = skl.LinearRegression()
model.fit(X, np.random.randn(50000))
print(f"After model fitting: {monitor_memory():.1f} MB")
```

### Large Dataset Strategies

#### Streaming Processing

```python
import numpy as np
import sklears as skl

def fit_model_streaming(data_generator, n_features):
    """Fit model on streaming data"""
    # Initialize with first batch
    X_batch, y_batch = next(data_generator)
    model = skl.LinearRegression()
    model.fit(X_batch, y_batch)
    
    # Update with subsequent batches (conceptual - actual incremental learning may vary)
    for X_batch, y_batch in data_generator:
        # In practice, you might need to accumulate statistics
        # or use online learning algorithms
        pass
    
    return model

# Example data generator
def data_generator(n_batches=10, batch_size=1000, n_features=50):
    for _ in range(n_batches):
        X = np.random.randn(batch_size, n_features)
        y = np.random.randn(batch_size)
        yield X, y

model = fit_model_streaming(data_generator(), 50)
```

## Benchmarking and Profiling

### Performance Measurement

#### Accurate Timing

```python
import time
import numpy as np
import sklears as skl

def benchmark_function(func, *args, n_runs=5, **kwargs):
    """Benchmark a function with multiple runs"""
    times = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min(times),
        'max_time': max(times),
        'result': result
    }

# Benchmark linear regression
X = np.random.randn(5000, 100)
y = np.random.randn(5000)

def fit_predict(X, y):
    model = skl.LinearRegression()
    model.fit(X, y)
    return model.predict(X)

benchmark_result = benchmark_function(fit_predict, X, y)
print(f"Mean time: {benchmark_result['mean_time']:.4f} ± {benchmark_result['std_time']:.4f} seconds")
```

#### Detailed Profiling

```python
import cProfile
import pstats
import sklears as skl
import numpy as np

def profile_code():
    """Profile sklears code to identify bottlenecks"""
    X = np.random.randn(10000, 100)
    y = np.random.randn(10000)

    # Linear regression
    model = skl.LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    # NOTE: StandardScaler - Coming Soon; using numpy workaround
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std

    # Clustering
    kmeans = skl.KMeans(n_clusters=5)
    labels = kmeans.fit_predict(X)

# Profile the code
profiler = cProfile.Profile()
profiler.enable()
profile_code()
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Comparative Benchmarking

```python
import time
import numpy as np
import sklears as skl
from sklearn.linear_model import LinearRegression as SklearnLR

def compare_implementations(dataset_sizes):
    """Compare Sklears vs Scikit-learn performance"""
    results = []
    
    for n_samples, n_features in dataset_sizes:
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Benchmark Sklears
        start = time.perf_counter()
        skl_model = skl.LinearRegression()
        skl_model.fit(X, y)
        skl_pred = skl_model.predict(X)
        skl_time = time.perf_counter() - start
        
        # Benchmark Scikit-learn
        start = time.perf_counter()
        sklearn_model = SklearnLR()
        sklearn_model.fit(X, y)
        sklearn_pred = sklearn_model.predict(X)
        sklearn_time = time.perf_counter() - start
        
        speedup = sklearn_time / skl_time
        
        results.append({
            'dataset_size': (n_samples, n_features),
            'sklears_time': skl_time,
            'sklearn_time': sklearn_time,
            'speedup': speedup
        })
        
        print(f"Size {n_samples}×{n_features}: Sklears {skl_time:.4f}s, "
              f"Sklearn {sklearn_time:.4f}s, Speedup {speedup:.2f}x")
    
    return results

# Run comparison
sizes = [(1000, 10), (5000, 50), (10000, 100)]
comparison_results = compare_implementations(sizes)
```

## Common Performance Pitfalls

### Avoid These Anti-patterns

#### 1. Unnecessary Data Copying

```python
# Bad: Creates unnecessary copies
X_bad = X.copy()
X_bad = np.array(X_bad)

# Good: Minimize copying - use in-place numpy operations
# (NOTE: StandardScaler with copy=False - Coming Soon)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_scaled = (X - X_mean) / X_std  # single allocation
```

#### 2. Wrong Data Types

```python
# Bad: Using object arrays or wrong dtypes
X_bad = np.array([[1, 2.0], [3, 4.0]], dtype=object)

# Good: Use appropriate numeric dtypes
X_good = np.array([[1, 2.0], [3, 4.0]], dtype=np.float64)
```

#### 3. Non-contiguous Arrays

```python
# Bad: Non-contiguous arrays
X_bad = X[:, ::2]  # Creates a view with gaps
print(f"Contiguous: {X_bad.flags.c_contiguous}")

# Good: Ensure contiguity for performance-critical operations
X_good = np.ascontiguousarray(X_bad)
print(f"Contiguous: {X_good.flags.c_contiguous}")
```

#### 4. Inefficient Loops

```python
# Bad: Processing one sample at a time
predictions = []
for i in range(len(X_test)):
    pred = model.predict(X_test[i:i+1])
    predictions.append(pred[0])

# Good: Batch processing
predictions = model.predict(X_test)
```

### Performance Debugging

#### Check Data Properties

```python
def check_data_properties(X, name="Data"):
    """Check data properties that affect performance"""
    print(f"{name} properties:")
    print(f"  Shape: {X.shape}")
    print(f"  Dtype: {X.dtype}")
    print(f"  C-contiguous: {X.flags.c_contiguous}")
    print(f"  F-contiguous: {X.flags.f_contiguous}")
    print(f"  Memory usage: {X.nbytes / 1024 / 1024:.2f} MB")
    print(f"  Min/Max: {X.min():.3f} / {X.max():.3f}")
    
    if X.dtype == np.float64 or X.dtype == np.float32:
        print(f"  Mean/Std: {X.mean():.3f} / {X.std():.3f}")

# Check your data
X = np.random.randn(1000, 50)
check_data_properties(X)
```

#### Performance Checklist

Before optimizing, verify:

1. **Data format**: Contiguous, correct dtype, reasonable size
2. **Hardware**: SIMD available, sufficient memory, appropriate thread count
3. **Algorithm choice**: Right algorithm for your problem size and type
4. **Memory usage**: No unnecessary copies, efficient data structures
5. **Measurement**: Proper benchmarking with multiple runs

```python
import os
import platform

def performance_checklist(X, y=None):
    """Run through performance optimization checklist"""
    print("Performance Optimization Checklist:")
    print("=" * 40)

    # Data checks
    print(f"  Data is contiguous: {X.flags.c_contiguous}")
    print(f"  Data type is numeric: {np.issubdtype(X.dtype, np.number)}")
    print(f"  Data size: {X.nbytes / 1024 / 1024:.1f} MB")

    # NOTE: get_hardware_info() - Coming Soon (not yet available)
    # Use platform info as a fallback:
    cpu_count = os.cpu_count() or 1
    print(f"  Machine arch: {platform.machine()}")
    print(f"  CPU count: {cpu_count}")
    print(f"  Multiple cores available: {cpu_count > 1}")

    # NOTE: get_config() - Coming Soon (not yet available)
    # Thread count via environment variable:
    n_threads = os.environ.get('RAYON_NUM_THREADS', 'auto (all cores)')
    print(f"  RAYON_NUM_THREADS: {n_threads}")

    print("\nRecommendations:")
    if not X.flags.c_contiguous:
        print("- Make data contiguous with np.ascontiguousarray()")
    if X.dtype not in [np.float32, np.float64]:
        print(f"- Consider using float32/float64 instead of {X.dtype}")
    if X.nbytes > 1024 * 1024 * 1024:  # > 1GB
        print("- Consider batch processing for large datasets")

# Run checklist
X = np.random.randn(5000, 100)
performance_checklist(X)
```

By following these optimization guidelines, you can achieve maximum performance from Sklears and get the most out of your machine learning workflows.