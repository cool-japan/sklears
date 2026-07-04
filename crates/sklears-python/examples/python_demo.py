#!/usr/bin/env python3
"""
Demo script for sklears Python bindings

This script demonstrates the basic usage of sklears through PyO3 bindings,
showing how to use the library as a drop-in replacement for scikit-learn
with significant performance improvements.

Requirements:
- Build the sklears-python crate with: maturin develop
- Install numpy: pip install numpy
"""

import numpy as np
import sklears as skl
from sklearn import datasets
import time

def demo_linear_regression():
    """Demonstrate linear regression with performance comparison"""
    print("=== Linear Regression Demo ===")

    # Generate sample data
    X, y = datasets.make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = skl.train_test_split(X, y, test_size=0.2, random_state=42)

    # Train sklears model
    start_time = time.time()
    model = skl.LinearRegression()
    model.fit(X_train, y_train)
    sklears_predictions = model.predict(X_test)
    sklears_time = time.time() - start_time

    # Calculate metrics using numpy (skl.metrics not yet exposed)
    ss_res = np.sum((y_test - sklears_predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    sklears_mse = ss_res / len(y_test)
    sklears_r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    print(f"Sklears - Time: {sklears_time:.4f}s, MSE: {sklears_mse:.4f}, R2: {sklears_r2:.4f}")

    # Compare with scikit-learn if available
    try:
        from sklearn.linear_model import LinearRegression as SklearnLR
        from sklearn.metrics import mean_squared_error, r2_score

        start_time = time.time()
        sklearn_model = SklearnLR()
        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict(X_test)
        sklearn_time = time.time() - start_time

        sklearn_mse = mean_squared_error(y_test, sklearn_predictions)
        sklearn_r2 = r2_score(y_test, sklearn_predictions)

        print(f"Sklearn - Time: {sklearn_time:.4f}s, MSE: {sklearn_mse:.4f}, R2: {sklearn_r2:.4f}")
        print(f"Speedup: {sklearn_time / sklears_time:.2f}x")

    except ImportError:
        print("Scikit-learn not available for comparison")

    print()

def demo_clustering():
    """Demonstrate clustering algorithms"""
    print("=== Clustering Demo ===")

    # Generate sample data
    X, _ = datasets.make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

    # K-Means clustering
    start_time = time.time()
    kmeans = skl.KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    kmeans_time = time.time() - start_time

    print(f"K-Means - Time: {kmeans_time:.4f}s, Clusters found: {len(np.unique(labels))}")

    # DBSCAN clustering
    start_time = time.time()
    dbscan = skl.DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    dbscan_time = time.time() - start_time

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"DBSCAN - Time: {dbscan_time:.4f}s, Clusters: {n_clusters}, Noise points: {n_noise}")
    print()

def demo_preprocessing():
    """Demonstrate preprocessing utilities"""
    print("=== Preprocessing Demo ===")

    X = np.random.randn(100, 5) * 10 + 5

    # Standard scaling
    scaler = skl.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Original data - Mean: {X.mean():.2f}, Std: {X.std():.2f}")
    print(f"Scaled data - Mean: {X_scaled.mean():.2f}, Std: {X_scaled.std():.2f}")

    # Min-Max scaling
    minmax_scaler = skl.MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    print(f"MinMax scaled - Min: {X_minmax.min():.2f}, Max: {X_minmax.max():.2f}")

    # Label encoding
    labels = ['cat', 'dog', 'cat', 'fish', 'dog', 'fish']
    encoder = skl.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    print(f"Original labels: {labels}")
    print(f"Encoded labels: {encoded_labels}")
    print(f"Classes: {encoder.classes_}")
    print()

def demo_model_selection():
    """Demonstrate model selection utilities"""
    print("=== Model Selection Demo ===")

    # Generate sample data
    X, y = datasets.make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)

    # Train-test split. train_test_split's PyO3 signature requires exact-dtype
    # PyReadonlyArray1<f64> for y; make_classification() returns int64 labels
    # on 64-bit platforms, so the dtype must be made explicit here (same
    # reasoning as the accuracy_score dtype note in demo_metrics() below).
    X_train, X_test, y_train, y_test = skl.train_test_split(
        X, y.astype(np.float64), test_size=0.3, random_state=42
    )

    print(f"Dataset split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Cross-validation
    kfold = skl.KFold(n_splits=5, shuffle=True, random_state=42)
    splits = kfold.split(X, y)

    print(f"K-Fold cross-validation with {kfold.get_n_splits()} splits")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")

    print()

def demo_metrics():
    """Demonstrate metrics functions"""
    print("=== Metrics Demo ===")

    # accuracy_score's PyO3 signature requires exact-dtype PyReadonlyArray1<i32>;
    # NumPy defaults to int64 on 64-bit platforms, so the dtype must be explicit.
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int32)
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0], dtype=np.int32)

    accuracy = skl.accuracy_score(y_true, y_pred)
    print(f"Classification Accuracy: {accuracy:.3f}")

    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])

    mse = skl.mean_squared_error(y_true_reg, y_pred_reg)
    r2 = skl.r2_score(y_true_reg, y_pred_reg)

    print(f"Regression MSE: {mse:.3f}")
    print(f"Regression R2: {r2:.3f}")
    print()

def demo_system_info():
    """Demonstrate system information utilities"""
    print("=== System Information ===")

    print(f"Sklears version: {skl.get_version()}")

    # Build information
    build_info = skl.get_build_info()
    print("Build information:")
    for key, value in build_info.items():
        print(f"  {key}: {value}")

    hardware_info = skl.get_hardware_info()
    print("Hardware capabilities:")
    for key, value in hardware_info.items():
        print(f"  {key}: {value}")

    print("Running basic performance benchmarks...")
    benchmarks = skl.benchmark_basic_operations()
    print("Benchmark results:")
    for operation, time_ms in benchmarks.items():
        print(f"  {operation}: {time_ms:.2f}")

    print()

def main():
    """Run all demos"""
    print("Sklears Python Bindings Demo")
    print("=" * 40)
    print()

    try:
        demo_linear_regression()
        demo_clustering()
        demo_preprocessing()
        demo_model_selection()
        demo_metrics()
        demo_system_info()

        print("All demos completed successfully!")

    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
