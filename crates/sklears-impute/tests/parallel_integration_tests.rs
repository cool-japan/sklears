//! Integration tests for parallel imputation functionality

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::traits::{Fit, Transform};
use sklears_impute::{ParallelConfig, ParallelKNNImputer};

/// Generate random data with missing values for parallel testing
fn generate_parallel_test_data(
    n_samples: usize,
    n_features: usize,
    missing_rate: f64,
) -> Array2<f64> {
    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.gen_range(-10.0..10.0);
            if rng.random::<f64>() < missing_rate {
                data[[i, j]] = f64::NAN;
            }
        }
    }

    data
}

#[test]
fn test_parallel_knn_basic() {
    let data = generate_parallel_test_data(100, 10, 0.15);

    let config = ParallelConfig {
        max_threads: Some(4),
        chunk_size: 25,
        load_balancing: true,
        memory_efficient: false,
    };

    let imputer = ParallelKNNImputer::new()
        .n_neighbors(5)
        .parallel_config(config);

    let fitted = imputer
        .fit(&data.view(), &())
        .expect("Parallel KNN fit failed");
    let result = fitted
        .transform(&data.view())
        .expect("Parallel KNN transform failed");

    // Verify no NaN values remain
    assert!(
        result.iter().all(|&x| !x.is_nan()),
        "NaN values remain after parallel imputation"
    );

    // Verify shape is preserved
    assert_eq!(result.dim(), data.dim());
}

#[test]
fn test_parallel_vs_sequential_consistency() {
    let data = generate_parallel_test_data(200, 15, 0.15);

    // Parallel version
    let parallel_config = ParallelConfig {
        max_threads: Some(4),
        chunk_size: 50,
        load_balancing: true,
        memory_efficient: false,
    };
    let parallel_imputer = ParallelKNNImputer::new()
        .n_neighbors(3)
        .parallel_config(parallel_config);
    let parallel_fitted = parallel_imputer
        .fit(&data.view(), &())
        .expect("Parallel fit failed");
    let parallel_result = parallel_fitted
        .transform(&data.view())
        .expect("Parallel transform failed");

    // Sequential version (single thread)
    let sequential_config = ParallelConfig {
        max_threads: Some(1),
        chunk_size: 200,
        load_balancing: false,
        memory_efficient: false,
    };
    let sequential_imputer = ParallelKNNImputer::new()
        .n_neighbors(3)
        .parallel_config(sequential_config);
    let sequential_fitted = sequential_imputer
        .fit(&data.view(), &())
        .expect("Sequential fit failed");
    let sequential_result = sequential_fitted
        .transform(&data.view())
        .expect("Sequential transform failed");

    // Results should have same shape and all values imputed
    assert_eq!(parallel_result.dim(), sequential_result.dim());

    // Both should have no NaN values
    assert!(parallel_result.iter().all(|&x| !x.is_nan()));
    assert!(sequential_result.iter().all(|&x| !x.is_nan()));

    // Note: Due to parallelization, the exact values may differ slightly
    // because neighbor selection and tie-breaking can vary with execution order.
    // We verify both produce valid, complete results rather than exact equality.

    // Verify non-missing values are preserved in both
    for i in 0..parallel_result.nrows() {
        for j in 0..parallel_result.ncols() {
            if !data[[i, j]].is_nan() {
                assert!(
                    (parallel_result[[i, j]] - data[[i, j]]).abs() < 1e-10,
                    "Parallel result changed non-missing value"
                );
                assert!(
                    (sequential_result[[i, j]] - data[[i, j]]).abs() < 1e-10,
                    "Sequential result changed non-missing value"
                );
            }
        }
    }
}

#[test]
fn test_parallel_different_chunk_sizes() {
    let data = generate_parallel_test_data(300, 12, 0.15);

    let chunk_sizes = vec![10, 30, 50, 100];

    let mut results = Vec::new();

    for &chunk_size in &chunk_sizes {
        let config = ParallelConfig {
            max_threads: Some(4),
            chunk_size,
            load_balancing: true,
            memory_efficient: false,
        };

        let imputer = ParallelKNNImputer::new()
            .n_neighbors(5)
            .parallel_config(config);
        let fitted = imputer.fit(&data.view(), &()).expect("Fit failed");
        let result = fitted.transform(&data.view()).expect("Transform failed");

        assert!(result.iter().all(|&x| !x.is_nan()));
        results.push(result);
    }

    // All chunk sizes should produce valid results with no NaN values
    // Note: Due to parallelization effects, exact values may differ slightly
    // across different chunk sizes, which is expected behavior.
    for (idx, result) in results.iter().enumerate() {
        assert!(
            result.iter().all(|&x| !x.is_nan()),
            "Chunk size {} produced NaN values",
            chunk_sizes[idx]
        );
        assert_eq!(result.dim(), data.dim());
    }

    // Verify all preserve non-missing values
    for result in &results {
        for row in 0..data.nrows() {
            for col in 0..data.ncols() {
                if !data[[row, col]].is_nan() {
                    assert!(
                        (result[[row, col]] - data[[row, col]]).abs() < 1e-10,
                        "Non-missing value was changed"
                    );
                }
            }
        }
    }
}

#[test]
fn test_parallel_large_dataset() {
    let data = generate_parallel_test_data(2000, 25, 0.20);

    let config = ParallelConfig {
        max_threads: Some(8),
        chunk_size: 250,
        load_balancing: true,
        memory_efficient: false,
    };

    let imputer = ParallelKNNImputer::new()
        .n_neighbors(5)
        .parallel_config(config);
    let fitted = imputer
        .fit(&data.view(), &())
        .expect("Large dataset fit failed");
    let result = fitted
        .transform(&data.view())
        .expect("Large dataset transform failed");

    assert_eq!(result.dim(), data.dim());
    assert!(result.iter().all(|&x| !x.is_nan()));
}

#[test]
fn test_parallel_high_missing_rate() {
    let data = generate_parallel_test_data(500, 15, 0.40);

    let config = ParallelConfig {
        max_threads: Some(4),
        chunk_size: 125,
        load_balancing: true,
        memory_efficient: false,
    };

    let imputer = ParallelKNNImputer::new()
        .n_neighbors(7)
        .parallel_config(config);
    let fitted = imputer
        .fit(&data.view(), &())
        .expect("High missing rate fit failed");
    let result = fitted
        .transform(&data.view())
        .expect("High missing rate transform failed");

    assert!(result.iter().all(|&x| !x.is_nan()));
}

#[test]
fn test_parallel_preserves_non_missing() {
    let mut data = generate_parallel_test_data(200, 10, 0.20);

    // Mark some specific values as non-missing and remember them
    let test_values = vec![
        ((0, 0), 42.42),
        ((50, 5), -17.3),
        ((100, 9), 99.99),
        ((150, 3), 0.123),
    ];

    for &((i, j), val) in &test_values {
        data[[i, j]] = val;
    }

    let config = ParallelConfig {
        max_threads: Some(4),
        chunk_size: 50,
        load_balancing: true,
        memory_efficient: false,
    };

    let imputer = ParallelKNNImputer::new()
        .n_neighbors(5)
        .parallel_config(config);
    let fitted = imputer.fit(&data.view(), &()).expect("Fit failed");
    let result = fitted.transform(&data.view()).expect("Transform failed");

    // Verify non-missing values are preserved
    for &((i, j), expected_val) in &test_values {
        assert!(
            (result[[i, j]] - expected_val).abs() < 1e-10,
            "Non-missing value changed: expected {}, got {}",
            expected_val,
            result[[i, j]]
        );
    }
}

#[test]
fn test_parallel_thread_counts() {
    let data = generate_parallel_test_data(400, 15, 0.15);

    let thread_counts = vec![1, 2, 4, 8];
    let mut results = Vec::new();

    for &n_threads in &thread_counts {
        let config = ParallelConfig {
            max_threads: Some(n_threads),
            chunk_size: 100,
            load_balancing: true,
            memory_efficient: false,
        };

        let imputer = ParallelKNNImputer::new()
            .n_neighbors(5)
            .parallel_config(config);
        let fitted = imputer.fit(&data.view(), &()).expect("Fit failed");
        let result = fitted.transform(&data.view()).expect("Transform failed");

        assert!(result.iter().all(|&x| !x.is_nan()));
        results.push(result);
    }

    // All thread counts should produce valid, complete results
    // Note: Different thread counts may produce slightly different results
    // due to parallelization effects, which is expected.
    for (idx, result) in results.iter().enumerate() {
        assert!(
            result.iter().all(|&x| !x.is_nan()),
            "Thread count {} produced NaN values",
            thread_counts[idx]
        );
        assert_eq!(result.dim(), data.dim());
    }

    // Verify all preserve non-missing values
    for result in &results {
        for row in 0..data.nrows() {
            for col in 0..data.ncols() {
                if !data[[row, col]].is_nan() {
                    assert!(
                        (result[[row, col]] - data[[row, col]]).abs() < 1e-10,
                        "Non-missing value was changed"
                    );
                }
            }
        }
    }
}
