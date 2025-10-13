//! SIMD operations tests
//!
//! This module contains tests for SIMD-accelerated operations including
//! vectorized dot products, vector addition, matrix-vector multiplication,
//! distance calculations, and similarity measures.

use crate::simd_ops;
use scirs2_core::ndarray::{Array1, Array2};

#[test]
fn test_simd_dot_product() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    let result = simd_ops::simd_dot_product(&a.view(), &b.view()).unwrap();

    // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8 + 8*9 = 240
    let expected = 240.0;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_add_vectors() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

    let result = simd_ops::simd_add_vectors(&a.view(), &b.view()).unwrap();

    let expected = vec![6.0, 8.0, 10.0, 12.0];
    assert_eq!(result.len(), expected.len());

    for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-10,
            "Mismatch at index {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_simd_matrix_vector_multiply() {
    let matrix_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let matrix = Array2::from_shape_vec((3, 3), matrix_data).unwrap();
    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    let result = simd_ops::simd_matrix_vector_multiply(&matrix.view(), &vector.view()).unwrap();

    // Expected: [1*1+2*2+3*3, 4*1+5*2+6*3, 7*1+8*2+9*3] = [14, 32, 50]
    let expected = vec![14.0, 32.0, 50.0];
    assert_eq!(result.len(), expected.len());

    for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-10,
            "Mismatch at index {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_simd_euclidean_distance() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 6.0, 8.0]);

    let result = simd_ops::simd_euclidean_distance(&a.view(), &b.view()).unwrap();

    // Expected: sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
    let expected = (50.0_f64).sqrt();
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_cosine_similarity() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![2.0, 4.0, 6.0]); // b is 2*a, so cosine similarity should be 1.0

    let result = simd_ops::simd_cosine_similarity(&a.view(), &b.view()).unwrap();

    assert!((result - 1.0).abs() < 1e-10);

    // Test with orthogonal vectors
    let c = Array1::from_vec(vec![1.0, 0.0]);
    let d = Array1::from_vec(vec![0.0, 1.0]);

    let result_orthogonal = simd_ops::simd_cosine_similarity(&c.view(), &d.view()).unwrap();
    assert!(result_orthogonal.abs() < 1e-10); // Should be close to 0
}

#[test]
fn test_simd_error_cases() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![1.0, 2.0]); // Different length

    // These should all fail due to length mismatch
    assert!(simd_ops::simd_dot_product(&a.view(), &b.view()).is_err());
    assert!(simd_ops::simd_add_vectors(&a.view(), &b.view()).is_err());
    assert!(simd_ops::simd_euclidean_distance(&a.view(), &b.view()).is_err());
    assert!(simd_ops::simd_cosine_similarity(&a.view(), &b.view()).is_err());

    // Test matrix-vector multiplication with wrong dimensions
    let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Wrong size
    assert!(simd_ops::simd_matrix_vector_multiply(&matrix.view(), &vector.view()).is_err());
}

#[test]
fn test_simd_subtract_vectors() {
    let a = Array1::from_vec(vec![10.0, 8.0, 6.0, 4.0]);
    let b = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

    let result = simd_ops::simd_subtract_vectors(&a.view(), &b.view()).unwrap();

    let expected = vec![9.0, 6.0, 3.0, 0.0];
    assert_eq!(result.len(), expected.len());

    for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-10,
            "Mismatch at index {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_simd_multiply_vectors() {
    let a = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

    let result = simd_ops::simd_multiply_vectors(&a.view(), &b.view()).unwrap();

    let expected = vec![2.0, 6.0, 12.0, 20.0];
    assert_eq!(result.len(), expected.len());

    for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-10,
            "Mismatch at index {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_simd_vector_norm() {
    let vector = Array1::from_vec(vec![3.0, 4.0]); // 3-4-5 triangle

    let l2_norm = simd_ops::simd_vector_norm(&vector.view(), 2).unwrap();
    assert!((l2_norm - 5.0).abs() < 1e-10);

    let l1_norm = simd_ops::simd_vector_norm(&vector.view(), 1).unwrap();
    assert!((l1_norm - 7.0).abs() < 1e-10);

    // Test with larger vector
    let large_vector = Array1::from_vec(vec![1.0; 100]);
    let l2_norm_large = simd_ops::simd_vector_norm(&large_vector.view(), 2).unwrap();
    assert!((l2_norm_large - 10.0).abs() < 1e-10); // sqrt(100) = 10
}

#[test]
fn test_simd_manhattan_distance() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 6.0, 8.0]);

    let result = simd_ops::simd_manhattan_distance(&a.view(), &b.view()).unwrap();

    // Expected: |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
    let expected = 12.0;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_squared_euclidean_distance() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 6.0, 8.0]);

    let result = simd_ops::simd_squared_euclidean_distance(&a.view(), &b.view()).unwrap();

    // Expected: (4-1)^2 + (6-2)^2 + (8-3)^2 = 9 + 16 + 25 = 50
    let expected = 50.0;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_batch_dot_product() {
    let vectors_vec = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let vectors =
        Array2::from_shape_vec((3, 3), vectors_vec.into_iter().flatten().collect()).unwrap();
    let query = Array1::from_vec(vec![1.0, 1.0, 1.0]);

    let results = simd_ops::simd_batch_dot_product(&vectors.view(), &query.view()).unwrap();

    let expected = vec![6.0, 15.0, 24.0]; // 1+2+3, 4+5+6, 7+8+9
    assert_eq!(results.len(), expected.len());

    for (i, (&res, &exp)) in results.iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-10,
            "Mismatch at index {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_simd_matrix_multiply() {
    let a_vec = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b_vec = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let a = Array2::from_shape_vec((2, 2), a_vec.into_iter().flatten().collect()).unwrap();
    let b = Array2::from_shape_vec((2, 2), b_vec.into_iter().flatten().collect()).unwrap();

    let result = simd_ops::simd_matrix_multiply(&a.view(), &b.view()).unwrap();

    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let expected = Array2::from_shape_vec((2, 2), vec![19.0, 22.0, 43.0, 50.0]).unwrap();

    assert_eq!(result.dim(), expected.dim());
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(
                (result[[i, j]] - expected[[i, j]]).abs() < 1e-10,
                "Mismatch at [{}, {}]: {} != {}",
                i,
                j,
                result[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

#[test]
fn test_simd_vector_sum() {
    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let result = simd_ops::simd_vector_sum(&vector.view()).unwrap();

    let expected = 15.0; // 1+2+3+4+5
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_vector_mean() {
    let vector = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let result = simd_ops::simd_vector_mean(&vector.view()).unwrap();

    let expected = 5.0; // (2+4+6+8)/4
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_vector_variance() {
    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let result = simd_ops::simd_vector_variance(&vector.view()).unwrap();

    // Mean = 3.0, variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4 = 2.5
    let expected = 2.5;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_performance() {
    // Test with larger vectors to ensure SIMD benefits
    let size = 10000;
    let a = Array1::from_vec((0..size).map(|i| i as f64).collect());
    let b = Array1::from_vec((0..size).map(|i| (i + 1) as f64).collect());

    let start = std::time::Instant::now();
    let result = simd_ops::simd_dot_product(&a.view(), &b.view()).unwrap();
    let duration = start.elapsed();

    // Should complete quickly
    assert!(duration.as_millis() < 100);

    // Verify result is correct
    let expected: f64 = (0..size).map(|i| i as f64 * (i + 1) as f64).sum();
    assert!((result - expected).abs() < 1e-6); // Allow for some floating point error
}

#[test]
fn test_simd_edge_cases() {
    // Test with empty vectors
    let empty = Array1::from_vec(vec![]);
    assert!(simd_ops::simd_vector_sum(&empty.view()).is_err());
    assert!(simd_ops::simd_vector_mean(&empty.view()).is_err());

    // Test with single element
    let single = Array1::from_vec(vec![42.0]);
    assert!((simd_ops::simd_vector_sum(&single.view()).unwrap() - 42.0).abs() < 1e-10);
    assert!((simd_ops::simd_vector_mean(&single.view()).unwrap() - 42.0).abs() < 1e-10);

    // Test with very small numbers
    let small = Array1::from_vec(vec![1e-100, 2e-100, 3e-100]);
    let result = simd_ops::simd_vector_sum(&small.view()).unwrap();
    assert!(result > 0.0);
    assert!(result.is_finite());

    // Test with very large numbers
    let large = Array1::from_vec(vec![1e50, 2e50, 3e50]);
    let result = simd_ops::simd_vector_sum(&large.view()).unwrap();
    assert!(result > 0.0);
    assert!(result.is_finite());
}

#[test]
fn test_simd_numerical_stability() {
    // Test with values that might cause numerical issues
    let a = Array1::from_vec(vec![1e10, 1e-10, 1e10, 1e-10]);
    let b = Array1::from_vec(vec![1e-10, 1e10, 1e-10, 1e10]);

    let dot_result = simd_ops::simd_dot_product(&a.view(), &b.view()).unwrap();
    assert!(dot_result.is_finite());

    let add_result = simd_ops::simd_add_vectors(&a.view(), &b.view()).unwrap();
    for &val in add_result.iter() {
        assert!(val.is_finite());
    }

    let distance_result = simd_ops::simd_euclidean_distance(&a.view(), &b.view()).unwrap();
    assert!(distance_result.is_finite());
    assert!(distance_result >= 0.0);
}

#[test]
fn test_simd_consistency_with_scalar() {
    // Compare SIMD results with scalar implementations for consistency
    let a = Array1::from_vec(vec![1.5, 2.7, 3.1, 4.8, 5.2]);
    let b = Array1::from_vec(vec![2.1, 1.3, 4.6, 3.9, 2.7]);

    // SIMD dot product
    let simd_dot = simd_ops::simd_dot_product(&a.view(), &b.view()).unwrap();

    // Scalar dot product
    let scalar_dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    assert!((simd_dot - scalar_dot).abs() < 1e-10);

    // SIMD euclidean distance
    let simd_dist = simd_ops::simd_euclidean_distance(&a.view(), &b.view()).unwrap();

    // Scalar euclidean distance
    let scalar_dist: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt();

    assert!((simd_dist - scalar_dist).abs() < 1e-10);
}

#[test]
fn test_simd_different_alignments() {
    // Test with vectors of different sizes to ensure alignment handling
    let sizes = vec![1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100, 128, 1000];

    for size in sizes {
        let a = Array1::from_vec((0..size).map(|i| i as f64).collect());
        let b = Array1::from_vec((0..size).map(|i| (i + 1) as f64).collect());

        let dot_result = simd_ops::simd_dot_product(&a.view(), &b.view()).unwrap();
        assert!(dot_result.is_finite());

        let add_result = simd_ops::simd_add_vectors(&a.view(), &b.view()).unwrap();
        assert_eq!(add_result.len(), size);
        for &val in add_result.iter() {
            assert!(val.is_finite());
        }

        let norm_result = simd_ops::simd_vector_norm(&a.view(), 2).unwrap();
        assert!(norm_result.is_finite());
        assert!(norm_result >= 0.0);
    }
}
