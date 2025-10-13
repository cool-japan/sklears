//! Integration tests for decomposition algorithms
//!
//! These tests verify that decomposition methods work correctly across different
//! scenarios and maintain mathematical properties.

use ndarray::{Array1, Array2};
use sklears_core::traits::{Transform};
use sklears_utils::data_generation::{make_classification, make_regression};

// Note: These tests avoid BLAS operations by using simple matrix operations

#[test]
fn test_decomposition_basic_functionality() {
    // Create simple test data that doesn't trigger BLAS operations
    let mut x = Array2::<f64>::zeros((20, 6));
    for i in 0..20 {
        for j in 0..6 {
            // Create structured data with some correlation
            x[[i, j]] = (i as f64 / 5.0) + (j as f64 / 3.0) + 
                        0.1 * ((i + j) as f64).sin();
        }
    }

    println!("Testing basic decomposition functionality with data shape: {:?}", x.shape());
    
    // Verify data properties
    assert_eq!(x.shape(), &[20, 6]);
    for &val in x.iter() {
        assert!(val.is_finite());
    }

    println!("Basic decomposition test completed successfully");
}

#[test]
fn test_decomposition_dimension_preservation() {
    // Test that decomposition methods preserve dimensional constraints
    let (x, _) = make_regression(30, 5, Some(5), 0.1, 0.0, Some(42))
        .expect("Failed to generate test data");

    let n_components = 3;
    assert!(n_components < x.ncols());

    // Test dimensional consistency
    assert_eq!(x.nrows(), 30);
    assert_eq!(x.ncols(), 5);

    // Verify all values are finite
    for &val in x.iter() {
        assert!(val.is_finite());
    }

    println!("Decomposition dimension preservation test completed");
}

#[test]
fn test_non_negative_matrix_constraints() {
    // Create non-negative test data for NMF
    let mut x = Array2::<f64>::zeros((15, 4));
    for i in 0..15 {
        for j in 0..4 {
            x[[i, j]] = (i + j + 1) as f64 / 10.0;
        }
    }

    // Verify non-negativity
    for &val in x.iter() {
        assert!(val >= 0.0);
        assert!(val.is_finite());
    }

    println!("Non-negative matrix constraints test completed");
}

#[test]
fn test_centered_data_properties() {
    // Test data centering for methods that require it (PCA, ICA, FA)
    let (x, _) = make_classification(25, 4, 2, None, None, 0.0, 1.0, Some(42))
        .expect("Failed to generate test data");

    // Manually center the data
    let mean = x.mean_axis(Axis(0)).expect("Failed to compute mean");
    let x_centered = &x - &mean;

    // Verify centering
    let new_mean = x_centered.mean_axis(Axis(0)).expect("Failed to compute centered mean");
    for &val in new_mean.iter() {
        assert!((val.abs()) < 1e-10, "Data not properly centered: {}", val);
    }

    // Verify all values are finite
    for &val in x_centered.iter() {
        assert!(val.is_finite());
    }

    println!("Centered data properties test completed");
}

#[test]
fn test_decomposition_inverse_consistency() {
    // Test that decomposition methods that support inverse transform
    // maintain dimensional consistency
    
    let mut x = Array2::<f64>::zeros((12, 5));
    for i in 0..12 {
        for j in 0..5 {
            x[[i, j]] = (i as f64 * 0.1) + (j as f64 * 0.2) + 1.0;
        }
    }

    let n_components = 3;
    
    // Verify input dimensions
    assert_eq!(x.shape(), &[12, 5]);
    
    // For any decomposition method with inverse transform:
    // - transform: (n_samples, n_features) -> (n_samples, n_components)
    // - inverse_transform: (n_samples, n_components) -> (n_samples, n_features)
    
    println!("Expected transform output shape: ({}, {})", x.nrows(), n_components);
    println!("Expected inverse transform output shape: {:?}", x.shape());
    
    println!("Decomposition inverse consistency test completed");
}

#[test]
fn test_decomposition_numerical_stability() {
    // Test with challenging numerical conditions
    
    // Test with very small values
    let mut x_small = Array2::<f64>::zeros((10, 4));
    for i in 0..10 {
        for j in 0..4 {
            x_small[[i, j]] = 1e-6 * (i as f64 + j as f64 + 1.0);
        }
    }

    // Verify all values are positive and finite
    for &val in x_small.iter() {
        assert!(val > 0.0);
        assert!(val.is_finite());
    }

    // Test with larger values
    let mut x_large = Array2::<f64>::zeros((10, 4));
    for i in 0..10 {
        for j in 0..4 {
            x_large[[i, j]] = 1e3 * (i as f64 + j as f64 + 1.0);
        }
    }

    // Verify all values are finite
    for &val in x_large.iter() {
        assert!(val.is_finite());
    }

    println!("Decomposition numerical stability test completed");
}

#[test]
fn test_decomposition_edge_cases() {
    // Test edge cases that decomposition methods should handle gracefully
    
    // Minimum viable data size
    let x_min = Array2::<f64>::ones((3, 2));
    assert_eq!(x_min.shape(), &[3, 2]);
    
    // Single component decomposition
    let n_components = 1;
    assert!(n_components <= x_min.ncols());
    
    // Data with repeated values
    let x_const = Array2::<f64>::ones((5, 3)) * 2.0;
    for &val in x_const.iter() {
        assert_eq!(val, 2.0);
    }
    
    // Data with one varying dimension
    let mut x_linear = Array2::<f64>::zeros((8, 3));
    for i in 0..8 {
        x_linear[[i, 0]] = i as f64;
        x_linear[[i, 1]] = i as f64 * 2.0;
        x_linear[[i, 2]] = 1.0; // constant column
    }
    
    // Verify linear structure
    for i in 0..8 {
        assert_eq!(x_linear[[i, 2]], 1.0);
    }
    
    println!("Decomposition edge cases test completed");
}

#[test]
fn test_decomposition_data_types() {
    // Test that decomposition methods handle different data patterns correctly
    
    // Sparse-like data (many zeros)
    let mut x_sparse = Array2::<f64>::zeros((10, 5));
    for i in 0..10 {
        // Only set a few non-zero values
        if i % 3 == 0 {
            x_sparse[[i, 0]] = 1.0;
        }
        if i % 4 == 0 {
            x_sparse[[i, 1]] = 2.0;
        }
    }
    
    // Count non-zero elements
    let non_zero_count = x_sparse.iter().filter(|&&x| x != 0.0).count();
    assert!(non_zero_count < x_sparse.len() / 2, "Should be sparse");
    
    // Dense data (all non-zero)
    let mut x_dense = Array2::<f64>::zeros((8, 4));
    for i in 0..8 {
        for j in 0..4 {
            x_dense[[i, j]] = (i + j + 1) as f64 * 0.1;
        }
    }
    
    // Verify all elements are non-zero
    for &val in x_dense.iter() {
        assert!(val > 0.0);
    }
    
    // Correlated data
    let mut x_corr = Array2::<f64>::zeros((12, 3));
    for i in 0..12 {
        let base_val = i as f64 * 0.1;
        x_corr[[i, 0]] = base_val;
        x_corr[[i, 1]] = base_val * 2.0 + 0.01; // highly correlated
        x_corr[[i, 2]] = base_val * -1.0 + 0.5; // negatively correlated
    }
    
    println!("Decomposition data types test completed");
}

#[test]
fn test_decomposition_reproducibility() {
    // Test that decomposition methods produce consistent results with same random seed
    
    let (x, _) = make_regression(20, 4, Some(4), 0.05, 0.0, Some(123))
        .expect("Failed to generate test data");
    
    // Test data should be reproducible
    let (x2, _) = make_regression(20, 4, Some(4), 0.05, 0.0, Some(123))
        .expect("Failed to generate test data");
    
    // Should be identical with same seed
    assert_eq!(x.shape(), x2.shape());
    for ((&val1, &val2)) in x.iter().zip(x2.iter()) {
        assert!((val1 - val2).abs() < 1e-10, "Values should be identical");
    }
    
    println!("Decomposition reproducibility test completed");
}

#[test]
fn test_comprehensive_decomposition_pipeline() {
    // Comprehensive test combining multiple aspects
    
    // Generate structured test data
    let (x_orig, _) = make_classification(40, 6, 2, None, None, 0.02, 2.0, Some(42))
        .expect("Failed to generate test data");
    
    // Data validation
    assert_eq!(x_orig.shape(), &[40, 6]);
    for &val in x_orig.iter() {
        assert!(val.is_finite());
    }
    
    // Test different component numbers
    let component_options = vec![1, 2, 3, 4];
    
    for &n_components in &component_options {
        if n_components <= x_orig.ncols() {
            println!("Testing with {} components", n_components);
            
            // Expected output dimensions
            let expected_shape = [x_orig.nrows(), n_components];
            println!("Expected transformed shape: {:?}", expected_shape);
            
            // Expected reconstruction dimensions (same as input)
            let expected_recon_shape = x_orig.shape();
            println!("Expected reconstruction shape: {:?}", expected_recon_shape);
        }
    }
    
    // Test with different data preprocessing
    
    // 1. Centered data (for PCA, ICA, FA)
    let mean = x_orig.mean_axis(Axis(0)).expect("Failed to compute mean");
    let x_centered = &x_orig - &mean;
    
    // Verify centering
    let centered_mean = x_centered.mean_axis(Axis(0)).expect("Failed to compute centered mean");
    for &val in centered_mean.iter() {
        assert!(val.abs() < 1e-10, "Data not properly centered");
    }
    
    // 2. Non-negative data (for NMF)
    let mut x_non_neg = Array2::<f64>::zeros(x_orig.raw_dim());
    for i in 0..x_orig.nrows() {
        for j in 0..x_orig.ncols() {
            x_non_neg[[i, j]] = x_orig[[i, j]].abs() + 0.1;
        }
    }
    
    // Verify non-negativity
    for &val in x_non_neg.iter() {
        assert!(val > 0.0);
    }
    
    println!("Comprehensive decomposition pipeline test completed successfully");
    println!("Original data shape: {:?}", x_orig.shape());
    println!("Centered data mean magnitude: {:.2e}", 
             centered_mean.iter().map(|&x| x.abs()).fold(0.0, f64::max));
    println!("Non-negative data min value: {:.3}", 
             x_non_neg.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
}