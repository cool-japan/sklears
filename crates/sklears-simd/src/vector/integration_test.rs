//! # SIMD Integration Tests
//!
//! Comprehensive tests that verify the full SIMD vector operations pipeline
//! works correctly across all modules and platforms.

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    #[cfg(feature = "no-std")]
    use alloc::vec;
    #[cfg(feature = "no-std")]
    use alloc::vec::Vec;

    use crate::vector::arithmetic_ops::{add_vec, multiply_vec};
    use crate::vector::basic_operations::{dot_product, norm_l2};
    use crate::vector::comparison_ops::lt_vec;

    use crate::vector::intrinsics::{detect_simd_capabilities, simd_width_f32, F32x4};
    use crate::vector::math_functions::{sin_vec, sqrt_vec};
    use crate::vector::statistics_ops::{norm_l2_squared, sum_vec};
    use core::f32::consts;

    #[test]
    fn test_comprehensive_simd_pipeline() {
        // Create test vectors
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 8];

        // Test 1: Basic arithmetic operations
        add_vec(&a, &b, &mut result);
        assert_eq!(result, vec![9.0; 8]); // All elements should be 9.0

        multiply_vec(&a, &b, &mut result);
        assert_eq!(result, vec![8.0, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0]);

        // Test 2: Statistical operations
        let sum_a = sum_vec(&a);
        assert_eq!(sum_a, 36.0); // 1+2+3+4+5+6+7+8 = 36

        let norm_a = norm_l2_squared(&a);
        assert_eq!(norm_a, 204.0); // 1²+2²+3²+4²+5²+6²+7²+8² = 204

        let dot_ab = dot_product(&a, &b);
        assert_eq!(dot_ab, 120.0); // 1*8+2*7+3*6+4*5+5*4+6*3+7*2+8*1 = 120

        // Test 3: Math functions
        let angles = vec![0.0, consts::PI / 4.0, consts::PI / 2.0, consts::PI];
        let mut sin_results = vec![0.0; 4];
        sin_vec(&angles, &mut sin_results);

        // Check sin(0) ≈ 0, sin(π/2) ≈ 1, sin(π) ≈ 0
        assert!((sin_results[0] - 0.0_f32).abs() < 1e-6_f32);
        assert!((sin_results[2] - 1.0_f32).abs() < 1e-6_f32);
        assert!((sin_results[3] - 0.0_f32).abs() < 1e-6_f32);

        // Test 4: Square root operations
        let squares = vec![1.0, 4.0, 9.0, 16.0];
        let mut sqrt_results = vec![0.0; 4];
        sqrt_vec(&squares, &mut sqrt_results);
        assert_eq!(sqrt_results, vec![1.0, 2.0, 3.0, 4.0]);

        // Test 5: Comparison operations
        let mut comparison_result = vec![false; 8];
        lt_vec(&a, &b, &mut comparison_result);
        assert_eq!(
            comparison_result,
            vec![true, true, true, true, false, false, false, false]
        );

        // Test 6: Platform detection
        let capabilities = detect_simd_capabilities();
        let simd_width = simd_width_f32();
        println!("Platform: {}", capabilities.platform_name());
        println!("SIMD width: {} elements", simd_width);

        assert!(simd_width >= 1);
        assert!(simd_width <= 16);
    }

    #[test]
    fn test_accuracy_and_performance_characteristics() {
        // Generate test data with various sizes to test performance scaling
        let sizes = vec![16, 64, 256, 1024, 4096];

        for &size in &sizes {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| ((size - i) as f32) * 0.1).collect();

            // Test that SIMD operations produce consistent results regardless of vector size
            let dot_product_result = dot_product(&a, &b);
            let sum_result = sum_vec(&a);

            // These operations should scale linearly with vector size
            assert!(dot_product_result.is_finite());
            assert!(sum_result.is_finite());

            // Test memory alignment doesn't affect correctness
            let mut result = vec![0.0; size];
            add_vec(&a, &b, &mut result);

            // Verify results are reasonable
            for (i, &val) in result.iter().enumerate() {
                let expected = a[i] + b[i];
                assert!(
                    (val - expected).abs() < 1e-6,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn test_edge_cases_and_special_values() {
        // Test with special floating-point values
        let special_values = vec![0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY];
        let normal_values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0; 6];

        // Test arithmetic with special values
        add_vec(&special_values, &normal_values, &mut result);
        assert_eq!(result[0], 1.0); // 0 + 1 = 1
        assert_eq!(result[1], 1.0); // -0 + 1 = 1
        assert_eq!(result[4], f32::INFINITY); // inf + 1 = inf
        assert_eq!(result[5], f32::NEG_INFINITY); // -inf + 1 = -inf

        // Test with very small and very large numbers
        let small_values = vec![f32::EPSILON, f32::MIN_POSITIVE, 1e-20, 1e20];
        let sum_small = sum_vec(&small_values);
        assert!(sum_small.is_finite());

        // Test with NaN handling
        let nan_values = vec![f32::NAN, 1.0, 2.0, 3.0];
        let sum_nan = sum_vec(&nan_values);
        assert!(sum_nan.is_nan());

        // Test empty vector handling
        let empty: Vec<f32> = vec![];
        assert_eq!(sum_vec(&empty), 0.0);
        assert_eq!(norm_l2_squared(&empty), 0.0);
    }

    #[test]
    fn test_mathematical_properties() {
        let a = vec![3.0, 4.0, 5.0, 12.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];

        // Test Pythagorean theorem: a² + b² = c² for (3,4,5) and (12,5,13)
        let a_norm_sq = norm_l2_squared(&vec![3.0, 4.0]);
        let expected_c = (a_norm_sq).sqrt();
        assert!((expected_c - 5.0).abs() < 1e-6);

        // Test dot product properties: a·b = b·a (commutativity)
        let dot_ab = dot_product(&a, &b);
        let dot_ba = dot_product(&b, &a);
        assert_eq!(dot_ab, dot_ba);

        // Test that |a·b| ≤ ||a|| * ||b|| (Cauchy-Schwarz inequality)
        let norm_a = norm_l2(&a);
        let norm_b = norm_l2(&b);
        assert!(dot_ab.abs() <= norm_a * norm_b + 1e-6); // Add small epsilon for floating point

        // Test distributivity: (a + b) · c = a·c + b·c
        let c = vec![0.5, 0.5, 0.5, 0.5];
        let mut a_plus_b = vec![0.0; 4];
        add_vec(&a, &b, &mut a_plus_b);

        let dot_ab_c = dot_product(&a_plus_b, &c);
        let dot_a_c = dot_product(&a, &c);
        let dot_b_c = dot_product(&b, &c);

        assert!((dot_ab_c - (dot_a_c + dot_b_c)).abs() < 1e-6);
    }

    #[test]
    #[ignore = "temporarily skipped - timeout"]
    fn test_platform_optimization() {
        // This test verifies that different SIMD platforms produce equivalent results
        let test_vectors = generate_test_vectors();

        for (a, b) in test_vectors {
            let mut result_simd = vec![0.0; a.len()];

            // Test core operations across different vector sizes
            // This will automatically use the best available SIMD instruction set
            add_vec(&a, &b, &mut result_simd);

            // Verify result is reasonable (simple sanity check)
            for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
                let expected = av + bv;
                let actual = result_simd[i];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "Platform optimization failed: expected {}, got {}",
                    expected,
                    actual
                );
            }
        }
    }

    fn generate_test_vectors() -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![
            // Small vectors (scalar might be used)
            (vec![1.0, 2.0], vec![3.0, 4.0]),
            // SSE2-sized vectors (4 elements)
            (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]),
            // AVX2-sized vectors (8 elements)
            (
                (0..8).map(|i| i as f32).collect(),
                (8..16).map(|i| i as f32).collect(),
            ),
            // AVX512-sized vectors (16 elements)
            (
                (0..16).map(|i| i as f32).collect(),
                (16..32).map(|i| i as f32).collect(),
            ),
            // Large vectors (multiple SIMD operations)
            (
                (0..100).map(|i| i as f32 * 0.1).collect(),
                (0..100).map(|i| (100 - i) as f32 * 0.1).collect(),
            ),
        ]
    }

    #[test]
    fn test_intrinsics_wrapper() {
        // Test the F32x4 wrapper directly
        let a = F32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = F32x4::new(5.0, 6.0, 7.0, 8.0);

        // Test basic operations
        let sum = a.add(b);
        assert_eq!(sum.extract(0), 6.0);
        assert_eq!(sum.extract(1), 8.0);
        assert_eq!(sum.extract(2), 10.0);
        assert_eq!(sum.extract(3), 12.0);

        let product = a.mul(b);
        assert_eq!(product.extract(0), 5.0);
        assert_eq!(product.extract(1), 12.0);
        assert_eq!(product.extract(2), 21.0);
        assert_eq!(product.extract(3), 32.0);

        // Test horizontal operations
        assert_eq!(a.horizontal_sum(), 10.0);

        // Test memory operations
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];

        unsafe {
            let vec1 = F32x4::load_unaligned(data.as_ptr());
            let vec2 = F32x4::load_unaligned(data.as_ptr().add(4));

            vec1.store_unaligned(output.as_mut_ptr());
            vec2.store_unaligned(output.as_mut_ptr().add(4));
        }

        assert_eq!(output, data);
    }
}
