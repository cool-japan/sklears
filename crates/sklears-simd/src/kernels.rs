//! SIMD-optimized kernel functions for machine learning

use crate::vector::dot_product;

#[cfg(feature = "no-std")]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "no-std"))]
use std::{vec, vec::Vec};

/// SIMD-optimized RBF (Gaussian) kernel function
pub fn rbf_kernel(x: &[f32], y: &[f32], gamma: f32) -> f32 {
    let distance_squared = euclidean_distance_squared(x, y);
    (-gamma * distance_squared).exp()
}

/// SIMD-optimized polynomial kernel function
pub fn polynomial_kernel(x: &[f32], y: &[f32], degree: f32, coef0: f32, gamma: f32) -> f32 {
    let dot_prod = dot_product(x, y);
    (gamma * dot_prod + coef0).powf(degree)
}

/// SIMD-optimized linear kernel function
pub fn linear_kernel(x: &[f32], y: &[f32]) -> f32 {
    dot_product(x, y)
}

/// SIMD-optimized sigmoid kernel function
pub fn sigmoid_kernel(x: &[f32], y: &[f32], gamma: f32, coef0: f32) -> f32 {
    let dot_prod = dot_product(x, y);
    (gamma * dot_prod + coef0).tanh()
}

/// Helper function to compute squared Euclidean distance
fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { euclidean_distance_squared_avx2(a, b) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { euclidean_distance_squared_sse2(a, b) };
        }
    }

    euclidean_distance_squared_scalar(a, b)
}

fn euclidean_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn euclidean_distance_squared_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(a_vec, b_vec);
        let squared = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, squared);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum
}

/// Kernel matrix computation for batch processing
pub fn kernel_matrix(
    x_data: &[Vec<f32>],
    y_data: &[Vec<f32>],
    kernel_type: KernelType,
) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0; y_data.len()]; x_data.len()];

    for i in 0..x_data.len() {
        for j in 0..y_data.len() {
            matrix[i][j] = match kernel_type {
                KernelType::Linear => linear_kernel(&x_data[i], &y_data[j]),
                KernelType::Rbf { gamma } => rbf_kernel(&x_data[i], &y_data[j], gamma),
                KernelType::Polynomial {
                    degree,
                    gamma,
                    coef0,
                } => polynomial_kernel(&x_data[i], &y_data[j], degree, coef0, gamma),
                KernelType::Sigmoid { gamma, coef0 } => {
                    sigmoid_kernel(&x_data[i], &y_data[j], gamma, coef0)
                }
            };
        }
    }

    matrix
}

/// Kernel types for different kernel functions
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Linear,
    Rbf { gamma: f32 },
    Polynomial { degree: f32, gamma: f32, coef0: f32 },
    Sigmoid { gamma: f32, coef0: f32 },
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::Rbf { gamma: 1.0 }
    }
}

/// Compute kernel values for a single point against multiple points
pub fn kernel_vector(x: &[f32], y_data: &[Vec<f32>], kernel_type: KernelType) -> Vec<f32> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        y_data
            .par_iter()
            .map(|y| match kernel_type {
                KernelType::Linear => linear_kernel(x, y),
                KernelType::Rbf { gamma } => rbf_kernel(x, y, gamma),
                KernelType::Polynomial {
                    degree,
                    gamma,
                    coef0,
                } => polynomial_kernel(x, y, degree, coef0, gamma),
                KernelType::Sigmoid { gamma, coef0 } => sigmoid_kernel(x, y, gamma, coef0),
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        y_data
            .iter()
            .map(|y| match kernel_type {
                KernelType::Linear => linear_kernel(x, y),
                KernelType::Rbf { gamma } => rbf_kernel(x, y, gamma),
                KernelType::Polynomial {
                    degree,
                    gamma,
                    coef0,
                } => polynomial_kernel(x, y, degree, coef0, gamma),
                KernelType::Sigmoid { gamma, coef0 } => sigmoid_kernel(x, y, gamma, coef0),
            })
            .collect()
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_kernel() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = linear_kernel(&x, &y);
        let expected = 32.0; // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_rbf_kernel() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        let gamma = 1.0;

        let result = rbf_kernel(&x, &y, gamma);
        let expected = 1.0; // Same points should give 1.0

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_polynomial_kernel() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let degree = 2.0;
        let gamma = 1.0;
        let coef0 = 0.0;

        let result = polynomial_kernel(&x, &y, degree, coef0, gamma);
        let dot_prod: f32 = 1.0 * 3.0 + 2.0 * 4.0; // = 11
        let expected = dot_prod.powf(degree); // = 121

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let gamma = 1.0;
        let coef0 = 0.0;

        let result = sigmoid_kernel(&x, &y, gamma, coef0);
        let dot_prod: f32 = 1.0 * 3.0 + 2.0 * 4.0; // = 11
        let expected = (gamma * dot_prod + coef0).tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_kernel_matrix() {
        let x_data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y_data = vec![vec![1.0, 2.0], vec![5.0, 6.0]];

        let matrix = kernel_matrix(&x_data, &y_data, KernelType::Linear);

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);

        // Check diagonal element (same vectors)
        assert_relative_eq!(matrix[0][0], 5.0, epsilon = 1e-6); // 1*1 + 2*2 = 5
    }

    #[test]
    fn test_kernel_vector() {
        let x = vec![1.0, 2.0];
        let y_data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = kernel_vector(&x, &y_data, KernelType::Linear);

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 5.0, epsilon = 1e-6); // 1*1 + 2*2 = 5
        assert_relative_eq!(result[1], 11.0, epsilon = 1e-6); // 1*3 + 2*4 = 11
    }
}
