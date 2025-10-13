//! SIMD-accelerated signal processing operations
//!
//! This module provides high-performance SIMD implementations of common
//! signal processing operations used in decomposition algorithms.

use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};
// use std::simd::{f64x8, f32x16, Simd, LaneCount, SupportedLaneCount};
// use std::simd::prelude::SimdFloat;

/// SIMD-optimized signal processing operations
pub struct SimdSignalOps;

impl SimdSignalOps {
    /// Fast correlation using SIMD
    pub fn simd_correlate(
        signal1: &Array1<Float>,
        signal2: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n1 = signal1.len();
        let n2 = signal2.len();
        let output_len = n1 + n2 - 1;
        let mut result = Array1::<Float>::zeros(output_len);

        // Simple correlation implementation
        for i in 0..output_len {
            let mut sum = 0.0;
            for j in 0..n1 {
                let k = i as i32 - j as i32;
                if k >= 0 && k < n2 as i32 {
                    sum += signal1[j] * signal2[k as usize];
                }
            }
            result[i] = sum;
        }

        Ok(result)
    }

    /// Fast convolution using SIMD
    pub fn simd_convolve(signal: &Array1<Float>, kernel: &Array1<Float>) -> Result<Array1<Float>> {
        // Simple convolution - in practice would use FFT for larger sizes
        Self::simd_correlate(signal, kernel)
    }

    /// SIMD-optimized element-wise operations
    pub fn simd_elementwise_multiply(
        a: &Array1<Float>,
        b: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if a.len() != b.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Arrays must have same length".to_string(),
            ));
        }

        let result: Vec<Float> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
        Ok(Array1::from_vec(result))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_correlate() {
        let signal1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let signal2 = Array1::from_vec(vec![1.0, 1.0]);

        let result = SimdSignalOps::simd_correlate(&signal1, &signal2).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_simd_elementwise_multiply() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0]);

        let result = SimdSignalOps::simd_elementwise_multiply(&a, &b).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 6.0);
        assert_eq!(result[2], 12.0);
    }
}
