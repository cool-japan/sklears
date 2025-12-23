//! SIMD-optimized probability distributions and sampling algorithms
//!
//! This module provides high-performance implementations of common probability
//! distributions and sampling algorithms using SIMD instructions for maximum
//! performance in machine learning applications.

use scirs2_autograd::ndarray::{Array1, Array2};

#[cfg(feature = "no-std")]
use core::f32::consts::{SQRT_2, TAU};
#[cfg(not(feature = "no-std"))]
use std::f32::consts::{SQRT_2, TAU};

#[cfg(feature = "no-std")]
use alloc::vec;

/// SIMD-optimized random number generator using LCG (Linear Congruential Generator)
pub struct SimdRng {
    state: u64,
    multiplier: u64,
    increment: u64,
}

impl SimdRng {
    /// Create a new SIMD random number generator
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            multiplier: 1103515245,
            increment: 12345,
        }
    }

    /// Generate a single random u32
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(self.multiplier)
            .wrapping_add(self.increment);
        (self.state >> 16) as u32
    }

    /// Generate multiple random u32 values using SIMD
    pub fn fill_u32(&mut self, output: &mut [u32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                unsafe { self.fill_u32_avx2(output) };
                return;
            } else if crate::simd_feature_detected!("sse2") {
                unsafe { self.fill_u32_sse2(output) };
                return;
            }
        }

        // Scalar fallback
        for val in output.iter_mut() {
            *val = self.next_u32();
        }
    }

    /// Generate uniform random floats in [0, 1)
    pub fn uniform_f32(&mut self, output: &mut [f32]) {
        let mut u32_buffer = vec![0u32; output.len()];
        self.fill_u32(&mut u32_buffer);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                unsafe { convert_u32_to_f32_avx2(&u32_buffer, output) };
                return;
            } else if crate::simd_feature_detected!("sse2") {
                unsafe { convert_u32_to_f32_sse2(&u32_buffer, output) };
                return;
            }
        }

        // Scalar fallback
        for (i, &val) in u32_buffer.iter().enumerate() {
            output[i] = (val as f32) / (u32::MAX as f32);
        }
    }
}

/// SIMD-optimized normal (Gaussian) distribution
pub struct Normal {
    mean: f32,
    std_dev: f32,
}

impl Normal {
    /// Create a new normal distribution
    pub fn new(mean: f32, std_dev: f32) -> Self {
        assert!(std_dev > 0.0, "Standard deviation must be positive");
        Self { mean, std_dev }
    }

    /// Generate samples using Box-Muller transform
    pub fn sample(&self, rng: &mut SimdRng, output: &mut [f32]) {
        let mut uniform_samples = vec![0.0f32; output.len() * 2];
        rng.uniform_f32(&mut uniform_samples);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                unsafe { self.box_muller_avx2(&uniform_samples, output) };
                return;
            } else if crate::simd_feature_detected!("sse2") {
                unsafe { self.box_muller_sse2(&uniform_samples, output) };
                return;
            }
        }

        // Scalar fallback
        self.box_muller_scalar(&uniform_samples, output);
    }

    /// Compute probability density function
    pub fn pdf(&self, values: &[f32], output: &mut [f32]) {
        assert_eq!(values.len(), output.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                unsafe { self.pdf_avx2(values, output) };
                return;
            } else if crate::simd_feature_detected!("sse2") {
                unsafe { self.pdf_sse2(values, output) };
                return;
            }
        }

        // Scalar fallback
        self.pdf_scalar(values, output);
    }

    /// Compute cumulative distribution function using error function approximation
    pub fn cdf(&self, values: &[f32], output: &mut [f32]) {
        assert_eq!(values.len(), output.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                unsafe { self.cdf_avx2(values, output) };
                return;
            } else if crate::simd_feature_detected!("sse2") {
                unsafe { self.cdf_sse2(values, output) };
                return;
            }
        }

        // Scalar fallback
        self.cdf_scalar(values, output);
    }

    fn box_muller_scalar(&self, uniform: &[f32], output: &mut [f32]) {
        let mut i = 0;
        let mut out_idx = 0;

        while out_idx < output.len() && i + 1 < uniform.len() {
            let u1 = uniform[i].max(1e-10); // Avoid log(0)
            let u2 = uniform[i + 1];

            let magnitude = (-2.0 * u1.ln()).sqrt() * self.std_dev;
            let angle = TAU * u2;

            let z0 = magnitude * angle.cos() + self.mean;
            let z1 = magnitude * angle.sin() + self.mean;

            output[out_idx] = z0;
            if out_idx + 1 < output.len() {
                output[out_idx + 1] = z1;
            }

            i += 2;
            out_idx += 2;
        }
    }

    fn pdf_scalar(&self, values: &[f32], output: &mut [f32]) {
        let inv_sqrt_2pi = 1.0 / (TAU).sqrt();
        let inv_std = 1.0 / self.std_dev;
        let inv_var_2 = 1.0 / (2.0 * self.std_dev * self.std_dev);

        for (i, &x) in values.iter().enumerate() {
            let z = (x - self.mean) * inv_std;
            output[i] = inv_sqrt_2pi * inv_std * (-z * z * inv_var_2).exp();
        }
    }

    fn cdf_scalar(&self, values: &[f32], output: &mut [f32]) {
        for (i, &x) in values.iter().enumerate() {
            let z = (x - self.mean) / (self.std_dev * SQRT_2);
            output[i] = 0.5 * (1.0 + erf_approximation(z));
        }
    }
}

/// SIMD-optimized exponential distribution
pub struct Exponential {
    rate: f32,
}

impl Exponential {
    /// Create a new exponential distribution
    pub fn new(rate: f32) -> Self {
        assert!(rate > 0.0, "Rate parameter must be positive");
        Self { rate }
    }

    /// Generate samples using inverse transform sampling
    pub fn sample(&self, rng: &mut SimdRng, output: &mut [f32]) {
        let mut uniform_samples = vec![0.0f32; output.len()];
        rng.uniform_f32(&mut uniform_samples);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                unsafe { self.inverse_transform_avx2(&uniform_samples, output) };
                return;
            } else if crate::simd_feature_detected!("sse2") {
                unsafe { self.inverse_transform_sse2(&uniform_samples, output) };
                return;
            }
        }

        // Scalar fallback
        for (i, &u) in uniform_samples.iter().enumerate() {
            output[i] = -(1.0 - u).ln() / self.rate;
        }
    }

    /// Compute probability density function
    pub fn pdf(&self, values: &[f32], output: &mut [f32]) {
        for (i, &x) in values.iter().enumerate() {
            if x >= 0.0 {
                output[i] = self.rate * (-self.rate * x).exp();
            } else {
                output[i] = 0.0;
            }
        }
    }
}

/// SIMD-optimized beta distribution (simplified using rejection sampling)
pub struct Beta {
    alpha: f32,
    beta: f32,
}

impl Beta {
    /// Create a new beta distribution
    pub fn new(alpha: f32, beta: f32) -> Self {
        assert!(alpha > 0.0 && beta > 0.0, "Alpha and beta must be positive");
        Self { alpha, beta }
    }

    /// Generate samples using rejection sampling (simplified)
    pub fn sample(&self, rng: &mut SimdRng, output: &mut [f32]) {
        // For simplicity, we'll use a basic approach
        // A more sophisticated implementation would use more efficient algorithms
        let mut uniform_samples = vec![0.0f32; output.len() * 2];
        rng.uniform_f32(&mut uniform_samples);

        for i in 0..output.len() {
            let u1 = uniform_samples[i * 2];
            let u2 = uniform_samples[i * 2 + 1];

            // Simple transformation (not the most efficient for all parameter values)
            let x = u1.powf(1.0 / self.alpha);
            let y = u2.powf(1.0 / self.beta);

            output[i] = x / (x + y);
        }
    }
}

/// Error function approximation using polynomial approximation
fn erf_approximation(x: f32) -> f32 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254_829_6;
    let a2 = -0.284_496_72;
    let a3 = 1.421_413_8;
    let a4 = -1.453_152_1;
    let a5 = 1.061_405_4;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();

    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

    sign * y
}

// SIMD implementations for x86/x86_64

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdRng {
    #[target_feature(enable = "sse2")]
    unsafe fn fill_u32_sse2(&mut self, output: &mut [u32]) {
        // SSE2 doesn't have 32-bit multiply (_mm_mullo_epi32 is SSE4.1)
        // Fall back to scalar implementation
        for val in output.iter_mut() {
            *val = self.next_u32();
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn fill_u32_avx2(&mut self, output: &mut [u32]) {
        // The previous SIMD implementation had a fundamental flaw:
        // using _mm256_set1_epi32 broadcasts the same state to all lanes,
        // causing all lanes to generate identical values.
        // A proper SIMD RNG requires different states per lane or
        // a counter-based approach. For correctness, use scalar code.
        for val in output.iter_mut() {
            *val = self.next_u32();
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn convert_u32_to_f32_sse2(input: &[u32], output: &mut [f32]) {
    // _mm_cvtepi32_ps converts signed i32, not unsigned u32
    // This causes values > 2^31 to be interpreted as negative
    // Use scalar conversion for correctness
    for (i, &val) in input.iter().enumerate() {
        output[i] = (val as f32) / (u32::MAX as f32);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn convert_u32_to_f32_avx2(input: &[u32], output: &mut [f32]) {
    // _mm256_cvtepi32_ps converts signed i32, not unsigned u32
    // This causes values > 2^31 to be interpreted as negative
    // Use scalar conversion for correctness
    for (i, &val) in input.iter().enumerate() {
        output[i] = (val as f32) / (u32::MAX as f32);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Normal {
    #[target_feature(enable = "sse2")]
    unsafe fn box_muller_sse2(&self, uniform: &[f32], output: &mut [f32]) {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;

        let mut i = 0;
        let mut out_idx = 0;

        while out_idx + 4 <= output.len() && i + 8 <= uniform.len() {
            let u1 = _mm_loadu_ps(&uniform[i]);
            let u2 = _mm_loadu_ps(&uniform[i + 4]);

            // Simplified - use scalar math for trigonometric functions
            // magnitude = sqrt(-2 * ln(u1)) * std_dev
            let mut u1_vals = [0.0f32; 4];
            let mut u2_vals = [0.0f32; 4];
            _mm_storeu_ps(u1_vals.as_mut_ptr(), u1);
            _mm_storeu_ps(u2_vals.as_mut_ptr(), u2);

            let mut z0_vals = [0.0f32; 4];
            for k in 0..4 {
                let magnitude = (-2.0 * u1_vals[k].ln()).sqrt() * self.std_dev;
                let angle = TAU * u2_vals[k];
                z0_vals[k] = magnitude * angle.cos() + self.mean;
            }

            let z0 = _mm_loadu_ps(z0_vals.as_ptr());

            _mm_storeu_ps(&mut output[out_idx], z0);

            i += 8;
            out_idx += 4;
        }

        // Handle remaining elements with scalar code
        while out_idx < output.len() && i + 1 < uniform.len() {
            let u1 = uniform[i].max(1e-10);
            let u2 = uniform[i + 1];

            let magnitude = (-2.0 * u1.ln()).sqrt() * self.std_dev;
            let angle = TAU * u2;

            output[out_idx] = magnitude * angle.cos() + self.mean;

            i += 2;
            out_idx += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn box_muller_avx2(&self, uniform: &[f32], output: &mut [f32]) {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;

        let mut i = 0;
        let mut out_idx = 0;

        while out_idx + 8 <= output.len() && i + 16 <= uniform.len() {
            let u1 = _mm256_loadu_ps(&uniform[i]);
            let u2 = _mm256_loadu_ps(&uniform[i + 8]);

            // Simplified - use scalar math for trigonometric functions
            // magnitude = sqrt(-2 * ln(u1)) * std_dev
            let mut u1_vals = [0.0f32; 8];
            let mut u2_vals = [0.0f32; 8];
            _mm256_storeu_ps(u1_vals.as_mut_ptr(), u1);
            _mm256_storeu_ps(u2_vals.as_mut_ptr(), u2);

            let mut z0_vals = [0.0f32; 8];
            for k in 0..8 {
                let magnitude = (-2.0 * u1_vals[k].ln()).sqrt() * self.std_dev;
                let angle = TAU * u2_vals[k];
                z0_vals[k] = magnitude * angle.cos() + self.mean;
            }

            let z0 = _mm256_loadu_ps(z0_vals.as_ptr());

            _mm256_storeu_ps(&mut output[out_idx], z0);

            i += 16;
            out_idx += 8;
        }

        // Handle remaining elements with scalar code
        while out_idx < output.len() && i + 1 < uniform.len() {
            let u1 = uniform[i].max(1e-10);
            let u2 = uniform[i + 1];

            let magnitude = (-2.0 * u1.ln()).sqrt() * self.std_dev;
            let angle = TAU * u2;

            output[out_idx] = magnitude * angle.cos() + self.mean;

            i += 2;
            out_idx += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn pdf_sse2(&self, values: &[f32], output: &mut [f32]) {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;

        let inv_sqrt_2pi = _mm_set1_ps(1.0 / (TAU).sqrt());
        let mean_vec = _mm_set1_ps(self.mean);
        let inv_std = _mm_set1_ps(1.0 / self.std_dev);
        let inv_var_2 = _mm_set1_ps(1.0 / (2.0 * self.std_dev * self.std_dev));

        let mut i = 0;
        while i + 4 <= values.len() {
            let x = _mm_loadu_ps(&values[i]);
            let z = _mm_mul_ps(_mm_sub_ps(x, mean_vec), inv_std);
            let exp_arg = _mm_mul_ps(_mm_mul_ps(z, z), inv_var_2);
            let mut exp_arg_vals = [0.0f32; 4];
            _mm_storeu_ps(exp_arg_vals.as_mut_ptr(), exp_arg);
            let mut exp_vals = [0.0f32; 4];
            for k in 0..4 {
                exp_vals[k] = (-exp_arg_vals[k]).exp();
            }
            let exp_result = _mm_loadu_ps(exp_vals.as_ptr());
            let result = _mm_mul_ps(_mm_mul_ps(inv_sqrt_2pi, inv_std), exp_result);
            _mm_storeu_ps(&mut output[i], result);
            i += 4;
        }

        // Handle remaining elements
        while i < values.len() {
            let z = (values[i] - self.mean) / self.std_dev;
            output[i] = (1.0 / (TAU).sqrt()) / self.std_dev * (-z * z / 2.0).exp();
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn pdf_avx2(&self, values: &[f32], output: &mut [f32]) {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;

        let inv_sqrt_2pi = _mm256_set1_ps(1.0 / (TAU).sqrt());
        let mean_vec = _mm256_set1_ps(self.mean);
        let inv_std = _mm256_set1_ps(1.0 / self.std_dev);
        let inv_var_2 = _mm256_set1_ps(1.0 / (2.0 * self.std_dev * self.std_dev));

        let mut i = 0;
        while i + 8 <= values.len() {
            let x = _mm256_loadu_ps(&values[i]);
            let z = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std);
            let exp_arg = _mm256_mul_ps(_mm256_mul_ps(z, z), inv_var_2);
            let mut exp_arg_vals = [0.0f32; 8];
            _mm256_storeu_ps(exp_arg_vals.as_mut_ptr(), exp_arg);
            let mut exp_vals = [0.0f32; 8];
            for k in 0..8 {
                exp_vals[k] = (-exp_arg_vals[k]).exp();
            }
            let exp_result = _mm256_loadu_ps(exp_vals.as_ptr());
            let result = _mm256_mul_ps(_mm256_mul_ps(inv_sqrt_2pi, inv_std), exp_result);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }

        // Handle remaining elements
        while i < values.len() {
            let z = (values[i] - self.mean) / self.std_dev;
            output[i] = (1.0 / (TAU).sqrt()) / self.std_dev * (-z * z / 2.0).exp();
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn cdf_sse2(&self, values: &[f32], output: &mut [f32]) {
        // Implementation would use SIMD error function approximation
        self.cdf_scalar(values, output);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cdf_avx2(&self, values: &[f32], output: &mut [f32]) {
        // Implementation would use SIMD error function approximation
        self.cdf_scalar(values, output);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Exponential {
    #[target_feature(enable = "sse2")]
    unsafe fn inverse_transform_sse2(&self, uniform: &[f32], output: &mut [f32]) {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;

        let one = _mm_set1_ps(1.0);
        let rate_vec = _mm_set1_ps(self.rate);

        let mut i = 0;
        while i + 4 <= uniform.len() {
            let u = _mm_loadu_ps(&uniform[i]);
            let one_minus_u = _mm_sub_ps(one, u);
            let mut one_minus_u_vals = [0.0f32; 4];
            _mm_storeu_ps(one_minus_u_vals.as_mut_ptr(), one_minus_u);
            let mut ln_vals = [0.0f32; 4];
            for k in 0..4 {
                ln_vals[k] = one_minus_u_vals[k].ln();
            }
            let ln_result = _mm_loadu_ps(ln_vals.as_ptr());
            let neg_ln = _mm_sub_ps(_mm_setzero_ps(), ln_result);
            let result = _mm_div_ps(neg_ln, rate_vec);
            _mm_storeu_ps(&mut output[i], result);
            i += 4;
        }

        // Handle remaining elements
        while i < uniform.len() {
            output[i] = -(1.0 - uniform[i]).ln() / self.rate;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn inverse_transform_avx2(&self, uniform: &[f32], output: &mut [f32]) {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;

        let one = _mm256_set1_ps(1.0);
        let rate_vec = _mm256_set1_ps(self.rate);

        let mut i = 0;
        while i + 8 <= uniform.len() {
            let u = _mm256_loadu_ps(&uniform[i]);
            let one_minus_u = _mm256_sub_ps(one, u);
            let mut one_minus_u_vals = [0.0f32; 8];
            _mm256_storeu_ps(one_minus_u_vals.as_mut_ptr(), one_minus_u);
            let mut ln_vals = [0.0f32; 8];
            for k in 0..8 {
                ln_vals[k] = one_minus_u_vals[k].ln();
            }
            let ln_result = _mm256_loadu_ps(ln_vals.as_ptr());
            let neg_ln = _mm256_sub_ps(_mm256_setzero_ps(), ln_result);
            let result = _mm256_div_ps(neg_ln, rate_vec);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }

        // Handle remaining elements
        while i < uniform.len() {
            output[i] = -(1.0 - uniform[i]).ln() / self.rate;
            i += 1;
        }
    }
}

/// Multivariate normal distribution sampling
pub fn multivariate_normal_sample(
    mean: &Array1<f32>,
    covariance: &Array2<f32>,
    rng: &mut SimdRng,
    num_samples: usize,
) -> Array2<f32> {
    let dim = mean.len();
    assert_eq!(covariance.shape(), &[dim, dim]);

    // Cholesky decomposition of covariance matrix (simplified)
    let chol = cholesky_decomposition(covariance);

    let mut samples = Array2::zeros((num_samples, dim));
    let normal = Normal::new(0.0, 1.0);

    for i in 0..num_samples {
        let mut standard_normal = vec![0.0f32; dim];
        normal.sample(rng, &mut standard_normal);

        // Transform standard normal to desired distribution
        let z = Array1::from_vec(standard_normal);
        let transformed = crate::matrix::matrix_vector_multiply_f32(&chol, &z);

        for j in 0..dim {
            samples[[i, j]] = transformed[j] + mean[j];
        }
    }

    samples
}

/// Simplified Cholesky decomposition
fn cholesky_decomposition(matrix: &Array2<f32>) -> Array2<f32> {
    let n = matrix.nrows();
    let mut chol = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            if i == j {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += chol[[j, k]] * chol[[j, k]];
                }
                chol[[j, j]] = (matrix[[j, j]] - sum).sqrt();
            } else {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += chol[[i, k]] * chol[[j, k]];
                }
                chol[[i, j]] = (matrix[[i, j]] - sum) / chol[[j, j]];
            }
        }
    }

    chol
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_simd_rng() {
        let mut rng = SimdRng::new(12345);
        let mut output = vec![0u32; 16];
        rng.fill_u32(&mut output);

        // Check that we get different values
        assert!(output.iter().any(|&x| x != output[0]));
    }

    #[test]
    fn test_uniform_f32() {
        let mut rng = SimdRng::new(12345);
        let mut output = vec![0.0f32; 100];
        rng.uniform_f32(&mut output);

        // Check range [0, 1)
        for &val in &output {
            assert!(val >= 0.0 && val < 1.0);
        }

        // Check some variability
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!(mean > 0.4 && mean < 0.6); // Should be around 0.5
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = SimdRng::new(42);
        let normal = Normal::new(5.0, 2.0);
        let mut samples = vec![0.0f32; 1000];
        normal.sample(&mut rng, &mut samples);

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert_relative_eq!(mean, 5.0, epsilon = 0.2);
    }

    #[test]
    fn test_normal_pdf() {
        let normal = Normal::new(0.0, 1.0);
        let values = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0f32; 3];
        normal.pdf(&values, &mut output);

        // At x=0, PDF should be 1/sqrt(2π) ≈ 0.3989
        assert_relative_eq!(output[0], 0.3989, epsilon = 0.01);

        // At x=1 and x=-1, should be equal (symmetric)
        assert_relative_eq!(output[1], output[2], epsilon = 1e-6);
    }

    #[test]
    fn test_exponential_distribution() {
        let mut rng = SimdRng::new(123);
        let exp_dist = Exponential::new(2.0);
        let mut samples = vec![0.0f32; 1000];
        exp_dist.sample(&mut rng, &mut samples);

        // All samples should be non-negative
        for &sample in &samples {
            assert!(sample >= 0.0);
        }

        // Mean should be approximately 1/rate = 0.5
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert_relative_eq!(mean, 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_beta_distribution() {
        let mut rng = SimdRng::new(456);
        let beta = Beta::new(2.0, 3.0);
        let mut samples = vec![0.0f32; 100];
        beta.sample(&mut rng, &mut samples);

        // All samples should be in [0, 1]
        for &sample in &samples {
            assert!(sample >= 0.0 && sample <= 1.0);
        }
    }

    #[test]
    fn test_erf_approximation() {
        assert_relative_eq!(erf_approximation(0.0), 0.0, epsilon = 1e-4);
        assert_relative_eq!(erf_approximation(1.0), 0.8427, epsilon = 1e-3);
        assert_relative_eq!(erf_approximation(-1.0), -0.8427, epsilon = 1e-3);
    }

    #[test]
    fn test_rng_uniform() {
        let mut rng = SimdRng::new(123);
        let mut samples = vec![0.0f32; 10];
        rng.uniform_f32(&mut samples);

        eprintln!("Uniform samples: {:?}", samples);
        let sum: f32 = samples.iter().sum();
        eprintln!("Sum: {}, Mean: {}", sum, sum / samples.len() as f32);

        // At least some variance
        assert!(sum > 0.1);
    }

    #[test]
    fn test_multivariate_normal() {
        let mut rng = SimdRng::new(789);
        let mean = Array1::from_vec(vec![1.0, 2.0]);
        let cov = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap();

        let samples = multivariate_normal_sample(&mean, &cov, &mut rng, 10);
        assert_eq!(samples.shape(), &[10, 2]);
    }
}
