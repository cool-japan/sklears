//! SIMD-optimized kernel functions for enhanced performance
//!
//! This module provides SIMD (Single Instruction, Multiple Data) optimized
//! implementations of common kernel functions. These optimizations can provide
//! significant speedups for kernel matrix computations in SVM training and prediction.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{error::Result, types::Float};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Configuration for SIMD kernel operations
#[derive(Debug, Clone)]
pub struct SimdKernelConfig {
    /// Whether to use SIMD optimizations
    pub use_simd: bool,
    /// Minimum vector size to enable SIMD
    pub min_simd_size: usize,
    /// Alignment for SIMD operations
    pub alignment: usize,
}

impl Default for SimdKernelConfig {
    fn default() -> Self {
        Self {
            use_simd: is_simd_available(),
            min_simd_size: 8,
            alignment: 32, // AVX2 alignment
        }
    }
}

/// Check if SIMD instructions are available
fn is_simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON is available on most aarch64 systems
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// SIMD-optimized kernel implementations
pub struct SimdKernels {
    config: SimdKernelConfig,
}

impl SimdKernels {
    /// Create a new SIMD kernel instance
    pub fn new(config: SimdKernelConfig) -> Self {
        Self { config }
    }

    /// Compute linear kernel with SIMD optimization
    pub fn linear_kernel(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
        if !self.config.use_simd || x.len() < self.config.min_simd_size {
            return Ok(self.linear_kernel_scalar(x, y));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Ok(unsafe { self.linear_kernel_avx2(x, y) });
            }
            if is_x86_feature_detected!("sse2") {
                return Ok(unsafe { self.linear_kernel_sse2(x, y) });
            }
        }

        #[cfg(target_arch = "aarch64")]
        return Ok(unsafe { self.linear_kernel_neon(x, y) });

        #[cfg(not(target_arch = "aarch64"))]
        Ok(self.linear_kernel_scalar(x, y))
    }

    /// Scalar implementation of linear kernel
    fn linear_kernel_scalar(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
    }

    /// AVX2 implementation of linear kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn linear_kernel_avx2(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        let len = x.len().min(y.len());
        let mut sum = _mm256_setzero_pd();
        let mut i = 0;

        // Process 4 doubles at a time with AVX2
        while i + 4 <= len {
            let x_vec = _mm256_loadu_pd(x.as_ptr().add(i));
            let y_vec = _mm256_loadu_pd(y.as_ptr().add(i));
            let prod = _mm256_mul_pd(x_vec, y_vec);
            sum = _mm256_add_pd(sum, prod);
            i += 4;
        }

        // Extract sum from SIMD register
        let mut result_array = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<Float>();

        // Handle remaining elements
        while i < len {
            total += x[i] * y[i];
            i += 1;
        }

        total
    }

    /// SSE2 implementation of linear kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn linear_kernel_sse2(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        let len = x.len().min(y.len());
        let mut sum = _mm_setzero_pd();
        let mut i = 0;

        // Process 2 doubles at a time with SSE2
        while i + 2 <= len {
            let x_vec = _mm_loadu_pd(x.as_ptr().add(i));
            let y_vec = _mm_loadu_pd(y.as_ptr().add(i));
            let prod = _mm_mul_pd(x_vec, y_vec);
            sum = _mm_add_pd(sum, prod);
            i += 2;
        }

        // Extract sum from SIMD register
        let mut result_array = [0.0; 2];
        _mm_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<Float>();

        // Handle remaining elements
        while i < len {
            total += x[i] * y[i];
            i += 1;
        }

        total
    }

    /// ARM NEON implementation of linear kernel
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn linear_kernel_neon(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        let len = x.len().min(y.len());
        let mut sum = vdupq_n_f64(0.0);
        let mut i = 0;

        // Process 2 doubles at a time with NEON
        while i + 2 <= len {
            let x_vec = vld1q_f64(x.as_ptr().add(i));
            let y_vec = vld1q_f64(y.as_ptr().add(i));
            let prod = vmulq_f64(x_vec, y_vec);
            sum = vaddq_f64(sum, prod);
            i += 2;
        }

        // Extract sum from SIMD register
        let total = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);

        // Handle remaining elements
        let mut result = total;
        while i < len {
            result += x[i] * y[i];
            i += 1;
        }

        result
    }

    /// Compute RBF kernel with SIMD optimization
    pub fn rbf_kernel(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        gamma: Float,
    ) -> Result<Float> {
        if !self.config.use_simd || x.len() < self.config.min_simd_size {
            return Ok(self.rbf_kernel_scalar(x, y, gamma));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Ok(unsafe { self.rbf_kernel_avx2(x, y, gamma) });
            }
        }

        #[cfg(target_arch = "aarch64")]
        return Ok(unsafe { self.rbf_kernel_neon(x, y, gamma) });

        #[cfg(not(target_arch = "aarch64"))]
        Ok(self.rbf_kernel_scalar(x, y, gamma))
    }

    /// Scalar implementation of RBF kernel
    fn rbf_kernel_scalar(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        gamma: Float,
    ) -> Float {
        let squared_distance: Float = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
        (-gamma * squared_distance).exp()
    }

    /// AVX2 implementation of RBF kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn rbf_kernel_avx2(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        gamma: Float,
    ) -> Float {
        let len = x.len().min(y.len());
        let mut sum = _mm256_setzero_pd();
        let mut i = 0;

        // Process 4 doubles at a time with AVX2
        while i + 4 <= len {
            let x_vec = _mm256_loadu_pd(x.as_ptr().add(i));
            let y_vec = _mm256_loadu_pd(y.as_ptr().add(i));
            let diff = _mm256_sub_pd(x_vec, y_vec);
            let squared = _mm256_mul_pd(diff, diff);
            sum = _mm256_add_pd(sum, squared);
            i += 4;
        }

        // Extract sum from SIMD register
        let mut result_array = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut squared_distance = result_array.iter().sum::<Float>();

        // Handle remaining elements
        while i < len {
            let diff = x[i] - y[i];
            squared_distance += diff * diff;
            i += 1;
        }

        (-gamma * squared_distance).exp()
    }

    /// ARM NEON implementation of RBF kernel
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn rbf_kernel_neon(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        gamma: Float,
    ) -> Float {
        let len = x.len().min(y.len());
        let mut sum = vdupq_n_f64(0.0);
        let mut i = 0;

        // Process 2 doubles at a time with NEON
        while i + 2 <= len {
            let x_vec = vld1q_f64(x.as_ptr().add(i));
            let y_vec = vld1q_f64(y.as_ptr().add(i));
            let diff = vsubq_f64(x_vec, y_vec);
            let squared = vmulq_f64(diff, diff);
            sum = vaddq_f64(sum, squared);
            i += 2;
        }

        // Extract sum from SIMD register
        let squared_distance = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);

        // Handle remaining elements
        let mut result = squared_distance;
        while i < len {
            let diff = x[i] - y[i];
            result += diff * diff;
            i += 1;
        }

        (-gamma * result).exp()
    }

    /// Compute polynomial kernel with SIMD optimization
    pub fn polynomial_kernel(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        degree: u32,
        gamma: Float,
        coef0: Float,
    ) -> Result<Float> {
        let dot_product = self.linear_kernel(x, y)?;
        Ok((gamma * dot_product + coef0).powf(degree as Float))
    }

    /// Batch kernel computation with SIMD optimization
    pub fn compute_kernel_matrix(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
        kernel_fn: impl Fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Result<Float> + Sync,
    ) -> Result<Array2<Float>> {
        let n_x = x.nrows();
        let n_y = y.nrows();
        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            kernel_matrix
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .try_for_each(|(i, mut row)| -> Result<()> {
                    let x_i = x.row(i);
                    for (j, cell) in row.iter_mut().enumerate() {
                        let y_j = y.row(j);
                        *cell = kernel_fn(&x_i, &y_j)?;
                    }
                    Ok(())
                })?;
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n_x {
                let x_i = x.row(i);
                for j in 0..n_y {
                    let y_j = y.row(j);
                    kernel_matrix[[i, j]] = kernel_fn(&x_i, &y_j)?;
                }
            }
        }

        Ok(kernel_matrix)
    }

    /// Compute diagonal of kernel matrix (efficient for RBF and similar kernels)
    pub fn compute_kernel_diagonal(
        &self,
        x: &Array2<Float>,
        kernel_fn: impl Fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Result<Float>,
    ) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut diagonal = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let x_i = x.row(i);
            diagonal[i] = kernel_fn(&x_i, &x_i)?;
        }

        Ok(diagonal)
    }

    /// Optimized kernel computation for symmetric matrices
    pub fn compute_symmetric_kernel_matrix(
        &self,
        x: &Array2<Float>,
        kernel_fn: impl Fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Result<Float> + Sync,
    ) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let mut kernel_matrix = Array2::zeros((n_samples, n_samples));

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Compute upper triangle in parallel
            let indices: Vec<(usize, usize)> = (0..n_samples)
                .flat_map(|i| (i..n_samples).map(move |j| (i, j)))
                .collect();

            let values: Result<Vec<Float>> = indices
                .par_iter()
                .map(|&(i, j)| {
                    let x_i = x.row(i);
                    let x_j = x.row(j);
                    kernel_fn(&x_i, &x_j)
                })
                .collect();

            let values = values?;

            // Fill matrix (symmetric)
            for ((i, j), &value) in indices.iter().zip(values.iter()) {
                kernel_matrix[[*i, *j]] = value;
                if i != j {
                    kernel_matrix[[*j, *i]] = value;
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n_samples {
                let x_i = x.row(i);
                for j in i..n_samples {
                    let x_j = x.row(j);
                    let value = kernel_fn(&x_i, &x_j)?;
                    kernel_matrix[[i, j]] = value;
                    if i != j {
                        kernel_matrix[[j, i]] = value;
                    }
                }
            }
        }

        Ok(kernel_matrix)
    }
}

/// Optimized kernel function wrappers that use SIMD when available
pub struct SimdKernelFunctions {
    simd_kernels: SimdKernels,
}

impl SimdKernelFunctions {
    /// Create new SIMD kernel functions
    pub fn new() -> Self {
        Self {
            simd_kernels: SimdKernels::new(SimdKernelConfig::default()),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SimdKernelConfig) -> Self {
        Self {
            simd_kernels: SimdKernels::new(config),
        }
    }

    /// Linear kernel
    pub fn linear(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
        self.simd_kernels.linear_kernel(x, y)
    }

    /// RBF kernel
    pub fn rbf(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>, gamma: Float) -> Result<Float> {
        self.simd_kernels.rbf_kernel(x, y, gamma)
    }

    /// Polynomial kernel
    pub fn polynomial(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        degree: u32,
        gamma: Float,
        coef0: Float,
    ) -> Result<Float> {
        self.simd_kernels
            .polynomial_kernel(x, y, degree, gamma, coef0)
    }
}

impl Default for SimdKernelFunctions {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_linear_kernel() {
        let config = SimdKernelConfig::default();
        let simd_kernels = SimdKernels::new(config);

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = array![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = simd_kernels.linear_kernel(&x.view(), &y.view()).unwrap();
        let expected: Float = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_rbf_kernel() {
        let config = SimdKernelConfig::default();
        let simd_kernels = SimdKernels::new(config);

        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 3.0, 4.0, 5.0];
        let gamma = 0.5;

        let result = simd_kernels
            .rbf_kernel(&x.view(), &y.view(), gamma)
            .unwrap();

        let squared_distance: Float = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
        let expected = (-gamma * squared_distance).exp();

        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_matrix_computation() {
        let config = SimdKernelConfig::default();
        let simd_kernels = SimdKernels::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[2.0, 3.0], [4.0, 5.0]];

        let kernel_matrix = simd_kernels
            .compute_kernel_matrix(&x, &y, |x_i, y_j| simd_kernels.linear_kernel(x_i, y_j))
            .unwrap();

        assert_eq!(kernel_matrix.shape(), &[3, 2]);

        // Verify some values
        let expected_00 = 1.0 * 2.0 + 2.0 * 3.0; // 8.0
        let expected_11 = 3.0 * 4.0 + 4.0 * 5.0; // 32.0

        assert_abs_diff_eq!(kernel_matrix[[0, 0]], expected_00, epsilon = 1e-10);
        assert_abs_diff_eq!(kernel_matrix[[1, 1]], expected_11, epsilon = 1e-10);
    }

    #[test]
    fn test_symmetric_kernel_matrix() {
        let config = SimdKernelConfig::default();
        let simd_kernels = SimdKernels::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let kernel_matrix = simd_kernels
            .compute_symmetric_kernel_matrix(&x, |x_i, x_j| simd_kernels.linear_kernel(x_i, x_j))
            .unwrap();

        assert_eq!(kernel_matrix.shape(), &[3, 3]);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    kernel_matrix[[i, j]],
                    kernel_matrix[[j, i]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_kernel_diagonal() {
        let config = SimdKernelConfig::default();
        let simd_kernels = SimdKernels::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let gamma = 1.0;

        let diagonal = simd_kernels
            .compute_kernel_diagonal(&x, |x_i, x_j| simd_kernels.rbf_kernel(x_i, x_j, gamma))
            .unwrap();

        assert_eq!(diagonal.len(), 3);

        // Diagonal elements of RBF kernel should be 1.0
        for &value in diagonal.iter() {
            assert_abs_diff_eq!(value, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_simd_availability() {
        let available = is_simd_available();

        // This test just ensures the function doesn't panic
        // Actual availability depends on the hardware
        println!("SIMD available: {available}");
    }

    #[test]
    fn test_simd_kernel_functions() {
        let kernel_funcs = SimdKernelFunctions::new();

        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 3.0, 4.0, 5.0];

        let linear_result = kernel_funcs.linear(&x.view(), &y.view()).unwrap();
        let rbf_result = kernel_funcs.rbf(&x.view(), &y.view(), 0.5).unwrap();
        let poly_result = kernel_funcs
            .polynomial(&x.view(), &y.view(), 2, 1.0, 1.0)
            .unwrap();

        assert!(linear_result > 0.0);
        assert!(rbf_result > 0.0 && rbf_result <= 1.0);
        assert!(poly_result > 0.0);
    }
}
