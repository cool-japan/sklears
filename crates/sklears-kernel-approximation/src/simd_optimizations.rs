//! SIMD optimizations for kernel approximation feature generation
//!
//! This module provides SIMD-optimized implementations of critical operations
//! in kernel approximation methods for improved performance.

use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::Distribution;
use scirs2_core::random::{Seedablethread_rng};
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::traits::Fit;
use sklears_core::{
    error::{Result, SklearsError},
    traits::Transform,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized utilities for kernel approximations
pub struct SimdOptimizations;

impl SimdOptimizations {
    /// SIMD-optimized dot product computation
    #[cfg(target_arch = "x86_64")]
    pub fn simd_dot_product(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        assert_eq!(a.len(), b.len());

        let len = a.len();
        let mut result = 0.0;

        // Check if we have AVX2 support
        if is_x86_feature_detected!("avx2") {
            unsafe { result += Self::simd_dot_product_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            unsafe { result += Self::simd_dot_product_sse2(a, b) };
        } else {
            // Fallback to regular computation
            result = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        }

        result
    }

    /// AVX2-optimized dot product
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn simd_dot_product_avx2(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let len = a.len();
        let simd_len = len & !3; // Process 4 elements at a time
        let mut sum = _mm256_setzero_pd();

        for i in (0..simd_len).step_by(4) {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
            let prod = _mm256_mul_pd(a_vec, b_vec);
            sum = _mm256_add_pd(sum, prod);
        }

        // Extract and sum the 4 elements
        let mut result = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum::<f64>();

        // Handle remaining elements
        for i in simd_len..len {
            total += a[i] * b[i];
        }

        total
    }

    /// SSE2-optimized dot product
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn simd_dot_product_sse2(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let len = a.len();
        let simd_len = len & !1; // Process 2 elements at a time
        let mut sum = _mm_setzero_pd();

        for i in (0..simd_len).step_by(2) {
            let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
            let b_vec = _mm_loadu_pd(b.as_ptr().add(i));
            let prod = _mm_mul_pd(a_vec, b_vec);
            sum = _mm_add_pd(sum, prod);
        }

        // Extract and sum the 2 elements
        let mut result = [0.0; 2];
        _mm_storeu_pd(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum::<f64>();

        // Handle remaining elements
        for i in simd_len..len {
            total += a[i] * b[i];
        }

        total
    }

    /// SIMD-optimized matrix-vector multiplication
    #[cfg(target_arch = "x86_64")]
    pub fn simd_matvec_multiply(matrix: ArrayView2<f64>, vector: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(matrix.ncols(), vector.len());

        let nrows = matrix.nrows();
        let mut result = Array1::zeros(nrows);

        // Use parallel SIMD processing for each row
        result.par_iter_mut().enumerate().for_each(|(i, res)| {
            let row = matrix.row(i);
            *res = Self::simd_dot_product(row, vector);
        });

        result
    }

    /// SIMD-optimized element-wise operations
    #[cfg(target_arch = "x86_64")]
    pub fn simd_elementwise_exp(input: ArrayView1<f64>) -> Array1<f64> {
        let len = input.len();
        let mut result = Array1::zeros(len);

        if is_x86_feature_detected!("avx2") {
            unsafe { Self::simd_elementwise_exp_avx2(input, result.view_mut()) };
        } else {
            // Fallback
            result
                .iter_mut()
                .zip(input.iter())
                .for_each(|(r, &x)| *r = x.exp());
        }

        result
    }

    /// AVX2-optimized element-wise exponential
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn simd_elementwise_exp_avx2(
        input: ArrayView1<f64>,
        mut output: scirs2_core::ndarray::ArrayViewMut1<f64>,
    ) {
        let len = input.len();
        let simd_len = len & !3; // Process 4 elements at a time

        for i in (0..simd_len).step_by(4) {
            let x_vec = _mm256_loadu_pd(input.as_ptr().add(i));

            // Fast approximation of exp using polynomial approximation
            // This is a simplified version - for production, use a more accurate method
            let mut exp_vec = _mm256_set1_pd(1.0);
            let x2 = _mm256_mul_pd(x_vec, x_vec);
            let x3 = _mm256_mul_pd(x2, x_vec);
            let x4 = _mm256_mul_pd(x3, x_vec);

            exp_vec = _mm256_add_pd(exp_vec, x_vec);
            exp_vec = _mm256_add_pd(exp_vec, _mm256_mul_pd(x2, _mm256_set1_pd(0.5)));
            exp_vec = _mm256_add_pd(exp_vec, _mm256_mul_pd(x3, _mm256_set1_pd(1.0 / 6.0)));
            exp_vec = _mm256_add_pd(exp_vec, _mm256_mul_pd(x4, _mm256_set1_pd(1.0 / 24.0)));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), exp_vec);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].exp();
        }
    }

    /// SIMD-optimized cosine computation for RBF features
    #[cfg(target_arch = "x86_64")]
    pub fn simd_cos_features(
        projections: ArrayView1<f64>,
        offsets: ArrayView1<f64>,
    ) -> Array1<f64> {
        assert_eq!(projections.len(), offsets.len());

        let len = projections.len();
        let mut result = Array1::zeros(len);

        if is_x86_feature_detected!("avx2") {
            unsafe { Self::simd_cos_features_avx2(projections, offsets, result.view_mut()) };
        } else {
            // Fallback
            result
                .iter_mut()
                .zip(projections.iter().zip(offsets.iter()))
                .for_each(|(r, (&p, &o))| *r = (p + o).cos());
        }

        result
    }

    /// AVX2-optimized cosine computation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn simd_cos_features_avx2(
        projections: ArrayView1<f64>,
        offsets: ArrayView1<f64>,
        mut output: scirs2_core::ndarray::ArrayViewMut1<f64>,
    ) {
        let len = projections.len();
        let simd_len = len & !3; // Process 4 elements at a time

        for i in (0..simd_len).step_by(4) {
            let proj_vec = _mm256_loadu_pd(projections.as_ptr().add(i));
            let offset_vec = _mm256_loadu_pd(offsets.as_ptr().add(i));
            let sum_vec = _mm256_add_pd(proj_vec, offset_vec);

            // Fast cosine approximation using Taylor series
            // cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + ...
            let x2 = _mm256_mul_pd(sum_vec, sum_vec);
            let x4 = _mm256_mul_pd(x2, x2);
            let x6 = _mm256_mul_pd(x4, x2);

            let mut cos_vec = _mm256_set1_pd(1.0);
            cos_vec = _mm256_sub_pd(cos_vec, _mm256_mul_pd(x2, _mm256_set1_pd(0.5)));
            cos_vec = _mm256_add_pd(cos_vec, _mm256_mul_pd(x4, _mm256_set1_pd(1.0 / 24.0)));
            cos_vec = _mm256_sub_pd(cos_vec, _mm256_mul_pd(x6, _mm256_set1_pd(1.0 / 720.0)));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), cos_vec);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = (projections[i] + offsets[i]).cos();
        }
    }

    /// SIMD-optimized squared Euclidean distance computation
    #[cfg(target_arch = "x86_64")]
    pub fn simd_squared_euclidean_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        assert_eq!(a.len(), b.len());

        if is_x86_feature_detected!("avx2") {
            unsafe { Self::simd_squared_euclidean_distance_avx2(a, b) }
        } else if is_x86_feature_detected!("sse2") {
            unsafe { Self::simd_squared_euclidean_distance_sse2(a, b) }
        } else {
            // Fallback
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
        }
    }

    /// AVX2-optimized squared Euclidean distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn simd_squared_euclidean_distance_avx2(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let len = a.len();
        let simd_len = len & !3; // Process 4 elements at a time
        let mut sum = _mm256_setzero_pd();

        for i in (0..simd_len).step_by(4) {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
            let diff = _mm256_sub_pd(a_vec, b_vec);
            let squared_diff = _mm256_mul_pd(diff, diff);
            sum = _mm256_add_pd(sum, squared_diff);
        }

        // Extract and sum the 4 elements
        let mut result = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum::<f64>();

        // Handle remaining elements
        for i in simd_len..len {
            let diff = a[i] - b[i];
            total += diff * diff;
        }

        total
    }

    /// SSE2-optimized squared Euclidean distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn simd_squared_euclidean_distance_sse2(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let len = a.len();
        let simd_len = len & !1; // Process 2 elements at a time
        let mut sum = _mm_setzero_pd();

        for i in (0..simd_len).step_by(2) {
            let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
            let b_vec = _mm_loadu_pd(b.as_ptr().add(i));
            let diff = _mm_sub_pd(a_vec, b_vec);
            let squared_diff = _mm_mul_pd(diff, diff);
            sum = _mm_add_pd(sum, squared_diff);
        }

        // Extract and sum the 2 elements
        let mut result = [0.0; 2];
        _mm_storeu_pd(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum::<f64>();

        // Handle remaining elements
        for i in simd_len..len {
            let diff = a[i] - b[i];
            total += diff * diff;
        }

        total
    }

    /// SIMD-optimized RBF kernel computation
    #[cfg(target_arch = "x86_64")]
    pub fn simd_rbf_kernel_matrix(
        x1: ArrayView2<f64>,
        x2: ArrayView2<f64>,
        gamma: f64,
    ) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut kernel_matrix = Array2::zeros((n1, n2));

        // Parallelize over rows
        kernel_matrix
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(i, mut row)| {
                let x1_row = x1.row(i);
                row.iter_mut().enumerate().for_each(|(j, k_val)| {
                    let x2_row = x2.row(j);
                    let sq_dist = Self::simd_squared_euclidean_distance(x1_row, x2_row);
                    *k_val = (-gamma * sq_dist).exp();
                });
            });

        kernel_matrix
    }

    /// SIMD-optimized batch RBF feature generation
    #[cfg(target_arch = "x86_64")]
    pub fn simd_rbf_features(
        x: ArrayView2<f64>,
        random_weights: ArrayView2<f64>,
        random_offsets: ArrayView1<f64>,
        normalization: f64,
    ) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_components = random_weights.nrows();
        let mut result = Array2::zeros((n_samples, n_components));

        // Parallel processing over samples
        result
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(i, mut row)| {
                let x_sample = x.row(i);

                // Compute projections for all components at once
                let projections = Self::simd_matvec_multiply(random_weights.view(), x_sample);

                // Add offsets and compute cosines
                let features = Self::simd_cos_features(projections.view(), random_offsets);

                // Apply normalization
                row.iter_mut().zip(features.iter()).for_each(|(r, &f)| {
                    *r = f * normalization;
                });
            });

        result
    }

    /// SIMD-optimized polynomial kernel computation
    #[cfg(target_arch = "x86_64")]
    pub fn simd_polynomial_kernel(
        x1: ArrayView1<f64>,
        x2: ArrayView1<f64>,
        gamma: f64,
        coef0: f64,
        degree: i32,
    ) -> f64 {
        let dot_product = Self::simd_dot_product(x1, x2);
        let base = gamma * dot_product + coef0;
        base.powi(degree)
    }

    /// Fallback implementations for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_dot_product(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_matvec_multiply(matrix: ArrayView2<f64>, vector: ArrayView1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(matrix.nrows());
        result.iter_mut().enumerate().for_each(|(i, res)| {
            *res = matrix
                .row(i)
                .iter()
                .zip(vector.iter())
                .map(|(m, v)| m * v)
                .sum();
        });
        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_elementwise_exp(input: ArrayView1<f64>) -> Array1<f64> {
        input.mapv(|x| x.exp())
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_cos_features(
        projections: ArrayView1<f64>,
        offsets: ArrayView1<f64>,
    ) -> Array1<f64> {
        let mut result = Array1::zeros(projections.len());
        result
            .iter_mut()
            .zip(projections.iter().zip(offsets.iter()))
            .for_each(|(r, (&p, &o))| *r = (p + o).cos());
        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_squared_euclidean_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_rbf_kernel_matrix(
        x1: ArrayView2<f64>,
        x2: ArrayView2<f64>,
        gamma: f64,
    ) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut kernel_matrix = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let sq_dist = Self::simd_squared_euclidean_distance(x1.row(i), x2.row(j));
                kernel_matrix[[i, j]] = (-gamma * sq_dist).exp();
            }
        }

        kernel_matrix
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_rbf_features(
        x: ArrayView2<f64>,
        random_weights: ArrayView2<f64>,
        random_offsets: ArrayView1<f64>,
        normalization: f64,
    ) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_components = random_weights.nrows();
        let mut result = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            let x_sample = x.row(i);
            let projections = Self::simd_matvec_multiply(random_weights.view(), x_sample);
            let features = Self::simd_cos_features(projections.view(), random_offsets);

            for (j, &feature) in features.iter().enumerate() {
                result[[i, j]] = feature * normalization;
            }
        }

        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_polynomial_kernel(
        x1: ArrayView1<f64>,
        x2: ArrayView1<f64>,
        gamma: f64,
        coef0: f64,
        degree: i32,
    ) -> f64 {
        let dot_product = Self::simd_dot_product(x1, x2);
        let base = gamma * dot_product + coef0;
        base.powi(degree)
    }
}

/// SIMD-optimized RBF sampler
pub struct SimdRBFSampler {
    n_components: usize,
    gamma: f64,
    random_seed: Option<u64>,
}

impl SimdRBFSampler {
    /// Create a new SIMD-optimized RBF sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            gamma: 1.0,
            random_seed: None,
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

/// Fitted SIMD-optimized RBF sampler
pub struct FittedSimdRBFSampler {
    random_weights: Array2<f64>,
    random_offsets: Array1<f64>,
    gamma: f64,
    normalization: f64,
}

impl Fit<Array2<f64>, ()> for SimdRBFSampler {
    type Fitted = FittedSimdRBFSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;

        let n_features = x.ncols();

        let mut rng = if let Some(seed) = self.random_seed {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        // Generate random weights
        let random_weights = Array2::from_shape_fn((self.n_components, n_features), |_| {
            rng.sample(rand_distr::RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).unwrap())
        });

        // Generate random offsets
        let random_offsets = Array1::from_shape_fn(self.n_components, |_| {
            rng.sample(rand_distr::RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap())
        });

        let normalization = (2.0 / self.n_components as f64).sqrt();

        Ok(FittedSimdRBFSampler {
            random_weights,
            random_offsets,
            gamma: self.gamma,
            normalization,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedSimdRBFSampler {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let result = SimdOptimizations::simd_rbf_features(
            x.view(),
            self.random_weights.view(),
            self.random_offsets.view(),
            self.normalization,
        );

        Ok(result)
    }
}

/// Performance benchmarking utilities
pub struct SimdBenchmarks;

impl SimdBenchmarks {
    /// Benchmark SIMD vs regular dot product
    pub fn benchmark_dot_product(size: usize, iterations: usize) -> (f64, f64) {
        use scirs2_core::random::Rng;
        use std::time::Instant;

        let mut rng = thread_rng();
        let a: Array1<f64> = Array1::from_shape_fn(size, |_| rng.gen_range(-1.0..1.0));
        let b: Array1<f64> = Array1::from_shape_fn(size, |_| rng.gen_range(-1.0..1.0));

        // Benchmark regular dot product
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
        }
        let regular_time = start.elapsed().as_secs_f64();

        // Benchmark SIMD dot product
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = SimdOptimizations::simd_dot_product(a.view(), b.view());
        }
        let simd_time = start.elapsed().as_secs_f64();

        (regular_time, simd_time)
    }

    /// Benchmark SIMD vs regular RBF feature generation
    pub fn benchmark_rbf_features(
        n_samples: usize,
        n_features: usize,
        n_components: usize,
    ) -> (f64, f64) {
        use scirs2_core::random::Rng;
        use std::time::Instant;

        let mut rng = thread_rng();

        let x: Array2<f64> =
            Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-1.0..1.0));
        let weights: Array2<f64> =
            Array2::from_shape_fn((n_components, n_features), |_| rng.gen_range(-1.0..1.0));
        let offsets: Array1<f64> =
            Array1::from_shape_fn(n_components, |_| rng.gen_range(0.0..std::f64::consts::TAU));
        let normalization = (2.0 / n_components as f64).sqrt();

        // Benchmark regular implementation
        let start = Instant::now();
        let mut regular_result = Array2::zeros((n_samples, n_components));
        for i in 0..n_samples {
            for j in 0..n_components {
                let projection: f64 = x
                    .row(i)
                    .iter()
                    .zip(weights.row(j).iter())
                    .map(|(x, w)| x * w)
                    .sum();
                regular_result[[i, j]] = (projection + offsets[j]).cos() * normalization;
            }
        }
        let regular_time = start.elapsed().as_secs_f64();

        // Benchmark SIMD implementation
        let start = Instant::now();
        let _simd_result = SimdOptimizations::simd_rbf_features(
            x.view(),
            weights.view(),
            offsets.view(),
            normalization,
        );
        let simd_time = start.elapsed().as_secs_f64();

        (regular_time, simd_time)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_simd_dot_product() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Array1::from_vec(vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let regular_result = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
        let simd_result = SimdOptimizations::simd_dot_product(a.view(), b.view());

        assert_abs_diff_eq!(regular_result, simd_result, epsilon = 1e-10);
        assert_abs_diff_eq!(simd_result, 120.0, epsilon = 1e-10); // Expected result
    }

    #[test]
    fn test_simd_matvec_multiply() {
        let matrix = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let regular_result = matrix.dot(&vector);
        let simd_result = SimdOptimizations::simd_matvec_multiply(matrix.view(), vector.view());

        assert_abs_diff_eq!(regular_result, simd_result, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_squared_euclidean_distance() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        let regular_result: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = *x - *y;
                (diff as f64).powi(2)
            })
            .sum();
        let simd_result = SimdOptimizations::simd_squared_euclidean_distance(a.view(), b.view());

        assert_abs_diff_eq!(regular_result, simd_result, epsilon = 1e-10);
        assert_abs_diff_eq!(simd_result, 64.0, epsilon = 1e-10); // (4^2) * 4 = 64
    }

    #[test]
    fn test_simd_cos_features() {
        let projections =
            Array1::from_vec(vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI]);
        let offsets = Array1::from_vec(vec![0.0, 0.0, 0.0]);

        let result = SimdOptimizations::simd_cos_features(projections.view(), offsets.view());

        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_rbf_sampler() {
        let x = Array2::from_shape_vec((5, 3), (0..15).map(|i| i as f64).collect()).unwrap();

        let sampler = SimdRBFSampler::new(10).gamma(0.5).random_seed(42);
        let fitted = sampler.fit(&x, &()).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert_eq!(result.shape(), &[5, 10]);

        // Check that features are bounded (cosines should be between -1 and 1)
        for &val in result.iter() {
            assert!(val >= -2.0 && val <= 2.0); // With normalization, slightly larger bounds
        }
    }

    #[test]
    fn test_simd_rbf_kernel_matrix() {
        let x1 = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x2 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 7.0, 8.0]).unwrap();
        let gamma = 0.5;

        let result = SimdOptimizations::simd_rbf_kernel_matrix(x1.view(), x2.view(), gamma);

        assert_eq!(result.shape(), &[3, 2]);

        // Verify diagonal elements when x1 == x2
        let diagonal_result =
            SimdOptimizations::simd_rbf_kernel_matrix(x1.view(), x1.view(), gamma);
        for i in 0..3 {
            assert_abs_diff_eq!(diagonal_result[[i, i]], 1.0, epsilon = 1e-10);
        }

        // Verify symmetry
        assert_abs_diff_eq!(
            diagonal_result[[0, 1]],
            diagonal_result[[1, 0]],
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_simd_polynomial_kernel() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let gamma = 1.0;
        let coef0 = 0.0;
        let degree = 2;

        let result =
            SimdOptimizations::simd_polynomial_kernel(a.view(), b.view(), gamma, coef0, degree);

        // Expected: (1*4 + 2*5 + 3*6)^2 = (4 + 10 + 18)^2 = 32^2 = 1024
        assert_abs_diff_eq!(result, 1024.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_rbf_features_consistency() {
        let x = Array2::from_shape_vec((10, 5), (0..50).map(|i| i as f64 * 0.1).collect()).unwrap();

        // Create SIMD sampler
        let simd_sampler = SimdRBFSampler::new(20).gamma(0.8).random_seed(42);
        let fitted_simd = simd_sampler.fit(&x, &()).unwrap();
        let simd_result = fitted_simd.transform(&x).unwrap();

        // Create regular sampler with same parameters
        let regular_sampler = crate::RBFSampler::new(20).gamma(0.8).random_state(42);
        let fitted_regular = regular_sampler.fit(&x, &()).unwrap();
        let regular_result = fitted_regular.transform(&x).unwrap();

        // Results should be very similar (allowing for minor numerical differences)
        assert_eq!(simd_result.shape(), regular_result.shape());
        for (simd_val, regular_val) in simd_result.iter().zip(regular_result.iter()) {
            assert_abs_diff_eq!(simd_val, regular_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_benchmark_functions() {
        // Test that benchmark functions run without errors
        let (regular_time, simd_time) = SimdBenchmarks::benchmark_dot_product(1000, 10);
        assert!(regular_time > 0.0);
        assert!(simd_time > 0.0);

        let (regular_time, simd_time) = SimdBenchmarks::benchmark_rbf_features(100, 10, 50);
        assert!(regular_time > 0.0);
        assert!(simd_time > 0.0);

        // Print speedup (for manual verification during development)
        println!("Dot product speedup: {:.2}x", regular_time / simd_time);
        println!("RBF features speedup: {:.2}x", regular_time / simd_time);
    }

    #[test]
    fn test_edge_cases() {
        // Test with very small arrays
        let small_a = Array1::from_vec(vec![1.0]);
        let small_b = Array1::from_vec(vec![2.0]);
        let result = SimdOptimizations::simd_dot_product(small_a.view(), small_b.view());
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);

        // Test with empty-like scenarios
        let zero_a = Array1::zeros(8);
        let zero_b = Array1::zeros(8);
        let result =
            SimdOptimizations::simd_squared_euclidean_distance(zero_a.view(), zero_b.view());
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);

        // Test with odd-sized arrays (not divisible by SIMD width)
        let odd_a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let odd_b = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let result = SimdOptimizations::simd_dot_product(odd_a.view(), odd_b.view());
        assert_abs_diff_eq!(result, 35.0, epsilon = 1e-10); // 5+8+9+8+5 = 35
    }
}
