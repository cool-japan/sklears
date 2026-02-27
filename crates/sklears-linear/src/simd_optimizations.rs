//! SIMD optimizations for performance-critical linear algebra operations
//!
//! This module provides SIMD-accelerated implementations of common linear algebra
//! operations used in linear models, offering significant performance improvements
//! for large datasets.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// SIMD-optimized operations configuration
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Enable AVX2 optimizations (if available)
    pub use_avx2: bool,
    /// Enable SSE optimizations (if available)
    pub use_sse: bool,
    /// Minimum size threshold for using SIMD operations
    pub simd_threshold: usize,
    /// Block size for cache-friendly operations
    pub block_size: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            use_avx2: is_x86_feature_detected!("avx2"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            use_avx2: false,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            use_sse: is_x86_feature_detected!("sse4.1"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            use_sse: false,
            simd_threshold: 64,
            block_size: 256,
        }
    }
}

/// SIMD-optimized linear algebra operations
pub struct SimdOps {
    config: SimdConfig,
}

impl SimdOps {
    /// Create a new SIMD operations instance
    pub fn new(config: SimdConfig) -> Self {
        Self { config }
    }

    /// Compute dot product of two vectors using SIMD
    pub fn dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        assert_eq!(a.len(), b.len());

        if a.len() < self.config.simd_threshold {
            return self.dot_product_scalar(a, b);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                unsafe { self.dot_product_avx2(a, b) }
            } else if self.config.use_sse && is_x86_feature_detected!("sse4.1") {
                unsafe { self.dot_product_sse(a, b) }
            } else {
                self.dot_product_scalar(a, b)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.dot_product_scalar(a, b)
        }
    }

    /// Scalar dot product implementation
    fn dot_product_scalar(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// AVX2-optimized dot product
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let len = a.len();
        let mut sum = _mm256_setzero_pd();
        let mut i = 0;

        let a_slice = a.as_slice().expect("Array must be contiguous for SIMD");
        let b_slice = b.as_slice().expect("Array must be contiguous for SIMD");

        // Process 4 elements at a time with AVX2
        while i + 4 <= len {
            let va = _mm256_loadu_pd(a_slice.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b_slice.as_ptr().add(i));
            let prod = _mm256_mul_pd(va, vb);
            sum = _mm256_add_pd(sum, prod);
            i += 4;
        }

        // Extract the sum from the vector
        let mut result_array = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut result = result_array.iter().sum::<f64>();

        // Handle remaining elements
        while i < len {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    /// SSE-optimized dot product
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn dot_product_sse(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let len = a.len();
        let mut sum = _mm_setzero_pd();
        let mut i = 0;

        let a_slice = a.as_slice().expect("Array must be contiguous for SIMD");
        let b_slice = b.as_slice().expect("Array must be contiguous for SIMD");

        // Process 2 elements at a time with SSE
        while i + 2 <= len {
            let va = _mm_loadu_pd(a_slice.as_ptr().add(i));
            let vb = _mm_loadu_pd(b_slice.as_ptr().add(i));
            let prod = _mm_mul_pd(va, vb);
            sum = _mm_add_pd(sum, prod);
            i += 2;
        }

        // Extract the sum from the vector
        let mut result_array = [0.0; 2];
        _mm_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut result = result_array.iter().sum::<f64>();

        // Handle remaining elements
        while i < len {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    /// SIMD-optimized vector addition
    pub fn vector_add(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len());

        if a.len() < self.config.simd_threshold {
            return self.vector_add_scalar(a, b);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                unsafe { self.vector_add_avx2(a, b) }
            } else if self.config.use_sse && is_x86_feature_detected!("sse4.1") {
                unsafe { self.vector_add_sse(a, b) }
            } else {
                self.vector_add_scalar(a, b)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.vector_add_scalar(a, b)
        }
    }

    /// Scalar vector addition
    fn vector_add_scalar(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        a + b
    }

    /// AVX2-optimized vector addition
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_add_avx2(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let len = a.len();
        let mut result = vec![0.0; len];
        let mut i = 0;

        // Process 4 elements at a time with AVX2
        while i + 4 <= len {
            let va = _mm256_loadu_pd(
                a.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let vb = _mm256_loadu_pd(
                b.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let sum = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(result.as_mut_ptr().add(i), sum);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] + b[i];
            i += 1;
        }

        Array1::from_vec(result)
    }

    /// SSE-optimized vector addition
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn vector_add_sse(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let len = a.len();
        let mut result = vec![0.0; len];
        let mut i = 0;

        // Process 2 elements at a time with SSE
        while i + 2 <= len {
            let va = _mm_loadu_pd(
                a.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let vb = _mm_loadu_pd(
                b.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let sum = _mm_add_pd(va, vb);
            _mm_storeu_pd(result.as_mut_ptr().add(i), sum);
            i += 2;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] + b[i];
            i += 1;
        }

        Array1::from_vec(result)
    }

    /// SIMD-optimized vector subtraction
    pub fn vector_sub(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len());

        if a.len() < self.config.simd_threshold {
            return self.vector_sub_scalar(a, b);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                unsafe { self.vector_sub_avx2(a, b) }
            } else if self.config.use_sse && is_x86_feature_detected!("sse4.1") {
                unsafe { self.vector_sub_sse(a, b) }
            } else {
                self.vector_sub_scalar(a, b)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.vector_sub_scalar(a, b)
        }
    }

    /// Scalar vector subtraction
    fn vector_sub_scalar(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        a - b
    }

    /// AVX2-optimized vector subtraction
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_sub_avx2(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let len = a.len();
        let mut result = vec![0.0; len];
        let mut i = 0;

        // Process 4 elements at a time with AVX2
        while i + 4 <= len {
            let va = _mm256_loadu_pd(
                a.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let vb = _mm256_loadu_pd(
                b.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let diff = _mm256_sub_pd(va, vb);
            _mm256_storeu_pd(result.as_mut_ptr().add(i), diff);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] - b[i];
            i += 1;
        }

        Array1::from_vec(result)
    }

    /// SSE-optimized vector subtraction
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn vector_sub_sse(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let len = a.len();
        let mut result = vec![0.0; len];
        let mut i = 0;

        // Process 2 elements at a time with SSE
        while i + 2 <= len {
            let va = _mm_loadu_pd(
                a.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let vb = _mm_loadu_pd(
                b.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let diff = _mm_sub_pd(va, vb);
            _mm_storeu_pd(result.as_mut_ptr().add(i), diff);
            i += 2;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] - b[i];
            i += 1;
        }

        Array1::from_vec(result)
    }

    /// SIMD-optimized scalar multiplication
    pub fn vector_scale(&self, a: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
        if a.len() < self.config.simd_threshold {
            return self.vector_scale_scalar(a, scalar);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                unsafe { self.vector_scale_avx2(a, scalar) }
            } else if self.config.use_sse && is_x86_feature_detected!("sse4.1") {
                unsafe { self.vector_scale_sse(a, scalar) }
            } else {
                self.vector_scale_scalar(a, scalar)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.vector_scale_scalar(a, scalar)
        }
    }

    /// Scalar vector scaling
    fn vector_scale_scalar(&self, a: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
        a.mapv(|x| x * scalar)
    }

    /// AVX2-optimized vector scaling
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_scale_avx2(&self, a: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
        let len = a.len();
        let mut result = vec![0.0; len];
        let vs = _mm256_set1_pd(scalar);
        let mut i = 0;

        // Process 4 elements at a time with AVX2
        while i + 4 <= len {
            let va = _mm256_loadu_pd(
                a.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let prod = _mm256_mul_pd(va, vs);
            _mm256_storeu_pd(result.as_mut_ptr().add(i), prod);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * scalar;
            i += 1;
        }

        Array1::from_vec(result)
    }

    /// SSE-optimized vector scaling
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn vector_scale_sse(&self, a: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
        let len = a.len();
        let mut result = vec![0.0; len];
        let vs = _mm_set1_pd(scalar);
        let mut i = 0;

        // Process 2 elements at a time with SSE
        while i + 2 <= len {
            let va = _mm_loadu_pd(
                a.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let prod = _mm_mul_pd(va, vs);
            _mm_storeu_pd(result.as_mut_ptr().add(i), prod);
            i += 2;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * scalar;
            i += 1;
        }

        Array1::from_vec(result)
    }

    /// SIMD-optimized L2 norm computation
    pub fn l2_norm(&self, a: &ArrayView1<f64>) -> f64 {
        if a.len() < self.config.simd_threshold {
            return self.l2_norm_scalar(a);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.config.use_avx2 && is_x86_feature_detected!("avx2") {
                unsafe { self.l2_norm_avx2(a) }
            } else if self.config.use_sse && is_x86_feature_detected!("sse4.1") {
                unsafe { self.l2_norm_sse(a) }
            } else {
                self.l2_norm_scalar(a)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.l2_norm_scalar(a)
        }
    }

    /// Scalar L2 norm computation
    fn l2_norm_scalar(&self, a: &ArrayView1<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// AVX2-optimized L2 norm computation
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn l2_norm_avx2(&self, a: &ArrayView1<f64>) -> f64 {
        let len = a.len();
        let mut sum = _mm256_setzero_pd();
        let mut i = 0;

        // Process 4 elements at a time with AVX2
        while i + 4 <= len {
            let va = _mm256_loadu_pd(
                a.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let sq = _mm256_mul_pd(va, va);
            sum = _mm256_add_pd(sum, sq);
            i += 4;
        }

        // Extract the sum from the vector
        let mut result_array = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut result = result_array.iter().sum::<f64>();

        // Handle remaining elements
        while i < len {
            result += a[i] * a[i];
            i += 1;
        }

        result.sqrt()
    }

    /// SSE-optimized L2 norm computation
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn l2_norm_sse(&self, a: &ArrayView1<f64>) -> f64 {
        let len = a.len();
        let mut sum = _mm_setzero_pd();
        let mut i = 0;

        // Process 2 elements at a time with SSE
        while i + 2 <= len {
            let va = _mm_loadu_pd(
                a.as_slice()
                    .expect("Array must be contiguous for SIMD")
                    .as_ptr()
                    .add(i),
            );
            let sq = _mm_mul_pd(va, va);
            sum = _mm_add_pd(sum, sq);
            i += 2;
        }

        // Extract the sum from the vector
        let mut result_array = [0.0; 2];
        _mm_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut result = result_array.iter().sum::<f64>();

        // Handle remaining elements
        while i < len {
            result += a[i] * a[i];
            i += 1;
        }

        result.sqrt()
    }

    /// SIMD-optimized matrix-vector multiplication
    pub fn matrix_vector_mul(&self, matrix: &Array2<f64>, vector: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(matrix.ncols(), vector.len());

        let n_rows = matrix.nrows();
        let n_cols = matrix.ncols();

        if n_cols < self.config.simd_threshold {
            return self.matrix_vector_mul_scalar(matrix, vector);
        }

        let mut result = vec![0.0; n_rows];

        for i in 0..n_rows {
            let row = matrix.row(i);
            let row_vec = Array1::from_iter(row.iter().cloned());
            result[i] = self.dot_product(&row_vec.view(), vector);
        }

        Array1::from_vec(result)
    }

    /// Scalar matrix-vector multiplication
    fn matrix_vector_mul_scalar(
        &self,
        matrix: &Array2<f64>,
        vector: &ArrayView1<f64>,
    ) -> Array1<f64> {
        matrix.dot(vector)
    }

    /// SIMD-optimized matrix-matrix multiplication for small matrices
    pub fn matrix_matrix_mul_small(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        assert_eq!(a.ncols(), b.nrows());

        let n_rows = a.nrows();
        let n_cols = b.ncols();
        let n_inner = a.ncols();

        if n_inner < self.config.simd_threshold {
            return a * b;
        }

        let mut result = Array2::zeros((n_rows, n_cols));

        // Use block multiplication for better cache performance
        let block_size = self.config.block_size.min(n_rows).min(n_cols).min(n_inner);

        for i_block in (0..n_rows).step_by(block_size) {
            for j_block in (0..n_cols).step_by(block_size) {
                for k_block in (0..n_inner).step_by(block_size) {
                    let i_end = (i_block + block_size).min(n_rows);
                    let j_end = (j_block + block_size).min(n_cols);
                    let k_end = (k_block + block_size).min(n_inner);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;
                            for k in k_block..k_end {
                                sum += a[(i, k)] * b[(k, j)];
                            }
                            result[(i, j)] += sum;
                        }
                    }
                }
            }
        }

        result
    }

    /// SIMD-optimized coordinate descent update
    pub fn coordinate_descent_update(
        &self,
        residual: &ArrayView1<f64>,
        feature: &ArrayView1<f64>,
        _old_coeff: f64,
        alpha: f64,
        l1_ratio: f64,
    ) -> f64 {
        let correlation = self.dot_product(feature, residual);
        let feature_norm_sq = self.dot_product(feature, feature);

        if feature_norm_sq == 0.0 {
            return 0.0;
        }

        // Soft thresholding for L1 regularization
        let lasso_penalty = alpha * l1_ratio;
        let ridge_penalty = alpha * (1.0 - l1_ratio);

        let threshold = lasso_penalty;
        let denominator = feature_norm_sq + ridge_penalty;

        if correlation > threshold {
            (correlation - threshold) / denominator
        } else if correlation < -threshold {
            (correlation + threshold) / denominator
        } else {
            0.0
        }
    }

    /// SIMD-optimized residual update for coordinate descent
    pub fn update_residual(
        &self,
        residual: &mut Array1<f64>,
        feature: &ArrayView1<f64>,
        coeff_change: f64,
    ) {
        if coeff_change.abs() < f64::EPSILON {
            return;
        }

        let scaled_feature = self.vector_scale(feature, -coeff_change);
        let new_residual = self.vector_add(&residual.view(), &scaled_feature.view());
        *residual = new_residual;
    }

    /// Get information about available SIMD features
    pub fn simd_features() -> SimdFeatures {
        SimdFeatures {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            avx2: false,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse4_1: is_x86_feature_detected!("sse4.1"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse4_1: false,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            fma: is_x86_feature_detected!("fma"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            fma: false,
        }
    }
}

impl Default for SimdOps {
    fn default() -> Self {
        Self::new(SimdConfig::default())
    }
}

/// Information about available SIMD features
#[derive(Debug, Clone)]
pub struct SimdFeatures {
    pub avx2: bool,
    pub sse4_1: bool,
    pub fma: bool,
}

/// SIMD-accelerated linear regression solver
pub struct SimdLinearRegression {
    simd_ops: SimdOps,
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
}

impl SimdLinearRegression {
    /// Create a new SIMD-accelerated linear regression
    pub fn new() -> Self {
        Self {
            simd_ops: SimdOps::default(),
            coefficients: None,
            intercept: None,
        }
    }

    /// Create with custom SIMD configuration
    pub fn with_config(config: SimdConfig) -> Self {
        Self {
            simd_ops: SimdOps::new(config),
            coefficients: None,
            intercept: None,
        }
    }

    /// Fit the model using SIMD-optimized operations
    pub fn fit(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<f64>,
        fit_intercept: bool,
    ) -> Result<(), String> {
        if X.nrows() != y.len() {
            return Err("Number of samples in X and y must match".to_string());
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Center data if fitting intercept
        let (X_centered, y_centered, X_mean, y_mean) = if fit_intercept {
            let X_mean = X.mean_axis(Axis(0)).expect("Failed to compute mean");
            let y_mean = y.mean().expect("Failed to compute mean");

            let mut X_centered = X.clone();
            for i in 0..n_samples {
                for j in 0..n_features {
                    X_centered[(i, j)] -= X_mean[j];
                }
            }

            let y_mean_array = Array1::from_elem(n_samples, y_mean);
            let y_centered = self.simd_ops.vector_sub(&y.view(), &y_mean_array.view());

            (X_centered, y_centered, Some(X_mean), Some(y_mean))
        } else {
            (X.clone(), y.clone(), None, None)
        };

        // Solve normal equations: (X^T X) β = X^T y
        let XtX = self.compute_gram_matrix(&X_centered);
        let X_centered_t = X_centered.t().to_owned();
        let Xty = self
            .simd_ops
            .matrix_vector_mul(&X_centered_t, &y_centered.view());

        // Add small ridge regularization to ensure positive definiteness
        let ridge_eps = 1e-10_f64;
        let mut XtX_reg = XtX;
        for i in 0..n_features {
            XtX_reg[(i, i)] += ridge_eps;
        }

        // Solve using Cholesky decomposition
        let chol = XtX_reg
            .cholesky()
            .map_err(|e| format!("Cholesky decomposition failed: {:?}", e))?;
        let coefficients = chol
            .solve(&Xty)
            .map_err(|e| format!("Solve failed: {:?}", e))?;

        // Compute intercept if needed
        let intercept = if let (Some(x_mean), Some(y_mean)) = (X_mean, y_mean) {
            let x_mean_vec = Array1::from_iter(x_mean.iter().cloned());
            Some(
                y_mean
                    - self
                        .simd_ops
                        .dot_product(&coefficients.view(), &x_mean_vec.view()),
            )
        } else {
            None
        };

        self.coefficients = Some(coefficients);
        self.intercept = intercept;

        Ok(())
    }

    /// Predict using SIMD-optimized operations
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>, String> {
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("Model must be fitted before prediction")?;

        let mut predictions = self.simd_ops.matrix_vector_mul(X, &coefficients.view());

        if let Some(intercept) = self.intercept {
            let intercept_vec = Array1::from_elem(X.nrows(), intercept);
            predictions = self
                .simd_ops
                .vector_add(&predictions.view(), &intercept_vec.view());
        }

        Ok(predictions)
    }

    /// Get fitted coefficients
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Get fitted intercept
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Compute Gram matrix (X^T X) using SIMD
    fn compute_gram_matrix(&self, X: &Array2<f64>) -> Array2<f64> {
        let n_features = X.ncols();
        let mut gram = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in i..n_features {
                let col_i = X.column(i);
                let col_j = X.column(j);
                let col_i_vec = Array1::from_iter(col_i.iter().cloned());
                let col_j_vec = Array1::from_iter(col_j.iter().cloned());

                let value = self
                    .simd_ops
                    .dot_product(&col_i_vec.view(), &col_j_vec.view());
                gram[(i, j)] = value;
                if i != j {
                    gram[(j, i)] = value; // Symmetric
                }
            }
        }

        gram
    }
}

impl Default for SimdLinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD-accelerated coordinate descent solver
pub struct SimdCoordinateDescent {
    simd_ops: SimdOps,
    max_iter: usize,
    tolerance: f64,
}

impl SimdCoordinateDescent {
    /// Create a new SIMD coordinate descent solver
    pub fn new(max_iter: usize, tolerance: f64) -> Self {
        Self {
            simd_ops: SimdOps::default(),
            max_iter,
            tolerance,
        }
    }

    /// Solve Lasso using SIMD-optimized coordinate descent
    pub fn solve_lasso(&self, X: &Array2<f64>, y: &Array1<f64>, alpha: f64) -> Array1<f64> {
        let n_features = X.ncols();
        let mut coefficients = Array1::zeros(n_features);
        let mut residual = y.clone();

        for _iteration in 0..self.max_iter {
            let mut max_change: f64 = 0.0;

            for j in 0..n_features {
                let old_coeff: f64 = coefficients[j];
                let feature = X.column(j);
                let feature_vec = Array1::from_iter(feature.iter().cloned());

                // Add back the effect of the old coefficient
                if old_coeff.abs() > f64::EPSILON {
                    let old_effect = self.simd_ops.vector_scale(&feature_vec.view(), old_coeff);
                    residual = self
                        .simd_ops
                        .vector_add(&residual.view(), &old_effect.view());
                }

                // Compute new coefficient using soft thresholding
                let new_coeff = self.simd_ops.coordinate_descent_update(
                    &residual.view(),
                    &feature_vec.view(),
                    old_coeff,
                    alpha,
                    1.0,
                );

                // Update residual with new coefficient
                if new_coeff.abs() > f64::EPSILON {
                    let new_effect = self.simd_ops.vector_scale(&feature_vec.view(), new_coeff);
                    residual = self
                        .simd_ops
                        .vector_sub(&residual.view(), &new_effect.view());
                }

                coefficients[j] = new_coeff;
                max_change = max_change.max((new_coeff - old_coeff).abs());
            }

            // Check convergence
            if max_change < self.tolerance {
                break;
            }
        }

        coefficients
    }

    /// Solve Elastic Net using SIMD-optimized coordinate descent
    pub fn solve_elastic_net(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
        alpha: f64,
        l1_ratio: f64,
    ) -> Array1<f64> {
        let n_features = X.ncols();
        let mut coefficients = Array1::zeros(n_features);
        let mut residual = y.clone();

        for _iteration in 0..self.max_iter {
            let mut max_change: f64 = 0.0;

            for j in 0..n_features {
                let old_coeff: f64 = coefficients[j];
                let feature = X.column(j);
                let feature_vec = Array1::from_iter(feature.iter().cloned());

                // Add back the effect of the old coefficient
                if old_coeff.abs() > f64::EPSILON {
                    let old_effect = self.simd_ops.vector_scale(&feature_vec.view(), old_coeff);
                    residual = self
                        .simd_ops
                        .vector_add(&residual.view(), &old_effect.view());
                }

                // Compute new coefficient using elastic net update
                let new_coeff = self.simd_ops.coordinate_descent_update(
                    &residual.view(),
                    &feature_vec.view(),
                    old_coeff,
                    alpha,
                    l1_ratio,
                );

                // Update residual with new coefficient
                if new_coeff.abs() > f64::EPSILON {
                    let new_effect = self.simd_ops.vector_scale(&feature_vec.view(), new_coeff);
                    residual = self
                        .simd_ops
                        .vector_sub(&residual.view(), &new_effect.view());
                }

                coefficients[j] = new_coeff;
                max_change = max_change.max((new_coeff - old_coeff).abs());
            }

            // Check convergence
            if max_change < self.tolerance {
                break;
            }
        }

        coefficients
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        // Create data with low feature correlation to ensure XtX is positive definite
        let mut data = Vec::with_capacity(1000);
        for i in 0..100usize {
            for j in 0..10usize {
                // Use a combination of row and column indices to reduce correlation
                let val = ((i * 7 + j * 13) as f64 * 0.1).sin() * 10.0
                    + (i as f64 * 0.3 + j as f64 * 1.7).cos() * 5.0;
                data.push(val);
            }
        }
        let X = Array2::from_shape_vec((100, 10), data).expect("Shape mismatch");
        let y = Array1::from_iter((0..100).map(|i| i as f64));
        (X, y)
    }

    #[test]
    fn test_simd_features() {
        let features = SimdOps::simd_features();
        println!("SIMD Features: {:#?}", features);
        // Just ensure the function runs without panicking
        // Just ensure the function runs without panicking - no assertion needed
    }

    #[test]
    fn test_dot_product_consistency() {
        let simd_ops = SimdOps::default();
        let a = Array1::from_iter((0..100).map(|i| i as f64));
        let b = Array1::from_iter((0..100).map(|i| (i * 2) as f64));

        let simd_result = simd_ops.dot_product(&a.view(), &b.view());
        let scalar_result = simd_ops.dot_product_scalar(&a.view(), &b.view());

        assert!((simd_result - scalar_result).abs() < 1e-10);
    }

    #[test]
    fn test_vector_operations_consistency() {
        let simd_ops = SimdOps::default();
        let a = Array1::from_iter((0..100).map(|i| i as f64));
        let b = Array1::from_iter((0..100).map(|i| (i * 2) as f64));

        let simd_add = simd_ops.vector_add(&a.view(), &b.view());
        let scalar_add = simd_ops.vector_add_scalar(&a.view(), &b.view());

        for i in 0..100 {
            assert!((simd_add[i] - scalar_add[i]).abs() < 1e-10);
        }

        let simd_sub = simd_ops.vector_sub(&a.view(), &b.view());
        let scalar_sub = simd_ops.vector_sub_scalar(&a.view(), &b.view());

        for i in 0..100 {
            assert!((simd_sub[i] - scalar_sub[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_vector_scaling_consistency() {
        let simd_ops = SimdOps::default();
        let a = Array1::from_iter((0..100).map(|i| i as f64));
        let scalar = 2.5;

        let simd_result = simd_ops.vector_scale(&a.view(), scalar);
        let scalar_result = simd_ops.vector_scale_scalar(&a.view(), scalar);

        for i in 0..100 {
            assert!((simd_result[i] - scalar_result[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_l2_norm_consistency() {
        let simd_ops = SimdOps::default();
        let a = Array1::from_iter((0..100).map(|i| i as f64));

        let simd_result = simd_ops.l2_norm(&a.view());
        let scalar_result = simd_ops.l2_norm_scalar(&a.view());

        assert!((simd_result - scalar_result).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_vector_mul_consistency() {
        let simd_ops = SimdOps::default();
        let (X, _y) = create_test_data();
        let v = Array1::from_iter((0..10).map(|i| i as f64));

        let simd_result = simd_ops.matrix_vector_mul(&X, &v.view());
        let scalar_result = simd_ops.matrix_vector_mul_scalar(&X, &v.view());

        for i in 0..simd_result.len() {
            assert!((simd_result[i] - scalar_result[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simd_linear_regression() {
        let (X, y) = create_test_data();
        let mut model = SimdLinearRegression::new();

        let result = model.fit(&X, &y, false);
        assert!(result.is_ok());

        let predictions = model.predict(&X);
        assert!(predictions.is_ok());

        let pred = predictions.unwrap();
        assert_eq!(pred.len(), y.len());

        for &p in pred.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_simd_coordinate_descent() {
        let (X, y) = create_test_data();
        let solver = SimdCoordinateDescent::new(1000, 1e-6);

        let lasso_coeffs = solver.solve_lasso(&X, &y, 0.1);
        assert_eq!(lasso_coeffs.len(), X.ncols());

        for &coeff in lasso_coeffs.iter() {
            assert!(coeff.is_finite());
        }

        let elastic_coeffs = solver.solve_elastic_net(&X, &y, 0.1, 0.5);
        assert_eq!(elastic_coeffs.len(), X.ncols());

        for &coeff in elastic_coeffs.iter() {
            assert!(coeff.is_finite());
        }
    }

    #[test]
    fn test_small_vectors_fallback() {
        let simd_ops = SimdOps::default();

        // Test with very small vectors (should use scalar fallback)
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let result = simd_ops.dot_product(&a.view(), &b.view());
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_config() {
        let config = SimdConfig {
            use_avx2: false,
            use_sse: true,
            simd_threshold: 32,
            block_size: 128,
        };

        let simd_ops = SimdOps::new(config);
        let a = Array1::from_iter((0..100).map(|i| i as f64));
        let b = Array1::from_iter((0..100).map(|i| (i * 2) as f64));

        let result = simd_ops.dot_product(&a.view(), &b.view());
        assert!(result.is_finite());
    }
}
