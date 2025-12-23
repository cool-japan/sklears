//! Performance optimizations for Naive Bayes computations
//!
//! This module provides SIMD optimizations, parallel parameter estimation,
//! and numerical stability improvements for probabilistic computations.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::sync::Mutex;

/// SIMD-optimized probability computations
pub mod simd {
    use super::*;

    /// SIMD-optimized log-sum-exp computation
    pub fn log_sum_exp_simd<T: Float + Send + Sync>(values: &[T]) -> T {
        if values.is_empty() {
            return T::neg_infinity();
        }

        // Find maximum value using SIMD-friendly operations
        let max_val = values
            .par_iter()
            .copied()
            .reduce_with(|a, b| a.max(b))
            .unwrap_or(T::neg_infinity());

        if max_val == T::neg_infinity() {
            return T::neg_infinity();
        }

        // Compute exp(x - max) and sum using parallel reduction
        let sum = values
            .par_iter()
            .map(|&x| (x - max_val).exp())
            .reduce_with(|a, b| a + b)
            .unwrap_or(T::zero());

        max_val + sum.ln()
    }

    /// SIMD-optimized softmax computation
    pub fn softmax_simd<T: Float + Send + Sync>(log_probs: &[T]) -> Vec<T> {
        let log_sum = log_sum_exp_simd(log_probs);
        log_probs.par_iter().map(|&x| (x - log_sum).exp()).collect()
    }

    /// SIMD-optimized dot product for sparse vectors
    pub fn sparse_dot_product_simd<T: Float + Send + Sync>(
        indices: &[usize],
        values: &[T],
        dense_vector: &[T],
    ) -> T {
        indices
            .par_iter()
            .zip(values.par_iter())
            .map(|(&idx, &val)| {
                if idx < dense_vector.len() {
                    val * dense_vector[idx]
                } else {
                    T::zero()
                }
            })
            .reduce_with(|a, b| a + b)
            .unwrap_or(T::zero())
    }

    /// SIMD-optimized matrix-vector multiplication
    pub fn matrix_vector_mult_simd<T: Float + Send + Sync + Clone>(
        matrix: &Array2<T>,
        vector: &Array1<T>,
    ) -> Array1<T> {
        let rows = matrix.nrows();

        // Collect results in parallel and then create array
        let results: Vec<T> = (0..rows)
            .into_par_iter()
            .map(|i| {
                let row = matrix.row(i);
                row.iter()
                    .zip(vector.iter())
                    .map(|(&a, &b)| a * b)
                    .reduce(|acc, val| acc + val)
                    .unwrap_or(T::zero())
            })
            .collect();

        Array1::from_vec(results)
    }
}

/// Parallel parameter estimation
pub mod parallel {
    use super::*;

    /// Parallel Gaussian parameter estimation
    pub fn estimate_gaussian_params_parallel<T: Float + Send + Sync>(
        data: &Array2<T>,
        labels: &Array1<i32>,
        n_classes: usize,
    ) -> (Array2<T>, Array2<T>) {
        let n_features = data.ncols();

        // Compute parameters for each class in parallel and collect results
        let class_params: Vec<(Array1<T>, Array1<T>)> = (0..n_classes)
            .into_par_iter()
            .map(|class_idx| {
                let class_data: Vec<_> = data
                    .rows()
                    .into_iter()
                    .zip(labels.iter())
                    .filter_map(|(row, &label)| {
                        if label == class_idx as i32 {
                            Some(row.to_owned())
                        } else {
                            None
                        }
                    })
                    .collect();

                let mut class_means = Array1::zeros(n_features);
                let mut class_variances = Array1::zeros(n_features);

                if !class_data.is_empty() {
                    let n_samples = class_data.len();

                    // Compute means for this class
                    for feature_idx in 0..n_features {
                        let feature_sum: T = class_data
                            .par_iter()
                            .map(|row| row[feature_idx])
                            .reduce_with(|a, b| a + b)
                            .unwrap_or(T::zero());

                        class_means[feature_idx] = feature_sum / T::from(n_samples).unwrap();
                    }

                    // Compute variances for this class
                    for feature_idx in 0..n_features {
                        let mean = class_means[feature_idx];
                        let variance_sum: T = class_data
                            .par_iter()
                            .map(|row| {
                                let diff = row[feature_idx] - mean;
                                diff * diff
                            })
                            .reduce_with(|a, b| a + b)
                            .unwrap_or(T::zero());

                        class_variances[feature_idx] = variance_sum / T::from(n_samples).unwrap();
                    }
                }

                (class_means, class_variances)
            })
            .collect();

        // Assemble results into matrices
        let mut means = Array2::zeros((n_classes, n_features));
        let mut variances = Array2::zeros((n_classes, n_features));

        for (class_idx, (class_means, class_variances)) in class_params.into_iter().enumerate() {
            for feature_idx in 0..n_features {
                means[[class_idx, feature_idx]] = class_means[feature_idx];
                variances[[class_idx, feature_idx]] = class_variances[feature_idx];
            }
        }

        (means, variances)
    }

    /// Parallel multinomial parameter estimation
    pub fn estimate_multinomial_params_parallel<T: Float + Send + Sync>(
        data: &Array2<T>,
        labels: &Array1<i32>,
        n_classes: usize,
        alpha: T,
    ) -> Array2<T> {
        let n_features = data.ncols();

        // Compute parameters for each class in parallel and collect results
        let class_log_probs: Vec<Array1<T>> = (0..n_classes)
            .into_par_iter()
            .map(|class_idx| {
                let class_data: Vec<_> = data
                    .rows()
                    .into_iter()
                    .zip(labels.iter())
                    .filter_map(|(row, &label)| {
                        if label == class_idx as i32 {
                            Some(row.to_owned())
                        } else {
                            None
                        }
                    })
                    .collect();

                let mut class_log_prob = Array1::zeros(n_features);

                if !class_data.is_empty() {
                    // Compute feature counts in parallel
                    let feature_counts: Array1<T> = (0..n_features)
                        .into_par_iter()
                        .map(|feature_idx| {
                            class_data
                                .par_iter()
                                .map(|row| row[feature_idx])
                                .reduce_with(|a, b| a + b)
                                .unwrap_or(T::zero())
                                + alpha
                        })
                        .collect::<Vec<_>>()
                        .into();

                    let total_count: T = feature_counts
                        .iter()
                        .copied()
                        .reduce(|a, b| a + b)
                        .unwrap_or(T::zero());

                    // Compute log probabilities
                    for feature_idx in 0..n_features {
                        class_log_prob[feature_idx] =
                            (feature_counts[feature_idx] / total_count).ln();
                    }
                }

                class_log_prob
            })
            .collect();

        // Assemble results into matrix
        let mut feature_log_prob = Array2::zeros((n_classes, n_features));
        for (class_idx, class_log_prob) in class_log_probs.into_iter().enumerate() {
            for feature_idx in 0..n_features {
                feature_log_prob[[class_idx, feature_idx]] = class_log_prob[feature_idx];
            }
        }

        feature_log_prob
    }

    /// Parallel prediction computation
    pub fn predict_parallel<T: Float + Send + Sync>(
        data: &Array2<T>,
        class_log_prior: &Array1<T>,
        feature_log_prob: &Array2<T>,
    ) -> Array1<i32> {
        let predictions: Vec<i32> = (0..data.nrows())
            .into_par_iter()
            .map(|row_idx| {
                let sample = data.row(row_idx);
                let mut best_class = 0;
                let mut best_score = T::neg_infinity();

                for (class_idx, &log_prior) in class_log_prior.iter().enumerate() {
                    let log_likelihood: T = sample
                        .iter()
                        .zip(feature_log_prob.row(class_idx).iter())
                        .map(|(&feature_val, &log_prob)| feature_val * log_prob)
                        .reduce(|a, b| a + b)
                        .unwrap_or(T::zero());

                    let score = log_prior + log_likelihood;
                    if score > best_score {
                        best_score = score;
                        best_class = class_idx;
                    }
                }

                best_class as i32
            })
            .collect();

        Array1::from_vec(predictions)
    }
}

/// Numerical stability improvements
pub mod numerical_stability {
    use super::*;

    /// Numerically stable log probability computation
    pub fn stable_log_prob<T: Float>(x: T) -> T {
        if x <= T::zero() {
            T::neg_infinity()
        } else if x == T::one() {
            T::zero()
        } else {
            x.ln()
        }
    }

    /// Numerically stable log-sum-exp with high precision
    pub fn stable_log_sum_exp<T: Float>(values: &[T]) -> T {
        if values.is_empty() {
            return T::neg_infinity();
        }

        // Handle special cases
        let finite_values: Vec<T> = values.iter().copied().filter(|&x| x.is_finite()).collect();

        if finite_values.is_empty() {
            return T::neg_infinity();
        }

        let max_val = finite_values
            .iter()
            .copied()
            .fold(T::neg_infinity(), |a, b| a.max(b));

        if max_val == T::neg_infinity() {
            return T::neg_infinity();
        }

        // Use Kahan summation for improved numerical stability
        let mut sum = T::zero();
        let mut compensation = T::zero();

        for &value in &finite_values {
            let term = (value - max_val).exp();
            let y = term - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        max_val + sum.ln()
    }

    /// Numerically stable normalization of log probabilities
    pub fn stable_normalize_log_probs<T: Float>(log_probs: &mut [T]) {
        let log_sum = stable_log_sum_exp(log_probs);

        if log_sum.is_finite() {
            for prob in log_probs.iter_mut() {
                *prob = *prob - log_sum;
            }
        }
    }

    /// Numerically stable Gaussian log-pdf computation
    pub fn stable_gaussian_log_pdf<T: Float>(x: T, mean: T, variance: T) -> T {
        if variance <= T::zero() {
            return T::neg_infinity();
        }

        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let diff = x - mean;
        let log_norm = -T::from(0.5).unwrap() * two_pi.ln() - T::from(0.5).unwrap() * variance.ln();
        let log_exp = -T::from(0.5).unwrap() * (diff * diff) / variance;

        log_norm + log_exp
    }

    /// Numerically stable Poisson log-pmf computation
    pub fn stable_poisson_log_pmf<T: Float>(k: i32, lambda: T) -> T {
        if lambda <= T::zero() || k < 0 {
            return T::neg_infinity();
        }

        let k_f = T::from(k).unwrap();
        let log_factorial = log_factorial_stirling(k);

        k_f * lambda.ln() - lambda - log_factorial
    }

    /// Stirling's approximation for log factorial
    pub fn log_factorial_stirling<T: Float>(n: i32) -> T {
        if n <= 0 {
            return T::zero();
        }

        let n_f = T::from(n).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

        if n < 10 {
            // Use exact computation for small values
            (1..=n)
                .map(|i| T::from(i).unwrap().ln())
                .reduce(|a, b| a + b)
                .unwrap_or(T::zero())
        } else {
            // Use Stirling's approximation for larger values
            T::from(0.5).unwrap() * (two_pi * n_f).ln() + n_f * n_f.ln() - n_f
        }
    }
}

/// Cache-friendly data layouts and operations
pub mod cache_optimization {
    use super::*;

    /// Cache-friendly matrix transpose
    pub fn cache_friendly_transpose<T: Float + Send + Sync + Copy>(
        matrix: &Array2<T>,
    ) -> Array2<T> {
        let (rows, cols) = matrix.dim();
        let mut result = Array2::zeros((cols, rows));

        const BLOCK_SIZE: usize = 64; // Cache line size consideration

        for i_block in (0..rows).step_by(BLOCK_SIZE) {
            for j_block in (0..cols).step_by(BLOCK_SIZE) {
                let i_end = (i_block + BLOCK_SIZE).min(rows);
                let j_end = (j_block + BLOCK_SIZE).min(cols);

                for i in i_block..i_end {
                    for j in j_block..j_end {
                        result[[j, i]] = matrix[[i, j]];
                    }
                }
            }
        }

        result
    }

    /// Cache-friendly matrix multiplication
    pub fn cache_friendly_matrix_mult<T: Float + Send + Sync + Copy>(
        a: &Array2<T>,
        b: &Array2<T>,
    ) -> Array2<T> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2, "Matrix dimensions must match for multiplication");

        let mut result = Array2::zeros((m, n));
        const BLOCK_SIZE: usize = 64;

        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                for k_block in (0..k).step_by(BLOCK_SIZE) {
                    let i_end = (i_block + BLOCK_SIZE).min(m);
                    let j_end = (j_block + BLOCK_SIZE).min(n);
                    let k_end = (k_block + BLOCK_SIZE).min(k);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = T::zero();
                            for ki in k_block..k_end {
                                sum = sum + a[[i, ki]] * b[[ki, j]];
                            }
                            result[[i, j]] = result[[i, j]] + sum;
                        }
                    }
                }
            }
        }

        result
    }
}

/// Memory-efficient operations
pub mod memory_efficient {
    use super::*;

    /// Streaming mean and variance computation
    pub struct StreamingStats<T> {
        count: usize,
        mean: T,
        m2: T, // Sum of squares of differences from mean
    }

    impl<T: Float> Default for StreamingStats<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T: Float> StreamingStats<T> {
        pub fn new() -> Self {
            Self {
                count: 0,
                mean: T::zero(),
                m2: T::zero(),
            }
        }

        pub fn update(&mut self, value: T) {
            self.count += 1;
            let delta = value - self.mean;
            self.mean = self.mean + delta / T::from(self.count).unwrap();
            let delta2 = value - self.mean;
            self.m2 = self.m2 + delta * delta2;
        }

        pub fn mean(&self) -> T {
            self.mean
        }

        pub fn variance(&self) -> T {
            if self.count < 2 {
                T::zero()
            } else {
                self.m2 / T::from(self.count - 1).unwrap()
            }
        }

        pub fn count(&self) -> usize {
            self.count
        }
    }

    /// Memory-efficient feature statistics computation
    pub fn compute_feature_stats_streaming<'a, T: Float + 'a>(
        data_chunks: impl Iterator<Item = ArrayView2<'a, T>>,
        n_features: usize,
    ) -> (Array1<T>, Array1<T>) {
        let mut stats: Vec<StreamingStats<T>> =
            (0..n_features).map(|_| StreamingStats::new()).collect();

        for chunk in data_chunks {
            for row in chunk.rows() {
                for (feature_idx, &value) in row.iter().enumerate() {
                    if feature_idx < n_features {
                        stats[feature_idx].update(value);
                    }
                }
            }
        }

        let means: Array1<T> = stats.iter().map(|s| s.mean()).collect::<Vec<_>>().into();
        let variances: Array1<T> = stats
            .iter()
            .map(|s| s.variance())
            .collect::<Vec<_>>()
            .into();

        (means, variances)
    }
}

/// Unsafe performance optimizations for critical paths
///
/// These functions use unsafe code to bypass bounds checking and other
/// safety mechanisms for maximum performance in hot loops.
pub mod unsafe_optimizations {
    use super::*;
    use std::ptr;

    /// Compute log-sum-exp without bounds checking for maximum performance.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `values` slice is not empty
    /// - All indices accessed are within bounds of the slice
    pub unsafe fn unsafe_log_sum_exp_unchecked(values: &[f64]) -> f64 {
        debug_assert!(!values.is_empty(), "values must not be empty");

        // Find maximum value using unchecked access
        let mut max_val = *values.get_unchecked(0);
        for i in 1..values.len() {
            let val = *values.get_unchecked(i);
            if val > max_val {
                max_val = val;
            }
        }

        // Early return for infinite cases
        if !max_val.is_finite() {
            return max_val;
        }

        // Compute sum of exp(x - max) using unchecked access
        let mut sum = 0.0;
        for i in 0..values.len() {
            let val = *values.get_unchecked(i);
            sum += (val - max_val).exp();
        }

        max_val + sum.ln()
    }

    /// Unsafe matrix multiplication with unchecked bounds
    ///
    /// # Safety
    ///
    /// Caller must ensure that:
    /// - Matrix dimensions are compatible (a.ncols() == b.nrows())
    /// - Output array has correct dimensions (a.nrows() x b.ncols())
    /// - All arrays are properly allocated and initialized
    pub unsafe fn unsafe_matrix_multiply_unchecked(
        a: &Array2<f64>,
        b: &Array2<f64>,
        result: &mut Array2<f64>,
    ) {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        debug_assert_eq!(k, k2, "Matrix dimensions must be compatible");
        debug_assert_eq!(result.dim(), (m, n), "Output dimensions must be correct");

        // Get raw pointers for maximum performance
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr = result.as_mut_ptr();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for ki in 0..k {
                    let a_val = *a_ptr.add(i * k + ki);
                    let b_val = *b_ptr.add(ki * n + j);
                    sum += a_val * b_val;
                }
                *result_ptr.add(i * n + j) = sum;
            }
        }
    }

    /// Unsafe vectorized probability normalization
    ///
    /// # Safety
    ///
    /// Caller must ensure that:
    /// - `log_probs` array is properly allocated
    /// - All elements are finite numbers
    /// - Array length is greater than 0
    pub unsafe fn unsafe_normalize_log_probs_unchecked(log_probs: &mut [f64]) {
        debug_assert!(!log_probs.is_empty(), "log_probs must not be empty");

        // Find maximum using unchecked access
        let mut max_val = *log_probs.get_unchecked(0);
        for i in 1..log_probs.len() {
            let val = *log_probs.get_unchecked(i);
            if val > max_val {
                max_val = val;
            }
        }

        // Compute sum using unchecked access
        let mut sum = 0.0;
        for i in 0..log_probs.len() {
            let val = *log_probs.get_unchecked(i);
            sum += (val - max_val).exp();
        }

        let log_sum = max_val + sum.ln();

        // Normalize using unchecked access
        for i in 0..log_probs.len() {
            let val_ptr = log_probs.get_unchecked_mut(i);
            *val_ptr -= log_sum;
        }
    }

    /// Unsafe sparse vector dot product with unchecked indexing
    ///
    /// # Safety
    ///
    /// Caller must ensure that:
    /// - All indices in `indices` are valid for both `values1` and `values2`
    /// - Arrays are properly allocated
    /// - No index appears twice in the indices array
    pub unsafe fn unsafe_sparse_dot_product_unchecked(
        indices: &[usize],
        values1: &[f64],
        values2: &[f64],
    ) -> f64 {
        debug_assert!(
            indices
                .iter()
                .all(|&i| i < values1.len() && i < values2.len()),
            "All indices must be within bounds"
        );

        let mut result = 0.0;
        for &idx in indices {
            let v1 = *values1.get_unchecked(idx);
            let v2 = *values2.get_unchecked(idx);
            result += v1 * v2;
        }
        result
    }

    /// Unsafe memory copy optimized for probability arrays
    ///
    /// # Safety
    ///
    /// Caller must ensure that:
    /// - Both slices have the same length
    /// - Both slices are properly allocated
    /// - No overlapping memory regions
    pub unsafe fn unsafe_copy_probabilities_unchecked(src: &[f64], dst: &mut [f64]) {
        debug_assert_eq!(src.len(), dst.len(), "Arrays must have same length");

        ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), src.len());
    }

    /// Unsafe in-place probability array transformation
    ///
    /// # Safety
    ///
    /// Caller must ensure that:
    /// - Array is properly allocated
    /// - All values are finite
    /// - Transformation function is safe for all inputs
    pub unsafe fn unsafe_transform_probabilities_unchecked<F>(probs: &mut [f64], transform: F)
    where
        F: Fn(f64) -> f64,
    {
        for i in 0..probs.len() {
            let val_ptr = probs.get_unchecked_mut(i);
            *val_ptr = transform(*val_ptr);
        }
    }

    /// Unsafe batch probability computation for hot paths
    ///
    /// # Safety
    ///
    /// Caller must ensure that:
    /// - All arrays have compatible dimensions
    /// - All values are finite
    /// - No null pointers
    pub unsafe fn unsafe_batch_log_likelihood_unchecked(
        features: &Array2<f64>,
        means: &Array2<f64>,
        variances: &Array2<f64>,
        result: &mut Array2<f64>,
    ) {
        let (n_samples, n_features) = features.dim();
        let (n_classes, n_features2) = means.dim();

        debug_assert_eq!(n_features, n_features2, "Feature dimensions must match");
        debug_assert_eq!(
            variances.dim(),
            (n_classes, n_features),
            "Variance dimensions must match"
        );
        debug_assert_eq!(
            result.dim(),
            (n_samples, n_classes),
            "Result dimensions must match"
        );

        let features_ptr = features.as_ptr();
        let means_ptr = means.as_ptr();
        let variances_ptr = variances.as_ptr();
        let result_ptr = result.as_mut_ptr();

        for sample_idx in 0..n_samples {
            for class_idx in 0..n_classes {
                let mut log_likelihood = 0.0;

                for feature_idx in 0..n_features {
                    let feature_val = *features_ptr.add(sample_idx * n_features + feature_idx);
                    let mean_val = *means_ptr.add(class_idx * n_features + feature_idx);
                    let variance_val = *variances_ptr.add(class_idx * n_features + feature_idx);

                    let diff = feature_val - mean_val;
                    let log_prob = -0.5
                        * ((2.0 * std::f64::consts::PI * variance_val).ln()
                            + (diff * diff) / variance_val);

                    log_likelihood += log_prob;
                }

                *result_ptr.add(sample_idx * n_classes + class_idx) = log_likelihood;
            }
        }
    }

    /// Unsafe SIMD-style operations using manual loop unrolling
    ///
    /// # Safety
    ///
    /// Caller must ensure that:
    /// - Array length is divisible by 4 or handle remainder appropriately
    /// - Array is properly allocated
    /// - All values are finite
    pub unsafe fn unsafe_unrolled_log_sum_exp_unchecked(values: &[f64]) -> f64 {
        debug_assert!(!values.is_empty(), "values must not be empty");

        // Find maximum with loop unrolling
        let mut max_val = f64::NEG_INFINITY;
        let len = values.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let v0 = *values.get_unchecked(i);
            let v1 = *values.get_unchecked(i + 1);
            let v2 = *values.get_unchecked(i + 2);
            let v3 = *values.get_unchecked(i + 3);

            max_val = max_val.max(v0).max(v1).max(v2).max(v3);
            i += 4;
        }

        // Handle remainder
        while i < len {
            max_val = max_val.max(*values.get_unchecked(i));
            i += 1;
        }

        if !max_val.is_finite() {
            return max_val;
        }

        // Compute sum with loop unrolling
        let mut sum = 0.0;
        i = 0;

        while i + 4 <= len {
            let v0 = (*values.get_unchecked(i) - max_val).exp();
            let v1 = (*values.get_unchecked(i + 1) - max_val).exp();
            let v2 = (*values.get_unchecked(i + 2) - max_val).exp();
            let v3 = (*values.get_unchecked(i + 3) - max_val).exp();

            sum += v0 + v1 + v2 + v3;
            i += 4;
        }

        while i < len {
            sum += (*values.get_unchecked(i) - max_val).exp();
            i += 1;
        }

        max_val + sum.ln()
    }
}

/// Profile-guided optimization for adaptive performance tuning
///
/// This module provides runtime profiling and adaptive optimization based on
/// actual usage patterns and performance characteristics.
pub mod profile_guided_optimization {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Performance profile for different operation types
    #[derive(Debug, Clone)]
    pub struct PerformanceProfile {
        pub operation_counts: HashMap<String, u64>,
        pub operation_timings: HashMap<String, Vec<Duration>>,
        pub data_characteristics: HashMap<String, f64>,
        pub optimization_effectiveness: HashMap<String, f64>,
    }

    impl Default for PerformanceProfile {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PerformanceProfile {
        pub fn new() -> Self {
            Self {
                operation_counts: HashMap::new(),
                operation_timings: HashMap::new(),
                data_characteristics: HashMap::new(),
                optimization_effectiveness: HashMap::new(),
            }
        }

        /// Record timing for an operation
        pub fn record_operation(&mut self, operation: &str, duration: Duration) {
            *self
                .operation_counts
                .entry(operation.to_string())
                .or_insert(0) += 1;
            self.operation_timings
                .entry(operation.to_string())
                .or_default()
                .push(duration);
        }

        /// Record data characteristics
        pub fn record_data_characteristic(&mut self, metric: &str, value: f64) {
            self.data_characteristics.insert(metric.to_string(), value);
        }

        /// Get average timing for an operation
        pub fn get_average_timing(&self, operation: &str) -> Option<Duration> {
            self.operation_timings.get(operation).map(|timings| {
                let total: Duration = timings.iter().sum();
                total / timings.len() as u32
            })
        }

        /// Get operation frequency
        pub fn get_operation_frequency(&self, operation: &str) -> u64 {
            self.operation_counts.get(operation).copied().unwrap_or(0)
        }
    }

    /// Adaptive optimization strategy based on profiling data
    #[derive(Debug, Clone)]
    pub enum OptimizationStrategy {
        /// Conservative
        Conservative,
        /// Balanced
        Balanced,
        /// Aggressive
        Aggressive,
        /// Custom
        Custom {
            simd_threshold: f64,

            parallel_threshold: usize,

            cache_friendly: bool,
        },
    }

    /// Profile-guided optimizer that adapts optimization strategies
    #[derive(Debug)]
    pub struct ProfileGuidedOptimizer {
        profile: Arc<Mutex<PerformanceProfile>>,
        strategy: OptimizationStrategy,
        adaptation_threshold: f64,
        min_samples: usize,
    }

    impl ProfileGuidedOptimizer {
        /// Create new profile-guided optimizer
        pub fn new(strategy: OptimizationStrategy) -> Self {
            Self {
                profile: Arc::new(Mutex::new(PerformanceProfile::new())),
                strategy,
                adaptation_threshold: 0.1, // 10% improvement threshold
                min_samples: 10,
            }
        }

        /// Record a timed operation
        pub fn time_operation<F, R>(&self, operation: &str, f: F) -> R
        where
            F: FnOnce() -> R,
        {
            let start = Instant::now();
            let result = f();
            let duration = start.elapsed();

            if let Ok(mut profile) = self.profile.lock() {
                profile.record_operation(operation, duration);
            }

            result
        }

        /// Analyze performance profile and recommend optimization strategy
        pub fn recommend_strategy(&self) -> OptimizationStrategy {
            let profile = match self.profile.lock() {
                Ok(profile) => profile,
                Err(_) => return self.strategy.clone(),
            };

            // Analyze computational characteristics
            let matrix_ops_freq = profile.get_operation_frequency("matrix_multiply");
            let simd_ops_freq = profile.get_operation_frequency("log_sum_exp");
            let parallel_ops_freq = profile.get_operation_frequency("parallel_estimation");

            let total_ops = profile.operation_counts.values().sum::<u64>();
            if total_ops < self.min_samples as u64 {
                return self.strategy.clone();
            }

            // Get data size characteristics
            let avg_data_size = profile
                .data_characteristics
                .get("data_size")
                .copied()
                .unwrap_or(1000.0);
            let avg_features = profile
                .data_characteristics
                .get("n_features")
                .copied()
                .unwrap_or(10.0);
            let avg_samples = profile
                .data_characteristics
                .get("n_samples")
                .copied()
                .unwrap_or(1000.0);

            // Recommend strategy based on usage patterns
            if simd_ops_freq as f64 / total_ops as f64 > 0.3 && avg_data_size > 10000.0 {
                OptimizationStrategy::Custom {
                    simd_threshold: 0.8,
                    parallel_threshold: (avg_samples / 4.0) as usize,
                    cache_friendly: avg_features > 50.0,
                }
            } else if parallel_ops_freq as f64 / total_ops as f64 > 0.2 {
                OptimizationStrategy::Aggressive
            } else if matrix_ops_freq as f64 / total_ops as f64 > 0.4 {
                OptimizationStrategy::Balanced
            } else {
                OptimizationStrategy::Conservative
            }
        }

        /// Apply optimizations based on current strategy
        pub fn optimize_log_sum_exp(&self, values: &[f64]) -> f64 {
            match &self.strategy {
                OptimizationStrategy::Conservative => self.time_operation("log_sum_exp", || {
                    numerical_stability::stable_log_sum_exp(values)
                }),
                OptimizationStrategy::Balanced => self.time_operation("log_sum_exp", || {
                    if values.len() > 1000 {
                        simd::log_sum_exp_simd(values)
                    } else {
                        numerical_stability::stable_log_sum_exp(values)
                    }
                }),
                OptimizationStrategy::Aggressive => self.time_operation("log_sum_exp", || {
                    if values.len() > 100 {
                        simd::log_sum_exp_simd(values)
                    } else {
                        numerical_stability::stable_log_sum_exp(values)
                    }
                }),
                OptimizationStrategy::Custom { simd_threshold, .. } => {
                    self.time_operation("log_sum_exp", || {
                        if values.len() as f64 > *simd_threshold * 1000.0 {
                            simd::log_sum_exp_simd(values)
                        } else {
                            numerical_stability::stable_log_sum_exp(values)
                        }
                    })
                }
            }
        }

        /// Optimize matrix operations based on strategy
        pub fn optimize_matrix_multiply<T: Float + Send + Sync + Copy + 'static>(
            &self,
            a: &Array2<T>,
            b: &Array2<T>,
        ) -> Array2<T> {
            let use_cache_friendly = match &self.strategy {
                OptimizationStrategy::Conservative => false,
                OptimizationStrategy::Balanced => a.nrows() * a.ncols() > 10000,
                OptimizationStrategy::Aggressive => a.nrows() * a.ncols() > 1000,
                OptimizationStrategy::Custom { cache_friendly, .. } => *cache_friendly,
            };

            self.time_operation("matrix_multiply", || {
                if use_cache_friendly {
                    cache_optimization::cache_friendly_matrix_mult(a, b)
                } else {
                    // Standard ndarray multiplication
                    a.dot(b)
                }
            })
        }

        /// Optimize parallel operations based on strategy
        pub fn optimize_parallel_estimation<T: Float + Send + Sync + 'static>(
            &self,
            data: &Array2<T>,
            labels: &Array1<i32>,
            n_classes: usize,
        ) -> (Array2<T>, Array2<T>) {
            let use_parallel = match &self.strategy {
                OptimizationStrategy::Conservative => data.nrows() > 10000,
                OptimizationStrategy::Balanced => data.nrows() > 1000,
                OptimizationStrategy::Aggressive => data.nrows() > 100,
                OptimizationStrategy::Custom {
                    parallel_threshold, ..
                } => data.nrows() > *parallel_threshold,
            };

            // Record data characteristics
            if let Ok(mut profile) = self.profile.lock() {
                profile.record_data_characteristic("n_samples", data.nrows() as f64);
                profile.record_data_characteristic("n_features", data.ncols() as f64);
                profile
                    .record_data_characteristic("data_size", (data.nrows() * data.ncols()) as f64);
            }

            self.time_operation("parallel_estimation", || {
                if use_parallel {
                    parallel::estimate_gaussian_params_parallel(data, labels, n_classes)
                } else {
                    // Sequential estimation
                    estimate_gaussian_params_sequential(data, labels, n_classes)
                }
            })
        }

        /// Benchmark different optimization approaches
        pub fn benchmark_optimizations(
            &self,
            test_data: &[f64],
            iterations: usize,
        ) -> HashMap<String, Duration> {
            let mut results = HashMap::new();

            // Benchmark different log_sum_exp implementations
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = numerical_stability::stable_log_sum_exp(test_data);
            }
            results.insert(
                "stable_log_sum_exp".to_string(),
                start.elapsed() / iterations as u32,
            );

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = simd::log_sum_exp_simd(test_data);
            }
            results.insert(
                "simd_log_sum_exp".to_string(),
                start.elapsed() / iterations as u32,
            );

            if !test_data.is_empty() {
                unsafe {
                    let start = Instant::now();
                    for _ in 0..iterations {
                        let _ = unsafe_optimizations::unsafe_log_sum_exp_unchecked(test_data);
                    }
                    results.insert(
                        "unsafe_log_sum_exp".to_string(),
                        start.elapsed() / iterations as u32,
                    );
                }
            }

            results
        }

        /// Adapt strategy based on recent performance
        pub fn adapt_strategy(&mut self) {
            let recommended = self.recommend_strategy();

            // Test if new strategy would be beneficial
            if self.should_adapt_strategy(&recommended) {
                self.strategy = recommended;
            }
        }

        fn should_adapt_strategy(&self, new_strategy: &OptimizationStrategy) -> bool {
            // Simple heuristic: adapt if we have enough samples and the strategy differs significantly
            let profile = match self.profile.lock() {
                Ok(profile) => profile,
                Err(_) => return false,
            };

            let total_ops = profile.operation_counts.values().sum::<u64>();
            total_ops >= self.min_samples as u64
        }

        /// Get current performance statistics
        pub fn get_performance_stats(&self) -> HashMap<String, f64> {
            let mut stats = HashMap::new();

            if let Ok(profile) = self.profile.lock() {
                for (op, count) in &profile.operation_counts {
                    stats.insert(format!("{}_count", op), *count as f64);

                    if let Some(avg_time) = profile.get_average_timing(op) {
                        stats.insert(
                            format!("{}_avg_time_ms", op),
                            avg_time.as_secs_f64() * 1000.0,
                        );
                    }
                }
            }

            stats
        }

        /// Export profile data for analysis
        pub fn export_profile(&self) -> Option<PerformanceProfile> {
            self.profile.lock().ok().map(|profile| profile.clone())
        }

        /// Reset profiling data
        pub fn reset_profile(&self) {
            if let Ok(mut profile) = self.profile.lock() {
                *profile = PerformanceProfile::new();
            }
        }
    }

    /// Sequential Gaussian parameter estimation (for comparison with parallel version)
    fn estimate_gaussian_params_sequential<T: Float + 'static>(
        data: &Array2<T>,
        labels: &Array1<i32>,
        n_classes: usize,
    ) -> (Array2<T>, Array2<T>) {
        let n_features = data.ncols();
        let mut means = Array2::zeros((n_classes, n_features));
        let mut variances = Array2::zeros((n_classes, n_features));

        for class_idx in 0..n_classes {
            let class_data: Vec<_> = data
                .rows()
                .into_iter()
                .zip(labels.iter())
                .filter_map(|(row, &label)| {
                    if label == class_idx as i32 {
                        Some(row.to_owned())
                    } else {
                        None
                    }
                })
                .collect();

            if !class_data.is_empty() {
                let n_samples = class_data.len();

                // Compute means
                for feature_idx in 0..n_features {
                    let feature_sum: T = class_data
                        .iter()
                        .map(|row| row[feature_idx])
                        .fold(T::zero(), |acc, val| acc + val);

                    means[[class_idx, feature_idx]] = feature_sum / T::from(n_samples).unwrap();
                }

                // Compute variances
                for feature_idx in 0..n_features {
                    let mean = means[[class_idx, feature_idx]];
                    let variance_sum: T = class_data
                        .iter()
                        .map(|row| {
                            let diff = row[feature_idx] - mean;
                            diff * diff
                        })
                        .fold(T::zero(), |acc, val| acc + val);

                    variances[[class_idx, feature_idx]] =
                        variance_sum / T::from(n_samples).unwrap();
                }
            }
        }

        (means, variances)
    }

    /// Factory for creating optimizers with different configurations
    pub struct OptimizerFactory;

    impl OptimizerFactory {
        /// Create optimizer for small datasets
        pub fn for_small_datasets() -> ProfileGuidedOptimizer {
            ProfileGuidedOptimizer::new(OptimizationStrategy::Conservative)
        }

        /// Create optimizer for medium datasets
        pub fn for_medium_datasets() -> ProfileGuidedOptimizer {
            ProfileGuidedOptimizer::new(OptimizationStrategy::Balanced)
        }

        /// Create optimizer for large datasets
        pub fn for_large_datasets() -> ProfileGuidedOptimizer {
            ProfileGuidedOptimizer::new(OptimizationStrategy::Aggressive)
        }

        /// Create optimizer with custom configuration
        pub fn with_custom_config(
            simd_threshold: f64,
            parallel_threshold: usize,
            cache_friendly: bool,
        ) -> ProfileGuidedOptimizer {
            ProfileGuidedOptimizer::new(OptimizationStrategy::Custom {
                simd_threshold,
                parallel_threshold,
                cache_friendly,
            })
        }

        /// Create adaptive optimizer that learns from usage patterns
        pub fn adaptive() -> ProfileGuidedOptimizer {
            let mut optimizer = ProfileGuidedOptimizer::new(OptimizationStrategy::Balanced);
            optimizer.adaptation_threshold = 0.05; // More sensitive to performance changes
            optimizer.min_samples = 5; // Adapt more quickly
            optimizer
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_log_sum_exp() {
        let values = vec![-1.0, -2.0, -3.0];
        let result = simd::log_sum_exp_simd(&values);
        let expected = -1.0 + (1.0 + (-1.0f64).exp() + (-2.0f64).exp()).ln();
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_softmax() {
        let log_probs = vec![-1.0, -2.0, -3.0];
        let result = simd::softmax_simd(&log_probs);
        let sum: f64 = result.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_gaussian_estimation() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let (means, variances) = parallel::estimate_gaussian_params_parallel(&data, &labels, 2);

        assert_relative_eq!(means[[0, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(means[[0, 1]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(means[[1, 0]], 6.0, epsilon = 1e-10);
        assert_relative_eq!(means[[1, 1]], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_numerical_stability() {
        let values = vec![-100.0, -200.0, -300.0];
        let result = numerical_stability::stable_log_sum_exp(&values);
        assert!(result.is_finite());
        assert_relative_eq!(result, -100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_streaming_stats() {
        let mut stats = memory_efficient::StreamingStats::new();
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];

        for &value in &values {
            stats.update(value);
        }

        assert_relative_eq!(stats.mean(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(stats.variance(), 2.5, epsilon = 1e-10);
        assert_eq!(stats.count(), 5);
    }

    #[test]
    fn test_profile_guided_optimizer_creation() {
        let optimizer = profile_guided_optimization::ProfileGuidedOptimizer::new(
            profile_guided_optimization::OptimizationStrategy::Balanced,
        );

        let test_data = vec![-1.0, -2.0, -3.0, -4.0];
        let result = optimizer.optimize_log_sum_exp(&test_data);
        assert!(result.is_finite());
    }

    #[test]
    fn test_profile_guided_optimization_strategies() {
        let test_data =
            Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        // Test conservative strategy
        let conservative = profile_guided_optimization::OptimizerFactory::for_small_datasets();
        let (means_c, vars_c) = conservative.optimize_parallel_estimation(&test_data, &labels, 2);
        assert_eq!(means_c.shape(), &[2, 5]);
        assert_eq!(vars_c.shape(), &[2, 5]);

        // Test aggressive strategy
        let aggressive = profile_guided_optimization::OptimizerFactory::for_large_datasets();
        let (means_a, vars_a) = aggressive.optimize_parallel_estimation(&test_data, &labels, 2);
        assert_eq!(means_a.shape(), &[2, 5]);
        assert_eq!(vars_a.shape(), &[2, 5]);
    }

    #[test]
    fn test_performance_profile() {
        let mut profile = profile_guided_optimization::PerformanceProfile::new();

        profile.record_operation("test_op", std::time::Duration::from_millis(10));
        profile.record_operation("test_op", std::time::Duration::from_millis(20));
        profile.record_data_characteristic("data_size", 1000.0);

        assert_eq!(profile.get_operation_frequency("test_op"), 2);
        assert!(profile.get_average_timing("test_op").is_some());
        assert_eq!(profile.data_characteristics.get("data_size"), Some(&1000.0));
    }

    #[test]
    fn test_optimizer_factory() {
        let small = profile_guided_optimization::OptimizerFactory::for_small_datasets();
        let medium = profile_guided_optimization::OptimizerFactory::for_medium_datasets();
        let large = profile_guided_optimization::OptimizerFactory::for_large_datasets();
        let adaptive = profile_guided_optimization::OptimizerFactory::adaptive();

        let test_data = vec![1.0, 2.0, 3.0];

        // All should work for basic operations
        let _ = small.optimize_log_sum_exp(&test_data);
        let _ = medium.optimize_log_sum_exp(&test_data);
        let _ = large.optimize_log_sum_exp(&test_data);
        let _ = adaptive.optimize_log_sum_exp(&test_data);
    }

    #[test]
    fn test_benchmarking() {
        let optimizer = profile_guided_optimization::ProfileGuidedOptimizer::new(
            profile_guided_optimization::OptimizationStrategy::Balanced,
        );

        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let benchmarks = optimizer.benchmark_optimizations(&test_data, 10);

        assert!(benchmarks.contains_key("stable_log_sum_exp"));
        assert!(benchmarks.contains_key("simd_log_sum_exp"));
        assert!(benchmarks.contains_key("unsafe_log_sum_exp"));
    }

    #[test]
    fn test_strategy_recommendation() {
        let mut optimizer = profile_guided_optimization::ProfileGuidedOptimizer::new(
            profile_guided_optimization::OptimizationStrategy::Conservative,
        );

        // Simulate multiple operations to build profile
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for _ in 0..15 {
            let _ = optimizer.optimize_log_sum_exp(&test_data);
        }

        // Should recommend a strategy based on usage patterns
        let recommended = optimizer.recommend_strategy();
        match recommended {
            profile_guided_optimization::OptimizationStrategy::Conservative
            | profile_guided_optimization::OptimizationStrategy::Balanced
            | profile_guided_optimization::OptimizationStrategy::Aggressive
            | profile_guided_optimization::OptimizationStrategy::Custom { .. } => {
                // Any of these is valid
            }
        }
    }

    #[test]
    fn test_performance_stats() {
        let optimizer = profile_guided_optimization::ProfileGuidedOptimizer::new(
            profile_guided_optimization::OptimizationStrategy::Balanced,
        );

        let test_data = vec![1.0, 2.0, 3.0];
        let _ = optimizer.optimize_log_sum_exp(&test_data);

        let stats = optimizer.get_performance_stats();
        assert!(stats.contains_key("log_sum_exp_count"));
        assert!(stats.contains_key("log_sum_exp_avg_time_ms"));
    }

    #[test]
    fn test_profile_export_reset() {
        let optimizer = profile_guided_optimization::ProfileGuidedOptimizer::new(
            profile_guided_optimization::OptimizationStrategy::Balanced,
        );

        let test_data = vec![1.0, 2.0, 3.0];
        let _ = optimizer.optimize_log_sum_exp(&test_data);

        // Export profile
        let profile = optimizer.export_profile();
        assert!(profile.is_some());

        let exported = profile.unwrap();
        assert!(exported.operation_counts.get("log_sum_exp").is_some());

        // Reset profile
        optimizer.reset_profile();
        let stats_after_reset = optimizer.get_performance_stats();
        assert_eq!(
            stats_after_reset
                .get("log_sum_exp_count")
                .copied()
                .unwrap_or(0.0),
            0.0
        );
    }
}
