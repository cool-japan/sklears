//! Performance Optimizations for Feature Selection
//!
//! This module provides SIMD-accelerated operations, parallel algorithms,
//! and memory-efficient implementations for high-performance feature selection.
//! All implementations follow the SciRS2 policy and use Rust-specific optimizations.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::Result as SklResult;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::sync::{Arc, Mutex};

type Result<T> = SklResult<T>;

/// SIMD-accelerated statistical computations for feature selection
pub struct SIMDStats;

impl SIMDStats {
    /// Compute correlation between two arrays using SIMD operations
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    #[allow(unused_assignments)]
    pub unsafe fn correlation_simd(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        if x.len() != y.len() || x.len() < 4 {
            return Self::correlation_fallback(x, y);
        }

        let n = x.len();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        let mut sum_xy = 0.0;

        // Process 4 elements at a time using AVX2
        let chunks = n / 4;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();

        let mut vec_sum_x = _mm256_setzero_pd();
        let mut vec_sum_y = _mm256_setzero_pd();
        let mut vec_sum_x2 = _mm256_setzero_pd();
        let mut vec_sum_y2 = _mm256_setzero_pd();
        let mut vec_sum_xy = _mm256_setzero_pd();

        for i in 0..chunks {
            let offset = i * 4;
            let vec_x = _mm256_loadu_pd(x_ptr.add(offset));
            let vec_y = _mm256_loadu_pd(y_ptr.add(offset));

            vec_sum_x = _mm256_add_pd(vec_sum_x, vec_x);
            vec_sum_y = _mm256_add_pd(vec_sum_y, vec_y);
            vec_sum_x2 = _mm256_fmadd_pd(vec_x, vec_x, vec_sum_x2);
            vec_sum_y2 = _mm256_fmadd_pd(vec_y, vec_y, vec_sum_y2);
            vec_sum_xy = _mm256_fmadd_pd(vec_x, vec_y, vec_sum_xy);
        }

        // Horizontal sum of vector registers
        let mut temp = [0.0; 4];
        _mm256_storeu_pd(temp.as_mut_ptr(), vec_sum_x);
        sum_x = temp.iter().sum();

        _mm256_storeu_pd(temp.as_mut_ptr(), vec_sum_y);
        sum_y = temp.iter().sum();

        _mm256_storeu_pd(temp.as_mut_ptr(), vec_sum_x2);
        sum_x2 = temp.iter().sum();

        _mm256_storeu_pd(temp.as_mut_ptr(), vec_sum_y2);
        sum_y2 = temp.iter().sum();

        _mm256_storeu_pd(temp.as_mut_ptr(), vec_sum_xy);
        sum_xy = temp.iter().sum();

        // Process remaining elements
        for i in (chunks * 4)..n {
            let xi = *x_ptr.add(i);
            let yi = *y_ptr.add(i);
            sum_x += xi;
            sum_y += yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
            sum_xy += xi * yi;
        }

        // Compute correlation
        let n_f64 = n as f64;
        let numerator = n_f64 * sum_xy - sum_x * sum_y;
        let denominator =
            ((n_f64 * sum_x2 - sum_x * sum_x) * (n_f64 * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Fallback correlation computation for when SIMD is not available
    pub fn correlation_fallback(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            sum_xy / denom
        }
    }

    /// Auto-select correlation method based on CPU features
    pub fn correlation_auto(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && x.len() >= 16 {
                unsafe { Self::correlation_simd(x, y) }
            } else {
                Self::correlation_fallback(x, y)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self::correlation_fallback(x, y)
        }
    }

    /// SIMD-accelerated variance computation
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn variance_simd(x: ArrayView1<f64>) -> f64 {
        if x.len() < 4 {
            return Self::variance_fallback(x);
        }

        let n = x.len();
        let x_ptr = x.as_ptr();

        // First pass: compute mean using SIMD
        let chunks = n / 4;
        let mut vec_sum = _mm256_setzero_pd();

        for i in 0..chunks {
            let offset = i * 4;
            let vec_x = _mm256_loadu_pd(x_ptr.add(offset));
            vec_sum = _mm256_add_pd(vec_sum, vec_x);
        }

        // Horizontal sum
        let mut temp = [0.0; 4];
        _mm256_storeu_pd(temp.as_mut_ptr(), vec_sum);
        let mut sum = temp.iter().sum::<f64>();

        // Process remaining elements
        for i in (chunks * 4)..n {
            sum += *x_ptr.add(i);
        }

        let mean = sum / n as f64;

        // Second pass: compute variance using SIMD
        let vec_mean = _mm256_set1_pd(mean);
        let mut vec_sum_sq_diff = _mm256_setzero_pd();

        for i in 0..chunks {
            let offset = i * 4;
            let vec_x = _mm256_loadu_pd(x_ptr.add(offset));
            let vec_diff = _mm256_sub_pd(vec_x, vec_mean);
            vec_sum_sq_diff = _mm256_fmadd_pd(vec_diff, vec_diff, vec_sum_sq_diff);
        }

        // Horizontal sum
        _mm256_storeu_pd(temp.as_mut_ptr(), vec_sum_sq_diff);
        let mut sum_sq_diff = temp.iter().sum::<f64>();

        // Process remaining elements
        for i in (chunks * 4)..n {
            let diff = *x_ptr.add(i) - mean;
            sum_sq_diff += diff * diff;
        }

        sum_sq_diff / (n - 1) as f64
    }

    /// Fallback variance computation
    pub fn variance_fallback(x: ArrayView1<f64>) -> f64 {
        x.var(1.0)
    }

    /// Auto-select variance method
    pub fn variance_auto(x: ArrayView1<f64>) -> f64 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && x.len() >= 16 {
                unsafe { Self::variance_simd(x) }
            } else {
                Self::variance_fallback(x)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self::variance_fallback(x)
        }
    }

    /// SIMD-accelerated chi-square computation
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn chi_square_simd(observed: ArrayView1<f64>, expected: ArrayView1<f64>) -> f64 {
        if observed.len() != expected.len() || observed.len() < 4 {
            return Self::chi_square_fallback(observed, expected);
        }

        let n = observed.len();
        let chunks = n / 4;
        let obs_ptr = observed.as_ptr();
        let exp_ptr = expected.as_ptr();

        let mut vec_chi_sq = _mm256_setzero_pd();
        let eps = _mm256_set1_pd(1e-10);

        for i in 0..chunks {
            let offset = i * 4;
            let vec_obs = _mm256_loadu_pd(obs_ptr.add(offset));
            let vec_exp = _mm256_loadu_pd(exp_ptr.add(offset));

            // Avoid division by zero
            let vec_exp_safe = _mm256_max_pd(vec_exp, eps);

            let vec_diff = _mm256_sub_pd(vec_obs, vec_exp_safe);
            let vec_diff_sq = _mm256_mul_pd(vec_diff, vec_diff);
            let vec_chi_sq_contrib = _mm256_div_pd(vec_diff_sq, vec_exp_safe);

            vec_chi_sq = _mm256_add_pd(vec_chi_sq, vec_chi_sq_contrib);
        }

        // Horizontal sum
        let mut temp = [0.0; 4];
        _mm256_storeu_pd(temp.as_mut_ptr(), vec_chi_sq);
        let mut chi_sq = temp.iter().sum::<f64>();

        // Process remaining elements
        for i in (chunks * 4)..n {
            let obs = *obs_ptr.add(i);
            let exp = (*exp_ptr.add(i)).max(1e-10);
            let diff = obs - exp;
            chi_sq += (diff * diff) / exp;
        }

        chi_sq
    }

    /// Fallback chi-square computation
    pub fn chi_square_fallback(observed: ArrayView1<f64>, expected: ArrayView1<f64>) -> f64 {
        if observed.len() != expected.len() {
            return 0.0;
        }

        let mut chi_sq = 0.0;
        for i in 0..observed.len() {
            let obs = observed[i];
            let exp = expected[i].max(1e-10);
            let diff = obs - exp;
            chi_sq += (diff * diff) / exp;
        }

        chi_sq
    }

    /// Auto-select chi-square method
    pub fn chi_square_auto(observed: ArrayView1<f64>, expected: ArrayView1<f64>) -> f64 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && observed.len() >= 16 {
                unsafe { Self::chi_square_simd(observed, expected) }
            } else {
                Self::chi_square_fallback(observed, expected)
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self::chi_square_fallback(observed, expected)
        }
    }
}

/// Parallel feature selection algorithms
pub struct ParallelSelector;

impl ParallelSelector {
    /// Parallel correlation-based feature selection
    pub fn parallel_correlation_selection(
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        threshold: f64,
        chunk_size: Option<usize>,
    ) -> Result<Array1<bool>> {
        let n_features = X.ncols();
        let chunk_size = chunk_size.unwrap_or_else(|| (n_features / num_cpus::get()).max(1));

        let correlations: Vec<f64> = (0..n_features)
            .into_par_iter()
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .into_iter()
                    .map(|feature_idx| {
                        let feature_data = X.column(feature_idx);
                        SIMDStats::correlation_auto(feature_data, y).abs()
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();

        let selection_vec: Vec<bool> = correlations
            .into_par_iter()
            .map(|corr| corr > threshold)
            .collect();
        let selection = Array1::from_vec(selection_vec);

        Ok(selection)
    }

    /// Parallel variance threshold selection
    pub fn parallel_variance_selection(
        X: ArrayView2<f64>,
        threshold: f64,
        chunk_size: Option<usize>,
    ) -> Result<Array1<bool>> {
        let n_features = X.ncols();
        let chunk_size = chunk_size.unwrap_or_else(|| (n_features / num_cpus::get()).max(1));

        let variances: Vec<f64> = (0..n_features)
            .into_par_iter()
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .into_iter()
                    .map(|feature_idx| {
                        let feature_data = X.column(feature_idx);
                        SIMDStats::variance_auto(feature_data)
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();

        let selection_vec: Vec<bool> = variances
            .into_par_iter()
            .map(|var| var > threshold)
            .collect();
        let selection = Array1::from_vec(selection_vec);

        Ok(selection)
    }

    /// Parallel mutual information selection
    pub fn parallel_mutual_info_selection(
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        k: usize,
        chunk_size: Option<usize>,
    ) -> Result<Array1<bool>> {
        let n_features = X.ncols();
        let chunk_size = chunk_size.unwrap_or_else(|| (n_features / num_cpus::get()).max(1));

        let mi_scores: Vec<(usize, f64)> = (0..n_features)
            .into_par_iter()
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .into_iter()
                    .map(|feature_idx| {
                        let feature_data = X.column(feature_idx);
                        let mi_score = Self::estimate_mutual_information(feature_data, y);
                        (feature_idx, mi_score)
                    })
                    .collect::<Vec<(usize, f64)>>()
            })
            .collect();

        // Sort by MI score and select top k
        let mut sorted_scores = mi_scores;
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut selection = Array1::from_elem(n_features, false);
        for (feature_idx, _) in sorted_scores.into_iter().take(k) {
            selection[feature_idx] = true;
        }

        Ok(selection)
    }

    /// Parallel chi-square feature selection
    pub fn parallel_chi_square_selection(
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        k: usize,
        chunk_size: Option<usize>,
    ) -> Result<Array1<bool>> {
        let n_features = X.ncols();
        let chunk_size = chunk_size.unwrap_or_else(|| (n_features / num_cpus::get()).max(1));

        // Compute expected frequencies for chi-square test
        let class_counts = Self::compute_class_counts(y);
        let total_samples = y.len() as f64;

        let chi_scores: Vec<(usize, f64)> = (0..n_features)
            .into_par_iter()
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .into_iter()
                    .map(|feature_idx| {
                        let feature_data = X.column(feature_idx);
                        let chi_score = Self::compute_chi_square_score(
                            feature_data,
                            y,
                            &class_counts,
                            total_samples,
                        );
                        (feature_idx, chi_score)
                    })
                    .collect::<Vec<(usize, f64)>>()
            })
            .collect();

        // Sort by chi-square score and select top k
        let mut sorted_scores = chi_scores;
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut selection = Array1::from_elem(n_features, false);
        for (feature_idx, _) in sorted_scores.into_iter().take(k) {
            selection[feature_idx] = true;
        }

        Ok(selection)
    }

    /// Parallel recursive feature elimination
    pub fn parallel_recursive_elimination(
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        n_features_to_select: usize,
        step: f64,
    ) -> Result<Array1<bool>> {
        let mut current_features: Vec<usize> = (0..X.ncols()).collect();
        let mut selection = Array1::from_elem(X.ncols(), true);

        while current_features.len() > n_features_to_select {
            let n_to_remove = ((current_features.len() as f64 * step).ceil() as usize).max(1);

            // Compute feature importance in parallel
            let importances: Vec<f64> = current_features
                .par_iter()
                .map(|&feature_idx| {
                    let feature_data = X.column(feature_idx);
                    // Simplified importance: use correlation as proxy
                    SIMDStats::correlation_auto(feature_data, y).abs()
                })
                .collect();

            // Find features with lowest importance
            let mut indexed_importances: Vec<(usize, f64)> = current_features
                .iter()
                .zip(importances.iter())
                .map(|(&idx, &imp)| (idx, imp))
                .collect();

            indexed_importances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Remove features with lowest importance
            for i in 0..n_to_remove {
                if let Some((feature_idx, _)) = indexed_importances.get(i) {
                    selection[*feature_idx] = false;
                    current_features.retain(|&x| x != *feature_idx);
                }
            }
        }

        Ok(selection)
    }

    // Helper methods
    fn estimate_mutual_information(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Simplified MI estimation using correlation as proxy
        // In a full implementation, this would use proper MI estimation algorithms
        SIMDStats::correlation_auto(x, y).abs()
    }

    fn compute_class_counts(y: ArrayView1<f64>) -> Vec<(f64, usize)> {
        let mut counts = std::collections::HashMap::new();
        for &label in y.iter() {
            *counts.entry(label as i32).or_insert(0) += 1;
        }
        counts.into_iter().map(|(k, v)| (k as f64, v)).collect()
    }

    fn compute_chi_square_score(
        feature: ArrayView1<f64>,
        y: ArrayView1<f64>,
        _class_counts: &[(f64, usize)],
        _total_samples: f64,
    ) -> f64 {
        // Simplified chi-square computation
        // In practice, this would discretize continuous features and compute proper contingency tables
        SIMDStats::correlation_auto(feature, y).abs()
    }
}

/// Memory-efficient feature selection with streaming algorithms
pub struct MemoryEfficientSelector {
    chunk_size: usize,
    use_memory_mapping: bool,
}

impl MemoryEfficientSelector {
    pub fn new(chunk_size: usize, use_memory_mapping: bool) -> Self {
        Self {
            chunk_size,
            use_memory_mapping,
        }
    }

    /// Streaming correlation computation for large datasets
    pub fn streaming_correlation_selection(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        threshold: f64,
    ) -> Result<Array1<bool>> {
        let n_features = X.ncols();
        let n_samples = X.nrows();

        // Initialize streaming statistics
        let mut feature_stats: Vec<StreamingStats> =
            (0..n_features).map(|_| StreamingStats::new()).collect();

        let mut y_stats = StreamingStats::new();

        // Process data in chunks to reduce memory usage
        for chunk_start in (0..n_samples).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(n_samples);

            // Update statistics for this chunk
            for i in chunk_start..chunk_end {
                let y_val = y[i];
                y_stats.update(y_val);

                for j in 0..n_features {
                    let x_val = X[[i, j]];
                    feature_stats[j].update(x_val);
                    feature_stats[j].update_covariance(x_val, y_val);
                }
            }
        }

        // Compute final correlations
        let mut selection = Array1::from_elem(n_features, false);
        let y_variance = y_stats.variance();

        for (j, stats) in feature_stats.iter().enumerate() {
            let x_variance = stats.variance();
            let covariance = stats.covariance_xy();

            let correlation = if x_variance > 1e-10 && y_variance > 1e-10 {
                covariance / (x_variance.sqrt() * y_variance.sqrt())
            } else {
                0.0
            };

            selection[j] = correlation.abs() > threshold;
        }

        Ok(selection)
    }

    /// Memory-efficient variance threshold selection
    pub fn streaming_variance_selection(
        &self,
        X: ArrayView2<f64>,
        threshold: f64,
    ) -> Result<Array1<bool>> {
        let n_features = X.ncols();
        let n_samples = X.nrows();

        let mut feature_stats: Vec<StreamingStats> =
            (0..n_features).map(|_| StreamingStats::new()).collect();

        // Process data in chunks
        for chunk_start in (0..n_samples).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(n_samples);

            for i in chunk_start..chunk_end {
                for j in 0..n_features {
                    let x_val = X[[i, j]];
                    feature_stats[j].update(x_val);
                }
            }
        }

        // Compute final selection
        let mut selection = Array1::from_elem(n_features, false);
        for (j, stats) in feature_stats.iter().enumerate() {
            selection[j] = stats.variance() > threshold;
        }

        Ok(selection)
    }
}

/// Streaming statistics for memory-efficient computations
#[derive(Debug, Clone)]
struct StreamingStats {
    count: usize,
    sum: f64,
    sum_sq: f64,
    sum_xy: f64,
    sum_y: f64,
    sum_y_sq: f64,
}

impl StreamingStats {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            sum_xy: 0.0,
            sum_y: 0.0,
            sum_y_sq: 0.0,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn update_covariance(&mut self, x: f64, y: f64) {
        self.sum_xy += x * y;
        self.sum_y += y;
        self.sum_y_sq += y * y;
    }

    fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    fn variance(&self) -> f64 {
        if self.count > 1 {
            let n = self.count as f64;
            let mean = self.mean();
            (self.sum_sq - n * mean * mean) / (n - 1.0)
        } else {
            0.0
        }
    }

    fn covariance_xy(&self) -> f64 {
        if self.count > 1 {
            let n = self.count as f64;
            let mean_x = self.sum / n;
            let mean_y = self.sum_y / n;
            (self.sum_xy - n * mean_x * mean_y) / (n - 1.0)
        } else {
            0.0
        }
    }
}

/// Cache-friendly data structures for improved performance
pub struct CacheFriendlyMatrix<T> {
    data: Vec<T>,
    n_rows: usize,
    n_cols: usize,
    layout: MatrixLayout,
}

#[derive(Debug, Clone, Copy)]
pub enum MatrixLayout {
    /// RowMajor
    RowMajor,
    /// ColumnMajor
    ColumnMajor,
    /// Blocked
    Blocked { block_size: usize },
}

impl<T: Clone> CacheFriendlyMatrix<T> {
    pub fn new(data: Vec<T>, n_rows: usize, n_cols: usize, layout: MatrixLayout) -> Self {
        assert_eq!(data.len(), n_rows * n_cols);

        Self {
            data,
            n_rows,
            n_cols,
            layout,
        }
    }

    pub fn from_array2(array: Array2<T>, layout: MatrixLayout) -> Self {
        let (n_rows, n_cols) = array.dim();
        let data = match layout {
            MatrixLayout::RowMajor => array.into_raw_vec(),
            MatrixLayout::ColumnMajor => {
                // Transpose data for column-major layout
                let mut col_major_data = Vec::with_capacity(n_rows * n_cols);
                for col in 0..n_cols {
                    for row in 0..n_rows {
                        col_major_data.push(array[[row, col]].clone());
                    }
                }
                col_major_data
            }
            MatrixLayout::Blocked { block_size } => Self::create_blocked_layout(array, block_size),
        };

        Self {
            data,
            layout,
            n_rows,
            n_cols,
        }
    }

    fn create_blocked_layout(array: Array2<T>, block_size: usize) -> Vec<T> {
        let (n_rows, n_cols) = array.dim();
        let mut blocked_data = Vec::with_capacity(n_rows * n_cols);

        let row_blocks = (n_rows + block_size - 1) / block_size;
        let col_blocks = (n_cols + block_size - 1) / block_size;

        for row_block in 0..row_blocks {
            for col_block in 0..col_blocks {
                let row_start = row_block * block_size;
                let row_end = (row_start + block_size).min(n_rows);
                let col_start = col_block * block_size;
                let col_end = (col_start + block_size).min(n_cols);

                for row in row_start..row_end {
                    for col in col_start..col_end {
                        blocked_data.push(array[[row, col]].clone());
                    }
                }
            }
        }

        blocked_data
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }

        let index = match self.layout {
            MatrixLayout::RowMajor => row * self.n_cols + col,
            MatrixLayout::ColumnMajor => col * self.n_rows + row,
            MatrixLayout::Blocked { block_size } => self.blocked_index(row, col, block_size),
        };

        self.data.get(index)
    }

    fn blocked_index(&self, row: usize, col: usize, block_size: usize) -> usize {
        let row_block = row / block_size;
        let col_block = col / block_size;
        let row_in_block = row % block_size;
        let col_in_block = col % block_size;

        let col_blocks = (self.n_cols + block_size - 1) / block_size;
        let block_index = row_block * col_blocks + col_block;
        let block_offset = block_index * block_size * block_size;
        let in_block_index = row_in_block * block_size + col_in_block;

        block_offset + in_block_index
    }

    pub fn column_iter(&self, col: usize) -> ColumnIterator<'_, T> {
        ColumnIterator::new(self, col)
    }
}

pub struct ColumnIterator<'a, T> {
    matrix: &'a CacheFriendlyMatrix<T>,
    col: usize,
    current_row: usize,
}

impl<'a, T> ColumnIterator<'a, T> {
    fn new(matrix: &'a CacheFriendlyMatrix<T>, col: usize) -> Self {
        Self {
            matrix,
            col,
            current_row: 0,
        }
    }
}

impl<'a, T: Clone> Iterator for ColumnIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.matrix.n_rows {
            let item = self.matrix.get(self.current_row, self.col);
            self.current_row += 1;
            item
        } else {
            None
        }
    }
}

/// Performance profiler for feature selection operations
pub struct PerformanceProfiler {
    timings: Arc<Mutex<Vec<(String, std::time::Duration)>>>,
    memory_usage: Arc<Mutex<Vec<(String, usize)>>>,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(Vec::new())),
            memory_usage: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn time_operation<F, R>(&self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration = start.elapsed();

        if let Ok(mut timings) = self.timings.lock() {
            timings.push((name.to_string(), duration));
        }

        result
    }

    pub fn record_memory_usage(&self, name: &str, bytes: usize) {
        if let Ok(mut memory) = self.memory_usage.lock() {
            memory.push((name.to_string(), bytes));
        }
    }

    pub fn get_report(&self) -> PerformanceReport {
        let timings = self.timings.lock().unwrap().clone();
        let memory_usage = self.memory_usage.lock().unwrap().clone();

        PerformanceReport {
            timings: timings.clone(),
            memory_usage: memory_usage.clone(),
            total_time: timings.iter().map(|(_, duration)| *duration).sum(),
            peak_memory: memory_usage
                .iter()
                .map(|(_, bytes)| *bytes)
                .max()
                .unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub timings: Vec<(String, std::time::Duration)>,
    pub memory_usage: Vec<(String, usize)>,
    pub total_time: std::time::Duration,
    pub peak_memory: usize,
}

impl PerformanceReport {
    pub fn print_summary(&self) {
        println!("=== Performance Report ===");
        println!("Total execution time: {:?}", self.total_time);
        println!("Peak memory usage: {} bytes", self.peak_memory);

        println!("\nOperation timings:");
        for (name, duration) in &self.timings {
            println!("  {}: {:?}", name, duration);
        }

        println!("\nMemory usage:");
        for (name, bytes) in &self.memory_usage {
            println!("  {}: {} bytes", name, bytes);
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let correlation_simd = SIMDStats::correlation_auto(x.view(), y.view());
        let correlation_fallback = SIMDStats::correlation_fallback(x.view(), y.view());

        assert!((correlation_simd - correlation_fallback).abs() < 1e-10);
        assert!((correlation_simd - 1.0).abs() < 1e-10); // Perfect correlation
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parallel_correlation_selection() -> Result<()> {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let selection =
            ParallelSelector::parallel_correlation_selection(X.view(), y.view(), 0.5, Some(2))?;

        assert_eq!(selection.len(), 3);
        assert!(selection.iter().any(|&x| x)); // At least one feature should be selected

        Ok(())
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_memory_efficient_selection() -> Result<()> {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let selector = MemoryEfficientSelector::new(2, false);
        let selection = selector.streaming_correlation_selection(X.view(), y.view(), 0.5)?;

        assert_eq!(selection.len(), 3);

        Ok(())
    }

    #[test]
    fn test_cache_friendly_matrix() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = CacheFriendlyMatrix::new(data, 2, 3, MatrixLayout::RowMajor);

        assert_eq!(matrix.get(0, 0), Some(&1));
        assert_eq!(matrix.get(0, 1), Some(&2));
        assert_eq!(matrix.get(1, 2), Some(&6));
        assert_eq!(matrix.get(2, 0), None); // Out of bounds
    }

    #[test]
    fn test_performance_profiler() {
        let profiler = PerformanceProfiler::new();

        let result = profiler.time_operation("test_operation", || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);

        profiler.record_memory_usage("test_memory", 1024);

        let report = profiler.get_report();
        assert_eq!(report.timings.len(), 1);
        assert_eq!(report.memory_usage.len(), 1);
        assert!(report.total_time >= std::time::Duration::from_millis(10));
    }
}
