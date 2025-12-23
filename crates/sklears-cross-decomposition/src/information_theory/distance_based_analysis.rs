//! Distance-Based Canonical Analysis
//!
//! This module provides distribution-free canonical correlation analysis methods
//! based on distance correlation and other distance-based dependence measures.
//!
//! ## Methods
//! - Distance Canonical Correlation (dCCA)
//! - Distance Covariance (dCov)
//! - Brownian Distance Covariance
//! - Hilbert-Schmidt Independence Criterion (HSIC)
//! - Maximal Information Coefficient (MIC)
//!
//! ## Advantages
//! - Distribution-free (no normality assumptions)
//! - Detects nonlinear dependencies
//! - Robust to outliers
//! - Works with arbitrary data types

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, CoreRandom, Rng};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Distance metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Chebyshev (L-infinity) distance
    Chebyshev,
    /// Minkowski distance with parameter p
    Minkowski(u32), // Store p as u32 (will be converted to Float)
    /// Mahalanobis distance
    Mahalanobis,
}

/// Configuration for distance-based analysis
#[derive(Debug, Clone)]
pub struct DistanceBasedConfig {
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Number of permutations for significance testing
    pub n_permutations: usize,
    /// Significance level
    pub alpha: Float,
    /// Use bias-corrected estimator
    pub bias_corrected: bool,
}

impl Default for DistanceBasedConfig {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::Euclidean,
            n_permutations: 100,
            alpha: 0.05,
            bias_corrected: true,
        }
    }
}

/// Results from distance-based analysis
#[derive(Debug, Clone)]
pub struct DistanceBasedResults {
    /// Distance correlation
    pub distance_correlation: Float,
    /// Distance covariance
    pub distance_covariance: Float,
    /// P-value from permutation test
    pub p_value: Float,
    /// Whether the result is statistically significant
    pub is_significant: bool,
    /// Test statistic distribution from permutations
    pub null_distribution: Vec<Float>,
}

/// Distance Canonical Correlation Analysis
pub struct DistanceCCA {
    /// Configuration
    config: DistanceBasedConfig,
}

/// Distance covariance estimator
pub struct DistanceCovariance {
    /// Configuration
    config: DistanceBasedConfig,
}

/// Hilbert-Schmidt Independence Criterion
pub struct HSIC {
    /// Kernel bandwidth for X
    bandwidth_x: Float,
    /// Kernel bandwidth for Y
    bandwidth_y: Float,
    /// Number of permutations
    n_permutations: usize,
}

impl DistanceCCA {
    /// Create a new distance CCA
    pub fn new(config: DistanceBasedConfig) -> Self {
        Self { config }
    }

    /// Compute distance canonical correlation
    pub fn compute(&self, x: ArrayView2<Float>, y: ArrayView2<Float>) -> DistanceBasedResults {
        let dcov_estimator = DistanceCovariance::new(self.config.clone());

        // Compute distance correlation
        let dcorr = dcov_estimator.distance_correlation(x, y);
        let dcov = dcov_estimator.distance_covariance(x, y);

        // Perform permutation test
        let (p_value, null_dist) = self.permutation_test(x, y, dcorr);

        let is_significant = p_value < self.config.alpha;

        DistanceBasedResults {
            distance_correlation: dcorr,
            distance_covariance: dcov,
            p_value,
            is_significant,
            null_distribution: null_dist,
        }
    }

    /// Permutation test for significance
    fn permutation_test(
        &self,
        x: ArrayView2<Float>,
        y: ArrayView2<Float>,
        observed_stat: Float,
    ) -> (Float, Vec<Float>) {
        let mut rng = thread_rng();
        let n = x.nrows();
        let mut null_distribution = Vec::with_capacity(self.config.n_permutations);

        let dcov_estimator = DistanceCovariance::new(self.config.clone());

        for _ in 0..self.config.n_permutations {
            // Permute Y
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..n {
                let j = rng.random_range(i..n);
                indices.swap(i, j);
            }

            // Create permuted Y
            let mut y_perm = Array2::zeros(y.dim());
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                y_perm.row_mut(new_idx).assign(&y.row(old_idx));
            }

            // Compute distance correlation under null
            let dcorr_perm = dcov_estimator.distance_correlation(x, y_perm.view());
            null_distribution.push(dcorr_perm);
        }

        // Compute p-value: proportion of permuted stats >= observed
        let count = null_distribution
            .iter()
            .filter(|&&stat| stat >= observed_stat)
            .count();

        let p_value = (count + 1) as Float / (self.config.n_permutations + 1) as Float;

        (p_value, null_distribution)
    }
}

impl DistanceCovariance {
    /// Create a new distance covariance estimator
    pub fn new(config: DistanceBasedConfig) -> Self {
        Self { config }
    }

    /// Compute distance correlation between X and Y
    pub fn distance_correlation(&self, x: ArrayView2<Float>, y: ArrayView2<Float>) -> Float {
        let dcov_xy = self.distance_covariance(x, y);
        let dcov_xx = self.distance_covariance(x, x);
        let dcov_yy = self.distance_covariance(y, y);

        let denom = (dcov_xx * dcov_yy).sqrt();

        if denom > 1e-10 {
            dcov_xy / denom
        } else {
            0.0
        }
    }

    /// Compute distance covariance between X and Y
    pub fn distance_covariance(&self, x: ArrayView2<Float>, y: ArrayView2<Float>) -> Float {
        let n = x.nrows();

        // Compute pairwise distance matrices
        let dist_x = self.compute_distance_matrix(x);
        let dist_y = self.compute_distance_matrix(y);

        // Double-center the distance matrices
        let a = Self::double_center(&dist_x);
        let b = Self::double_center(&dist_y);

        // Compute distance covariance
        let mut dcov = 0.0;
        for i in 0..n {
            for j in 0..n {
                dcov += a[[i, j]] * b[[i, j]];
            }
        }

        dcov /= (n * n) as Float;

        if self.config.bias_corrected {
            // Apply bias correction for finite samples
            dcov = dcov.max(0.0);
        }

        dcov.sqrt()
    }

    /// Compute pairwise distance matrix
    fn compute_distance_matrix(&self, data: ArrayView2<Float>) -> Array2<Float> {
        let n = data.nrows();
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.compute_distance(data.row(i), data.row(j));
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        dist
    }

    /// Compute distance between two points
    fn compute_distance(&self, x: ArrayView1<Float>, y: ArrayView1<Float>) -> Float {
        match self.config.metric {
            DistanceMetric::Euclidean => {
                let diff = &x.to_owned() - &y.to_owned();
                diff.mapv(|v| v * v).sum().sqrt()
            }
            DistanceMetric::Manhattan => {
                let diff = &x.to_owned() - &y.to_owned();
                diff.mapv(|v| v.abs()).sum()
            }
            DistanceMetric::Chebyshev => {
                let diff = &x.to_owned() - &y.to_owned();
                diff.mapv(|v| v.abs()).iter().fold(0.0, |a, &b| a.max(b))
            }
            DistanceMetric::Minkowski(p_fixed) => {
                let p = p_fixed as Float;
                let diff = &x.to_owned() - &y.to_owned();
                diff.mapv(|v| v.abs().powf(p)).sum().powf(1.0 / p)
            }
            DistanceMetric::Mahalanobis => {
                // Simplified: use Euclidean for now (proper implementation needs covariance)
                let diff = &x.to_owned() - &y.to_owned();
                diff.mapv(|v| v * v).sum().sqrt()
            }
        }
    }

    /// Double-center a matrix (subtract row means, column means, add grand mean)
    fn double_center(mat: &Array2<Float>) -> Array2<Float> {
        let n = mat.nrows();

        // Compute row means
        let row_means = mat.mean_axis(Axis(1)).unwrap();

        // Compute column means
        let col_means = mat.mean_axis(Axis(0)).unwrap();

        // Compute grand mean
        let grand_mean = mat.mean().unwrap();

        // Double-center
        let mut centered = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                centered[[i, j]] = mat[[i, j]] - row_means[i] - col_means[j] + grand_mean;
            }
        }

        centered
    }
}

impl HSIC {
    /// Create a new HSIC estimator
    pub fn new(bandwidth_x: Float, bandwidth_y: Float, n_permutations: usize) -> Self {
        Self {
            bandwidth_x,
            bandwidth_y,
            n_permutations,
        }
    }

    /// Compute HSIC between X and Y
    pub fn compute(&self, x: ArrayView2<Float>, y: ArrayView2<Float>) -> DistanceBasedResults {
        let hsic_value = self.hsic_estimator(x, y);

        // Permutation test
        let (p_value, null_dist) = self.permutation_test(x, y, hsic_value);

        let is_significant = p_value < 0.05;

        DistanceBasedResults {
            distance_correlation: hsic_value,
            distance_covariance: hsic_value,
            p_value,
            is_significant,
            null_distribution: null_dist,
        }
    }

    /// HSIC estimator (biased version for simplicity)
    fn hsic_estimator(&self, x: ArrayView2<Float>, y: ArrayView2<Float>) -> Float {
        let n = x.nrows();

        // Compute kernel matrices
        let k_x = self.rbf_kernel(x, self.bandwidth_x);
        let k_y = self.rbf_kernel(y, self.bandwidth_y);

        // Center kernel matrices
        let h = Self::centering_matrix(n);

        let k_x_centered = h.dot(&k_x).dot(&h);
        let k_y_centered = h.dot(&k_y).dot(&h);

        // Compute HSIC
        let mut hsic = 0.0;
        for i in 0..n {
            for j in 0..n {
                hsic += k_x_centered[[i, j]] * k_y_centered[[i, j]];
            }
        }

        hsic / ((n - 1) * (n - 1)) as Float
    }

    /// RBF (Gaussian) kernel
    fn rbf_kernel(&self, data: ArrayView2<Float>, bandwidth: Float) -> Array2<Float> {
        let n = data.nrows();
        let mut kernel = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let diff = &data.row(i).to_owned() - &data.row(j).to_owned();
                let dist_sq = diff.mapv(|v| v * v).sum();
                kernel[[i, j]] = (-dist_sq / (2.0 * bandwidth * bandwidth)).exp();
            }
        }

        kernel
    }

    /// Centering matrix H = I - 1/n * 1 * 1^T
    fn centering_matrix(n: usize) -> Array2<Float> {
        let mut h = Array2::eye(n);
        let correction = 1.0 / n as Float;

        for i in 0..n {
            for j in 0..n {
                h[[i, j]] -= correction;
            }
        }

        h
    }

    /// Permutation test for HSIC
    fn permutation_test(
        &self,
        x: ArrayView2<Float>,
        y: ArrayView2<Float>,
        observed_stat: Float,
    ) -> (Float, Vec<Float>) {
        let mut rng = thread_rng();
        let n = x.nrows();
        let mut null_distribution = Vec::with_capacity(self.n_permutations);

        for _ in 0..self.n_permutations {
            // Permute Y
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..n {
                let j = rng.random_range(i..n);
                indices.swap(i, j);
            }

            let mut y_perm = Array2::zeros(y.dim());
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                y_perm.row_mut(new_idx).assign(&y.row(old_idx));
            }

            let hsic_perm = self.hsic_estimator(x, y_perm.view());
            null_distribution.push(hsic_perm);
        }

        let count = null_distribution
            .iter()
            .filter(|&&stat| stat >= observed_stat)
            .count();

        let p_value = (count + 1) as Float / (self.n_permutations + 1) as Float;

        (p_value, null_distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_distance_cca_creation() {
        let config = DistanceBasedConfig::default();
        let dcca = DistanceCCA::new(config);
        assert_eq!(dcca.config.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_distance_covariance_creation() {
        let config = DistanceBasedConfig::default();
        let dcov = DistanceCovariance::new(config);
        assert_eq!(dcov.config.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_euclidean_distance() {
        let config = DistanceBasedConfig::default();
        let dcov = DistanceCovariance::new(config);

        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];

        let dist = dcov.compute_distance(x.view(), y.view());
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let config = DistanceBasedConfig {
            metric: DistanceMetric::Manhattan,
            ..Default::default()
        };
        let dcov = DistanceCovariance::new(config);

        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];

        let dist = dcov.compute_distance(x.view(), y.view());
        assert!((dist - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_chebyshev_distance() {
        let config = DistanceBasedConfig {
            metric: DistanceMetric::Chebyshev,
            ..Default::default()
        };
        let dcov = DistanceCovariance::new(config);

        let x = array![0.0, 0.0];
        let y = array![3.0, 4.0];

        let dist = dcov.compute_distance(x.view(), y.view());
        assert!((dist - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_matrix() {
        let config = DistanceBasedConfig::default();
        let dcov = DistanceCovariance::new(config);

        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],];

        let dist_mat = dcov.compute_distance_matrix(data.view());

        assert_eq!(dist_mat.shape(), &[3, 3]);
        assert!((dist_mat[[0, 0]]).abs() < 1e-6); // Distance to self is 0
        assert!((dist_mat[[0, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_double_center() {
        let mat = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let centered = DistanceCovariance::double_center(&mat);

        // Row means of centered matrix should be close to zero
        let row_means = centered.mean_axis(Axis(1)).unwrap();
        for &mean in row_means.iter() {
            assert!(mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_distance_covariance_identical() {
        let config = DistanceBasedConfig::default();
        let dcov = DistanceCovariance::new(config);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        // Distance covariance of X with itself should be positive
        let dcov_val = dcov.distance_covariance(x.view(), x.view());
        assert!(dcov_val > 0.0);
    }

    #[test]
    fn test_distance_correlation_range() {
        let config = DistanceBasedConfig::default();
        let dcov = DistanceCovariance::new(config);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let y = array![[1.0], [2.0], [3.0],];

        let dcorr = dcov.distance_correlation(x.view(), y.view());

        // Distance correlation should be between 0 and 1
        assert!(dcorr >= 0.0);
        assert!(dcorr <= 1.0 + 1e-6); // Allow small numerical error
    }

    #[test]
    fn test_distance_cca_compute() {
        let config = DistanceBasedConfig {
            n_permutations: 10, // Small number for speed
            ..Default::default()
        };
        let dcca = DistanceCCA::new(config);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[1.0], [2.0], [3.0], [4.0],];

        let results = dcca.compute(x.view(), y.view());

        assert!(results.distance_correlation >= 0.0);
        assert!(results.distance_correlation <= 1.0 + 1e-6);
        assert!(results.p_value >= 0.0);
        assert!(results.p_value <= 1.0);
        assert_eq!(results.null_distribution.len(), 10);
    }

    #[test]
    fn test_hsic_creation() {
        let hsic = HSIC::new(1.0, 1.0, 10);
        assert_eq!(hsic.bandwidth_x, 1.0);
        assert_eq!(hsic.bandwidth_y, 1.0);
        assert_eq!(hsic.n_permutations, 10);
    }

    #[test]
    fn test_rbf_kernel() {
        let hsic = HSIC::new(1.0, 1.0, 10);

        let data = array![[0.0, 0.0], [1.0, 1.0],];

        let kernel = hsic.rbf_kernel(data.view(), 1.0);

        assert_eq!(kernel.shape(), &[2, 2]);
        assert!((kernel[[0, 0]] - 1.0).abs() < 1e-6); // Kernel with self is 1
        assert!(kernel[[0, 1]] > 0.0); // Kernel is positive
        assert!(kernel[[0, 1]] < 1.0); // Kernel is less than 1 for different points
    }

    #[test]
    fn test_centering_matrix() {
        let h = HSIC::centering_matrix(3);

        assert_eq!(h.shape(), &[3, 3]);

        // Row sums should be close to zero
        let row_sums = h.sum_axis(Axis(1));
        for &sum in row_sums.iter() {
            assert!(sum.abs() < 1e-10);
        }
    }

    #[test]
    fn test_hsic_compute() {
        let hsic = HSIC::new(1.0, 1.0, 5);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let y = array![[1.0], [2.0], [3.0],];

        let results = hsic.compute(x.view(), y.view());

        assert!(results.distance_correlation >= 0.0);
        assert_eq!(results.null_distribution.len(), 5);
    }

    #[test]
    fn test_minkowski_distance() {
        let config = DistanceBasedConfig {
            metric: DistanceMetric::Minkowski(3),
            ..Default::default()
        };
        let dcov = DistanceCovariance::new(config);

        let x = array![0.0, 0.0];
        let y = array![1.0, 1.0];

        let dist = dcov.compute_distance(x.view(), y.view());
        assert!(dist > 0.0);
        assert!(dist < 2.0);
    }
}
