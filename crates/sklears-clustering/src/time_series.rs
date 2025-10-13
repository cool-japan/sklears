//! Time Series Clustering Algorithms
//!
//! This module provides specialized clustering algorithms designed for time series data,
//! including Dynamic Time Warping (DTW) clustering, shape-based clustering, and temporal
//! pattern recognition methods.
//!
//! # Algorithms Provided
//! - **DTW K-Means**: K-Means clustering with Dynamic Time Warping distance
//! - **Shape-based Clustering**: Clustering based on time series shape characteristics
//! - **Temporal Segmentation Clustering**: Segmentation-based clustering for long time series
//! - **Multi-scale Time Series Clustering**: Clustering at different temporal resolutions
//!
//! # Mathematical Background
//!
//! ## Dynamic Time Warping (DTW)
//! DTW finds an optimal alignment between two time series by computing the minimum
//! cumulative cost of aligning elements. For series X of length n and Y of length m:
//!
//! DTW(X,Y) = min Σ c(xi, yj) over all valid warping paths
//!
//! ## Shape Distance Measures
//! Various shape-based distances including:
//! - Euclidean Distance in shape space
//! - Derivative-based shape distance
//! - Angle-based shape similarity

use std::f64::INFINITY;

use numrs2::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::*;

/// Configuration for DTW K-Means clustering
#[derive(Debug, Clone)]
pub struct DTWKMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// DTW window size (None for no window constraint)
    pub dtw_window: Option<usize>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Averaging method for centroid computation
    pub averaging_method: CentroidAveraging,
}

impl Default for DTWKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 300,
            tolerance: 1e-4,
            dtw_window: None,
            random_seed: None,
            averaging_method: CentroidAveraging::DTWBarycenter,
        }
    }
}

/// Methods for computing centroid of time series
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CentroidAveraging {
    /// Simple element-wise mean
    ElementWise,
    /// DTW Barycenter Averaging (DBA)
    DTWBarycenter,
    /// Medoid-based centroid
    Medoid,
}

/// DTW K-Means clustering for time series
#[derive(Clone)]
pub struct DTWKMeans {
    config: DTWKMeansConfig,
}

/// Fitted DTW K-Means model
pub struct DTWKMeansFitted {
    /// Cluster centroids (time series)
    pub centroids: Array2<f64>,
    /// Cluster labels for training data
    pub labels: Vec<i32>,
    /// Configuration used
    pub config: DTWKMeansConfig,
    /// Final inertia
    pub inertia: f64,
    /// Number of iterations until convergence
    pub n_iterations: usize,
}

impl DTWKMeans {
    /// Create a new DTW K-Means clusterer
    pub fn new(config: DTWKMeansConfig) -> Self {
        Self { config }
    }

    /// Create DTW K-Means with default configuration
    pub fn default() -> Self {
        Self::new(DTWKMeansConfig::default())
    }

    /// Builder pattern: set number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.config.n_clusters = n_clusters;
        self
    }

    /// Builder pattern: set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Builder pattern: set DTW window constraint
    pub fn dtw_window(mut self, window: Option<usize>) -> Self {
        self.config.dtw_window = window;
        self
    }

    /// Builder pattern: set averaging method
    pub fn averaging_method(mut self, method: CentroidAveraging) -> Self {
        self.config.averaging_method = method;
        self
    }

    /// Compute Dynamic Time Warping distance between two time series
    fn dtw_distance(&self, ts1: ArrayView1<f64>, ts2: ArrayView1<f64>) -> f64 {
        let n = ts1.len();
        let m = ts2.len();

        if n == 0 || m == 0 {
            return INFINITY;
        }

        // Initialize DTW matrix with infinity
        let mut dtw_matrix = Array2::<f64>::from_elem((n + 1, m + 1), INFINITY);
        dtw_matrix[[0, 0]] = 0.0;

        // Apply window constraint if specified
        let window = self.config.dtw_window.unwrap_or(n.max(m));

        // Fill DTW matrix
        for i in 1..=n {
            let start_j = (1_usize).max(i.saturating_sub(window));
            let end_j = (m + 1).min(i + window + 1);

            for j in start_j..end_j {
                let cost = (ts1[i - 1] - ts2[j - 1]).powi(2);
                let min_prev = dtw_matrix[[i - 1, j]]
                    .min(dtw_matrix[[i, j - 1]])
                    .min(dtw_matrix[[i - 1, j - 1]]);
                dtw_matrix[[i, j]] = cost + min_prev;
            }
        }

        dtw_matrix[[n, m]].sqrt()
    }

    /// Compute centroid using DTW Barycenter Averaging
    fn compute_dtw_barycenter(&self, time_series: &[ArrayView1<f64>]) -> Array1<f64> {
        if time_series.is_empty() {
            return Array1::zeros(0);
        }

        // Use first series as initial barycenter
        let mut barycenter = time_series[0].to_owned();
        let max_iter = 10; // DBA iterations

        for _ in 0..max_iter {
            let mut new_barycenter = Array1::<f64>::zeros(barycenter.len());
            let mut weights = Array1::<f64>::zeros(barycenter.len());

            // Align each time series to current barycenter
            for ts in time_series {
                let path = self.compute_dtw_path(barycenter.view(), *ts);

                // Update barycenter based on alignment
                for (i, j) in path {
                    new_barycenter[i] += ts[j];
                    weights[i] += 1.0;
                }
            }

            // Normalize by weights
            for i in 0..barycenter.len() {
                if weights[i] > 0.0 {
                    new_barycenter[i] /= weights[i];
                } else {
                    new_barycenter[i] = barycenter[i];
                }
            }

            // Check convergence
            let change = (&new_barycenter - &barycenter)
                .mapv(|x| x.powi(2))
                .sum()
                .sqrt();

            barycenter = new_barycenter;

            if change < self.config.tolerance {
                break;
            }
        }

        barycenter
    }

    /// Compute DTW alignment path
    fn compute_dtw_path(&self, ts1: ArrayView1<f64>, ts2: ArrayView1<f64>) -> Vec<(usize, usize)> {
        let n = ts1.len();
        let m = ts2.len();

        if n == 0 || m == 0 {
            return vec![];
        }

        // Compute DTW matrix and backtrack path
        let mut dtw_matrix = Array2::<f64>::from_elem((n + 1, m + 1), INFINITY);
        dtw_matrix[[0, 0]] = 0.0;

        // Forward pass
        for i in 1..=n {
            for j in 1..=m {
                let cost = (ts1[i - 1] - ts2[j - 1]).powi(2);
                let min_prev = dtw_matrix[[i - 1, j]]
                    .min(dtw_matrix[[i, j - 1]])
                    .min(dtw_matrix[[i - 1, j - 1]]);
                dtw_matrix[[i, j]] = cost + min_prev;
            }
        }

        // Backtrack to find optimal path
        let mut path = Vec::new();
        let mut i = n;
        let mut j = m;

        while i > 0 || j > 0 {
            path.push((i - 1, j - 1));

            if i == 0 {
                j -= 1;
            } else if j == 0 {
                i -= 1;
            } else {
                // Find minimum predecessor
                let diag = dtw_matrix[[i - 1, j - 1]];
                let left = dtw_matrix[[i, j - 1]];
                let up = dtw_matrix[[i - 1, j]];

                if diag <= left && diag <= up {
                    i -= 1;
                    j -= 1;
                } else if up <= left {
                    i -= 1;
                } else {
                    j -= 1;
                }
            }
        }

        path.reverse();
        path
    }

    /// Compute medoid centroid (most representative time series)
    fn compute_medoid(&self, time_series: &[ArrayView1<f64>]) -> Array1<f64> {
        if time_series.is_empty() {
            return Array1::zeros(0);
        }

        if time_series.len() == 1 {
            return time_series[0].to_owned();
        }

        let mut min_total_distance = INFINITY;
        let mut medoid_idx = 0;

        // Find time series with minimum total DTW distance to all others
        for i in 0..time_series.len() {
            let mut total_distance = 0.0;

            for j in 0..time_series.len() {
                if i != j {
                    total_distance += self.dtw_distance(time_series[i], time_series[j]);
                }
            }

            if total_distance < min_total_distance {
                min_total_distance = total_distance;
                medoid_idx = i;
            }
        }

        time_series[medoid_idx].to_owned()
    }

    /// Initialize centroids using random selection
    fn initialize_centroids(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let ts_length = X.ncols();

        if n_samples < self.config.n_clusters {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples ({}) must be >= n_clusters ({})",
                n_samples, self.config.n_clusters
            )));
        }

        let mut rng = thread_rng();

        // Random initialization
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, ts_length));

        for (i, &idx) in indices.iter().take(self.config.n_clusters).enumerate() {
            centroids.row_mut(i).assign(&X.row(idx));
        }

        Ok(centroids)
    }

    /// Assign each time series to the nearest cluster
    fn assign_clusters(&self, X: &Array2<f64>, centroids: &Array2<f64>) -> (Vec<i32>, f64) {
        let n_samples = X.nrows();
        let mut labels = vec![0i32; n_samples];
        let mut inertia = 0.0;

        for i in 0..n_samples {
            let ts = X.row(i);
            let mut min_distance = INFINITY;
            let mut best_cluster = 0i32;

            for k in 0..self.config.n_clusters {
                let centroid = centroids.row(k);
                let distance = self.dtw_distance(ts, centroid);

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = k as i32;
                }
            }

            labels[i] = best_cluster;
            inertia += min_distance.powi(2);
        }

        (labels, inertia)
    }

    /// Update centroids based on cluster assignments
    fn update_centroids(&self, X: &Array2<f64>, labels: &[i32]) -> Array2<f64> {
        let ts_length = X.ncols();
        let mut new_centroids = Array2::<f64>::zeros((self.config.n_clusters, ts_length));

        for k in 0..self.config.n_clusters {
            // Collect time series for this cluster
            let cluster_series: Vec<ArrayView1<f64>> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == k as i32)
                .map(|(i, _)| X.row(i))
                .collect();

            if cluster_series.is_empty() {
                // Keep previous centroid if no points assigned
                continue;
            }

            // Compute centroid based on averaging method
            let centroid = match self.config.averaging_method {
                CentroidAveraging::ElementWise => {
                    // Simple element-wise mean
                    let mut sum = Array1::<f64>::zeros(ts_length);
                    for ts in &cluster_series {
                        sum += ts;
                    }
                    sum / cluster_series.len() as f64
                }
                CentroidAveraging::DTWBarycenter => self.compute_dtw_barycenter(&cluster_series),
                CentroidAveraging::Medoid => self.compute_medoid(&cluster_series),
            };

            new_centroids.row_mut(k).assign(&centroid);
        }

        new_centroids
    }
}

impl Estimator for DTWKMeans {
    type Config = DTWKMeansConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for DTWKMeans {
    type Fitted = DTWKMeansFitted;

    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<Self::Fitted> {
        let config = self.clone();
        if X.is_empty() || X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        // Initialize centroids
        let mut centroids = self.initialize_centroids(X)?;
        let mut previous_inertia = INFINITY;

        let mut final_labels = Vec::new();
        let mut final_inertia = 0.0;

        // Main clustering loop
        for iteration in 0..self.config.max_iter {
            // Assign clusters
            let (labels, inertia) = self.assign_clusters(X, &centroids);

            // Check convergence
            if (previous_inertia - inertia).abs() < self.config.tolerance {
                final_labels = labels;
                final_inertia = inertia;

                return Ok(DTWKMeansFitted {
                    centroids,
                    labels: final_labels,
                    config: config.config.clone(),
                    inertia: final_inertia,
                    n_iterations: iteration + 1,
                });
            }

            // Update centroids
            centroids = self.update_centroids(X, &labels);

            previous_inertia = inertia;
            final_labels = labels;
            final_inertia = inertia;
        }

        Ok(DTWKMeansFitted {
            centroids,
            labels: final_labels,
            config: self.config.clone(),
            inertia: final_inertia,
            n_iterations: self.config.max_iter,
        })
    }
}

impl Predict<Array2<f64>, Vec<i32>> for DTWKMeansFitted {
    fn predict(&self, X: &Array2<f64>) -> Result<Vec<i32>> {
        if X.is_empty() {
            return Ok(vec![]);
        }

        let clusterer = DTWKMeans::new(self.config.clone());
        let (labels, _) = clusterer.assign_clusters(X, &self.centroids);
        Ok(labels)
    }
}

/// Configuration for Shape-based clustering
#[derive(Debug, Clone)]
pub struct ShapeClusteringConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Shape distance metric
    pub shape_metric: ShapeDistanceMetric,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ShapeClusteringConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 300,
            tolerance: 1e-4,
            shape_metric: ShapeDistanceMetric::DerivativeBased,
            random_seed: None,
        }
    }
}

/// Shape distance metrics for time series
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeDistanceMetric {
    /// Distance based on first derivatives
    DerivativeBased,
    /// Distance based on shape angles
    AngleBased,
    /// Z-normalized Euclidean distance
    ZNormalizedEuclidean,
    /// Complexity-Invariant Distance (CID)
    ComplexityInvariant,
}

/// Shape-based clustering for time series
#[derive(Clone)]
pub struct ShapeClustering {
    config: ShapeClusteringConfig,
}

/// Fitted Shape-based clustering model
pub struct ShapeClusteringFitted {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Cluster labels for training data
    pub labels: Vec<i32>,
    /// Configuration used
    pub config: ShapeClusteringConfig,
    /// Final inertia
    pub inertia: f64,
    /// Number of iterations until convergence
    pub n_iterations: usize,
}

impl ShapeClustering {
    /// Create a new Shape-based clusterer
    pub fn new(config: ShapeClusteringConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(ShapeClusteringConfig::default())
    }

    /// Builder pattern: set number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.config.n_clusters = n_clusters;
        self
    }

    /// Builder pattern: set shape distance metric
    pub fn shape_metric(mut self, metric: ShapeDistanceMetric) -> Self {
        self.config.shape_metric = metric;
        self
    }

    /// Compute shape distance between two time series
    fn shape_distance(&self, ts1: ArrayView1<f64>, ts2: ArrayView1<f64>) -> f64 {
        match self.config.shape_metric {
            ShapeDistanceMetric::DerivativeBased => self.derivative_distance(ts1, ts2),
            ShapeDistanceMetric::AngleBased => self.angle_distance(ts1, ts2),
            ShapeDistanceMetric::ZNormalizedEuclidean => self.z_normalized_distance(ts1, ts2),
            ShapeDistanceMetric::ComplexityInvariant => {
                self.complexity_invariant_distance(ts1, ts2)
            }
        }
    }

    /// Compute derivative-based shape distance
    fn derivative_distance(&self, ts1: ArrayView1<f64>, ts2: ArrayView1<f64>) -> f64 {
        let deriv1 = self.compute_derivative(ts1);
        let deriv2 = self.compute_derivative(ts2);

        if deriv1.len() != deriv2.len() {
            return INFINITY;
        }

        deriv1
            .iter()
            .zip(deriv2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute first derivative of time series
    fn compute_derivative(&self, ts: ArrayView1<f64>) -> Vec<f64> {
        if ts.len() < 2 {
            return vec![];
        }

        (1..ts.len()).map(|i| ts[i] - ts[i - 1]).collect()
    }

    /// Compute angle-based shape distance
    fn angle_distance(&self, ts1: ArrayView1<f64>, ts2: ArrayView1<f64>) -> f64 {
        let angles1 = self.compute_angles(ts1);
        let angles2 = self.compute_angles(ts2);

        if angles1.len() != angles2.len() {
            return INFINITY;
        }

        angles1
            .iter()
            .zip(angles2.iter())
            .map(|(a, b)| {
                let diff = (a - b).abs();
                // Handle angle wrapping
                diff.min(2.0 * std::f64::consts::PI - diff)
            })
            .sum::<f64>()
            / angles1.len() as f64
    }

    /// Compute angles for shape representation
    fn compute_angles(&self, ts: ArrayView1<f64>) -> Vec<f64> {
        if ts.len() < 3 {
            return vec![];
        }

        (1..ts.len() - 1)
            .map(|i| {
                let prev = ts[i - 1];
                let curr = ts[i];
                let next = ts[i + 1];

                let v1 = curr - prev;
                let v2 = next - curr;

                v2.atan2(v1)
            })
            .collect()
    }

    /// Compute z-normalized Euclidean distance
    fn z_normalized_distance(&self, ts1: ArrayView1<f64>, ts2: ArrayView1<f64>) -> f64 {
        let norm1 = self.z_normalize(ts1);
        let norm2 = self.z_normalize(ts2);

        if norm1.len() != norm2.len() {
            return INFINITY;
        }

        norm1
            .iter()
            .zip(norm2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Z-normalize a time series
    fn z_normalize(&self, ts: ArrayView1<f64>) -> Vec<f64> {
        if ts.is_empty() {
            return vec![];
        }

        let mean = ts.sum() / ts.len() as f64;
        let variance = ts.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ts.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            vec![0.0; ts.len()]
        } else {
            ts.iter().map(|x| (x - mean) / std_dev).collect()
        }
    }

    /// Compute Complexity-Invariant Distance (CID)
    fn complexity_invariant_distance(&self, ts1: ArrayView1<f64>, ts2: ArrayView1<f64>) -> f64 {
        let ce1 = self.complexity_estimate(ts1);
        let ce2 = self.complexity_estimate(ts2);

        let euclidean_dist = self.z_normalized_distance(ts1, ts2);

        let complexity_factor = (ce1.max(ce2) / ce1.min(ce2).max(1e-10)).max(1.0);

        euclidean_dist * complexity_factor
    }

    /// Estimate complexity of a time series
    fn complexity_estimate(&self, ts: ArrayView1<f64>) -> f64 {
        if ts.len() < 2 {
            return 1.0;
        }

        let diff_sum: f64 = (1..ts.len()).map(|i| (ts[i] - ts[i - 1]).abs()).sum();

        diff_sum + 1.0 // Add 1 to avoid division by zero
    }
}

impl Estimator for ShapeClustering {
    type Config = ShapeClusteringConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for ShapeClustering {
    type Fitted = ShapeClusteringFitted;

    fn fit(self, X: &Array2<f64>, _y: &Array1<f64>) -> Result<Self::Fitted> {
        let config = self.clone();
        if X.is_empty() || X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        // Use standard K-Means with shape distance
        let mut rng = thread_rng();

        // Initialize centroids randomly
        let n_samples = X.nrows();
        let ts_length = X.ncols();

        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, ts_length));

        for (i, &idx) in indices.iter().take(self.config.n_clusters).enumerate() {
            centroids.row_mut(i).assign(&X.row(idx));
        }

        let mut previous_inertia = INFINITY;
        let mut final_labels = Vec::new();
        let mut final_inertia = 0.0;

        // Main clustering loop
        for iteration in 0..self.config.max_iter {
            // Assign clusters based on shape distance
            let (labels, inertia) = self.assign_clusters(X, &centroids);

            // Check convergence
            if (previous_inertia - inertia).abs() < self.config.tolerance {
                final_labels = labels;
                final_inertia = inertia;

                return Ok(ShapeClusteringFitted {
                    centroids,
                    labels: final_labels,
                    config: config.config.clone(),
                    inertia: final_inertia,
                    n_iterations: iteration + 1,
                });
            }

            // Update centroids (simple mean for now)
            for k in 0..self.config.n_clusters {
                let cluster_points: Vec<_> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &label)| label == k as i32)
                    .map(|(i, _)| X.row(i))
                    .collect();

                if !cluster_points.is_empty() {
                    let mut sum = Array1::<f64>::zeros(ts_length);
                    for ts in &cluster_points {
                        sum += ts;
                    }
                    let mean = sum / cluster_points.len() as f64;
                    centroids.row_mut(k).assign(&mean);
                }
            }

            previous_inertia = inertia;
            final_labels = labels;
            final_inertia = inertia;
        }

        Ok(ShapeClusteringFitted {
            centroids,
            labels: final_labels,
            config: self.config.clone(),
            inertia: final_inertia,
            n_iterations: self.config.max_iter,
        })
    }
}

impl ShapeClustering {
    /// Assign each time series to the nearest cluster based on shape distance
    fn assign_clusters(&self, X: &Array2<f64>, centroids: &Array2<f64>) -> (Vec<i32>, f64) {
        let n_samples = X.nrows();
        let mut labels = vec![0i32; n_samples];
        let mut inertia = 0.0;

        for i in 0..n_samples {
            let ts = X.row(i);
            let mut min_distance = INFINITY;
            let mut best_cluster = 0i32;

            for k in 0..self.config.n_clusters {
                let centroid = centroids.row(k);
                let distance = self.shape_distance(ts, centroid);

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = k as i32;
                }
            }

            labels[i] = best_cluster;
            inertia += min_distance.powi(2);
        }

        (labels, inertia)
    }
}

impl Predict<Array2<f64>, Vec<i32>> for ShapeClusteringFitted {
    fn predict(&self, X: &Array2<f64>) -> Result<Vec<i32>> {
        if X.is_empty() {
            return Ok(vec![]);
        }

        let clusterer = ShapeClustering::new(self.config.clone());
        let (labels, _) = clusterer.assign_clusters(X, &self.centroids);
        Ok(labels)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dtw_distance() {
        let config = DTWKMeansConfig::default();
        let clusterer = DTWKMeans::new(config);

        let ts1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let ts2 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);

        let distance = clusterer.dtw_distance(ts1.view(), ts2.view());
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dtw_kmeans_basic() {
        let X = Array2::from_shape_vec(
            (4, 5),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 1.1, 2.1, 3.1, 2.1, 1.1, 3.0, 4.0, 5.0, 4.0, 3.0, 3.1,
                4.1, 5.1, 4.1, 3.1,
            ],
        )
        .unwrap();

        let config = DTWKMeansConfig {
            n_clusters: 2,
            max_iter: 100,
            tolerance: 1e-4,
            dtw_window: None,
            random_seed: Some(42),
            averaging_method: CentroidAveraging::ElementWise,
        };

        let clusterer = DTWKMeans::new(config);
        let dummy_y = Array1::<f64>::zeros(X.nrows());
        let fitted = clusterer.fit(&X, &dummy_y).unwrap();

        assert_eq!(fitted.labels.len(), 4);
        assert!(fitted.n_iterations <= 100);
        assert!(fitted.inertia >= 0.0);
    }

    #[test]
    fn test_shape_distance_derivative() {
        let config = ShapeClusteringConfig {
            shape_metric: ShapeDistanceMetric::DerivativeBased,
            ..Default::default()
        };
        let clusterer = ShapeClustering::new(config);

        let ts1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let ts2 = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let distance = clusterer.shape_distance(ts1.view(), ts2.view());
        // Both series have constant derivative of 1.0, so distance should be 0
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_z_normalize() {
        let config = ShapeClusteringConfig::default();
        let clusterer = ShapeClustering::new(config);

        let ts = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = clusterer.z_normalize(ts.view());

        // Check that normalized series has mean ~0 and std ~1
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);

        let variance: f64 =
            normalized.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / normalized.len() as f64;
        assert_abs_diff_eq!(variance.sqrt(), 1.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_shape_clustering_basic() {
        let X = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 1.1, 2.1, 3.1, 4.1, 4.0, 3.0, 2.0, 1.0, 4.1, 3.1, 2.1, 1.1,
            ],
        )
        .unwrap();

        let config = ShapeClusteringConfig {
            n_clusters: 2,
            max_iter: 100,
            tolerance: 1e-4,
            shape_metric: ShapeDistanceMetric::DerivativeBased,
            random_seed: Some(42),
        };

        let clusterer = ShapeClustering::new(config);
        let dummy_y = Array1::<f64>::zeros(X.nrows());
        let fitted = clusterer.fit(&X, &dummy_y).unwrap();

        assert_eq!(fitted.labels.len(), 4);
        assert!(fitted.n_iterations <= 100);
        assert!(fitted.inertia >= 0.0);
    }
}

/// Regime Change Detection for Time Series
///
/// Detects points in time series where statistical properties change significantly,
/// useful for identifying structural breaks in temporal patterns.
#[derive(Debug, Clone)]
pub struct RegimeChangeDetector {
    pub config: RegimeChangeConfig,
}

/// Configuration for regime change detection
#[derive(Debug, Clone)]
pub struct RegimeChangeConfig {
    /// Window size for computing local statistics
    pub window_size: usize,
    /// Threshold for detecting significant changes (in standard deviations)
    pub change_threshold: f64,
    /// Minimum distance between detected change points
    pub min_distance: usize,
    /// Statistical test to use for change detection
    pub test_method: ChangeDetectionTest,
}

/// Methods for detecting regime changes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChangeDetectionTest {
    MeanChange,
    VarianceChange,
    DistributionChange,
    Combined,
}

impl Default for RegimeChangeConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            change_threshold: 2.0,
            min_distance: 10,
            test_method: ChangeDetectionTest::Combined,
        }
    }
}

/// Result of regime change detection
#[derive(Debug, Clone)]
pub struct RegimeChangeResult {
    /// Indices of detected change points
    pub change_points: Vec<usize>,
    /// Test statistics at each point
    pub test_statistics: Vec<f64>,
    /// P-values for change detection tests
    pub p_values: Vec<f64>,
    /// Regime labels for each time point
    pub regime_labels: Vec<usize>,
}

impl RegimeChangeDetector {
    /// Create a new regime change detector
    pub fn new(config: RegimeChangeConfig) -> Self {
        Self { config }
    }

    /// Detect regime changes in a time series
    pub fn detect_changes(&self, ts: ArrayView1<f64>) -> Result<RegimeChangeResult> {
        if ts.len() < 2 * self.config.window_size {
            return Err(SklearsError::InvalidInput(
                "Time series too short for regime change detection".to_string(),
            ));
        }

        let mut change_points = Vec::new();
        let mut test_statistics = Vec::new();
        let mut p_values = Vec::new();

        // Scan for change points using sliding window approach
        for i in self.config.window_size..(ts.len() - self.config.window_size) {
            let left_window = ts.slice(scirs2_core::ndarray::s![(i - self.config.window_size)..i]);
            let right_window = ts.slice(scirs2_core::ndarray::s![i..(i + self.config.window_size)]);

            let (test_stat, p_value) = self.compute_test_statistic(left_window, right_window);

            test_statistics.push(test_stat);
            p_values.push(p_value);

            // Check if this is a significant change point
            if test_stat > self.config.change_threshold {
                // Ensure minimum distance between change points
                if change_points.is_empty()
                    || i - change_points.last().unwrap() >= self.config.min_distance
                {
                    change_points.push(i);
                }
            }
        }

        // Assign regime labels
        let regime_labels = self.assign_regime_labels(ts.len(), &change_points);

        Ok(RegimeChangeResult {
            change_points,
            test_statistics,
            p_values,
            regime_labels,
        })
    }

    /// Compute test statistic for change detection
    fn compute_test_statistic(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> (f64, f64) {
        match self.config.test_method {
            ChangeDetectionTest::MeanChange => self.mean_change_test(left, right),
            ChangeDetectionTest::VarianceChange => self.variance_change_test(left, right),
            ChangeDetectionTest::DistributionChange => self.distribution_change_test(left, right),
            ChangeDetectionTest::Combined => {
                let (mean_stat, mean_p) = self.mean_change_test(left, right);
                let (var_stat, var_p) = self.variance_change_test(left, right);
                // Combine using Fisher's method (simplified)
                let combined_stat = mean_stat.max(var_stat);
                let combined_p = mean_p.min(var_p);
                (combined_stat, combined_p)
            }
        }
    }

    /// Test for change in mean (simplified t-test)
    fn mean_change_test(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> (f64, f64) {
        let mean_left = left.mean().unwrap_or(0.0);
        let mean_right = right.mean().unwrap_or(0.0);

        let var_left = left.var(0.0);
        let var_right = right.var(0.0);

        let n_left = left.len() as f64;
        let n_right = right.len() as f64;

        // Pooled variance estimate
        let pooled_var =
            ((n_left - 1.0) * var_left + (n_right - 1.0) * var_right) / (n_left + n_right - 2.0);

        let standard_error = (pooled_var * (1.0 / n_left + 1.0 / n_right)).sqrt();

        let t_stat = if standard_error > 1e-10 {
            (mean_left - mean_right).abs() / standard_error
        } else {
            0.0
        };

        // Simple approximation for p-value (normally would use t-distribution)
        let p_value = (-0.5 * t_stat.powi(2)).exp();

        (t_stat, p_value)
    }

    /// Test for change in variance (simplified F-test)
    fn variance_change_test(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> (f64, f64) {
        let var_left = left.var(0.0);
        let var_right = right.var(0.0);

        let f_stat = if var_right > 1e-10 {
            var_left / var_right
        } else {
            1.0
        };

        // Use log of F-statistic for better scale
        let log_f = f_stat.ln().abs();
        let p_value = (-0.5 * log_f).exp();

        (log_f, p_value)
    }

    /// Simplified Kolmogorov-Smirnov test for distribution change
    fn distribution_change_test(
        &self,
        left: ArrayView1<f64>,
        right: ArrayView1<f64>,
    ) -> (f64, f64) {
        // Sort both samples
        let mut left_sorted: Vec<f64> = left.to_vec();
        let mut right_sorted: Vec<f64> = right.to_vec();
        left_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        right_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute empirical CDFs and find maximum difference
        let mut max_diff: f64 = 0.0;
        let n_left = left_sorted.len() as f64;
        let n_right = right_sorted.len() as f64;

        let mut i_left = 0;
        let mut i_right = 0;

        while i_left < left_sorted.len() && i_right < right_sorted.len() {
            let x_left = left_sorted[i_left];
            let x_right = right_sorted[i_right];

            let cdf_left = (i_left as f64) / n_left;
            let cdf_right = (i_right as f64) / n_right;

            max_diff = max_diff.max((cdf_left - cdf_right).abs());

            if x_left <= x_right {
                i_left += 1;
            }
            if x_right <= x_left {
                i_right += 1;
            }
        }

        // KS test statistic
        let ks_stat = max_diff * (n_left * n_right / (n_left + n_right)).sqrt();

        // Approximate p-value
        let p_value = (-2.0 * ks_stat.powi(2)).exp();

        (ks_stat, p_value)
    }

    /// Assign regime labels based on change points
    fn assign_regime_labels(&self, ts_length: usize, change_points: &[usize]) -> Vec<usize> {
        let mut labels = vec![0; ts_length];
        let mut current_regime = 0;

        for &change_point in change_points {
            current_regime += 1;
            for i in change_point..ts_length {
                labels[i] = current_regime;
            }
        }

        labels
    }
}

/// Temporal Segmentation Clustering
///
/// Clusters long time series by first segmenting them into meaningful temporal chunks
/// based on change points, then clustering the segments.
#[derive(Debug, Clone)]
pub struct TemporalSegmentationClustering {
    pub config: TemporalSegmentationConfig,
}

/// Configuration for temporal segmentation clustering
#[derive(Debug, Clone)]
pub struct TemporalSegmentationConfig {
    /// Number of clusters for segment clustering
    pub n_clusters: usize,
    /// Configuration for regime change detection
    pub regime_config: RegimeChangeConfig,
    /// Minimum segment length to consider for clustering
    pub min_segment_length: usize,
    /// Maximum number of iterations for clustering
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for TemporalSegmentationConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            regime_config: RegimeChangeConfig::default(),
            min_segment_length: 10,
            max_iter: 300,
            tolerance: 1e-4,
            random_seed: None,
        }
    }
}

/// Result of temporal segmentation clustering
#[derive(Debug, Clone)]
pub struct TemporalSegmentationResult {
    /// Change points that define segment boundaries
    pub change_points: Vec<usize>,
    /// Cluster assignments for each segment
    pub segment_clusters: Vec<i32>,
    /// Cluster centroids (representative patterns)
    pub cluster_centroids: Array2<f64>,
    /// Labels for each time point
    pub time_point_labels: Vec<i32>,
    /// Number of iterations until convergence
    pub n_iterations: usize,
    /// Final inertia value
    pub inertia: f64,
}

impl TemporalSegmentationClustering {
    /// Create a new temporal segmentation clustering instance
    pub fn new(config: TemporalSegmentationConfig) -> Self {
        Self { config }
    }

    /// Perform temporal segmentation clustering on a time series
    pub fn fit_predict(&self, ts: ArrayView1<f64>) -> Result<TemporalSegmentationResult> {
        // Step 1: Detect change points to create segments
        let detector = RegimeChangeDetector::new(self.config.regime_config.clone());
        let change_result = detector.detect_changes(ts)?;

        // Step 2: Extract segments and filter by minimum length
        let segments = self.extract_segments(ts, &change_result.change_points)?;

        if segments.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid segments found for clustering".to_string(),
            ));
        }

        // Step 3: Cluster the segments using DTW K-means
        let segment_matrix = self.segments_to_matrix(&segments)?;
        let dtw_config = DTWKMeansConfig {
            n_clusters: self.config.n_clusters,
            max_iter: self.config.max_iter,
            tolerance: self.config.tolerance,
            dtw_window: None,
            random_seed: self.config.random_seed,
            averaging_method: CentroidAveraging::DTWBarycenter,
        };

        let dtw_kmeans = DTWKMeans::new(dtw_config);
        // For clustering, provide dummy Y parameter
        let dummy_y = Array1::zeros(segment_matrix.nrows());
        let dtw_result = dtw_kmeans.fit(&segment_matrix, &dummy_y)?;

        // Step 4: Create time point labels
        let time_point_labels = self.create_time_point_labels(
            ts.len(),
            &change_result.change_points,
            &dtw_result.labels,
        );

        Ok(TemporalSegmentationResult {
            change_points: change_result.change_points,
            segment_clusters: dtw_result.labels,
            cluster_centroids: dtw_result.centroids,
            time_point_labels,
            n_iterations: dtw_result.n_iterations,
            inertia: dtw_result.inertia,
        })
    }

    /// Extract segments from time series based on change points
    fn extract_segments(
        &self,
        ts: ArrayView1<f64>,
        change_points: &[usize],
    ) -> Result<Vec<Array1<f64>>> {
        let mut segments = Vec::new();
        let mut start = 0;

        for &change_point in change_points {
            if change_point > start && change_point - start >= self.config.min_segment_length {
                let segment = ts
                    .slice(scirs2_core::ndarray::s![start..change_point])
                    .to_owned();
                segments.push(segment);
            }
            start = change_point;
        }

        // Add final segment
        if ts.len() > start && ts.len() - start >= self.config.min_segment_length {
            let segment = ts.slice(scirs2_core::ndarray::s![start..]).to_owned();
            segments.push(segment);
        }

        Ok(segments)
    }

    /// Convert segments to matrix format for clustering
    fn segments_to_matrix(&self, segments: &[Array1<f64>]) -> Result<Array2<f64>> {
        if segments.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No segments provided".to_string(),
            ));
        }

        // Find maximum segment length for padding
        let max_length = segments.iter().map(|s| s.len()).max().unwrap();

        let mut matrix = Array2::<f64>::zeros((segments.len(), max_length));

        for (i, segment) in segments.iter().enumerate() {
            // Pad shorter segments with the last value
            let segment_len = segment.len();
            for j in 0..max_length {
                if j < segment_len {
                    matrix[[i, j]] = segment[j];
                } else {
                    // Pad with last value
                    matrix[[i, j]] = segment[segment_len - 1];
                }
            }
        }

        Ok(matrix)
    }

    /// Create time point labels from segment clustering results
    fn create_time_point_labels(
        &self,
        ts_length: usize,
        change_points: &[usize],
        segment_labels: &[i32],
    ) -> Vec<i32> {
        let mut labels = vec![-1i32; ts_length];
        let mut segment_idx = 0;
        let mut start = 0;

        for &change_point in change_points {
            if change_point > start
                && change_point - start >= self.config.min_segment_length
                && segment_idx < segment_labels.len()
            {
                for i in start..change_point {
                    labels[i] = segment_labels[segment_idx];
                }
                segment_idx += 1;
            }
            start = change_point;
        }

        // Handle final segment
        if ts_length > start
            && ts_length - start >= self.config.min_segment_length
            && segment_idx < segment_labels.len()
        {
            for i in start..ts_length {
                labels[i] = segment_labels[segment_idx];
            }
        }

        labels
    }
}
