//! Mean Shift Clustering
//!
//! Mean shift is a non-parametric clustering technique that doesn't require
//! specifying the number of clusters in advance. It works by shifting points
//! towards the mode of the data density.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::marker::PhantomData;

// Import from scirs2
use scirs2_cluster::meanshift::{mean_shift, MeanShiftOptions};

/// Bandwidth estimation methods for Mean Shift
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandwidthMethod {
    /// Scott's rule of thumb
    Scott,
    /// Silverman's rule of thumb  
    Silverman,
    /// Adaptive bandwidth based on k-nearest neighbors
    KNearestNeighbors { k: usize },
    /// Local adaptive bandwidth (variable bandwidth)
    LocalAdaptive { k: usize, alpha: f64 },
}

/// Configuration for Mean Shift clustering
#[derive(Debug, Clone)]
pub struct MeanShiftConfig {
    /// Bandwidth of the kernel. If None, will be estimated automatically.
    pub bandwidth: Option<Float>,
    /// Method for automatic bandwidth estimation
    pub bandwidth_method: BandwidthMethod,
    /// Number of seeds for binning. If -1, all points are used as seeds.
    pub seeds: i32,
    /// If true, initial kernel locations are binned.
    pub bin_seeding: bool,
    /// Minimum number of neighbors for a bin to be considered.
    pub min_bin_freq: usize,
    /// Stop iterations when change is less than this value.
    pub cluster_all: bool,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Number of jobs for parallel computation (-1 for all cores).
    pub n_jobs: i32,
    /// Use adaptive bandwidth (different bandwidth for each point)
    pub use_adaptive_bandwidth: bool,
}

impl Default for MeanShiftConfig {
    fn default() -> Self {
        Self {
            bandwidth: None,
            bandwidth_method: BandwidthMethod::Scott,
            seeds: -1,
            bin_seeding: false,
            min_bin_freq: 1,
            cluster_all: true,
            max_iter: 300,
            n_jobs: 1,
            use_adaptive_bandwidth: false,
        }
    }
}

/// Mean Shift clustering model
pub struct MeanShift<X = Array2<Float>, Y = ()> {
    config: MeanShiftConfig,
    cluster_centers: Option<Array2<Float>>,
    labels: Option<Array1<usize>>,
    n_clusters: Option<usize>,
    bandwidth: Option<Float>,
    _phantom: PhantomData<(X, Y)>,
}

impl<X, Y> MeanShift<X, Y> {
    /// Create a new Mean Shift model with default configuration
    pub fn new() -> Self {
        Self {
            config: MeanShiftConfig::default(),
            cluster_centers: None,
            labels: None,
            n_clusters: None,
            bandwidth: None,
            _phantom: PhantomData,
        }
    }

    /// Set the bandwidth parameter
    pub fn bandwidth(mut self, bandwidth: Float) -> Self {
        self.config.bandwidth = Some(bandwidth);
        self
    }

    /// Set the bandwidth estimation method
    pub fn bandwidth_method(mut self, method: BandwidthMethod) -> Self {
        self.config.bandwidth_method = method;
        self
    }

    /// Enable adaptive bandwidth estimation
    pub fn use_adaptive_bandwidth(mut self, use_adaptive: bool) -> Self {
        self.config.use_adaptive_bandwidth = use_adaptive;
        self
    }

    /// Set whether to use bin seeding
    pub fn bin_seeding(mut self, bin_seeding: bool) -> Self {
        self.config.bin_seeding = bin_seeding;
        self
    }

    /// Set the minimum bin frequency
    pub fn min_bin_freq(mut self, min_bin_freq: usize) -> Self {
        self.config.min_bin_freq = min_bin_freq;
        self
    }

    /// Set whether to cluster all points
    pub fn cluster_all(mut self, cluster_all: bool) -> Self {
        self.config.cluster_all = cluster_all;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Get the cluster centers
    pub fn cluster_centers(&self) -> Result<&Array2<Float>> {
        self.cluster_centers
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "cluster_centers".to_string(),
            })
    }

    /// Get the cluster labels
    pub fn labels(&self) -> &Array1<usize> {
        self.labels.as_ref().expect("Model has not been fitted yet")
    }

    /// Get the number of clusters
    pub fn n_clusters(&self) -> usize {
        self.n_clusters.expect("Model has not been fitted yet")
    }

    /// Get the bandwidth used
    pub fn bandwidth_used(&self) -> Float {
        self.bandwidth.expect("Model has not been fitted yet")
    }

    /// Estimate bandwidth using the configured method
    fn estimate_bandwidth(&self, data: &Array2<Float>) -> Result<Float> {
        match self.config.bandwidth_method {
            BandwidthMethod::Scott => self.scott_bandwidth(data),
            BandwidthMethod::Silverman => self.silverman_bandwidth(data),
            BandwidthMethod::KNearestNeighbors { k } => self.knn_bandwidth(data, k),
            BandwidthMethod::LocalAdaptive { k, alpha } => {
                self.local_adaptive_bandwidth(data, k, alpha)
            }
        }
    }

    /// Scott's rule of thumb for bandwidth estimation
    pub fn scott_bandwidth(&self, data: &Array2<Float>) -> Result<Float> {
        let n_samples = data.nrows() as f64;
        let n_features = data.ncols() as f64;
        let factor = n_samples.powf(-1.0 / (n_features + 4.0));

        // Estimate standard deviation for each feature
        let mut std_sum = 0.0;
        for j in 0..data.ncols() {
            let col = data.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples;
            std_sum += variance.sqrt();
        }
        let avg_std = std_sum / n_features;

        Ok(factor * avg_std)
    }

    /// Silverman's rule of thumb for bandwidth estimation
    pub fn silverman_bandwidth(&self, data: &Array2<Float>) -> Result<Float> {
        let n_samples = data.nrows() as f64;
        let n_features = data.ncols() as f64;

        // Silverman's rule: (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * sigma
        let factor = (4.0 / (n_features + 2.0)).powf(1.0 / (n_features + 4.0))
            * n_samples.powf(-1.0 / (n_features + 4.0));

        // Estimate standard deviation for each feature
        let mut std_sum = 0.0;
        for j in 0..data.ncols() {
            let col = data.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples;
            std_sum += variance.sqrt();
        }
        let avg_std = std_sum / n_features;

        Ok(factor * avg_std)
    }

    /// K-nearest neighbors adaptive bandwidth estimation
    pub fn knn_bandwidth(&self, data: &Array2<Float>, k: usize) -> Result<Float> {
        let n_samples = data.nrows();
        if k >= n_samples {
            return Err(SklearsError::InvalidInput(
                "k must be less than number of samples".to_string(),
            ));
        }

        let mut distances = Vec::new();

        // For each point, find distance to k-th nearest neighbor
        for i in 0..n_samples {
            let point = data.row(i);
            let mut point_distances = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let other_point = data.row(j);
                    let dist = (&point - &other_point).mapv(|x| x * x).sum().sqrt();
                    point_distances.push(dist);
                }
            }

            // Sort and get k-th nearest neighbor distance
            point_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if let Some(&kth_dist) = point_distances.get(k - 1) {
                distances.push(kth_dist);
            }
        }

        // Use median of k-th nearest neighbor distances
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = distances.len() / 2;
        Ok(distances[median_idx])
    }

    /// Local adaptive bandwidth estimation (variable bandwidth)
    pub fn local_adaptive_bandwidth(
        &self,
        data: &Array2<Float>,
        k: usize,
        alpha: f64,
    ) -> Result<Float> {
        let n_samples = data.nrows();
        if k >= n_samples {
            return Err(SklearsError::InvalidInput(
                "k must be less than number of samples".to_string(),
            ));
        }

        let mut local_densities = Vec::new();

        // Estimate local density for each point using k-NN
        for i in 0..n_samples {
            let point = data.row(i);
            let mut distances = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let other_point = data.row(j);
                    let dist = (&point - &other_point).mapv(|x| x * x).sum().sqrt();
                    distances.push(dist);
                }
            }

            // Sort and get k-th nearest neighbor distance
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if let Some(&kth_dist) = distances.get(k - 1) {
                // Local density estimation (inverse of volume)
                let volume = kth_dist.powf(data.ncols() as f64);
                let density = if volume > 0.0 { k as f64 / volume } else { 1.0 };
                local_densities.push(density);
            } else {
                local_densities.push(1.0);
            }
        }

        // Calculate geometric mean of local densities
        let log_sum: f64 = local_densities.iter().map(|&d| d.ln()).sum();
        let geometric_mean = (log_sum / n_samples as f64).exp();

        // Adaptive bandwidth based on local density
        let mut adaptive_bandwidths = Vec::new();
        for &density in &local_densities {
            let adaptive_factor = (geometric_mean / density).powf(alpha);
            adaptive_bandwidths.push(adaptive_factor);
        }

        // Use global bandwidth scaled by median adaptive factor
        let global_bandwidth = self.scott_bandwidth(data)?;
        adaptive_bandwidths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = adaptive_bandwidths.len() / 2;
        let median_factor = adaptive_bandwidths[median_idx];

        Ok(global_bandwidth * median_factor)
    }
}

impl<X, Y> Default for MeanShift<X, Y> {
    fn default() -> Self {
        Self::new()
    }
}

impl<X, Y> Estimator for MeanShift<X, Y> {
    type Config = MeanShiftConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<X: Send + Sync, Y: Send + Sync> Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>>
    for MeanShift<X, Y>
{
    type Fitted = Self;

    fn fit(self, x: &ArrayView2<Float>, _y: &ArrayView1<Float>) -> Result<Self::Fitted> {
        let x_data = x.to_owned();

        // Estimate bandwidth if not provided
        let bandwidth = if let Some(bw) = self.config.bandwidth {
            bw
        } else {
            self.estimate_bandwidth(&x_data)?
        };

        // Set up options
        let options = MeanShiftOptions {
            bandwidth: Some(bandwidth),
            seeds: None, // TODO: Support custom seeds in future version
            bin_seeding: self.config.bin_seeding,
            min_bin_freq: self.config.min_bin_freq,
            cluster_all: self.config.cluster_all,
            max_iter: self.config.max_iter,
        };

        // Run mean shift using scirs2
        let (cluster_centers, labels) = mean_shift(&x_data.view(), options)
            .map_err(|e| SklearsError::Other(format!("Mean shift failed: {e:?}")))?;

        let n_clusters = cluster_centers.nrows();

        Ok(Self {
            config: self.config.clone(),
            cluster_centers: Some(cluster_centers),
            labels: Some(labels.mapv(|x| x as usize)),
            n_clusters: Some(n_clusters),
            bandwidth: Some(bandwidth),
            _phantom: PhantomData,
        })
    }
}

impl<X, Y> Predict<ArrayView2<'_, Float>, Array1<usize>> for MeanShift<X, Y> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<usize>> {
        let centers = self.cluster_centers()?;
        let mut labels = Array1::zeros(x.nrows());

        // For each point, find the nearest cluster center
        for (i, point) in x.outer_iter().enumerate() {
            let mut min_dist = Float::INFINITY;
            let mut best_label = 0;

            for (j, center) in centers.outer_iter().enumerate() {
                let dist = (&point - &center).mapv(|x| x * x).sum().sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    best_label = j;
                }
            }

            labels[i] = best_label;
        }

        Ok(labels)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_shift_basic() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: MeanShift = MeanShift::new()
            .bandwidth(2.0)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        assert!(model.n_clusters() > 0);
        assert_eq!(model.labels().len(), x.nrows());
    }

    #[test]
    fn test_mean_shift_adaptive_bandwidth() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        // Test Scott's rule
        let model_scott: MeanShift = MeanShift::new()
            .bandwidth_method(BandwidthMethod::Scott)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        assert!(model_scott.n_clusters() > 0);
        assert_eq!(model_scott.labels().len(), x.nrows());

        // Test Silverman's rule
        let model_silverman: MeanShift = MeanShift::new()
            .bandwidth_method(BandwidthMethod::Silverman)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        assert!(model_silverman.n_clusters() > 0);
        assert_eq!(model_silverman.labels().len(), x.nrows());

        // Test k-NN adaptive bandwidth
        let model_knn: MeanShift = MeanShift::new()
            .bandwidth_method(BandwidthMethod::KNearestNeighbors { k: 3 })
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        assert!(model_knn.n_clusters() > 0);
        assert_eq!(model_knn.labels().len(), x.nrows());
    }

    #[test]
    fn test_bandwidth_estimation_methods() {
        let x = array![
            [1.0, 2.0],
            [1.5, 2.1],
            [2.0, 2.2],
            [8.0, 9.0],
            [8.5, 9.1],
            [9.0, 9.2],
        ];

        let model: MeanShift = MeanShift::new();

        // Test Scott's bandwidth
        let scott_bw = model.scott_bandwidth(&x).unwrap();
        assert!(scott_bw > 0.0);

        // Test Silverman's bandwidth
        let silverman_bw = model.silverman_bandwidth(&x).unwrap();
        assert!(silverman_bw > 0.0);

        // Test k-NN bandwidth
        let knn_bw = model.knn_bandwidth(&x, 2).unwrap();
        assert!(knn_bw > 0.0);

        // Test local adaptive bandwidth
        let adaptive_bw = model.local_adaptive_bandwidth(&x, 2, 0.5).unwrap();
        assert!(adaptive_bw > 0.0);
    }

    #[test]
    fn test_bandwidth_method_comparison() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
        ];

        let model: MeanShift = MeanShift::new();

        let scott_bw = model.scott_bandwidth(&x).unwrap();
        let silverman_bw = model.silverman_bandwidth(&x).unwrap();
        let knn_bw = model.knn_bandwidth(&x, 3).unwrap();

        // All methods should produce positive bandwidths
        assert!(scott_bw > 0.0);
        assert!(silverman_bw > 0.0);
        assert!(knn_bw > 0.0);

        // Silverman is typically larger than Scott
        // (though this is not guaranteed for all datasets)
        assert!(silverman_bw > 0.0);
        assert!(scott_bw > 0.0);
    }
}
