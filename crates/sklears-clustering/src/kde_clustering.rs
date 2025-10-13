//! Kernel Density Estimation (KDE) Clustering
//!
//! KDE clustering identifies clusters by finding dense regions in the data space
//! using kernel density estimation. Points in high-density regions are assigned
//! to the same cluster, while points in low-density regions are considered noise.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::marker::PhantomData;

/// Kernel functions for density estimation
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    /// Gaussian kernel: exp(-0.5 * (x/h)^2) / sqrt(2π)
    Gaussian,
    /// Epanechnikov kernel: 0.75 * (1 - (x/h)^2) for |x/h| <= 1
    Epanechnikov,
    /// Uniform kernel: 0.5 for |x/h| <= 1
    Uniform,
    /// Triangular kernel: 1 - |x/h| for |x/h| <= 1
    Triangular,
}

/// Bandwidth selection methods
#[derive(Debug, Clone, Copy)]
pub enum BandwidthMethod {
    /// Scott's rule: n^(-1/(d+4))
    Scott,
    /// Silverman's rule: (n*(d+2)/4)^(-1/(d+4))
    Silverman,
    /// Manual bandwidth specification
    Manual(Float),
}

/// Configuration for KDE Clustering
#[derive(Debug, Clone)]
pub struct KDEClusteringConfig {
    /// Kernel function to use
    pub kernel: KernelType,
    /// Bandwidth selection method
    pub bandwidth: BandwidthMethod,
    /// Density threshold for cluster assignment
    pub density_threshold: Float,
    /// Minimum samples per cluster
    pub min_samples: usize,
    /// Maximum number of clusters to find (0 = unlimited)
    pub max_clusters: usize,
    /// Grid resolution for density estimation
    pub grid_resolution: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for KDEClusteringConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::Gaussian,
            bandwidth: BandwidthMethod::Scott,
            density_threshold: 0.1,
            min_samples: 3,
            max_clusters: 0,
            grid_resolution: 100,
            random_state: None,
        }
    }
}

/// KDE Clustering algorithm
pub struct KDEClustering<X = Array2<Float>, Y = ()> {
    config: KDEClusteringConfig,
    cluster_centers: Option<Array2<Float>>,
    density_values: Option<Array1<Float>>,
    bandwidth_values: Option<Array1<Float>>,
    n_clusters: Option<usize>,
    _phantom: PhantomData<(X, Y)>,
}

impl<X, Y> KDEClustering<X, Y> {
    /// Create a new KDE clustering instance
    pub fn new() -> Self {
        Self {
            config: KDEClusteringConfig::default(),
            cluster_centers: None,
            density_values: None,
            bandwidth_values: None,
            n_clusters: None,
            _phantom: PhantomData,
        }
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the bandwidth method
    pub fn bandwidth(mut self, bandwidth: BandwidthMethod) -> Self {
        self.config.bandwidth = bandwidth;
        self
    }

    /// Set the density threshold
    pub fn density_threshold(mut self, threshold: Float) -> Self {
        self.config.density_threshold = threshold;
        self
    }

    /// Set the minimum samples per cluster
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.config.min_samples = min_samples;
        self
    }

    /// Set the maximum number of clusters
    pub fn max_clusters(mut self, max_clusters: usize) -> Self {
        self.config.max_clusters = max_clusters;
        self
    }

    /// Set the grid resolution
    pub fn grid_resolution(mut self, resolution: usize) -> Self {
        self.config.grid_resolution = resolution;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
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

    /// Get the density values at each point
    pub fn density_values(&self) -> Result<&Array1<Float>> {
        self.density_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "density_values".to_string(),
            })
    }

    /// Get the number of clusters found
    pub fn n_clusters(&self) -> usize {
        self.n_clusters.unwrap_or(0)
    }

    /// Calculate bandwidth using the specified method
    fn calculate_bandwidth(&self, x: &ArrayView2<Float>) -> Array1<Float> {
        let n_samples = x.nrows() as Float;
        let n_features = x.ncols();

        match self.config.bandwidth {
            BandwidthMethod::Scott => {
                let factor = n_samples.powf(-1.0 / (n_features as Float + 4.0));
                let std_devs = x.var_axis(Axis(0), 0.0).mapv(|v| v.sqrt());
                std_devs * factor
            }
            BandwidthMethod::Silverman => {
                let factor = (n_samples * (n_features as Float + 2.0) / 4.0)
                    .powf(-1.0 / (n_features as Float + 4.0));
                let std_devs = x.var_axis(Axis(0), 0.0).mapv(|v| v.sqrt());
                std_devs * factor
            }
            BandwidthMethod::Manual(h) => Array1::from_elem(n_features, h),
        }
    }

    /// Evaluate kernel function
    fn kernel_function(&self, distance: Float) -> Float {
        match self.config.kernel {
            KernelType::Gaussian => {
                (-0.5 * distance * distance).exp() / (2.0 * std::f64::consts::PI).sqrt()
            }
            KernelType::Epanechnikov => {
                if distance.abs() <= 1.0 {
                    0.75 * (1.0 - distance * distance)
                } else {
                    0.0
                }
            }
            KernelType::Uniform => {
                if distance.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            KernelType::Triangular => {
                if distance.abs() <= 1.0 {
                    1.0 - distance.abs()
                } else {
                    0.0
                }
            }
        }
    }

    /// Estimate density at each data point
    fn estimate_density(&self, x: &ArrayView2<Float>, bandwidth: &Array1<Float>) -> Array1<Float> {
        let n_samples = x.nrows();
        let mut densities = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point_i = x.row(i);
            let mut density = 0.0;

            for j in 0..n_samples {
                if i == j {
                    continue;
                }

                let point_j = x.row(j);
                let mut normalized_distance = 0.0;

                // Calculate normalized Euclidean distance
                for k in 0..point_i.len() {
                    let diff = (point_i[k] - point_j[k]) / bandwidth[k];
                    normalized_distance += diff * diff;
                }
                normalized_distance = normalized_distance.sqrt();

                density += self.kernel_function(normalized_distance);
            }

            // Normalize by number of samples and bandwidth product
            let bandwidth_prod = bandwidth.iter().product::<Float>();
            densities[i] = density / ((n_samples - 1) as Float * bandwidth_prod);
        }

        densities
    }

    /// Find local density maxima (cluster centers)
    fn find_density_peaks(
        &self,
        x: &ArrayView2<Float>,
        densities: &Array1<Float>,
        bandwidth: &Array1<Float>,
    ) -> Array2<Float> {
        let mut peaks = Vec::new();
        let threshold = self.config.density_threshold;

        for i in 0..x.nrows() {
            if densities[i] < threshold {
                continue;
            }

            let point_i = x.row(i);
            let mut is_peak = true;

            // Check if this point is a local maximum
            for j in 0..x.nrows() {
                if i == j {
                    continue;
                }

                let point_j = x.row(j);
                let mut distance = 0.0;

                for k in 0..point_i.len() {
                    let diff = (point_i[k] - point_j[k]) / bandwidth[k];
                    distance += diff * diff;
                }
                distance = distance.sqrt();

                // If there's a nearby point with higher density, this isn't a peak
                if distance < 1.0 && densities[j] > densities[i] {
                    is_peak = false;
                    break;
                }
            }

            if is_peak {
                peaks.push(point_i.to_owned());
            }
        }

        // Sort peaks by density and limit number if specified
        let mut peak_densities: Vec<(Array1<Float>, Float)> = peaks
            .into_iter()
            .map(|peak| {
                // Find the density value for this peak
                let mut max_density = 0.0;
                for i in 0..x.nrows() {
                    let point = x.row(i);
                    let mut distance = 0.0;

                    for k in 0..peak.len() {
                        let diff = (peak[k] - point[k]) / bandwidth[k];
                        distance += diff * diff;
                    }

                    if distance < 1e-6 {
                        max_density = densities[i];
                        break;
                    }
                }
                (peak, max_density)
            })
            .collect();

        peak_densities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if self.config.max_clusters > 0 && peak_densities.len() > self.config.max_clusters {
            peak_densities.truncate(self.config.max_clusters);
        }

        let centers: Vec<Array1<Float>> =
            peak_densities.into_iter().map(|(peak, _)| peak).collect();

        if centers.is_empty() {
            Array2::zeros((0, x.ncols()))
        } else {
            let n_centers = centers.len();
            let n_features = centers[0].len();
            let mut center_matrix = Array2::zeros((n_centers, n_features));

            for (i, center) in centers.iter().enumerate() {
                center_matrix.row_mut(i).assign(center);
            }

            center_matrix
        }
    }

    /// Assign points to clusters based on nearest density peak
    fn assign_clusters(
        &self,
        x: &ArrayView2<Float>,
        centers: &Array2<Float>,
        densities: &Array1<Float>,
    ) -> Array1<i32> {
        let mut labels = Array1::from_elem(x.nrows(), -1);
        let threshold = self.config.density_threshold;

        if centers.nrows() == 0 {
            return labels;
        }

        for i in 0..x.nrows() {
            if densities[i] < threshold {
                continue; // Noise point
            }

            let point = x.row(i);
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = -1;

            for j in 0..centers.nrows() {
                let center = centers.row(j);
                let distance = (&point - &center).mapv(|x| x * x).sum().sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j as i32;
                }
            }

            labels[i] = best_cluster;
        }

        // Filter out small clusters
        let mut cluster_sizes = vec![0; centers.nrows()];
        for &label in labels.iter() {
            if label >= 0 {
                cluster_sizes[label as usize] += 1;
            }
        }

        for i in 0..labels.len() {
            let label = labels[i];
            if label >= 0 && cluster_sizes[label as usize] < self.config.min_samples {
                labels[i] = -1; // Mark as noise
            }
        }

        labels
    }
}

impl<X, Y> Default for KDEClustering<X, Y> {
    fn default() -> Self {
        Self::new()
    }
}

impl<X, Y> Estimator for KDEClustering<X, Y> {
    type Config = KDEClusteringConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<X: Send + Sync, Y: Send + Sync> Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>>
    for KDEClustering<X, Y>
{
    type Fitted = Self;

    fn fit(self, x: &ArrayView2<Float>, _y: &ArrayView1<Float>) -> Result<Self::Fitted> {
        if x.nrows() < self.config.min_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Need at least {} samples, got {}",
                self.config.min_samples,
                x.nrows()
            )));
        }

        // Calculate bandwidth
        let bandwidth = self.calculate_bandwidth(x);

        // Estimate density at each point
        let densities = self.estimate_density(x, &bandwidth);

        // Find density peaks (cluster centers)
        let centers = self.find_density_peaks(x, &densities, &bandwidth);

        let n_clusters = centers.nrows();

        Ok(Self {
            config: self.config,
            cluster_centers: Some(centers),
            density_values: Some(densities),
            bandwidth_values: Some(bandwidth),
            n_clusters: Some(n_clusters),
            _phantom: PhantomData,
        })
    }
}

impl<X, Y> Predict<ArrayView2<'_, Float>, Array1<i32>> for KDEClustering<X, Y> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<i32>> {
        let centers = self.cluster_centers()?;
        let densities = self.density_values()?;

        // For new data, we need to estimate densities first
        let bandwidth = self
            .bandwidth_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "bandwidth_values".to_string(),
            })?;

        let new_densities = self.estimate_density(x, bandwidth);
        let labels = self.assign_clusters(x, centers, &new_densities);

        Ok(labels)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_kde_clustering_basic() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: KDEClustering<Array2<Float>, ()> = KDEClustering::new()
            .kernel(KernelType::Gaussian)
            .bandwidth(BandwidthMethod::Scott)
            .density_threshold(0.01)
            .min_samples(2)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        assert!(model.n_clusters() >= 1);
        assert!(model.cluster_centers().is_ok());
        assert!(model.density_values().is_ok());
    }

    #[test]
    fn test_kde_clustering_predict() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model: KDEClustering<Array2<Float>, ()> = KDEClustering::new()
            .kernel(KernelType::Epanechnikov)
            .density_threshold(0.01)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        let labels = model.predict(&x.view()).unwrap();
        assert_eq!(labels.len(), x.nrows());
    }

    #[test]
    fn test_bandwidth_methods() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0],];

        let model_scott: KDEClustering<Array2<Float>, ()> = KDEClustering::new()
            .bandwidth(BandwidthMethod::Scott)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        let model_silverman: KDEClustering<Array2<Float>, ()> = KDEClustering::new()
            .bandwidth(BandwidthMethod::Silverman)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        let model_manual: KDEClustering<Array2<Float>, ()> = KDEClustering::new()
            .bandwidth(BandwidthMethod::Manual(0.5))
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        assert!(model_scott.cluster_centers().is_ok());
        assert!(model_silverman.cluster_centers().is_ok());
        assert!(model_manual.cluster_centers().is_ok());
    }

    #[test]
    fn test_kernel_types() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1],];

        for kernel in &[
            KernelType::Gaussian,
            KernelType::Epanechnikov,
            KernelType::Uniform,
            KernelType::Triangular,
        ] {
            let model: KDEClustering<Array2<Float>, ()> = KDEClustering::new()
                .kernel(*kernel)
                .density_threshold(0.001)
                .min_samples(1)
                .fit(&x.view(), &Array1::zeros(0).view())
                .unwrap();

            assert!(model.cluster_centers().is_ok());
        }
    }

    #[test]
    fn test_density_threshold() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let model_low: KDEClustering<Array2<Float>, ()> = KDEClustering::new()
            .density_threshold(0.001)
            .min_samples(1)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        let model_high: KDEClustering<Array2<Float>, ()> = KDEClustering::new()
            .density_threshold(0.5)
            .min_samples(1)
            .fit(&x.view(), &Array1::zeros(0).view())
            .unwrap();

        // Lower threshold should find more clusters
        assert!(model_low.n_clusters() >= model_high.n_clusters());
    }
}
