//! Density estimation algorithms using nearest neighbors
//!
//! This module provides various density estimation methods based on k-nearest neighbors,
//! including kernel density estimation, adaptive bandwidth selection, and local density methods.

use crate::nearest_neighbors::NearestNeighbors;
use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::error::Result;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::{Features, Float};
use std::f64::consts::PI;

/// Kernel density estimation using k-nearest neighbors
///
/// This estimator uses k-nearest neighbors to perform kernel density estimation
/// with adaptive bandwidth based on local density.
pub struct KNeighborsDensityEstimator {
    /// Number of neighbors to use
    k: usize,
    /// Distance metric
    distance: Distance,
    /// Kernel type for density estimation
    kernel: KernelType,
    /// Bandwidth selection method
    bandwidth_method: BandwidthMethod,
    /// Fitted nearest neighbors model
    nn_model: Option<NearestNeighbors<sklears_core::traits::Trained>>,
    /// Training data (stored for density computation)
    X_train: Option<Array2<Float>>,
    /// Computed bandwidths for each training point
    bandwidths: Option<Array1<Float>>,
}

/// Kernel types for density estimation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
    Uniform,
    Triangular,
    Cosine,
}

/// Bandwidth selection methods
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BandwidthMethod {
    /// Fixed bandwidth for all points
    Fixed(Float),
    /// Adaptive bandwidth based on k-nearest neighbor distance
    KthNeighbor,
    /// Silverman's rule of thumb
    Silverman,
    /// Scott's rule
    Scott,
    /// Cross-validation optimized bandwidth
    CrossValidation,
}

impl KNeighborsDensityEstimator {
    /// Create a new KNN density estimator
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
            kernel: KernelType::Gaussian,
            bandwidth_method: BandwidthMethod::KthNeighbor,
            nn_model: None,
            X_train: None,
            bandwidths: None,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set the kernel type
    pub fn with_kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the bandwidth selection method
    pub fn with_bandwidth_method(mut self, method: BandwidthMethod) -> Self {
        self.bandwidth_method = method;
        self
    }

    /// Compute kernel function value
    fn kernel_function(&self, u: Float) -> Float {
        match self.kernel {
            KernelType::Gaussian => (-0.5 * u * u).exp() / (2.0 * PI).sqrt(),
            KernelType::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            KernelType::Uniform => {
                if u.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            KernelType::Triangular => {
                if u.abs() <= 1.0 {
                    1.0 - u.abs()
                } else {
                    0.0
                }
            }
            KernelType::Cosine => {
                if u.abs() <= 1.0 {
                    (PI * u / 4.0).cos() * PI / 4.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute adaptive bandwidth for each training point
    fn compute_adaptive_bandwidths(&self, X: &Features) -> NeighborsResult<Array1<Float>> {
        let n_samples = X.nrows();
        let mut bandwidths = Array1::zeros(n_samples);

        match self.bandwidth_method {
            BandwidthMethod::Fixed(h) => {
                bandwidths.fill(h);
            }
            BandwidthMethod::KthNeighbor => {
                let nn = NearestNeighbors::new(self.k + 1).with_metric(self.distance.clone());
                let fitted_nn = nn
                    .fit(X, &())
                    .map_err(|e| NeighborsError::InvalidInput(format!("Fit error: {:?}", e)))?;

                for i in 0..n_samples {
                    let query = X.slice(s![i..i + 1, ..]).to_owned();
                    let (distances_opt, _) =
                        fitted_nn.kneighbors(&query, Some(self.k + 1), true)?;
                    let distances = distances_opt.unwrap();
                    // Use k-th neighbor distance as bandwidth (excluding self)
                    bandwidths[i] = distances[[0, self.k]];
                }
            }
            BandwidthMethod::Silverman => {
                let d = X.ncols() as Float;
                let n = n_samples as Float;
                let std_dev = self.compute_std_dev(&X.view());
                let h = 1.06 * std_dev * n.powf(-1.0 / (d + 4.0));
                bandwidths.fill(h);
            }
            BandwidthMethod::Scott => {
                let d = X.ncols() as Float;
                let n = n_samples as Float;
                let std_dev = self.compute_std_dev(&X.view());
                let h = std_dev * n.powf(-1.0 / (d + 4.0));
                bandwidths.fill(h);
            }
            BandwidthMethod::CrossValidation => {
                // Simplified cross-validation: use Silverman's rule for now
                // In practice, you'd implement leave-one-out CV
                let d = X.ncols() as Float;
                let n = n_samples as Float;
                let std_dev = self.compute_std_dev(&X.view());
                let h = 1.06 * std_dev * n.powf(-1.0 / (d + 4.0));
                bandwidths.fill(h);
            }
        }

        Ok(bandwidths)
    }

    /// Compute standard deviation for bandwidth calculation
    fn compute_std_dev(&self, X: &ArrayView2<Float>) -> Float {
        let mean = X.mean_axis(Axis(0)).unwrap();
        let mut variance = 0.0;
        let n = X.nrows() as Float;

        for row in X.axis_iter(Axis(0)) {
            for (i, &val) in row.iter().enumerate() {
                variance += (val - mean[i]).powi(2);
            }
        }

        (variance / (n - 1.0)).sqrt()
    }

    /// Compute density at query points
    #[allow(non_snake_case)]
    fn compute_density(&self, X_query: &Features) -> NeighborsResult<Array1<Float>> {
        let X_train = self
            .X_train
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Model not fitted".to_string()))?;
        let bandwidths = self
            .bandwidths
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Bandwidths not computed".to_string()))?;

        let n_query = X_query.nrows();
        let n_train = X_train.nrows();
        let mut densities = Array1::zeros(n_query);

        for i in 0..n_query {
            let query_point = X_query.row(i);
            let mut density = 0.0;

            for j in 0..n_train {
                let train_point = X_train.row(j);
                let distance = self.compute_distance(&query_point, &train_point);
                let bandwidth = bandwidths[j];

                if bandwidth > 0.0 {
                    let u = distance / bandwidth;
                    let kernel_val = self.kernel_function(u);
                    density += kernel_val / bandwidth.powf(X_query.ncols() as Float);
                }
            }

            densities[i] = density / n_train as Float;
        }

        Ok(densities)
    }

    /// Compute distance between two points
    fn compute_distance(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        match &self.distance {
            Distance::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<Float>()
                .sqrt(),
            Distance::Manhattan => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .sum::<Float>(),
            Distance::Chebyshev => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, |acc, x| acc.max(x)),
            Distance::Minkowski(p) => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs().powf(*p))
                .sum::<Float>()
                .powf(1.0 / p),
            _ => {
                // For other distance types, fall back to Euclidean
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<Float>()
                    .sqrt()
            }
        }
    }
}

impl Fit<Features, ()> for KNeighborsDensityEstimator {
    type Fitted = KNeighborsDensityEstimator;

    fn fit(self, X: &Features, _y: &()) -> Result<Self::Fitted> {
        if X.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        // Fit nearest neighbors model
        let nn = NearestNeighbors::new(self.k + 1).with_metric(self.distance.clone());
        let fitted_nn = nn.fit(X, &())?;

        // Compute adaptive bandwidths
        let bandwidths = self
            .compute_adaptive_bandwidths(X)
            .map_err(sklears_core::error::SklearsError::from)?;

        let mut fitted_model = self;
        fitted_model.nn_model = Some(fitted_nn);
        fitted_model.X_train = Some(X.to_owned());
        fitted_model.bandwidths = Some(bandwidths);

        Ok(fitted_model)
    }
}

impl Predict<Features, Array1<Float>> for KNeighborsDensityEstimator {
    fn predict(&self, X: &Features) -> Result<Array1<Float>> {
        self.compute_density(X)
            .map_err(sklears_core::error::SklearsError::from)
    }
}

impl Clone for KNeighborsDensityEstimator {
    fn clone(&self) -> Self {
        Self {
            k: self.k,
            distance: self.distance.clone(),
            kernel: self.kernel,
            bandwidth_method: self.bandwidth_method,
            nn_model: self.nn_model.clone(),
            X_train: self.X_train.clone(),
            bandwidths: self.bandwidths.clone(),
        }
    }
}

/// Local density estimation using k-nearest neighbors
///
/// Estimates the local density around each point using the volume of the
/// k-nearest neighbor hypersphere.
pub struct LocalDensityEstimator {
    /// Number of neighbors to use
    k: usize,
    /// Distance metric
    distance: Distance,
    /// Fitted nearest neighbors model
    nn_model: Option<NearestNeighbors<sklears_core::traits::Trained>>,
}

impl LocalDensityEstimator {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
            nn_model: None,
        }
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Compute local density for query points
    fn compute_local_density(&self, X_query: &ArrayView2<Float>) -> NeighborsResult<Array1<Float>> {
        let nn_model = self
            .nn_model
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Model not fitted".to_string()))?;

        let n_query = X_query.nrows();
        let mut densities = Array1::zeros(n_query);

        for i in 0..n_query {
            let query = X_query.slice(s![i..i + 1, ..]);
            let (distances_opt, _) =
                nn_model.kneighbors(&query.to_owned(), Some(self.k + 1), true)?;
            let distances = distances_opt.unwrap();

            // Get k-th nearest neighbor distance
            let k_distance = distances[[0, self.k - 1]];

            if k_distance > 0.0 {
                // Estimate density as k / (volume of k-neighborhood)
                let d = X_query.ncols() as Float;
                let volume = self.hypersphere_volume(k_distance, d);
                densities[i] = self.k as Float / volume;
            }
        }

        Ok(densities)
    }

    /// Compute volume of d-dimensional hypersphere with radius r
    fn hypersphere_volume(&self, radius: Float, dimension: Float) -> Float {
        let d = dimension;
        let gamma_half_d = self.gamma_function(d / 2.0 + 1.0);
        (PI.powf(d / 2.0) / gamma_half_d) * radius.powf(d)
    }

    /// Simplified gamma function approximation
    fn gamma_function(&self, x: Float) -> Float {
        // Stirling's approximation for gamma function
        if x > 1.0 {
            (2.0 * PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
        } else {
            1.0 // Simplified for small values
        }
    }
}

impl Fit<Features, ()> for LocalDensityEstimator {
    type Fitted = LocalDensityEstimator;

    fn fit(self, X: &Features, _y: &()) -> Result<Self::Fitted> {
        if X.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let nn = NearestNeighbors::new(self.k).with_metric(self.distance.clone());
        let fitted_nn = nn.fit(X, &())?;

        let mut fitted_model = self;
        fitted_model.nn_model = Some(fitted_nn);

        Ok(fitted_model)
    }
}

impl Predict<Features, Array1<Float>> for LocalDensityEstimator {
    fn predict(&self, X: &Features) -> Result<Array1<Float>> {
        self.compute_local_density(&X.view())
            .map_err(sklears_core::error::SklearsError::from)
    }
}

impl Clone for LocalDensityEstimator {
    fn clone(&self) -> Self {
        Self {
            k: self.k,
            distance: self.distance.clone(),
            nn_model: self.nn_model.clone(),
        }
    }
}

/// Variable bandwidth kernel density estimation
///
/// Uses adaptive bandwidth that varies based on local density.
pub struct VariableBandwidthKDE {
    /// Number of neighbors for pilot density estimation
    k_pilot: usize,
    /// Alpha parameter for bandwidth adjustment
    alpha: Float,
    /// Distance metric
    distance: Distance,
    /// Kernel type
    kernel: KernelType,
    /// Training data
    X_train: Option<Array2<Float>>,
    /// Pilot densities for each training point
    pilot_densities: Option<Array1<Float>>,
    /// Adaptive bandwidths for each training point
    bandwidths: Option<Array1<Float>>,
}

impl VariableBandwidthKDE {
    /// Create a new variable bandwidth KDE
    pub fn new(k_pilot: usize) -> Self {
        Self {
            k_pilot,
            alpha: 0.5,
            distance: Distance::Euclidean,
            kernel: KernelType::Gaussian,
            X_train: None,
            pilot_densities: None,
            bandwidths: None,
        }
    }

    /// Set alpha parameter for bandwidth adaptation
    pub fn with_alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set kernel type
    pub fn with_kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    /// Compute pilot density estimates
    fn compute_pilot_densities(&self, X: &ArrayView2<Float>) -> NeighborsResult<Array1<Float>> {
        let pilot_kde = KNeighborsDensityEstimator::new(self.k_pilot)
            .with_distance(self.distance.clone())
            .with_kernel(self.kernel)
            .with_bandwidth_method(BandwidthMethod::KthNeighbor);

        let fitted_pilot = pilot_kde
            .fit(&X.to_owned(), &())
            .map_err(|e| NeighborsError::InvalidInput(format!("Pilot KDE fit error: {:?}", e)))?;
        fitted_pilot
            .predict(&X.to_owned())
            .map_err(|e| NeighborsError::InvalidInput(format!("Pilot KDE predict error: {:?}", e)))
    }

    /// Compute adaptive bandwidths based on pilot densities
    fn compute_adaptive_bandwidths(&self, pilot_densities: &Array1<Float>) -> Array1<Float> {
        let geometric_mean = pilot_densities
            .iter()
            .map(|&x| x.max(1e-10).ln())
            .sum::<Float>()
            / pilot_densities.len() as Float;
        let geometric_mean = geometric_mean.exp();

        pilot_densities.mapv(|density| {
            let ratio = density.max(1e-10) / geometric_mean;
            ratio.powf(-self.alpha)
        })
    }

    /// Kernel function (same as in KNeighborsDensityEstimator)
    fn kernel_function(&self, u: Float) -> Float {
        match self.kernel {
            KernelType::Gaussian => (-0.5 * u * u).exp() / (2.0 * PI).sqrt(),
            KernelType::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            KernelType::Uniform => {
                if u.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            KernelType::Triangular => {
                if u.abs() <= 1.0 {
                    1.0 - u.abs()
                } else {
                    0.0
                }
            }
            KernelType::Cosine => {
                if u.abs() <= 1.0 {
                    (PI * u / 4.0).cos() * PI / 4.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute distance between two points
    fn compute_distance(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        match &self.distance {
            Distance::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<Float>()
                .sqrt(),
            Distance::Manhattan => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .sum::<Float>(),
            _ => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<Float>()
                .sqrt(),
        }
    }

    /// Compute density at query points
    #[allow(non_snake_case)]
    fn compute_density(&self, X_query: &ArrayView2<Float>) -> NeighborsResult<Array1<Float>> {
        let X_train = self
            .X_train
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Model not fitted".to_string()))?;
        let bandwidths = self
            .bandwidths
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("Bandwidths not computed".to_string()))?;

        let n_query = X_query.nrows();
        let n_train = X_train.nrows();
        let mut densities = Array1::zeros(n_query);

        for i in 0..n_query {
            let query_point = X_query.row(i);
            let mut density = 0.0;

            for j in 0..n_train {
                let train_point = X_train.row(j);
                let distance = self.compute_distance(&query_point, &train_point);
                let bandwidth = bandwidths[j];

                if bandwidth > 0.0 {
                    let u = distance / bandwidth;
                    let kernel_val = self.kernel_function(u);
                    density += kernel_val / bandwidth.powf(X_query.ncols() as Float);
                }
            }

            densities[i] = density / n_train as Float;
        }

        Ok(densities)
    }
}

impl Fit<Features, ()> for VariableBandwidthKDE {
    type Fitted = VariableBandwidthKDE;

    fn fit(self, X: &Features, _y: &()) -> Result<Self::Fitted> {
        if X.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        // Compute pilot densities
        let pilot_densities = self
            .compute_pilot_densities(&X.view())
            .map_err(sklears_core::error::SklearsError::from)?;

        // Compute adaptive bandwidths
        let bandwidths = self.compute_adaptive_bandwidths(&pilot_densities);

        let mut fitted_model = self;
        fitted_model.X_train = Some(X.to_owned());
        fitted_model.pilot_densities = Some(pilot_densities);
        fitted_model.bandwidths = Some(bandwidths);

        Ok(fitted_model)
    }
}

impl Predict<Features, Array1<Float>> for VariableBandwidthKDE {
    fn predict(&self, X: &Features) -> Result<Array1<Float>> {
        self.compute_density(&X.view())
            .map_err(sklears_core::error::SklearsError::from)
    }
}

impl Clone for VariableBandwidthKDE {
    fn clone(&self) -> Self {
        Self {
            k_pilot: self.k_pilot,
            alpha: self.alpha,
            distance: self.distance.clone(),
            kernel: self.kernel,
            X_train: self.X_train.clone(),
            pilot_densities: self.pilot_densities.clone(),
            bandwidths: self.bandwidths.clone(),
        }
    }
}

/// Density-based clustering using KDE
///
/// This clustering algorithm uses kernel density estimation to identify dense regions
/// and assigns points to clusters based on local density peaks and valleys.
pub struct DensityBasedClustering {
    /// Density estimator
    density_estimator: KNeighborsDensityEstimator,
    /// Minimum density threshold for cluster formation
    min_density: Float,
    /// Minimum number of points in a cluster
    min_cluster_size: usize,
    /// Maximum distance for connecting density peaks
    connection_radius: Float,
    /// Fitted model state
    fitted: bool,
    /// Training data
    X_train: Option<Array2<Float>>,
    /// Density estimates for training data
    densities: Option<Array1<Float>>,
    /// Cluster labels for training data
    labels: Option<Array1<i32>>,
    /// Number of clusters found
    n_clusters: Option<usize>,
}

impl DensityBasedClustering {
    /// Create a new density-based clustering algorithm
    pub fn new(k: usize, min_density: Float) -> Self {
        Self {
            density_estimator: KNeighborsDensityEstimator::new(k),
            min_density,
            min_cluster_size: 2,
            connection_radius: 1.0,
            fitted: false,
            X_train: None,
            densities: None,
            labels: None,
            n_clusters: None,
        }
    }

    /// Set the minimum cluster size
    pub fn with_min_cluster_size(mut self, min_cluster_size: usize) -> Self {
        self.min_cluster_size = min_cluster_size;
        self
    }

    /// Set the connection radius for linking density peaks
    pub fn with_connection_radius(mut self, connection_radius: Float) -> Self {
        self.connection_radius = connection_radius;
        self
    }

    /// Set the kernel type for density estimation
    pub fn with_kernel(mut self, kernel: KernelType) -> Self {
        self.density_estimator = self.density_estimator.with_kernel(kernel);
        self
    }

    /// Set the bandwidth method for density estimation
    pub fn with_bandwidth_method(mut self, bandwidth_method: BandwidthMethod) -> Self {
        self.density_estimator = self
            .density_estimator
            .with_bandwidth_method(bandwidth_method);
        self
    }

    /// Set the distance metric
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.density_estimator = self.density_estimator.with_distance(distance);
        self
    }

    /// Get the cluster labels for training data
    pub fn labels(&self) -> Option<&Array1<i32>> {
        self.labels.as_ref()
    }

    /// Get the number of clusters found
    pub fn n_clusters(&self) -> Option<usize> {
        self.n_clusters
    }

    /// Get the density estimates for training data
    pub fn densities(&self) -> Option<&Array1<Float>> {
        self.densities.as_ref()
    }

    /// Find density peaks in the data
    fn find_density_peaks(&self, densities: &Array1<Float>, X: &ArrayView2<Float>) -> Vec<usize> {
        let mut peaks = Vec::new();
        let n_samples = X.nrows();

        for i in 0..n_samples {
            let density_i = densities[i];

            // Skip points below minimum density threshold
            if density_i < self.min_density {
                continue;
            }

            // Check if this point has higher density than its neighbors
            let mut is_peak = true;
            let point_i = X.row(i);

            for j in 0..n_samples {
                if i == j {
                    continue;
                }

                let point_j = X.row(j);
                let distance = self.compute_distance(&point_i, &point_j);

                if distance <= self.connection_radius && densities[j] > density_i {
                    is_peak = false;
                    break;
                }
            }

            if is_peak {
                peaks.push(i);
            }
        }

        peaks
    }

    /// Assign points to clusters based on density peaks
    fn assign_clusters(
        &self,
        densities: &Array1<Float>,
        X: &ArrayView2<Float>,
        peaks: &[usize],
    ) -> (Array1<i32>, usize) {
        let n_samples = X.nrows();
        let mut labels = Array1::from_elem(n_samples, -1); // -1 for noise
        let mut cluster_id = 0;

        // Assign peak points to clusters
        for &peak_idx in peaks {
            labels[peak_idx] = cluster_id;
            cluster_id += 1;
        }

        // Assign remaining points to nearest peak above threshold
        for i in 0..n_samples {
            if labels[i] != -1 || densities[i] < self.min_density {
                continue; // Already assigned or below threshold
            }

            let point_i = X.row(i);
            let mut nearest_peak = None;
            let mut min_distance = Float::INFINITY;

            for &peak_idx in peaks {
                let peak_point = X.row(peak_idx);
                let distance = self.compute_distance(&point_i, &peak_point);

                if distance < min_distance && distance <= self.connection_radius {
                    min_distance = distance;
                    nearest_peak = Some(peak_idx);
                }
            }

            if let Some(peak_idx) = nearest_peak {
                labels[i] = labels[peak_idx];
            }
        }

        // Filter out small clusters
        let mut cluster_sizes = std::collections::HashMap::new();
        for &label in labels.iter() {
            if label >= 0 {
                *cluster_sizes.entry(label).or_insert(0) += 1;
            }
        }

        // Reassign points from small clusters to noise
        for i in 0..n_samples {
            let label = labels[i];
            if label >= 0 {
                if let Some(&size) = cluster_sizes.get(&label) {
                    if size < self.min_cluster_size {
                        labels[i] = -1; // Mark as noise
                    }
                }
            }
        }

        // Relabel clusters to be contiguous
        let mut unique_labels: Vec<i32> = cluster_sizes
            .keys()
            .filter(|&&label| cluster_sizes[&label] >= self.min_cluster_size)
            .copied()
            .collect();
        unique_labels.sort();

        let mut label_map = std::collections::HashMap::new();
        for (new_label, &old_label) in unique_labels.iter().enumerate() {
            label_map.insert(old_label, new_label as i32);
        }

        for i in 0..n_samples {
            if labels[i] >= 0 {
                if let Some(&new_label) = label_map.get(&labels[i]) {
                    labels[i] = new_label;
                } else {
                    labels[i] = -1; // Small cluster, mark as noise
                }
            }
        }

        let n_clusters = label_map.len();
        (labels, n_clusters)
    }

    /// Compute distance between two points
    fn compute_distance(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<Float>()
            .sqrt()
    }
}

impl Fit<Features, ()> for DensityBasedClustering {
    type Fitted = DensityBasedClustering;

    fn fit(self, X: &Features, _y: &()) -> Result<Self::Fitted> {
        if X.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let mut fitted_model = self;
        fitted_model.X_train = Some(X.to_owned());

        // Fit density estimator
        let density_estimator = fitted_model.density_estimator.fit(X, &())?;
        fitted_model.density_estimator = density_estimator;

        // Compute densities for all points
        let densities = fitted_model.density_estimator.predict(X)?;
        fitted_model.densities = Some(densities.clone());

        // Find density peaks
        let peaks = fitted_model.find_density_peaks(&densities, &X.view());

        // Assign clusters
        let (labels, n_clusters) = fitted_model.assign_clusters(&densities, &X.view(), &peaks);
        fitted_model.labels = Some(labels);
        fitted_model.n_clusters = Some(n_clusters);
        fitted_model.fitted = true;

        Ok(fitted_model)
    }
}

impl Predict<Features, Array1<i32>> for DensityBasedClustering {
    #[allow(non_snake_case)]
    fn predict(&self, X: &Features) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(NeighborsError::InvalidInput("Model not fitted".to_string()).into());
        }

        let X_train = self
            .X_train
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("No training data".to_string()))?;

        let train_densities = self
            .densities
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("No density estimates".to_string()))?;

        let train_labels = self
            .labels
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("No cluster labels".to_string()))?;

        // For new points, find nearest training point and assign its cluster
        let n_test = X.nrows();
        let mut test_labels = Array1::from_elem(n_test, -1);

        for i in 0..n_test {
            let test_point = X.row(i);
            let mut min_distance = Float::INFINITY;
            let mut nearest_label = -1;

            for j in 0..X_train.nrows() {
                let train_point = X_train.row(j);
                let distance = self.compute_distance(&test_point, &train_point);

                if distance < min_distance {
                    min_distance = distance;
                    nearest_label = train_labels[j];
                }
            }

            // Only assign cluster if the nearest point is close enough and not noise
            if min_distance <= self.connection_radius && nearest_label >= 0 {
                test_labels[i] = nearest_label;
            }
        }

        Ok(test_labels)
    }
}

impl Clone for DensityBasedClustering {
    fn clone(&self) -> Self {
        Self {
            density_estimator: self.density_estimator.clone(),
            min_density: self.min_density,
            min_cluster_size: self.min_cluster_size,
            connection_radius: self.connection_radius,
            fitted: self.fitted,
            X_train: self.X_train.clone(),
            densities: self.densities.clone(),
            labels: self.labels.clone(),
            n_clusters: self.n_clusters,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    #[allow(non_snake_case)]
    fn test_knn_density_estimator() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        let kde = KNeighborsDensityEstimator::new(3)
            .with_kernel(KernelType::Gaussian)
            .with_bandwidth_method(BandwidthMethod::KthNeighbor);

        let fitted = kde.fit(&X, &()).unwrap();
        let densities = fitted.predict(&X).unwrap();

        assert_eq!(densities.len(), 6);
        // Density should be positive
        for &density in densities.iter() {
            assert!(density > 0.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_local_density_estimator() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        let lde = LocalDensityEstimator::new(2);
        let fitted = lde.fit(&X, &()).unwrap();
        let densities = fitted.predict(&X).unwrap();

        assert_eq!(densities.len(), 4);
        for &density in densities.iter() {
            assert!(density > 0.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_variable_bandwidth_kde() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        let vb_kde = VariableBandwidthKDE::new(2)
            .with_alpha(0.5)
            .with_kernel(KernelType::Gaussian);

        let fitted = vb_kde.fit(&X, &()).unwrap();
        let densities = fitted.predict(&X).unwrap();

        assert_eq!(densities.len(), 4);
        for &density in densities.iter() {
            assert!(density > 0.0);
        }
    }

    #[test]
    fn test_kernel_functions() {
        let kde = KNeighborsDensityEstimator::new(3);

        // Test Gaussian kernel
        let kde_gaussian = kde.clone().with_kernel(KernelType::Gaussian);
        let val = kde_gaussian.kernel_function(0.0);
        assert_abs_diff_eq!(val, 1.0 / (2.0 * PI).sqrt(), epsilon = 1e-10);

        // Test Epanechnikov kernel
        let kde_epan = kde.clone().with_kernel(KernelType::Epanechnikov);
        let val = kde_epan.kernel_function(0.0);
        assert_abs_diff_eq!(val, 0.75, epsilon = 1e-10);

        // Test Uniform kernel
        let kde_uniform = kde.clone().with_kernel(KernelType::Uniform);
        let val = kde_uniform.kernel_function(0.5);
        assert_abs_diff_eq!(val, 0.5, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bandwidth_methods() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();

        // Test fixed bandwidth
        let kde_fixed =
            KNeighborsDensityEstimator::new(2).with_bandwidth_method(BandwidthMethod::Fixed(1.0));
        let fitted = kde_fixed.fit(&X, &()).unwrap();
        assert!(fitted.bandwidths.is_some());

        // Test Silverman's rule
        let kde_silverman =
            KNeighborsDensityEstimator::new(2).with_bandwidth_method(BandwidthMethod::Silverman);
        let fitted = kde_silverman.fit(&X, &()).unwrap();
        assert!(fitted.bandwidths.is_some());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_density_based_clustering_basic() {
        // Create dataset with two clear clusters
        let X = Array2::from_shape_vec(
            (8, 2),
            vec![
                // Cluster 1
                1.0, 1.0, 1.1, 1.1, 1.2, 1.0, 1.0, 1.2, // Cluster 2
                5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
            ],
        )
        .unwrap();

        let clustering = DensityBasedClustering::new(2, 0.01)
            .with_min_cluster_size(2)
            .with_connection_radius(2.0);

        let fitted = clustering.fit(&X, &()).unwrap();
        let labels = fitted.predict(&X).unwrap();

        assert_eq!(labels.len(), 8);

        // Check that we found some clusters
        let n_clusters = fitted.n_clusters().unwrap_or(0);
        assert!(n_clusters > 0, "Should find at least one cluster");

        // Check that labels are valid
        for &label in labels.iter() {
            assert!(label >= -1, "Labels should be -1 (noise) or >= 0 (cluster)");
        }

        println!(
            "Density clustering found {} clusters with labels: {:?}",
            n_clusters, labels
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_density_based_clustering_configuration() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 5.0, 5.0, 5.1, 5.1, 10.0, 10.0, // Isolated point
            ],
        )
        .unwrap();

        // Test with different kernel types
        let clustering_gaussian = DensityBasedClustering::new(2, 0.01)
            .with_kernel(KernelType::Gaussian)
            .with_connection_radius(2.0);

        let fitted = clustering_gaussian.fit(&X, &()).unwrap();
        let labels = fitted.predict(&X).unwrap();
        assert_eq!(labels.len(), 6);

        // Test with Epanechnikov kernel
        let clustering_epan = DensityBasedClustering::new(2, 0.01)
            .with_kernel(KernelType::Epanechnikov)
            .with_connection_radius(2.0);

        let fitted = clustering_epan.fit(&X, &()).unwrap();
        let labels = fitted.predict(&X).unwrap();
        assert_eq!(labels.len(), 6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_density_based_clustering_edge_cases() {
        // Test with minimal data
        let X = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();

        let clustering = DensityBasedClustering::new(1, 0.001).with_min_cluster_size(1);

        let fitted = clustering.fit(&X, &()).unwrap();
        let labels = fitted.predict(&X).unwrap();

        assert_eq!(labels.len(), 3);

        // All labels should be valid
        for &label in labels.iter() {
            assert!(label >= -1);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_density_based_clustering_identical_points() {
        // Test with identical points
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let clustering = DensityBasedClustering::new(2, 0.001).with_connection_radius(0.1);

        let fitted = clustering.fit(&X, &()).unwrap();
        let labels = fitted.predict(&X).unwrap();

        assert_eq!(labels.len(), 4);

        // With identical points, should form one cluster
        let unique_labels: std::collections::HashSet<i32> = labels.iter().cloned().collect();
        assert!(
            unique_labels.len() <= 2,
            "Should have at most 2 unique labels (cluster or noise)"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_density_based_clustering_accessors() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();

        let clustering = DensityBasedClustering::new(2, 0.01);
        let fitted = clustering.fit(&X, &()).unwrap();

        // Test accessor methods
        assert!(fitted.labels().is_some());
        assert!(fitted.n_clusters().is_some());
        assert!(fitted.densities().is_some());

        let labels = fitted.labels().unwrap();
        let n_clusters = fitted.n_clusters().unwrap();
        let densities = fitted.densities().unwrap();

        assert_eq!(labels.len(), 4);
        assert_eq!(densities.len(), 4);

        // All densities should be positive
        for &density in densities.iter() {
            assert!(density >= 0.0, "Density should be non-negative");
            assert!(density.is_finite(), "Density should be finite");
        }

        println!(
            "Found {} clusters with densities: {:?}",
            n_clusters, densities
        );
    }
}
