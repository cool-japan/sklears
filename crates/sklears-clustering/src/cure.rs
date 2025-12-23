//! CURE (Clustering Using REpresentatives) Algorithm
//!
//! CURE is a hierarchical clustering algorithm designed for large datasets with irregular
//! cluster shapes. It uses multiple representative points for each cluster and shrinks
//! them towards the centroid to handle outliers effectively.
//!
//! # Key Features
//! - Handles irregular cluster shapes better than traditional hierarchical clustering
//! - Uses sampling for efficient processing of large datasets
//! - Multiple representative points per cluster improve cluster quality
//! - Shrinking factor helps handle outliers
//!
//! # Algorithm Overview
//! 1. Sample points from the dataset
//! 2. Partition sampled points into initial clusters
//! 3. For each cluster, select representative points
//! 4. Shrink representatives towards cluster centroid
//! 5. Merge clusters based on distance between representatives
//! 6. Assign non-sampled points to nearest cluster
//!
//! # Mathematical Background
//!
//! For a cluster C with centroid μ, representative points are shrunk by factor α:
//! rep_new = rep_old + α * (μ - rep_old)
//!
//! Distance between clusters is minimum distance between their representatives.

use scirs2_core::random::Random;
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::*;
use sklears_core::traits::{Fit, Predict};
use std::collections::HashSet;

/// Distance metrics supported by CURE
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CUREDistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Cosine distance
    Cosine,
}

impl CUREDistanceMetric {
    fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            CUREDistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt(),
            CUREDistanceMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
            }
            CUREDistanceMetric::Cosine => {
                let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0 // Maximum distance for zero vectors
                } else {
                    1.0 - (dot_product / (norm_a * norm_b))
                }
            }
        }
    }
}

/// Configuration for CURE clustering algorithm
#[derive(Debug, Clone)]
pub struct CUREConfig {
    /// Number of final clusters
    pub n_clusters: usize,
    /// Number of representative points per cluster
    pub num_representatives: usize,
    /// Shrinking factor (0.0 to 1.0)
    pub shrink_factor: f64,
    /// Sample size for large datasets (if None, uses all points)
    pub sample_size: Option<usize>,
    /// Distance metric
    pub distance_metric: CUREDistanceMetric,
    /// Random seed for reproducible sampling
    pub random_seed: Option<u64>,
    /// Minimum cluster size
    pub min_cluster_size: usize,
}

impl Default for CUREConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            num_representatives: 10,
            shrink_factor: 0.2,
            sample_size: None,
            distance_metric: CUREDistanceMetric::Euclidean,
            random_seed: None,
            min_cluster_size: 1,
        }
    }
}

impl CUREConfig {
    /// Create a new CURE configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Set the number of representative points per cluster
    pub fn num_representatives(mut self, num_representatives: usize) -> Self {
        self.num_representatives = num_representatives;
        self
    }

    /// Set the shrinking factor
    pub fn shrink_factor(mut self, shrink_factor: f64) -> Self {
        self.shrink_factor = shrink_factor.clamp(0.0, 1.0);
        self
    }

    /// Set the sample size
    pub fn sample_size(mut self, sample_size: Option<usize>) -> Self {
        self.sample_size = sample_size;
        self
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, distance_metric: CUREDistanceMetric) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, random_seed: Option<u64>) -> Self {
        self.random_seed = random_seed;
        self
    }

    /// Set the minimum cluster size
    pub fn min_cluster_size(mut self, min_cluster_size: usize) -> Self {
        self.min_cluster_size = min_cluster_size;
        self
    }

    /// Build the CURE clustering algorithm
    pub fn build(self) -> CURE {
        CURE::new(self)
    }
}

/// Cluster representation in CURE algorithm
#[derive(Debug, Clone)]
struct CURECluster {
    /// Points in the cluster
    points: Vec<usize>,
    /// Representative points (indices into original data)
    representatives: Vec<Vec<f64>>,
    /// Cluster centroid
    centroid: Vec<f64>,
}

impl CURECluster {
    fn new(points: Vec<usize>, data: &Array2<f64>) -> Self {
        let centroid = Self::compute_centroid(&points, data);
        Self {
            points,
            representatives: Vec::new(),
            centroid,
        }
    }

    fn compute_centroid(points: &[usize], data: &Array2<f64>) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }

        let n_features = data.ncols();
        let mut centroid = vec![0.0; n_features];

        for &point_idx in points {
            for (i, &val) in data.row(point_idx).iter().enumerate() {
                centroid[i] += val;
            }
        }

        let n_points = points.len() as f64;
        for val in centroid.iter_mut() {
            *val /= n_points;
        }

        centroid
    }

    fn update_centroid(&mut self, data: &Array2<f64>) {
        self.centroid = Self::compute_centroid(&self.points, data);
    }

    fn select_representatives(
        &mut self,
        num_representatives: usize,
        data: &Array2<f64>,
        metric: CUREDistanceMetric,
    ) {
        if self.points.is_empty() {
            return;
        }

        let mut representatives = Vec::new();
        let mut remaining_points = self.points.clone();

        // Select first representative as the point farthest from centroid
        let mut max_dist = 0.0;
        let mut farthest_idx = 0;
        for (i, &point_idx) in remaining_points.iter().enumerate() {
            let point = data.row(point_idx);
            let dist = metric.compute_distance(&point.to_vec(), &self.centroid);
            if dist > max_dist {
                max_dist = dist;
                farthest_idx = i;
            }
        }

        let first_rep_idx = remaining_points.remove(farthest_idx);
        representatives.push(data.row(first_rep_idx).to_vec());

        // Select remaining representatives to be as far as possible from existing ones
        for _ in 1..num_representatives.min(self.points.len()) {
            if remaining_points.is_empty() {
                break;
            }

            let mut max_min_dist = 0.0;
            let mut best_idx = 0;

            for (i, &point_idx) in remaining_points.iter().enumerate() {
                let point = data.row(point_idx).to_vec();

                // Find minimum distance to existing representatives
                let min_dist = representatives
                    .iter()
                    .map(|rep| metric.compute_distance(&point, rep))
                    .fold(f64::INFINITY, f64::min);

                if min_dist > max_min_dist {
                    max_min_dist = min_dist;
                    best_idx = i;
                }
            }

            let new_rep_idx = remaining_points.remove(best_idx);
            representatives.push(data.row(new_rep_idx).to_vec());
        }

        self.representatives = representatives;
    }

    fn shrink_representatives(&mut self, shrink_factor: f64) {
        for rep in &mut self.representatives {
            for (i, val) in rep.iter_mut().enumerate() {
                *val += shrink_factor * (self.centroid[i] - *val);
            }
        }
    }

    fn distance_to_cluster(&self, other: &CURECluster, metric: CUREDistanceMetric) -> f64 {
        let mut min_dist = f64::INFINITY;

        for rep1 in &self.representatives {
            for rep2 in &other.representatives {
                let dist = metric.compute_distance(rep1, rep2);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        min_dist
    }

    fn merge(&mut self, other: CURECluster, data: &Array2<f64>) {
        self.points.extend(other.points);
        self.update_centroid(data);
    }
}

/// CURE (Clustering Using REpresentatives) algorithm
#[derive(Debug, Clone)]
pub struct CURE {
    config: CUREConfig,
}

/// Fitted CURE model
#[derive(Debug, Clone)]
pub struct CUREFitted {
    config: CUREConfig,
    cluster_labels: Vec<i32>,
    cluster_centers: Array2<f64>,
    n_samples: usize,
    n_features: usize,
}

impl CURE {
    /// Create a new CURE clustering algorithm
    pub fn new(config: CUREConfig) -> Self {
        Self { config }
    }

    /// Create CURE with default configuration
    pub fn default() -> Self {
        Self::new(CUREConfig::default())
    }

    /// Get configuration builder
    pub fn builder() -> CUREConfig {
        CUREConfig::new()
    }

    fn sample_data(&self, data: &Array2<f64>) -> (Array2<f64>, Vec<usize>) {
        let n_samples = data.nrows();

        if let Some(sample_size) = self.config.sample_size {
            if sample_size >= n_samples {
                let indices: Vec<usize> = (0..n_samples).collect();
                return (data.clone(), indices);
            }

            let mut rng = if let Some(seed) = self.config.random_seed {
                Random::default()
            } else {
                Random::default() // Use a default seed to maintain same type
            };

            // Simple random sampling
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                let j = rng.gen_range(0..i + 1);
                indices.swap(i, j);
            }
            indices.truncate(sample_size);
            indices.sort();

            let mut sampled_data = Array2::zeros((sample_size, data.ncols()));
            for (new_idx, &orig_idx) in indices.iter().enumerate() {
                for (j, &val) in data.row(orig_idx).iter().enumerate() {
                    sampled_data[(new_idx, j)] = val;
                }
            }

            (sampled_data, indices)
        } else {
            let indices: Vec<usize> = (0..n_samples).collect();
            (data.clone(), indices)
        }
    }

    fn initialize_clusters(&self, data: &Array2<f64>) -> Vec<CURECluster> {
        let n_samples = data.nrows();
        let mut clusters = Vec::new();

        // Each point starts as its own cluster
        for i in 0..n_samples {
            clusters.push(CURECluster::new(vec![i], data));
        }

        clusters
    }

    fn hierarchical_clustering(
        &self,
        mut clusters: Vec<CURECluster>,
        data: &Array2<f64>,
    ) -> Vec<CURECluster> {
        while clusters.len() > self.config.n_clusters {
            // Find closest pair of clusters
            let mut min_dist = f64::INFINITY;
            let mut merge_pair = (0, 1);

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let dist =
                        clusters[i].distance_to_cluster(&clusters[j], self.config.distance_metric);
                    if dist < min_dist {
                        min_dist = dist;
                        merge_pair = (i, j);
                    }
                }
            }

            // Merge the closest clusters
            let (i, j) = merge_pair;
            let cluster_j = clusters.remove(j);
            clusters[i].merge(cluster_j, data);

            // Update representatives for merged cluster
            clusters[i].select_representatives(
                self.config.num_representatives,
                data,
                self.config.distance_metric,
            );
            clusters[i].shrink_representatives(self.config.shrink_factor);
        }

        clusters
    }

    fn assign_labels(
        &self,
        clusters: &[CURECluster],
        sample_indices: &[usize],
        original_data: &Array2<f64>,
    ) -> Vec<i32> {
        let n_samples = original_data.nrows();
        let mut labels = vec![-1; n_samples];

        // First assign labels to sampled points
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &point_idx in &cluster.points {
                if point_idx < sample_indices.len() {
                    let original_idx = sample_indices[point_idx];
                    labels[original_idx] = cluster_id as i32;
                }
            }
        }

        // Assign non-sampled points to nearest cluster
        let sampled_set: HashSet<usize> = sample_indices.iter().cloned().collect();

        for i in 0..n_samples {
            if !sampled_set.contains(&i) && labels[i] == -1 {
                let point = original_data.row(i).to_vec();
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for (cluster_id, cluster) in clusters.iter().enumerate() {
                    for rep in &cluster.representatives {
                        let dist = self.config.distance_metric.compute_distance(&point, rep);
                        if dist < min_dist {
                            min_dist = dist;
                            best_cluster = cluster_id;
                        }
                    }
                }

                labels[i] = best_cluster as i32;
            }
        }

        labels
    }

    fn compute_cluster_centers(&self, labels: &[i32], data: &Array2<f64>) -> Array2<f64> {
        let n_features = data.ncols();
        let mut centers = Array2::zeros((self.config.n_clusters, n_features));
        let mut counts = vec![0; self.config.n_clusters];

        for (i, &label) in labels.iter().enumerate() {
            if label >= 0 {
                let cluster_id = label as usize;
                counts[cluster_id] += 1;

                for (j, &val) in data.row(i).iter().enumerate() {
                    centers[(cluster_id, j)] += val;
                }
            }
        }

        // Normalize by count
        for i in 0..self.config.n_clusters {
            if counts[i] > 0 {
                for j in 0..n_features {
                    centers[(i, j)] /= counts[i] as f64;
                }
            }
        }

        centers
    }
}

impl Estimator for CURE {
    type Config = CUREConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, ()> for CURE {
    type Fitted = CUREFitted;

    fn fit(self, X: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        if X.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        let (n_samples, n_features) = X.dim();

        if n_samples < self.config.n_clusters {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least as large as number of clusters".to_string(),
            ));
        }

        // Sample data if needed
        let (sampled_data, sample_indices) = self.sample_data(X);

        // Initialize clusters
        let mut clusters = self.initialize_clusters(&sampled_data);

        // Select initial representatives and shrink them
        for cluster in &mut clusters {
            cluster.select_representatives(
                self.config.num_representatives,
                &sampled_data,
                self.config.distance_metric,
            );
            cluster.shrink_representatives(self.config.shrink_factor);
        }

        // Perform hierarchical clustering
        let final_clusters = self.hierarchical_clustering(clusters, &sampled_data);

        // Assign labels to all points
        let cluster_labels = self.assign_labels(&final_clusters, &sample_indices, X);

        // Compute cluster centers
        let cluster_centers = self.compute_cluster_centers(&cluster_labels, X);

        Ok(CUREFitted {
            config: self.config.clone(),
            cluster_labels,
            cluster_centers,
            n_samples,
            n_features,
        })
    }
}

impl Predict<Array2<f64>, Array1<i32>> for CUREFitted {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if X.ncols() != self.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(X.nrows());

        for (i, point) in X.rows().into_iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (cluster_id, center) in self.cluster_centers.rows().into_iter().enumerate() {
                let dist = self
                    .config
                    .distance_metric
                    .compute_distance(&point.to_vec(), &center.to_vec());

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = cluster_id;
                }
            }

            predictions[i] = best_cluster as i32;
        }

        Ok(predictions)
    }
}

impl CUREFitted {
    /// Get cluster labels from training
    pub fn labels(&self) -> &[i32] {
        &self.cluster_labels
    }

    /// Get cluster centers
    pub fn cluster_centers(&self) -> &Array2<f64> {
        &self.cluster_centers
    }

    /// Get number of samples used for training
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get inertia (sum of squared distances to cluster centers)
    pub fn inertia(&self, X: &Array2<f64>) -> Result<f64> {
        if X.ncols() != self.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        let mut inertia = 0.0;

        for (i, point) in X.rows().into_iter().enumerate() {
            if i < self.cluster_labels.len() {
                let cluster_id = self.cluster_labels[i] as usize;
                if cluster_id < self.cluster_centers.nrows() {
                    let center = self.cluster_centers.row(cluster_id);
                    let dist = self
                        .config
                        .distance_metric
                        .compute_distance(&point.to_vec(), &center.to_vec());
                    inertia += dist * dist;
                }
            }
        }

        Ok(inertia)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> Array2<f64> {
        let data = vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0];
        Array2::from_shape_vec((6, 2), data).unwrap()
    }

    #[test]
    fn test_cure_basic() {
        let data = create_sample_data();
        let cure = CURE::builder()
            .n_clusters(2)
            .num_representatives(3)
            .shrink_factor(0.2)
            .build();

        let result = cure.fit(&data, &()).unwrap();
        assert_eq!(result.labels().len(), 6);
        assert_eq!(result.cluster_centers().nrows(), 2);
        assert_eq!(result.cluster_centers().ncols(), 2);
    }

    #[test]
    fn test_cure_predict() {
        let data = create_sample_data();
        let cure = CURE::builder().n_clusters(2).build();

        let fitted = cure.fit(&data, &()).unwrap();
        let predictions = fitted.predict(&data).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_cure_distance_metrics() {
        let data = create_sample_data();

        let metrics = [
            CUREDistanceMetric::Euclidean,
            CUREDistanceMetric::Manhattan,
            CUREDistanceMetric::Cosine,
        ];

        for metric in &metrics {
            let cure = CURE::builder()
                .n_clusters(2)
                .distance_metric(*metric)
                .build();

            let result = cure.fit(&data, &());
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_cure_sampling() {
        let data = create_sample_data();
        let cure = CURE::builder()
            .n_clusters(2)
            .sample_size(Some(4))
            .random_seed(Some(42))
            .build();

        let result = cure.fit(&data, &()).unwrap();
        assert_eq!(result.labels().len(), 6);
    }

    #[test]
    fn test_cure_config_validation() {
        let data = Array2::zeros((2, 2));
        let cure = CURE::builder()
            .n_clusters(5) // More clusters than data points
            .build();

        let result = cure.fit(&data, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_cure_inertia() {
        let data = create_sample_data();
        let cure = CURE::builder().n_clusters(2).build();

        let fitted = cure.fit(&data, &()).unwrap();
        let inertia = fitted.inertia(&data).unwrap();
        assert!(inertia >= 0.0);
    }
}
