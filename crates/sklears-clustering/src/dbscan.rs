//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation using scirs2

use std::marker::PhantomData;
#[cfg(feature = "parallel")]
use std::sync::{Arc, Mutex};

use scirs2_core::ndarray::Array;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
    types::{Array1, Array2, Float},
};

use scirs2_cluster::density::{dbscan, DistanceMetric};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use sklears_core::parallel::{ParallelConfig, ParallelFit};

/// Configuration for DBSCAN
#[derive(Debug, Clone)]
pub struct DBSCANConfig {
    /// Maximum distance between two samples for them to be considered in the same neighborhood
    pub eps: Float,
    /// Number of samples in a neighborhood for a point to be considered as a core point
    pub min_samples: usize,
    /// Metric to use for distance computation
    pub metric: DistanceMetric,
}

impl Default for DBSCANConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
            metric: DistanceMetric::Euclidean,
        }
    }
}

/// DBSCAN clustering
#[derive(Debug, Clone)]
pub struct DBSCAN<State = Untrained> {
    config: DBSCANConfig,
    state: PhantomData<State>,
    // Trained state fields
    labels_: Option<Array1<i32>>,
    core_sample_mask_: Option<Array1<bool>>,
    n_features_: Option<usize>,
}

/// Label value for noise points
pub const NOISE: i32 = -1;

impl DBSCAN<Untrained> {
    /// Create a new DBSCAN model
    pub fn new() -> Self {
        Self {
            config: DBSCANConfig::default(),
            state: PhantomData,
            labels_: None,
            core_sample_mask_: None,
            n_features_: None,
        }
    }

    /// Set epsilon (maximum distance)
    pub fn eps(mut self, eps: Float) -> Self {
        self.config.eps = eps;
        self
    }

    /// Set minimum samples
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.config.min_samples = min_samples;
        self
    }

    /// Set metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.config.metric = metric;
        self
    }
}

impl Default for DBSCAN<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DBSCAN<Untrained> {
    type Config = DBSCANConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for DBSCAN<Untrained> {
    type Fitted = DBSCAN<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.config.min_samples > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "min_samples ({min_samples}) cannot exceed n_samples ({n_samples})",
                min_samples = self.config.min_samples,
                n_samples = n_samples
            )));
        }

        // Run DBSCAN using scirs2
        let labels = dbscan(
            x.view(),
            self.config.eps,
            self.config.min_samples,
            Some(self.config.metric),
        )
        .map_err(|e| SklearsError::Other(format!("DBSCAN failed: {e:?}")))?;

        // Identify core samples (those with labels >= 0)
        let core_sample_mask = labels.mapv(|label| label >= 0);

        Ok(DBSCAN {
            config: self.config,
            state: PhantomData,
            labels_: Some(labels),
            core_sample_mask_: Some(core_sample_mask),
            n_features_: Some(n_features),
        })
    }
}

impl DBSCAN<Trained> {
    /// Get cluster labels
    ///
    /// Returns -1 for noise points, and cluster IDs (0, 1, 2, ...) for clustered points
    pub fn labels(&self) -> &Array1<i32> {
        self.labels_.as_ref().expect("Model is trained")
    }

    /// Get core sample mask
    pub fn core_sample_mask(&self) -> &Array1<bool> {
        self.core_sample_mask_.as_ref().expect("Model is trained")
    }

    /// Get indices of core samples
    pub fn core_sample_indices(&self) -> Vec<usize> {
        let mask = self.core_sample_mask_.as_ref().expect("Model is trained");
        mask.iter()
            .enumerate()
            .filter_map(|(i, &is_core)| if is_core { Some(i) } else { None })
            .collect()
    }

    /// Get number of clusters found (excluding noise)
    pub fn n_clusters(&self) -> usize {
        let labels = self.labels_.as_ref().expect("Model is trained");
        let max_label = labels
            .iter()
            .filter(|&&l| l >= 0)
            .max()
            .copied()
            .unwrap_or(-1);
        if max_label >= 0 {
            (max_label + 1) as usize
        } else {
            0
        }
    }

    /// Get number of noise points
    pub fn n_noise_points(&self) -> usize {
        let labels = self.labels_.as_ref().expect("Model is trained");
        labels.iter().filter(|&&l| l == NOISE).count()
    }
}

/// DBSCAN doesn't support prediction on new data in the traditional sense
/// But we can provide a method to find if a new point would be noise or belong to a cluster
impl DBSCAN<Trained> {
    /// Determine if new points would be classified as noise or belong to existing clusters
    ///
    /// This is not a standard predict method - DBSCAN is inherently transductive
    pub fn predict_noise(
        &self,
        x_train: &Array2<Float>,
        x_new: &Array2<Float>,
    ) -> Result<Array1<bool>> {
        let n_features = self.n_features_.expect("Model is trained");
        if x_new.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {n_features} features, got {got_features}",
                got_features = x_new.ncols()
            )));
        }

        let n_new = x_new.nrows();
        let mut is_noise = Array::from_elem(n_new, true);

        // For each new point, check if it's within eps of any core point
        let core_indices = self.core_sample_indices();

        for i in 0..n_new {
            let new_point = x_new.row(i);

            for &core_idx in &core_indices {
                let core_point = x_train.row(core_idx);

                // Calculate distance based on metric
                let dist = match self.config.metric {
                    DistanceMetric::Euclidean => {
                        let diff = &new_point - &core_point;
                        diff.dot(&diff).sqrt()
                    }
                    DistanceMetric::Manhattan => new_point
                        .iter()
                        .zip(core_point.iter())
                        .map(|(&a, &b)| (a - b).abs())
                        .sum(),
                    DistanceMetric::Chebyshev => new_point
                        .iter()
                        .zip(core_point.iter())
                        .map(|(&a, &b)| (a - b).abs())
                        .fold(0.0, |max, val| if val > max { val } else { max }),
                    _ => {
                        // For other metrics, fall back to Euclidean
                        let diff = &new_point - &core_point;
                        diff.dot(&diff).sqrt()
                    }
                };

                if dist <= self.config.eps {
                    is_noise[i] = false;
                    break;
                }
            }
        }

        Ok(is_noise)
    }
}

/// Parallel DBSCAN implementation
#[cfg(feature = "parallel")]
impl ParallelFit<Array2<Float>, ()> for DBSCAN<Untrained> {
    type Fitted = DBSCAN<Trained>;

    fn fit_parallel(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let config = ParallelConfig::default();
        self.fit_parallel_with_config(x, _y, &config)
    }

    fn fit_parallel_with_config(
        self,
        x: &Array2<Float>,
        _y: &(),
        parallel_config: &ParallelConfig,
    ) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.config.min_samples > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "min_samples ({min_samples}) cannot exceed n_samples ({n_samples})",
                min_samples = self.config.min_samples,
                n_samples = n_samples
            )));
        }

        // Use parallel implementation for large datasets
        let labels =
            if parallel_config.enabled && n_samples >= parallel_config.min_parallel_batch_size {
                self.dbscan_parallel(x, parallel_config)?
            } else {
                // Fall back to sequential implementation
                dbscan(
                    x.view(),
                    self.config.eps,
                    self.config.min_samples,
                    Some(self.config.metric),
                )
                .map_err(|e| SklearsError::Other(format!("DBSCAN failed: {e:?}")))?
            };

        // Identify core samples (those with labels >= 0)
        let core_sample_mask = labels.mapv(|label| label >= 0);

        Ok(DBSCAN {
            config: self.config,
            state: PhantomData,
            labels_: Some(labels),
            core_sample_mask_: Some(core_sample_mask),
            n_features_: Some(n_features),
        })
    }
}

#[cfg(feature = "parallel")]
impl DBSCAN<Untrained> {
    /// Parallel DBSCAN implementation
    fn dbscan_parallel(&self, x: &Array2<Float>, config: &ParallelConfig) -> Result<Array1<i32>> {
        let n_samples = x.nrows();
        let mut labels = Array1::from_elem(n_samples, NOISE);
        let mut visited = vec![false; n_samples];
        let mut cluster_id = 0i32;

        // Thread-safe structures for parallel neighbor finding
        let x_arc = Arc::new(x.clone());
        let visited_arc = Arc::new(Mutex::new(&mut visited));

        for i in 0..n_samples {
            {
                let visited_guard = visited_arc.lock().unwrap();
                if visited_guard[i] {
                    continue;
                }
            }

            // Mark as visited
            {
                let mut visited_guard = visited_arc.lock().unwrap();
                visited_guard[i] = true;
            }

            // Find neighbors in parallel
            let neighbors = self.find_neighbors_parallel(i, &x_arc, config)?;

            if neighbors.len() < self.config.min_samples {
                // Mark as noise (already initialized to NOISE)
                continue;
            }

            // Start new cluster
            labels[i] = cluster_id;

            // Expand cluster using BFS with parallel neighbor finding
            let mut seed_set = neighbors;
            let mut j = 0;

            while j < seed_set.len() {
                let q = seed_set[j];

                let was_visited = {
                    let mut visited_guard = visited_arc.lock().unwrap();
                    let was_visited = visited_guard[q];
                    if !was_visited {
                        visited_guard[q] = true;
                    }
                    was_visited
                };

                if !was_visited {
                    let q_neighbors = self.find_neighbors_parallel(q, &x_arc, config)?;
                    if q_neighbors.len() >= self.config.min_samples {
                        // Add new neighbors to seed set
                        for &neighbor in &q_neighbors {
                            if !seed_set.contains(&neighbor) {
                                seed_set.push(neighbor);
                            }
                        }
                    }
                }

                if labels[q] == NOISE {
                    labels[q] = cluster_id;
                }

                j += 1;
            }

            cluster_id += 1;
        }

        Ok(labels)
    }

    /// Find neighbors of a point in parallel
    fn find_neighbors_parallel(
        &self,
        point_idx: usize,
        x: &Arc<Array2<Float>>,
        config: &ParallelConfig,
    ) -> Result<Vec<usize>> {
        let n_samples = x.nrows();
        let point = x.row(point_idx);

        if !config.enabled || n_samples < config.min_parallel_batch_size {
            // Sequential fallback
            return Ok(self.find_neighbors_sequential(point_idx, x, &self.config));
        }

        // Parallel neighbor search
        let neighbors: Vec<usize> = (0..n_samples)
            .into_par_iter()
            .filter(|&i| {
                if i == point_idx {
                    return false;
                }

                let other_point = x.row(i);
                let dist = self.calculate_distance(&point, &other_point);
                dist <= self.config.eps
            })
            .collect();

        Ok(neighbors)
    }

    /// Sequential neighbor finding (fallback)
    fn find_neighbors_sequential(
        &self,
        point_idx: usize,
        x: &Array2<Float>,
        config: &DBSCANConfig,
    ) -> Vec<usize> {
        let point = x.row(point_idx);
        let mut neighbors = Vec::new();

        for i in 0..x.nrows() {
            if i == point_idx {
                continue;
            }

            let other_point = x.row(i);
            let dist = self.calculate_distance(&point, &other_point);

            if dist <= config.eps {
                neighbors.push(i);
            }
        }

        neighbors
    }

    /// Calculate distance between two points based on the configured metric
    fn calculate_distance(
        &self,
        point1: &scirs2_core::ndarray::ArrayView1<Float>,
        point2: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Float {
        match self.config.metric {
            DistanceMetric::Euclidean => {
                let diff = point1.to_owned() - point2;
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::Manhattan => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum(),
            DistanceMetric::Chebyshev => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, |max, val| if val > max { val } else { max }),
            _ => {
                // Fall back to Euclidean for other metrics
                let diff = point1.to_owned() - point2;
                diff.dot(&diff).sqrt()
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dbscan_simple() {
        // Create two well-separated clusters and noise points
        let data = array![
            // Cluster 1
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [0.0, 0.2],
            [0.1, 0.2],
            // Cluster 2
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
            [5.0, 5.2],
            [5.1, 5.2],
            // Noise
            [2.5, 2.5],
            [7.5, 7.5],
        ];

        let model = DBSCAN::new()
            .eps(0.5)
            .min_samples(3)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();

        // Should find 2 clusters
        assert_eq!(model.n_clusters(), 2);

        // Points 0-4 should be in one cluster
        assert!(labels[0] >= 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[3], labels[4]);

        // Points 5-9 should be in another cluster
        assert!(labels[5] >= 0);
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[7], labels[8]);
        assert_eq!(labels[8], labels[9]);

        // Different clusters
        assert_ne!(labels[0], labels[5]);

        // Last two points should be noise
        assert_eq!(labels[10], NOISE);
        assert_eq!(labels[11], NOISE);
        assert_eq!(model.n_noise_points(), 2);
    }

    #[test]
    fn test_dbscan_all_noise() {
        // Points too far apart
        let data = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0],];

        let model = DBSCAN::new()
            .eps(1.0)
            .min_samples(2)
            .fit(&data, &())
            .unwrap();

        let labels = model.labels();

        // All points should be noise
        assert_eq!(model.n_clusters(), 0);
        assert_eq!(model.n_noise_points(), 4);
        for &label in labels.iter() {
            assert_eq!(label, NOISE);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_dbscan() {
        use sklears_core::parallel::ParallelFit;

        // Create test data with two clusters
        let data = array![
            // Cluster 1
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [0.0, 0.2],
            [0.1, 0.2],
            // Cluster 2
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
            [5.0, 5.2],
            [5.1, 5.2],
            // Noise
            [2.5, 2.5],
            [7.5, 7.5],
        ];

        let model = DBSCAN::new().eps(0.5).min_samples(3);

        let parallel_config = ParallelConfig {
            enabled: true,
            min_parallel_batch_size: 1, // Force parallel execution even for small data
            num_threads: None,
        };

        let fitted = model
            .fit_parallel_with_config(&data, &(), &parallel_config)
            .unwrap();
        let labels = fitted.labels();

        // Should find 2 clusters
        assert_eq!(fitted.n_clusters(), 2);

        // Points 0-4 should be in one cluster
        assert!(labels[0] >= 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[3], labels[4]);

        // Points 5-9 should be in another cluster
        assert!(labels[5] >= 0);
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[7], labels[8]);
        assert_eq!(labels[8], labels[9]);

        // Different clusters
        assert_ne!(labels[0], labels[5]);

        // Last two points should be noise
        assert_eq!(labels[10], NOISE);
        assert_eq!(labels[11], NOISE);
        assert_eq!(fitted.n_noise_points(), 2);
    }
}
