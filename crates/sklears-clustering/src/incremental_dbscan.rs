//! Incremental DBSCAN for Streaming Data
//!
//! This implementation allows DBSCAN clustering to process data incrementally,
//! making it suitable for streaming applications and large datasets that don't
//! fit in memory. The algorithm maintains cluster state and can handle new
//! points being added dynamically.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

/// Distance metrics supported by incremental DBSCAN
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Chebyshev distance (L-infinity norm)
    Chebyshev,
}

/// Configuration for Incremental DBSCAN
#[derive(Debug, Clone)]
pub struct IncrementalDBSCANConfig {
    /// Maximum distance between two samples for them to be considered neighbors
    pub eps: Float,
    /// Minimum number of samples in a neighborhood for a point to be core
    pub min_samples: usize,
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Maximum number of points to keep in memory for incremental processing
    pub max_memory_points: usize,
    /// Forgetting factor for old points (0.0 = never forget, 1.0 = immediate forgetting)
    pub forgetting_factor: Float,
    /// Window size for sliding window approach (0 = no windowing)
    pub window_size: usize,
}

impl Default for IncrementalDBSCANConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
            metric: DistanceMetric::Euclidean,
            max_memory_points: 10000,
            forgetting_factor: 0.0,
            window_size: 0,
        }
    }
}

/// Point state in incremental DBSCAN
#[derive(Debug, Clone, PartialEq)]
enum PointState {
    /// Core point that can form clusters
    Core,
    /// Border point that belongs to a cluster but is not core
    Border,
    /// Noise point that doesn't belong to any cluster
    Noise,
    /// Unprocessed point
    Unprocessed,
}

/// Cluster information
#[derive(Debug, Clone)]
struct ClusterInfo {
    /// Points in this cluster
    points: HashSet<usize>,
    /// Core points in this cluster
    core_points: HashSet<usize>,
    /// Cluster centroid
    centroid: Array1<Float>,
    /// Cluster ID
    id: i32,
    /// Last update timestamp
    last_update: usize,
}

/// Incremental DBSCAN algorithm
pub struct IncrementalDBSCAN<X = Array2<Float>, Y = ()> {
    config: IncrementalDBSCANConfig,
    /// Points processed so far
    points: Array2<Float>,
    /// Current cluster assignments (-1 for noise)
    labels: Array1<i32>,
    /// Point states
    point_states: Vec<PointState>,
    /// Active clusters
    clusters: HashMap<i32, ClusterInfo>,
    /// Next cluster ID
    next_cluster_id: i32,
    /// Current timestamp for tracking updates
    current_time: usize,
    /// Point insertion order (for windowing)
    insertion_order: VecDeque<usize>,
    /// Whether the model has been fitted
    is_fitted: bool,
    _phantom: PhantomData<(X, Y)>,
}

impl<X, Y> IncrementalDBSCAN<X, Y> {
    /// Create a new incremental DBSCAN instance
    pub fn new() -> Self {
        Self {
            config: IncrementalDBSCANConfig::default(),
            points: Array2::zeros((0, 0)),
            labels: Array1::zeros(0),
            point_states: Vec::new(),
            clusters: HashMap::new(),
            next_cluster_id: 0,
            current_time: 0,
            insertion_order: VecDeque::new(),
            is_fitted: false,
            _phantom: PhantomData,
        }
    }

    /// Set the eps parameter
    pub fn eps(mut self, eps: Float) -> Self {
        self.config.eps = eps;
        self
    }

    /// Set the minimum samples parameter
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.config.min_samples = min_samples;
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.config.metric = metric;
        self
    }

    /// Set the maximum memory points
    pub fn max_memory_points(mut self, max_points: usize) -> Self {
        self.config.max_memory_points = max_points;
        self
    }

    /// Set the forgetting factor
    pub fn forgetting_factor(mut self, factor: Float) -> Self {
        self.config.forgetting_factor = factor;
        self
    }

    /// Set the window size
    pub fn window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self
    }

    /// Calculate distance between two points
    fn distance(&self, p1: &ArrayView1<Float>, p2: &ArrayView1<Float>) -> Float {
        match self.config.metric {
            DistanceMetric::Euclidean => (p1 - p2).mapv(|x| x * x).sum().sqrt(),
            DistanceMetric::Manhattan => (p1 - p2).mapv(|x| x.abs()).sum(),
            DistanceMetric::Chebyshev => (p1 - p2).mapv(|x| x.abs()).fold(0.0, |a, &b| a.max(b)),
        }
    }

    /// Find neighbors of a point within eps distance
    fn find_neighbors(&self, point_idx: usize) -> Vec<usize> {
        let point = self.points.row(point_idx);
        let mut neighbors = Vec::new();

        for i in 0..self.points.nrows() {
            if i != point_idx {
                let neighbor = self.points.row(i);
                if self.distance(&point, &neighbor) <= self.config.eps {
                    neighbors.push(i);
                }
            }
        }

        neighbors
    }

    /// Check if a point is a core point
    fn is_core_point(&self, point_idx: usize) -> bool {
        let neighbors = self.find_neighbors(point_idx);
        neighbors.len() >= self.config.min_samples
    }

    /// Add new points incrementally
    pub fn partial_fit(&mut self, x: &ArrayView2<Float>) -> Result<()> {
        if !self.is_fitted {
            // Initialize if this is the first call
            self.points = Array2::zeros((0, x.ncols()));
            self.labels = Array1::zeros(0);
            self.point_states = Vec::new();
            self.is_fitted = true;
        }

        if x.ncols() != self.points.ncols() && self.points.nrows() > 0 {
            return Err(SklearsError::InvalidInput(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.points.ncols(),
                x.ncols()
            )));
        }

        for new_point in x.outer_iter() {
            self.add_point(&new_point)?;
        }

        Ok(())
    }

    /// Add a single point to the clustering
    fn add_point(&mut self, point: &ArrayView1<Float>) -> Result<()> {
        // Check memory limits - ensure we have space for the new point
        while self.points.nrows() >= self.config.max_memory_points {
            self.handle_memory_overflow()?;
        }

        // Add the new point
        let new_idx = self.points.nrows();

        // Extend arrays
        let mut new_points = Array2::zeros((new_idx + 1, point.len()));
        if new_idx > 0 {
            new_points
                .slice_mut(s![0..new_idx, ..])
                .assign(&self.points);
        }
        new_points.row_mut(new_idx).assign(point);
        self.points = new_points;

        let mut new_labels = Array1::zeros(new_idx + 1);
        if new_idx > 0 {
            new_labels.slice_mut(s![0..new_idx]).assign(&self.labels);
        }
        new_labels[new_idx] = -1; // Initially noise
        self.labels = new_labels;

        self.point_states.push(PointState::Unprocessed);
        self.insertion_order.push_back(new_idx);

        // Update timestamp
        self.current_time += 1;

        // Process the new point
        self.process_point(new_idx)?;

        // Handle windowing
        if self.config.window_size > 0 && self.insertion_order.len() > self.config.window_size {
            self.remove_old_point()?;
        }

        Ok(())
    }

    /// Process a single point for clustering
    fn process_point(&mut self, point_idx: usize) -> Result<()> {
        if self.is_core_point(point_idx) {
            self.point_states[point_idx] = PointState::Core;
            self.expand_cluster(point_idx)?;
        } else {
            // Check if the point can be assigned to an existing cluster
            let neighbors = self.find_neighbors(point_idx);
            let mut assigned_cluster = None;

            for &neighbor_idx in &neighbors {
                if self.point_states[neighbor_idx] == PointState::Core {
                    let cluster_id = self.labels[neighbor_idx];
                    if cluster_id >= 0 {
                        assigned_cluster = Some(cluster_id);
                        break;
                    }
                }
            }

            if let Some(cluster_id) = assigned_cluster {
                self.labels[point_idx] = cluster_id;
                self.point_states[point_idx] = PointState::Border;

                // Update cluster info
                if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
                    cluster.points.insert(point_idx);
                    cluster.last_update = self.current_time;
                    // Update centroid
                    self.update_cluster_centroid(cluster_id)?;
                }
            } else {
                self.point_states[point_idx] = PointState::Noise;
                self.labels[point_idx] = -1;
            }
        }

        Ok(())
    }

    /// Expand cluster from a core point
    fn expand_cluster(&mut self, core_point_idx: usize) -> Result<()> {
        let neighbors = self.find_neighbors(core_point_idx);

        // Check if this core point should join an existing cluster
        let mut existing_cluster = None;
        for &neighbor_idx in &neighbors {
            if self.labels[neighbor_idx] >= 0 {
                existing_cluster = Some(self.labels[neighbor_idx]);
                break;
            }
        }

        let cluster_id = if let Some(cluster_id) = existing_cluster {
            cluster_id
        } else {
            // Create new cluster
            let new_cluster_id = self.next_cluster_id;
            self.next_cluster_id += 1;

            let centroid = self.points.row(core_point_idx).to_owned();
            let mut cluster_info = ClusterInfo {
                points: HashSet::new(),
                core_points: HashSet::new(),
                centroid,
                id: new_cluster_id,
                last_update: self.current_time,
            };
            cluster_info.points.insert(core_point_idx);
            cluster_info.core_points.insert(core_point_idx);

            self.clusters.insert(new_cluster_id, cluster_info);
            new_cluster_id
        };

        // Assign core point to cluster
        self.labels[core_point_idx] = cluster_id;

        // Update cluster info
        if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
            cluster.points.insert(core_point_idx);
            cluster.core_points.insert(core_point_idx);
            cluster.last_update = self.current_time;
        }

        // Assign neighbors to cluster
        for &neighbor_idx in &neighbors {
            if self.labels[neighbor_idx] == -1 {
                // If it's noise
                self.labels[neighbor_idx] = cluster_id;
                self.point_states[neighbor_idx] = PointState::Border;

                if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
                    cluster.points.insert(neighbor_idx);
                }
            }
        }

        // Update cluster centroid
        self.update_cluster_centroid(cluster_id)?;

        Ok(())
    }

    /// Update cluster centroid
    fn update_cluster_centroid(&mut self, cluster_id: i32) -> Result<()> {
        if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
            if cluster.points.is_empty() {
                return Ok(());
            }

            let mut centroid = Array1::zeros(self.points.ncols());
            for &point_idx in &cluster.points {
                centroid += &self.points.row(point_idx);
            }
            centroid /= cluster.points.len() as Float;
            cluster.centroid = centroid;
        }
        Ok(())
    }

    /// Handle memory overflow by removing old points
    fn handle_memory_overflow(&mut self) -> Result<()> {
        if self.config.forgetting_factor > 0.0 {
            // Probabilistic forgetting
            let mut points_to_remove = Vec::new();
            for i in 0..self.points.nrows() {
                if thread_rng().random::<Float>() < self.config.forgetting_factor {
                    points_to_remove.push(i);
                }
            }

            for &idx in points_to_remove.iter().rev() {
                self.remove_point_at_index(idx)?;
            }
        } else {
            // Remove oldest points - ensure we free up space
            let num_to_remove = (self.points.nrows() / 10).max(1); // Remove at least 1, at most 10%
            for _ in 0..num_to_remove {
                if let Some(oldest_idx) = self.insertion_order.pop_front() {
                    self.remove_point_at_index(oldest_idx)?;
                } else {
                    break; // No more points to remove
                }
            }
        }
        Ok(())
    }

    /// Remove old point for windowing
    fn remove_old_point(&mut self) -> Result<()> {
        if let Some(old_idx) = self.insertion_order.pop_front() {
            self.remove_point_at_index(old_idx)?;
        }
        Ok(())
    }

    /// Remove a point at a specific index
    fn remove_point_at_index(&mut self, idx: usize) -> Result<()> {
        if idx >= self.points.nrows() {
            return Ok(());
        }

        // Remove from cluster if assigned
        let cluster_id = self.labels[idx];
        if cluster_id >= 0 {
            if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
                cluster.points.remove(&idx);
                cluster.core_points.remove(&idx);

                // If cluster becomes empty, remove it
                if cluster.points.is_empty() {
                    self.clusters.remove(&cluster_id);
                } else {
                    // Update centroid
                    self.update_cluster_centroid(cluster_id)?;
                }
            }
        }

        // Remove from arrays (this is expensive, but necessary for correctness)
        // In a production implementation, you might use a more efficient data structure
        let new_size = self.points.nrows() - 1;
        if new_size > 0 {
            let mut new_points = Array2::zeros((new_size, self.points.ncols()));
            let mut new_labels = Array1::zeros(new_size);
            let mut new_states = Vec::with_capacity(new_size);

            let mut new_idx = 0;
            for i in 0..self.points.nrows() {
                if i != idx {
                    new_points.row_mut(new_idx).assign(&self.points.row(i));
                    new_labels[new_idx] = self.labels[i];
                    new_states.push(self.point_states[i].clone());
                    new_idx += 1;
                }
            }

            self.points = new_points;
            self.labels = new_labels;
            self.point_states = new_states;

            // Update cluster point indices
            for cluster in self.clusters.values_mut() {
                let mut new_points = HashSet::new();
                let mut new_core_points = HashSet::new();

                for &point_idx in &cluster.points {
                    let new_point_idx = if point_idx > idx {
                        point_idx - 1
                    } else {
                        point_idx
                    };
                    if point_idx != idx {
                        new_points.insert(new_point_idx);
                    }
                }

                for &point_idx in &cluster.core_points {
                    let new_point_idx = if point_idx > idx {
                        point_idx - 1
                    } else {
                        point_idx
                    };
                    if point_idx != idx {
                        new_core_points.insert(new_point_idx);
                    }
                }

                cluster.points = new_points;
                cluster.core_points = new_core_points;
            }

            // Update insertion order
            self.insertion_order = self
                .insertion_order
                .iter()
                .filter_map(|&i| {
                    if i == idx {
                        None
                    } else if i > idx {
                        Some(i - 1)
                    } else {
                        Some(i)
                    }
                })
                .collect();
        } else {
            self.points = Array2::zeros((0, 0));
            self.labels = Array1::zeros(0);
            self.point_states.clear();
            self.clusters.clear();
            self.insertion_order.clear();
        }

        Ok(())
    }

    /// Get current cluster centers
    pub fn cluster_centers(&self) -> Result<Array2<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "cluster_centers".to_string(),
            });
        }

        let n_clusters = self.clusters.len();
        if n_clusters == 0 {
            return Ok(Array2::zeros((0, self.points.ncols())));
        }

        let mut centers = Array2::zeros((n_clusters, self.points.ncols()));
        for (i, cluster) in self.clusters.values().enumerate() {
            centers.row_mut(i).assign(&cluster.centroid);
        }

        Ok(centers)
    }

    /// Get current number of clusters
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Get cluster statistics
    pub fn cluster_stats(&self) -> HashMap<i32, (usize, usize)> {
        let mut stats = HashMap::new();
        for (&cluster_id, cluster) in &self.clusters {
            stats.insert(
                cluster_id,
                (cluster.points.len(), cluster.core_points.len()),
            );
        }
        stats
    }
}

impl<X, Y> Default for IncrementalDBSCAN<X, Y> {
    fn default() -> Self {
        Self::new()
    }
}

impl<X, Y> Estimator for IncrementalDBSCAN<X, Y> {
    type Config = IncrementalDBSCANConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<X: Send + Sync, Y: Send + Sync> Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>>
    for IncrementalDBSCAN<X, Y>
{
    type Fitted = Self;

    fn fit(mut self, x: &ArrayView2<Float>, _y: &ArrayView1<Float>) -> Result<Self::Fitted> {
        self.partial_fit(x)?;
        Ok(self)
    }
}

impl<X, Y> Predict<ArrayView2<'_, Float>, Array1<i32>> for IncrementalDBSCAN<X, Y> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<i32>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let mut labels = Array1::from_elem(x.nrows(), -1);

        for (i, point) in x.outer_iter().enumerate() {
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = -1;

            // Assign to nearest cluster center
            for (&cluster_id, cluster) in &self.clusters {
                let distance = self.distance(&point, &cluster.centroid.view());
                if distance <= self.config.eps && distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_id;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }
}

// Helper macros for slicing
use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_incremental_dbscan_basic() {
        let mut model: IncrementalDBSCAN<Array2<Float>, ()> =
            IncrementalDBSCAN::new().eps(0.5).min_samples(2);

        // Add first batch
        let x1 = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.0],];
        model.partial_fit(&x1.view()).unwrap();
        assert!(model.n_clusters() >= 1);

        // Add second batch
        let x2 = array![[5.0, 5.0], [5.1, 5.1], [5.2, 5.0],];
        model.partial_fit(&x2.view()).unwrap();
        assert!(model.n_clusters() >= 1);
    }

    #[test]
    fn test_incremental_dbscan_predict() {
        let mut model: IncrementalDBSCAN<Array2<Float>, ()> =
            IncrementalDBSCAN::new().eps(0.5).min_samples(2);

        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        model.partial_fit(&x.view()).unwrap();

        let test_data = array![[0.05, 0.05], [5.05, 5.05],];

        let labels = model.predict(&test_data.view()).unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_windowing() {
        let mut model: IncrementalDBSCAN<Array2<Float>, ()> = IncrementalDBSCAN::new()
            .eps(0.5)
            .min_samples(2)
            .window_size(4);

        // Add points that exceed window size
        for i in 0..6 {
            let point = array![[i as Float, 0.0]];
            model.partial_fit(&point.view()).unwrap();
        }

        // Should only keep the most recent points
        assert!(model.points.nrows() <= 4);
    }

    #[test]
    fn test_memory_limit() {
        let mut model: IncrementalDBSCAN<Array2<Float>, ()> = IncrementalDBSCAN::new()
            .eps(0.5)
            .min_samples(2)
            .max_memory_points(5);

        // Add more points than memory limit
        for i in 0..8 {
            let point = array![[i as Float, 0.0]];
            model.partial_fit(&point.view()).unwrap();
        }

        // Should respect memory limit
        assert!(model.points.nrows() <= 5);
    }

    #[test]
    fn test_distance_metrics() {
        for metric in &[
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Chebyshev,
        ] {
            let mut model: IncrementalDBSCAN<Array2<Float>, ()> = IncrementalDBSCAN::new()
                .eps(1.0)
                .min_samples(2)
                .metric(*metric);

            let x = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0],];

            model.partial_fit(&x.view()).unwrap();
            assert!(model.n_clusters() >= 0);
        }
    }

    #[test]
    fn test_cluster_stats() {
        let mut model: IncrementalDBSCAN<Array2<Float>, ()> =
            IncrementalDBSCAN::new().eps(0.5).min_samples(2);

        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        model.partial_fit(&x.view()).unwrap();
        let stats = model.cluster_stats();

        // Should have statistics for each cluster
        assert!(!stats.is_empty());
        for (points, core_points) in stats.values() {
            assert!(*core_points <= *points);
        }
    }
}
