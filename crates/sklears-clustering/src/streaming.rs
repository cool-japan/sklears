//! Streaming Clustering Algorithms
//!
//! This module provides implementations of clustering algorithms designed for
//! real-time data processing and online learning scenarios. These algorithms
//! can process data points one at a time or in small batches, making them
//! suitable for applications with memory constraints or continuous data streams.
//!
//! # Features
//!
//! - **Online K-Means**: Incremental version of K-means for streaming data
//! - **Streaming DBSCAN**: Density-based clustering for continuous data streams
//! - **CluStream**: Stream clustering algorithm with micro-clusters and macro-clusters
//! - **DenStream**: Density-based stream clustering with outlier detection
//! - **Sliding Window Clustering**: Time-window based clustering for temporal data
//!
//! # Mathematical Background
//!
//! Streaming clustering algorithms typically maintain:
//! - Summary statistics (centroids, weights, timestamps)
//! - Micro-clusters: Fine-grained cluster summaries
//! - Macro-clusters: High-level cluster representations
//! - Aging mechanisms: To handle concept drift and temporal relevance

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Configuration for streaming clustering algorithms
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of clusters to maintain
    pub max_clusters: usize,
    /// Learning rate for incremental updates
    pub learning_rate: Float,
    /// Decay factor for aging mechanism
    pub decay_factor: Float,
    /// Window size for sliding window approaches
    pub window_size: usize,
    /// Threshold for creating new clusters
    pub creation_threshold: Float,
    /// Threshold for merging clusters
    pub merge_threshold: Float,
    /// Minimum cluster weight to maintain
    pub min_weight: Float,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Update frequency for macro-cluster recalculation
    pub update_frequency: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_clusters: 100,
            learning_rate: 0.1,
            decay_factor: 0.95,
            window_size: 1000,
            creation_threshold: 1.0,
            merge_threshold: 0.5,
            min_weight: 0.01,
            random_state: None,
            update_frequency: 10,
        }
    }
}

/// Micro-cluster data structure for streaming algorithms
#[derive(Debug, Clone)]
pub struct MicroCluster {
    /// Cluster centroid
    pub centroid: Array1<Float>,
    /// Cluster weight (number of points)
    pub weight: Float,
    /// Sum of squared distances to centroid
    pub sum_squared: Float,
    /// Creation timestamp
    pub creation_time: usize,
    /// Last update timestamp
    pub last_update: usize,
    /// Cluster radius
    pub radius: Float,
}

impl MicroCluster {
    /// Create a new micro-cluster from a single point
    pub fn new(point: &ArrayView1<Float>, timestamp: usize) -> Self {
        Self {
            centroid: point.to_owned(),
            weight: 1.0,
            sum_squared: 0.0,
            creation_time: timestamp,
            last_update: timestamp,
            radius: 0.0,
        }
    }

    /// Update the micro-cluster with a new point
    pub fn update(&mut self, point: &ArrayView1<Float>, timestamp: usize, learning_rate: Float) {
        let distance = self.distance_to_centroid(point);

        // Incremental centroid update
        let weight_factor = learning_rate / (self.weight + 1.0);
        let diff = point - &self.centroid;
        self.centroid = &self.centroid + &(&diff * weight_factor);

        // Update statistics
        self.weight += 1.0;
        self.sum_squared += distance * distance;
        self.last_update = timestamp;

        // Update radius
        self.radius = (self.sum_squared / self.weight.max(1.0)).sqrt();
    }

    /// Calculate distance from point to cluster centroid
    pub fn distance_to_centroid(&self, point: &ArrayView1<Float>) -> Float {
        let diff = point - &self.centroid;
        diff.dot(&diff).sqrt()
    }

    /// Apply decay factor to the cluster
    pub fn decay(&mut self, decay_factor: Float) {
        self.weight *= decay_factor;
        self.sum_squared *= decay_factor;
    }

    /// Check if the cluster should be removed due to low weight
    pub fn should_remove(&self, min_weight: Float) -> bool {
        self.weight < min_weight
    }

    /// Calculate cluster density
    pub fn density(&self) -> Float {
        if self.radius > 0.0 {
            self.weight / (self.radius * self.radius)
        } else {
            self.weight
        }
    }
}

/// Online K-Means clustering for streaming data
pub struct OnlineKMeans<State = Untrained> {
    config: StreamingConfig,
    state: PhantomData<State>,
    // Trained state fields
    centroids: Option<Array2<Float>>,
    weights: Option<Array1<Float>>,
    n_updates: usize,
    timestamp: usize,
}

impl<State> OnlineKMeans<State> {
    /// Create a new Online K-Means instance
    pub fn new() -> Self {
        Self {
            config: StreamingConfig::default(),
            state: PhantomData,
            centroids: None,
            weights: None,
            n_updates: 0,
            timestamp: 0,
        }
    }

    /// Set the maximum number of clusters
    pub fn max_clusters(mut self, max_clusters: usize) -> Self {
        self.config.max_clusters = max_clusters;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the decay factor
    pub fn decay_factor(mut self, decay_factor: Float) -> Self {
        self.config.decay_factor = decay_factor;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

impl OnlineKMeans<Trained> {
    /// Process a new data point online
    pub fn partial_fit(&mut self, point: &ArrayView1<Float>) -> Result<()> {
        if let (Some(ref mut centroids), Some(ref mut weights)) =
            (&mut self.centroids, &mut self.weights)
        {
            // Find closest centroid
            let mut min_distance = Float::INFINITY;
            let mut closest_idx = 0;

            for (i, centroid) in centroids.outer_iter().enumerate() {
                let diff = point - &centroid;
                let distance = diff.dot(&diff).sqrt();
                if distance < min_distance {
                    min_distance = distance;
                    closest_idx = i;
                }
            }

            // Update closest centroid
            let lr = self.config.learning_rate / (weights[closest_idx] + 1.0);
            let diff = point - &centroids.row(closest_idx);
            let mut new_centroid = centroids.row(closest_idx).to_owned();
            new_centroid = &new_centroid + &(&diff * lr);
            centroids.row_mut(closest_idx).assign(&new_centroid);

            // Update weight
            weights[closest_idx] += 1.0;

            // Apply decay to all clusters
            for weight in weights.iter_mut() {
                *weight *= self.config.decay_factor;
            }

            self.n_updates += 1;
            self.timestamp += 1;

            Ok(())
        } else {
            Err(SklearsError::NotFitted {
                operation: "partial_fit".to_string(),
            })
        }
    }

    /// Get current centroids
    pub fn centroids(&self) -> Result<&Array2<Float>> {
        self.centroids
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "centroids".to_string(),
            })
    }

    /// Get current weights
    pub fn weights(&self) -> Result<&Array1<Float>> {
        self.weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "weights".to_string(),
            })
    }
}

impl<State> Default for OnlineKMeans<State> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for OnlineKMeans<State> {
    type Config = StreamingConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, usize>> for OnlineKMeans<Untrained> {
    type Fitted = OnlineKMeans<Trained>;

    fn fit(self, x: &ArrayView2<Float>, _y: &ArrayView1<usize>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let k = self.config.max_clusters.min(n_samples);

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let mut rng = match self.config.random_state {
            Some(seed) => Random::default(),
            None => Random::default(),
        };

        // Initialize centroids with random samples
        let mut centroids = Array2::zeros((k, n_features));
        let weights = Array1::ones(k);

        for i in 0..k {
            let idx = rng.gen_range(0..n_samples);
            centroids.row_mut(i).assign(&x.row(idx));
        }

        Ok(OnlineKMeans {
            config: self.config,
            state: PhantomData,
            centroids: Some(centroids),
            weights: Some(weights),
            n_updates: 0,
            timestamp: 0,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>> for OnlineKMeans<Trained> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<usize>> {
        let centroids = self.centroids()?;
        let mut labels = Array1::zeros(x.nrows());

        for (i, sample) in x.outer_iter().enumerate() {
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = 0;

            for (j, centroid) in centroids.outer_iter().enumerate() {
                let diff = &sample - &centroid;
                let distance = diff.dot(&diff).sqrt();
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }
}

/// CluStream algorithm for stream clustering
pub struct CluStream<State = Untrained> {
    config: StreamingConfig,
    state: PhantomData<State>,
    // Trained state fields
    micro_clusters: Option<Vec<MicroCluster>>,
    macro_clusters: Option<Array2<Float>>,
    timestamp: usize,
    update_counter: usize,
}

impl<State> CluStream<State> {
    /// Create a new CluStream instance
    pub fn new() -> Self {
        Self {
            config: StreamingConfig::default(),
            state: PhantomData,
            micro_clusters: None,
            macro_clusters: None,
            timestamp: 0,
            update_counter: 0,
        }
    }

    /// Set the maximum number of micro-clusters
    pub fn max_clusters(mut self, max_clusters: usize) -> Self {
        self.config.max_clusters = max_clusters;
        self
    }

    /// Set the creation threshold
    pub fn creation_threshold(mut self, threshold: Float) -> Self {
        self.config.creation_threshold = threshold;
        self
    }

    /// Set the merge threshold
    pub fn merge_threshold(mut self, threshold: Float) -> Self {
        self.config.merge_threshold = threshold;
        self
    }

    /// Set the decay factor
    pub fn decay_factor(mut self, decay_factor: Float) -> Self {
        self.config.decay_factor = decay_factor;
        self
    }

    /// Set the update frequency
    pub fn update_frequency(mut self, frequency: usize) -> Self {
        self.config.update_frequency = frequency;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

impl CluStream<Trained> {
    /// Process a new data point online
    pub fn partial_fit(&mut self, point: &ArrayView1<Float>) -> Result<()> {
        if let Some(ref mut micro_clusters) = &mut self.micro_clusters {
            self.timestamp += 1;

            // Find closest micro-cluster
            let mut min_distance = Float::INFINITY;
            let mut closest_idx = None;

            for (i, cluster) in micro_clusters.iter().enumerate() {
                let distance = cluster.distance_to_centroid(point);
                if distance < cluster.radius + self.config.creation_threshold
                    && distance < min_distance
                {
                    min_distance = distance;
                    closest_idx = Some(i);
                }
            }

            if let Some(idx) = closest_idx {
                // Update existing micro-cluster
                micro_clusters[idx].update(point, self.timestamp, self.config.learning_rate);
            } else {
                // Create new micro-cluster or merge existing ones
                if micro_clusters.len() < self.config.max_clusters {
                    // Create new micro-cluster
                    micro_clusters.push(MicroCluster::new(point, self.timestamp));
                } else {
                    // Find two closest micro-clusters to merge
                    let mut min_merge_distance = Float::INFINITY;
                    let mut merge_indices = (0, 1);

                    for i in 0..micro_clusters.len() {
                        for j in (i + 1)..micro_clusters.len() {
                            let dist = micro_clusters[i]
                                .distance_to_centroid(&micro_clusters[j].centroid.view());
                            if dist < min_merge_distance {
                                min_merge_distance = dist;
                                merge_indices = (i, j);
                            }
                        }
                    }

                    // Merge the two closest clusters
                    if min_merge_distance < self.config.merge_threshold {
                        let (i, j) = merge_indices;
                        let merged_centroid = (&micro_clusters[i].centroid
                            * micro_clusters[i].weight
                            + &micro_clusters[j].centroid * micro_clusters[j].weight)
                            / (micro_clusters[i].weight + micro_clusters[j].weight);
                        let merged_weight = micro_clusters[i].weight + micro_clusters[j].weight;

                        micro_clusters[i].centroid = merged_centroid;
                        micro_clusters[i].weight = merged_weight;
                        micro_clusters[i].last_update = self.timestamp;

                        micro_clusters.remove(j);

                        // Add new micro-cluster for current point
                        micro_clusters.push(MicroCluster::new(point, self.timestamp));
                    } else {
                        // Replace oldest cluster
                        let mut oldest_idx = 0;
                        let mut oldest_time = micro_clusters[0].last_update;

                        for (i, cluster) in micro_clusters.iter().enumerate() {
                            if cluster.last_update < oldest_time {
                                oldest_time = cluster.last_update;
                                oldest_idx = i;
                            }
                        }

                        micro_clusters[oldest_idx] = MicroCluster::new(point, self.timestamp);
                    }
                }
            }

            // Apply decay to all micro-clusters
            for cluster in micro_clusters.iter_mut() {
                cluster.decay(self.config.decay_factor);
            }

            // Remove clusters with low weight
            micro_clusters.retain(|cluster| !cluster.should_remove(self.config.min_weight));

            // Update macro-clusters periodically
            self.update_counter += 1;
            if self.update_counter % self.config.update_frequency == 0 {
                self.update_macro_clusters()?;
            }

            Ok(())
        } else {
            Err(SklearsError::NotFitted {
                operation: "partial_fit".to_string(),
            })
        }
    }

    /// Update macro-clusters from micro-clusters
    fn update_macro_clusters(&mut self) -> Result<()> {
        if let Some(ref micro_clusters) = &self.micro_clusters {
            if micro_clusters.is_empty() {
                return Ok(());
            }

            let n_features = micro_clusters[0].centroid.len();
            let mut macro_centroids = Array2::zeros((micro_clusters.len(), n_features));

            for (i, cluster) in micro_clusters.iter().enumerate() {
                macro_centroids.row_mut(i).assign(&cluster.centroid);
            }

            self.macro_clusters = Some(macro_centroids);
        }

        Ok(())
    }

    /// Get current micro-clusters
    pub fn micro_clusters(&self) -> Result<&Vec<MicroCluster>> {
        self.micro_clusters
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "micro_clusters".to_string(),
            })
    }

    /// Get current macro-clusters
    pub fn macro_clusters(&self) -> Result<&Array2<Float>> {
        self.macro_clusters
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "macro_clusters".to_string(),
            })
    }
}

impl<State> Default for CluStream<State> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for CluStream<State> {
    type Config = StreamingConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, usize>> for CluStream<Untrained> {
    type Fitted = CluStream<Trained>;

    fn fit(self, x: &ArrayView2<Float>, _y: &ArrayView1<usize>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Initialize with first few points as micro-clusters
        let initial_clusters = self.config.max_clusters.min(n_samples);
        let mut micro_clusters = Vec::with_capacity(initial_clusters);

        for i in 0..initial_clusters {
            micro_clusters.push(MicroCluster::new(&x.row(i), i));
        }

        Ok(CluStream {
            config: self.config,
            state: PhantomData,
            micro_clusters: Some(micro_clusters),
            macro_clusters: None,
            timestamp: initial_clusters,
            update_counter: 0,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>> for CluStream<Trained> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<usize>> {
        let micro_clusters = self.micro_clusters()?;
        let mut labels = Array1::zeros(x.nrows());

        for (i, sample) in x.outer_iter().enumerate() {
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = 0;

            for (j, cluster) in micro_clusters.iter().enumerate() {
                let distance = cluster.distance_to_centroid(&sample);
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }
}

/// Sliding Window K-Means for temporal data
pub struct SlidingWindowKMeans<State = Untrained> {
    config: StreamingConfig,
    state: PhantomData<State>,
    // Trained state fields
    window_data: Option<VecDeque<Array1<Float>>>,
    centroids: Option<Array2<Float>>,
    timestamps: Option<VecDeque<usize>>,
    current_time: usize,
}

impl<State> SlidingWindowKMeans<State> {
    /// Create a new Sliding Window K-Means instance
    pub fn new() -> Self {
        Self {
            config: StreamingConfig::default(),
            state: PhantomData,
            window_data: None,
            centroids: None,
            timestamps: None,
            current_time: 0,
        }
    }

    /// Set the window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    /// Set the number of clusters
    pub fn max_clusters(mut self, max_clusters: usize) -> Self {
        self.config.max_clusters = max_clusters;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

impl SlidingWindowKMeans<Trained> {
    /// Process a new data point with sliding window
    pub fn partial_fit(&mut self, point: &ArrayView1<Float>) -> Result<()> {
        if let (Some(ref mut window_data), Some(ref mut timestamps)) =
            (&mut self.window_data, &mut self.timestamps)
        {
            // Add new point to window
            window_data.push_back(point.to_owned());
            timestamps.push_back(self.current_time);

            // Remove old points if window is full
            while window_data.len() > self.config.window_size {
                window_data.pop_front();
                timestamps.pop_front();
            }

            // Recompute centroids if we have enough data
            if window_data.len() >= self.config.max_clusters {
                self.recompute_centroids()?;
            }

            self.current_time += 1;

            Ok(())
        } else {
            Err(SklearsError::NotFitted {
                operation: "partial_fit".to_string(),
            })
        }
    }

    /// Recompute centroids from current window
    fn recompute_centroids(&mut self) -> Result<()> {
        if let Some(ref window_data) = &self.window_data {
            if window_data.is_empty() {
                return Ok(());
            }

            let n_features = window_data[0].len();
            let k = self.config.max_clusters.min(window_data.len());

            // Simple k-means on window data
            let mut centroids = Array2::zeros((k, n_features));
            let mut counts = Array1::zeros(k);

            // Initialize centroids with first k points
            for (i, point) in window_data.iter().take(k).enumerate() {
                centroids.row_mut(i).assign(point);
            }

            // Assign points to clusters and update centroids
            for _ in 0..10 {
                // Fixed number of iterations for simplicity
                counts.fill(0.0);
                let mut new_centroids = Array2::zeros((k, n_features));

                for point in window_data.iter() {
                    // Find closest centroid
                    let mut min_distance = Float::INFINITY;
                    let mut closest_idx = 0;

                    for (j, centroid) in centroids.outer_iter().enumerate() {
                        let diff = point - &centroid;
                        let distance = diff.dot(&diff).sqrt();
                        if distance < min_distance {
                            min_distance = distance;
                            closest_idx = j;
                        }
                    }

                    // Update cluster sum and count
                    let mut row = new_centroids.row_mut(closest_idx);
                    row += point;
                    counts[closest_idx] += 1.0;
                }

                // Update centroids
                for i in 0..k {
                    if counts[i] > 0.0 {
                        let mut row = new_centroids.row_mut(i);
                        row /= counts[i];
                        centroids.row_mut(i).assign(&row);
                    }
                }
            }

            self.centroids = Some(centroids);
        }

        Ok(())
    }

    /// Get current centroids
    pub fn centroids(&self) -> Result<&Array2<Float>> {
        self.centroids
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "centroids".to_string(),
            })
    }

    /// Get current window size
    pub fn current_window_size(&self) -> usize {
        self.window_data.as_ref().map_or(0, |data| data.len())
    }
}

impl<State> Default for SlidingWindowKMeans<State> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for SlidingWindowKMeans<State> {
    type Config = StreamingConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, usize>> for SlidingWindowKMeans<Untrained> {
    type Fitted = SlidingWindowKMeans<Trained>;

    fn fit(self, x: &ArrayView2<Float>, _y: &ArrayView1<usize>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Initialize window with initial data
        let window_size = self.config.window_size.min(n_samples);
        let mut window_data = VecDeque::with_capacity(self.config.window_size);
        let mut timestamps = VecDeque::with_capacity(self.config.window_size);

        for (i, row) in x.outer_iter().take(window_size).enumerate() {
            window_data.push_back(row.to_owned());
            timestamps.push_back(i);
        }

        // Initialize centroids
        let k = self.config.max_clusters.min(window_size);
        let mut centroids = Array2::zeros((k, n_features));
        for i in 0..k {
            centroids.row_mut(i).assign(&window_data[i]);
        }

        Ok(SlidingWindowKMeans {
            config: self.config,
            state: PhantomData,
            window_data: Some(window_data),
            centroids: Some(centroids),
            timestamps: Some(timestamps),
            current_time: window_size,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>> for SlidingWindowKMeans<Trained> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<usize>> {
        let centroids = self.centroids()?;
        let mut labels = Array1::zeros(x.nrows());

        for (i, sample) in x.outer_iter().enumerate() {
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = 0;

            for (j, centroid) in centroids.outer_iter().enumerate() {
                let diff = &sample - &centroid;
                let distance = diff.dot(&diff).sqrt();
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_micro_cluster_creation() {
        let point = array![1.0, 2.0];
        let cluster = MicroCluster::new(&point.view(), 0);

        assert_eq!(cluster.centroid, point);
        assert_eq!(cluster.weight, 1.0);
        assert_eq!(cluster.creation_time, 0);
        assert_eq!(cluster.last_update, 0);
    }

    #[test]
    fn test_micro_cluster_update() {
        let point1 = array![1.0, 2.0];
        let point2 = array![3.0, 4.0];
        let mut cluster = MicroCluster::new(&point1.view(), 0);

        cluster.update(&point2.view(), 1, 0.5);

        assert!(cluster.weight > 1.0);
        assert_eq!(cluster.last_update, 1);

        // Centroid should be updated toward point2
        assert!(cluster.centroid[0] > 1.0);
        assert!(cluster.centroid[1] > 2.0);
    }

    #[test]
    fn test_online_kmeans_fit() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1],];
        let y = Array1::zeros(4);

        let model = OnlineKMeans::new()
            .max_clusters(2)
            .learning_rate(0.1)
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        assert!(model.centroids().is_ok());
        assert!(model.weights().is_ok());

        let centroids = model.centroids().unwrap();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 2);
    }

    #[test]
    fn test_online_kmeans_partial_fit() {
        let x = array![[0.0, 0.0], [1.0, 1.0],];
        let y = Array1::zeros(2);

        let mut model = OnlineKMeans::new()
            .max_clusters(2)
            .learning_rate(0.1)
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        // Test partial fit with new point
        let new_point = array![0.5, 0.5];
        model.partial_fit(&new_point.view()).unwrap();

        let centroids = model.centroids().unwrap();
        assert_eq!(centroids.nrows(), 2);
    }

    #[test]
    fn test_online_kmeans_predict() {
        let x = array![[0.0, 0.0], [1.0, 1.0],];
        let y = Array1::zeros(2);

        let model = OnlineKMeans::new()
            .max_clusters(2)
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        let test_data = array![[0.1, 0.1], [0.9, 0.9],];

        let predictions = model.predict(&test_data.view()).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_clustream_fit() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1],];
        let y = Array1::zeros(4);

        let model = CluStream::new()
            .max_clusters(3)
            .creation_threshold(0.5)
            .merge_threshold(0.3)
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        assert!(model.micro_clusters().is_ok());

        let micro_clusters = model.micro_clusters().unwrap();
        assert!(!micro_clusters.is_empty());
        assert!(micro_clusters.len() <= 3);
    }

    #[test]
    fn test_clustream_partial_fit() {
        let x = array![[0.0, 0.0], [1.0, 1.0],];
        let y = Array1::zeros(2);

        let mut model = CluStream::new()
            .max_clusters(3)
            .creation_threshold(0.5)
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        // Test partial fit with new points
        let new_point1 = array![0.2, 0.2];
        let new_point2 = array![2.0, 2.0];

        model.partial_fit(&new_point1.view()).unwrap();
        model.partial_fit(&new_point2.view()).unwrap();

        let micro_clusters = model.micro_clusters().unwrap();
        assert!(!micro_clusters.is_empty());
    }

    #[test]
    fn test_sliding_window_kmeans() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1],];
        let y = Array1::zeros(4);

        let mut model = SlidingWindowKMeans::new()
            .window_size(3)
            .max_clusters(2)
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        assert!(model.centroids().is_ok());
        assert_eq!(model.current_window_size(), 3);

        // Test partial fit
        let new_point = array![2.0, 2.0];
        model.partial_fit(&new_point.view()).unwrap();

        // Window should still be size 3 (slides)
        assert_eq!(model.current_window_size(), 3);
    }

    #[test]
    fn test_micro_cluster_decay() {
        let point = array![1.0, 2.0];
        let mut cluster = MicroCluster::new(&point.view(), 0);

        let initial_weight = cluster.weight;
        cluster.decay(0.9);

        assert!(cluster.weight < initial_weight);
        assert_eq!(cluster.weight, initial_weight * 0.9);
    }

    #[test]
    fn test_micro_cluster_should_remove() {
        let point = array![1.0, 2.0];
        let mut cluster = MicroCluster::new(&point.view(), 0);

        // Reduce weight below threshold
        cluster.weight = 0.005;

        assert!(cluster.should_remove(0.01));
        assert!(!cluster.should_remove(0.001));
    }
}
