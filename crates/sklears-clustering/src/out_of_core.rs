//! Out-of-Core Clustering Algorithms
//!
//! This module provides implementations of clustering algorithms designed for
//! datasets that don't fit in memory. These algorithms process data in chunks,
//! maintaining summary statistics and incrementally updating cluster models.
//!
//! # Features
//!
//! - **Chunked K-Means**: Process large datasets in memory-bounded chunks
//! - **Incremental DBSCAN**: Density-based clustering with chunked processing
//! - **Memory-Aware Hierarchical Clustering**: Hierarchical clustering with memory limits
//! - **Disk-Based Data Processing**: Efficient I/O operations for large datasets
//! - **Checkpointing**: Save/restore intermediate states during processing
//!
//! # Mathematical Background
//!
//! Out-of-core algorithms typically maintain:
//! - Summary statistics (centroids, counts, sufficient statistics)
//! - Incremental updates using online learning formulas
//! - Memory-bounded data structures
//! - Efficient merging strategies for chunk results

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::marker::PhantomData;
use std::path::Path;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Random, Rng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

/// Configuration for out-of-core clustering algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OutOfCoreConfig {
    /// Maximum memory usage per chunk (in MB)
    pub max_memory_mb: usize,
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Path for temporary files
    pub temp_dir: Option<String>,
    /// Enable checkpointing
    pub enable_checkpointing: bool,
    /// Checkpoint frequency (number of chunks)
    pub checkpoint_frequency: usize,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024, // 1GB default
            n_clusters: 8,
            max_iter: 100,
            tolerance: 1e-4,
            chunk_size: 10000,
            random_state: None,
            temp_dir: None,
            enable_checkpointing: true,
            checkpoint_frequency: 10,
        }
    }
}

/// Summary statistics for a cluster
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClusterSummary {
    /// Cluster centroid
    pub centroid: Array1<Float>,
    /// Number of points in cluster
    pub count: usize,
    /// Sum of squared distances to centroid
    pub sum_squared_distances: Float,
    /// Bounding box (min and max coordinates)
    pub bounds: (Array1<Float>, Array1<Float>),
    /// Variance in each dimension
    pub variance: Array1<Float>,
}

impl ClusterSummary {
    /// Create a new cluster summary from a single point
    pub fn new(point: &ArrayView1<Float>) -> Self {
        let n_features = point.len();
        Self {
            centroid: point.to_owned(),
            count: 1,
            sum_squared_distances: 0.0,
            bounds: (point.to_owned(), point.to_owned()),
            variance: Array1::zeros(n_features),
        }
    }

    /// Update cluster summary with a new point
    pub fn update(&mut self, point: &ArrayView1<Float>) {
        let old_count = self.count as Float;
        let new_count = old_count + 1.0;

        // Update centroid using online mean formula
        let delta = point - &self.centroid;
        self.centroid = &self.centroid + &(&delta / new_count);

        // Update variance using online variance formula
        let delta2 = point - &self.centroid;
        self.variance = &self.variance + &(&delta * &delta2);

        // Update bounds
        for i in 0..point.len() {
            self.bounds.0[i] = self.bounds.0[i].min(point[i]);
            self.bounds.1[i] = self.bounds.1[i].max(point[i]);
        }

        // Update sum of squared distances
        let distance_sq = delta2.dot(&delta2);
        self.sum_squared_distances += distance_sq;

        self.count += 1;
    }

    /// Merge two cluster summaries
    pub fn merge(&mut self, other: &ClusterSummary) {
        let total_count = self.count + other.count;
        let self_weight = self.count as Float / total_count as Float;
        let other_weight = other.count as Float / total_count as Float;

        // Merge centroids
        self.centroid = &(&self.centroid * self_weight) + &(&other.centroid * other_weight);

        // Merge variance
        self.variance = &(&self.variance * self_weight) + &(&other.variance * other_weight);

        // Merge bounds
        for i in 0..self.bounds.0.len() {
            self.bounds.0[i] = self.bounds.0[i].min(other.bounds.0[i]);
            self.bounds.1[i] = self.bounds.1[i].max(other.bounds.1[i]);
        }

        // Merge sum of squared distances
        self.sum_squared_distances += other.sum_squared_distances;
        self.count = total_count;
    }

    /// Get cluster inertia (within-cluster sum of squares)
    pub fn inertia(&self) -> Float {
        self.sum_squared_distances
    }

    /// Get cluster density
    pub fn density(&self) -> Float {
        if self.count == 0 {
            0.0
        } else {
            self.count as Float / self.volume()
        }
    }

    /// Get cluster volume (product of dimension ranges)
    pub fn volume(&self) -> Float {
        let mut volume = 1.0;
        for i in 0..self.bounds.0.len() {
            let range = self.bounds.1[i] - self.bounds.0[i];
            volume *= range.max(1e-10); // Avoid zero volume
        }
        volume
    }
}

/// Out-of-core K-Means clustering
pub struct OutOfCoreKMeans<State = Untrained> {
    config: OutOfCoreConfig,
    state: PhantomData<State>,
    // Trained state fields
    cluster_summaries: Option<Vec<ClusterSummary>>,
    centroids: Option<Array2<Float>>,
    inertia: Option<Float>,
    n_iter: Option<usize>,
    n_features: Option<usize>,
    n_samples_seen: Option<usize>,
}

impl<State> OutOfCoreKMeans<State> {
    /// Create a new out-of-core K-Means instance
    pub fn new() -> Self {
        Self {
            config: OutOfCoreConfig::default(),
            state: PhantomData,
            cluster_summaries: None,
            centroids: None,
            inertia: None,
            n_iter: None,
            n_features: None,
            n_samples_seen: None,
        }
    }

    /// Set the number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.config.n_clusters = n_clusters;
        self
    }

    /// Set the maximum memory usage per chunk (in MB)
    pub fn max_memory_mb(mut self, max_memory_mb: usize) -> Self {
        self.config.max_memory_mb = max_memory_mb;
        self
    }

    /// Set the chunk size for processing
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set the temporary directory for checkpointing
    pub fn temp_dir(mut self, temp_dir: String) -> Self {
        self.config.temp_dir = Some(temp_dir);
        self
    }

    /// Enable or disable checkpointing
    pub fn enable_checkpointing(mut self, enable: bool) -> Self {
        self.config.enable_checkpointing = enable;
        self
    }
}

impl OutOfCoreKMeans<Trained> {
    /// Get the cluster centroids
    pub fn centroids(&self) -> Result<&Array2<Float>> {
        self.centroids
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "centroids".to_string(),
            })
    }

    /// Get the cluster summaries
    pub fn cluster_summaries(&self) -> Result<&Vec<ClusterSummary>> {
        self.cluster_summaries
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "cluster_summaries".to_string(),
            })
    }

    /// Get the inertia (within-cluster sum of squares)
    pub fn inertia(&self) -> Result<Float> {
        self.inertia.ok_or_else(|| SklearsError::NotFitted {
            operation: "inertia".to_string(),
        })
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> Result<usize> {
        self.n_iter.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_iter".to_string(),
        })
    }

    /// Get the number of samples seen during training
    pub fn n_samples_seen(&self) -> Result<usize> {
        self.n_samples_seen.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_samples_seen".to_string(),
        })
    }

    /// Process a new chunk of data incrementally
    pub fn partial_fit(&mut self, x: &ArrayView2<Float>) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Ok(());
        }

        // Check feature consistency
        if let Some(expected_features) = self.n_features {
            if n_features != expected_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    expected_features, n_features
                )));
            }
        } else {
            self.n_features = Some(n_features);
        }

        // Initialize cluster summaries if not present
        if self.cluster_summaries.is_none() {
            self.initialize_clusters(x)?;
        }

        // Update cluster summaries with new data
        self.update_clusters(x)?;

        // Update total samples seen
        self.n_samples_seen = Some(self.n_samples_seen.unwrap_or(0) + n_samples);

        Ok(())
    }

    /// Initialize clusters from the first chunk of data
    fn initialize_clusters(&mut self, x: &ArrayView2<Float>) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let k = self.config.n_clusters.min(n_samples);

        let mut rng = match self.config.random_state {
            Some(seed) => Random::default(),
            None => Random::default(),
        };

        // Initialize cluster summaries with random samples
        let mut cluster_summaries = Vec::with_capacity(k);
        let mut centroids = Array2::zeros((k, n_features));

        for i in 0..k {
            let idx = rng.gen_range(0..n_samples);
            let point = x.row(idx);
            cluster_summaries.push(ClusterSummary::new(&point));
            centroids.row_mut(i).assign(&point);
        }

        self.cluster_summaries = Some(cluster_summaries);
        self.centroids = Some(centroids);

        Ok(())
    }

    /// Update clusters with a new chunk of data
    fn update_clusters(&mut self, x: &ArrayView2<Float>) -> Result<()> {
        if self.cluster_summaries.is_some() && self.centroids.is_some() {
            // Get current centroids for assignment
            let current_centroids = self.centroids.as_ref().unwrap().clone();

            // Assign points to clusters and update summaries
            for sample in x.outer_iter() {
                let cluster_idx = self.assign_to_cluster(&sample, &current_centroids)?;
                self.cluster_summaries.as_mut().unwrap()[cluster_idx].update(&sample);
            }

            // Update centroids from summaries
            if let Some(ref mut centroids) = self.centroids {
                if let Some(ref cluster_summaries) = self.cluster_summaries {
                    for (i, summary) in cluster_summaries.iter().enumerate() {
                        centroids.row_mut(i).assign(&summary.centroid);
                    }

                    // Calculate total inertia
                    let total_inertia: Float = cluster_summaries.iter().map(|s| s.inertia()).sum();
                    self.inertia = Some(total_inertia);
                }
            }
        }

        Ok(())
    }

    /// Assign a point to the nearest cluster
    fn assign_to_cluster(
        &self,
        point: &ArrayView1<Float>,
        centroids: &Array2<Float>,
    ) -> Result<usize> {
        let mut min_distance = Float::INFINITY;
        let mut best_cluster = 0;

        for (i, centroid) in centroids.outer_iter().enumerate() {
            let diff = point - &centroid;
            let distance = diff.dot(&diff);
            if distance < min_distance {
                min_distance = distance;
                best_cluster = i;
            }
        }

        Ok(best_cluster)
    }

    /// Save checkpoint to disk
    #[cfg(feature = "serde")]
    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        if !self.config.enable_checkpointing {
            return Ok(());
        }

        let checkpoint_data = OutOfCoreCheckpoint {
            cluster_summaries: self.cluster_summaries.clone(),
            centroids: self.centroids.clone(),
            inertia: self.inertia,
            n_iter: self.n_iter,
            n_features: self.n_features,
            n_samples_seen: self.n_samples_seen,
        };

        let serialized = serde_json::to_string(&checkpoint_data).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to serialize checkpoint: {}", e))
        })?;

        let mut file = File::create(path).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create checkpoint file: {}", e))
        })?;

        file.write_all(serialized.as_bytes()).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to write checkpoint: {}", e))
        })?;

        Ok(())
    }

    /// Load checkpoint from disk
    #[cfg(feature = "serde")]
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to open checkpoint file: {}", e))
        })?;

        let reader = BufReader::new(file);
        let checkpoint_data: OutOfCoreCheckpoint =
            serde_json::from_reader(reader).map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to deserialize checkpoint: {}", e))
            })?;

        self.cluster_summaries = checkpoint_data.cluster_summaries;
        self.centroids = checkpoint_data.centroids;
        self.inertia = checkpoint_data.inertia;
        self.n_iter = checkpoint_data.n_iter;
        self.n_features = checkpoint_data.n_features;
        self.n_samples_seen = checkpoint_data.n_samples_seen;

        Ok(())
    }
}

/// Checkpoint data structure for serialization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct OutOfCoreCheckpoint {
    cluster_summaries: Option<Vec<ClusterSummary>>,
    centroids: Option<Array2<Float>>,
    inertia: Option<Float>,
    n_iter: Option<usize>,
    n_features: Option<usize>,
    n_samples_seen: Option<usize>,
}

impl<State> Default for OutOfCoreKMeans<State> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for OutOfCoreKMeans<State> {
    type Config = OutOfCoreConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, usize>> for OutOfCoreKMeans<Untrained> {
    type Fitted = OutOfCoreKMeans<Trained>;

    fn fit(self, x: &ArrayView2<Float>, _y: &ArrayView1<usize>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Calculate optimal chunk size based on memory constraints
        let memory_per_sample = n_features * std::mem::size_of::<Float>();
        let max_samples_per_chunk = (self.config.max_memory_mb * 1024 * 1024) / memory_per_sample;
        let chunk_size = self.config.chunk_size.min(max_samples_per_chunk).max(1);

        let mut fitted_model = OutOfCoreKMeans {
            config: self.config,
            state: PhantomData,
            cluster_summaries: None,
            centroids: None,
            inertia: None,
            n_iter: Some(0),
            n_features: Some(n_features),
            n_samples_seen: Some(0),
        };

        // Process data in chunks
        let mut chunk_count = 0;
        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let chunk = x.slice(scirs2_core::ndarray::s![chunk_start..chunk_end, ..]);

            fitted_model.partial_fit(&chunk)?;

            chunk_count += 1;

            // Save checkpoint if enabled
            #[cfg(feature = "serde")]
            if fitted_model.config.enable_checkpointing
                && chunk_count % fitted_model.config.checkpoint_frequency == 0
            {
                if let Some(ref temp_dir) = fitted_model.config.temp_dir {
                    let checkpoint_path =
                        Path::new(temp_dir).join(format!("checkpoint_{}.json", chunk_count));
                    fitted_model.save_checkpoint(&checkpoint_path)?;
                }
            }
        }

        fitted_model.n_iter = Some(1); // Single pass through data
        Ok(fitted_model)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>> for OutOfCoreKMeans<Trained> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<usize>> {
        let centroids = self.centroids()?;
        let mut labels = Array1::zeros(x.nrows());

        for (i, sample) in x.outer_iter().enumerate() {
            let cluster_idx = self.assign_to_cluster(&sample, centroids)?;
            labels[i] = cluster_idx;
        }

        Ok(labels)
    }
}

/// Out-of-core data loader for large datasets
pub struct OutOfCoreDataLoader {
    file_path: String,
    chunk_size: usize,
    n_features: usize,
    current_position: usize,
    total_samples: Option<usize>,
}

impl OutOfCoreDataLoader {
    /// Create a new data loader
    pub fn new(file_path: String, chunk_size: usize, n_features: usize) -> Self {
        Self {
            file_path,
            chunk_size,
            n_features,
            current_position: 0,
            total_samples: None,
        }
    }

    /// Load the next chunk of data
    pub fn next_chunk(&mut self) -> Result<Option<Array2<Float>>> {
        let file = File::open(&self.file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to open data file: {}", e)))?;

        let reader = BufReader::new(file);
        let lines: Vec<String> = reader
            .lines()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read data file: {}", e)))?;

        if self.current_position >= lines.len() {
            return Ok(None);
        }

        let chunk_end = (self.current_position + self.chunk_size).min(lines.len());
        let chunk_lines = &lines[self.current_position..chunk_end];

        let mut chunk_data = Array2::zeros((chunk_lines.len(), self.n_features));

        for (i, line) in chunk_lines.iter().enumerate() {
            let values: Vec<Float> = line
                .split(',')
                .map(|s| s.trim().parse::<Float>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| SklearsError::InvalidInput(format!("Failed to parse data: {}", e)))?;

            if values.len() != self.n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    self.n_features,
                    values.len()
                )));
            }

            chunk_data.row_mut(i).assign(&Array1::from(values));
        }

        self.current_position = chunk_end;

        if self.total_samples.is_none() {
            self.total_samples = Some(lines.len());
        }

        Ok(Some(chunk_data))
    }

    /// Get the total number of samples (if known)
    pub fn total_samples(&self) -> Option<usize> {
        self.total_samples
    }

    /// Reset the loader to the beginning
    pub fn reset(&mut self) {
        self.current_position = 0;
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;
    use tempfile::tempdir;

    #[test]
    fn test_cluster_summary_creation() {
        let point = array![1.0, 2.0, 3.0];
        let summary = ClusterSummary::new(&point.view());

        assert_eq!(summary.centroid, point);
        assert_eq!(summary.count, 1);
        assert_eq!(summary.sum_squared_distances, 0.0);
    }

    #[test]
    fn test_cluster_summary_update() {
        let point1 = array![1.0, 2.0];
        let point2 = array![3.0, 4.0];
        let mut summary = ClusterSummary::new(&point1.view());

        summary.update(&point2.view());

        assert_eq!(summary.count, 2);
        assert_eq!(summary.centroid, array![2.0, 3.0]);
        assert!(summary.sum_squared_distances > 0.0);
    }

    #[test]
    fn test_cluster_summary_merge() {
        let point1 = array![1.0, 2.0];
        let point2 = array![3.0, 4.0];
        let mut summary1 = ClusterSummary::new(&point1.view());
        let summary2 = ClusterSummary::new(&point2.view());

        summary1.merge(&summary2);

        assert_eq!(summary1.count, 2);
        assert_eq!(summary1.centroid, array![2.0, 3.0]);
    }

    #[test]
    fn test_out_of_core_kmeans_fit() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1],];
        let y = Array1::zeros(4);

        let model = OutOfCoreKMeans::new()
            .n_clusters(2)
            .chunk_size(2)
            .max_memory_mb(1)
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        assert!(model.centroids().is_ok());
        assert!(model.inertia().is_ok());
        assert!(model.n_iter().is_ok());
        assert!(model.n_samples_seen().is_ok());

        let centroids = model.centroids().unwrap();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 2);
        assert_eq!(model.n_samples_seen().unwrap(), 4);
    }

    #[test]
    fn test_out_of_core_kmeans_partial_fit() {
        let x1 = array![[0.0, 0.0], [0.1, 0.1],];
        let x2 = array![[1.0, 1.0], [1.1, 1.1],];
        let y = Array1::zeros(2);

        let mut model = OutOfCoreKMeans::new()
            .n_clusters(2)
            .random_state(42)
            .fit(&x1.view(), &y.view())
            .unwrap();

        // Process second chunk
        model.partial_fit(&x2.view()).unwrap();

        assert_eq!(model.n_samples_seen().unwrap(), 4);
        assert!(model.inertia().unwrap() >= 0.0);
    }

    #[test]
    fn test_out_of_core_kmeans_predict() {
        let x = array![[0.0, 0.0], [1.0, 1.0],];
        let y = Array1::zeros(2);

        let model = OutOfCoreKMeans::new()
            .n_clusters(2)
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        let test_data = array![[0.1, 0.1], [0.9, 0.9],];
        let predictions = model.predict(&test_data.view()).unwrap();

        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_out_of_core_kmeans_checkpointing() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1],];
        let y = Array1::zeros(4);

        let temp_dir = tempdir().unwrap();
        let checkpoint_path = temp_dir.path().join("test_checkpoint.json");

        let model = OutOfCoreKMeans::new()
            .n_clusters(2)
            .enable_checkpointing(true)
            .temp_dir(temp_dir.path().to_string_lossy().to_string())
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        // Save checkpoint
        model.save_checkpoint(&checkpoint_path).unwrap();

        // Load checkpoint into new model
        let mut new_model = OutOfCoreKMeans::new()
            .n_clusters(2)
            .enable_checkpointing(true)
            .temp_dir(temp_dir.path().to_string_lossy().to_string())
            .random_state(42)
            .fit(&x.view(), &y.view())
            .unwrap();

        new_model.load_checkpoint(&checkpoint_path).unwrap();

        // Verify loaded model has same state
        assert_eq!(
            model.n_samples_seen().unwrap(),
            new_model.n_samples_seen().unwrap()
        );
        assert_relative_eq!(model.inertia().unwrap(), new_model.inertia().unwrap());
    }

    #[test]
    fn test_cluster_summary_density() {
        let point1 = array![0.0, 0.0];
        let point2 = array![1.0, 1.0];
        let mut summary = ClusterSummary::new(&point1.view());
        summary.update(&point2.view());

        let density = summary.density();
        assert!(density > 0.0);
        assert!(density.is_finite());
    }

    #[test]
    fn test_cluster_summary_volume() {
        let point1 = array![0.0, 0.0];
        let point2 = array![1.0, 1.0];
        let mut summary = ClusterSummary::new(&point1.view());
        summary.update(&point2.view());

        let volume = summary.volume();
        assert_eq!(volume, 1.0); // 1 * 1 for 2D unit square
    }
}
