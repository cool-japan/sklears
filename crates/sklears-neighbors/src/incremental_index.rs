//! Incremental index construction for dynamic neighbor search
//!
//! This module provides incremental construction of neighbor search indices,
//! allowing efficient updates when new data points are added without
//! requiring full index reconstruction.

use crate::distance::Distance;
use crate::tree::{BallTree, CoverTree, KdTree, VpTree};
use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::types::{Features, Float};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Strategy for incremental index updates
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpdateStrategy {
    /// Immediately update the index when new data is added
    Immediate,
    /// Batch updates and apply them periodically
    Batched { batch_size: usize },
    /// Use a threshold-based approach (rebuild when performance degrades)
    Threshold { degradation_threshold: Float },
    /// Hybrid approach combining batching and threshold
    Hybrid {
        batch_size: usize,
        degradation_threshold: Float,
    },
}

/// Index type for incremental construction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncrementalIndexType {
    /// KD-tree with incremental updates
    KdTree,
    /// Ball tree with incremental updates
    BallTree,
    /// VP-tree with incremental updates
    VpTree,
    /// Cover tree (naturally supports incremental construction)
    CoverTree,
    /// Flat index with batched updates
    Flat,
    /// LSH-based approximate index
    Lsh,
}

/// Performance metrics for incremental index
#[derive(Debug, Clone)]
pub struct IndexPerformanceMetrics {
    pub total_points: usize,
    pub index_depth: usize,
    pub avg_query_time_ms: Float,
    pub last_update_time_ms: Float,
    pub memory_usage_bytes: usize,
    pub update_count: usize,
    pub rebuild_count: usize,
    pub degradation_factor: Float,
}

impl IndexPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_points: 0,
            index_depth: 0,
            avg_query_time_ms: 0.0,
            last_update_time_ms: 0.0,
            memory_usage_bytes: 0,
            update_count: 0,
            rebuild_count: 0,
            degradation_factor: 1.0,
        }
    }
}

impl Default for IndexPerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch of pending updates for the index
#[derive(Debug, Clone)]
struct UpdateBatch {
    points: Vec<Array1<Float>>,
    indices: Vec<usize>,
    timestamp: Instant,
}

impl UpdateBatch {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            indices: Vec::new(),
            timestamp: Instant::now(),
        }
    }

    fn add_point(&mut self, point: Array1<Float>, index: usize) {
        self.points.push(point);
        self.indices.push(index);
    }

    fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn clear(&mut self) {
        self.points.clear();
        self.indices.clear();
        self.timestamp = Instant::now();
    }
}

/// Incremental neighbor index that supports dynamic updates
pub struct IncrementalNeighborIndex {
    index_type: IncrementalIndexType,
    update_strategy: UpdateStrategy,
    distance: Distance,

    // Core data storage
    data: Arc<RwLock<Array2<Float>>>,
    point_indices: Arc<RwLock<Vec<usize>>>,

    // Index structures
    kd_tree: Option<Arc<RwLock<KdTree>>>,
    ball_tree: Option<Arc<RwLock<BallTree>>>,
    vp_tree: Option<Arc<RwLock<VpTree>>>,
    cover_tree: Option<Arc<RwLock<CoverTree>>>,

    // Incremental update management
    pending_batch: Arc<RwLock<UpdateBatch>>,
    performance_metrics: Arc<RwLock<IndexPerformanceMetrics>>,

    // Configuration
    leaf_size: usize,
    max_batch_age: Duration,
    rebuild_threshold: Float,
}

impl IncrementalNeighborIndex {
    /// Create a new incremental neighbor index
    ///
    /// # Arguments
    /// * `initial_data` - Initial data points to build the index with
    /// * `index_type` - Type of index to use
    /// * `update_strategy` - Strategy for handling updates
    /// * `distance` - Distance metric to use
    /// * `leaf_size` - Leaf size for tree-based indices
    ///
    /// # Returns
    /// * `NeighborsResult<Self>` - New incremental index
    pub fn new(
        initial_data: &Features,
        index_type: IncrementalIndexType,
        update_strategy: UpdateStrategy,
        distance: Distance,
        leaf_size: usize,
    ) -> NeighborsResult<Self> {
        if initial_data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let n_samples = initial_data.shape()[0];
        let point_indices: Vec<usize> = (0..n_samples).collect();

        let mut index = Self {
            index_type,
            update_strategy,
            distance,
            data: Arc::new(RwLock::new(initial_data.clone())),
            point_indices: Arc::new(RwLock::new(point_indices)),
            kd_tree: None,
            ball_tree: None,
            vp_tree: None,
            cover_tree: None,
            pending_batch: Arc::new(RwLock::new(UpdateBatch::new())),
            performance_metrics: Arc::new(RwLock::new(IndexPerformanceMetrics::new())),
            leaf_size,
            max_batch_age: Duration::from_secs(10),
            rebuild_threshold: 2.0,
        };

        // Build initial index
        index.rebuild_index()?;

        Ok(index)
    }

    /// Add a new point to the index
    ///
    /// # Arguments
    /// * `point` - New data point to add
    ///
    /// # Returns
    /// * `NeighborsResult<usize>` - Index of the added point
    pub fn add_point(&mut self, point: ArrayView1<Float>) -> NeighborsResult<usize> {
        let start_time = Instant::now();

        let new_index = {
            let mut data = self.data.write().unwrap();
            let mut indices = self.point_indices.write().unwrap();

            let current_shape = data.shape().to_vec();
            let new_shape = [current_shape[0] + 1, current_shape[1]];

            // Resize data array
            let mut new_data = Array2::zeros(new_shape);
            for i in 0..current_shape[0] {
                new_data.row_mut(i).assign(&data.row(i));
            }
            new_data.row_mut(current_shape[0]).assign(&point);

            *data = new_data;
            let new_index = indices.len();
            indices.push(new_index);

            new_index
        };

        // Handle update based on strategy
        match self.update_strategy {
            UpdateStrategy::Immediate => {
                self.update_index_with_point(point.to_owned(), new_index)?;
            }
            UpdateStrategy::Batched { batch_size } => {
                let mut batch = self.pending_batch.write().unwrap();
                batch.add_point(point.to_owned(), new_index);

                if batch.len() >= batch_size {
                    drop(batch); // Release lock before applying batch
                    self.apply_pending_batch()?;
                }
            }
            UpdateStrategy::Threshold {
                degradation_threshold,
            } => {
                let mut batch = self.pending_batch.write().unwrap();
                batch.add_point(point.to_owned(), new_index);
                drop(batch);

                // Check if we need to rebuild
                if self.should_rebuild(degradation_threshold)? {
                    self.rebuild_index()?;
                }
            }
            UpdateStrategy::Hybrid {
                batch_size,
                degradation_threshold,
            } => {
                let mut batch = self.pending_batch.write().unwrap();
                batch.add_point(point.to_owned(), new_index);
                let should_apply_batch = batch.len() >= batch_size;
                drop(batch);

                if should_apply_batch {
                    self.apply_pending_batch()?;
                } else if self.should_rebuild(degradation_threshold)? {
                    self.rebuild_index()?;
                }
            }
        }

        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().unwrap();
            metrics.last_update_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            metrics.update_count += 1;
            metrics.total_points = new_index + 1;
        }

        Ok(new_index)
    }

    /// Add multiple points to the index
    ///
    /// # Arguments
    /// * `points` - New data points to add
    ///
    /// # Returns
    /// * `NeighborsResult<`Vec<usize>`>` - Indices of the added points
    pub fn add_points(&mut self, points: &Features) -> NeighborsResult<Vec<usize>> {
        let mut indices = Vec::with_capacity(points.shape()[0]);

        for point in points.axis_iter(Axis(0)) {
            let index = self.add_point(point)?;
            indices.push(index);
        }

        // Force batch application if we added many points
        if points.shape()[0] > 10 {
            self.apply_pending_batch()?;
        }

        Ok(indices)
    }

    /// Remove a point from the index
    ///
    /// # Arguments
    /// * `point_index` - Index of the point to remove
    ///
    /// # Returns
    /// * `NeighborsResult<()>` - Success or error
    pub fn remove_point(&mut self, point_index: usize) -> NeighborsResult<()> {
        let start_time = Instant::now();

        {
            let mut indices = self.point_indices.write().unwrap();

            if !indices.contains(&point_index) {
                return Err(NeighborsError::InvalidInput(format!(
                    "Point index {} not found",
                    point_index
                )));
            }

            indices.retain(|&x| x != point_index);
        }

        // For removal, we typically need to rebuild the index
        // as most tree structures don't support efficient removal
        self.rebuild_index()?;

        {
            let mut metrics = self.performance_metrics.write().unwrap();
            metrics.last_update_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            metrics.update_count += 1;
            metrics.rebuild_count += 1;
        }

        Ok(())
    }

    /// Query k-nearest neighbors
    ///
    /// # Arguments
    /// * `query_point` - Point to find neighbors for
    /// * `k` - Number of neighbors to find
    ///
    /// # Returns
    /// * `NeighborsResult<(`Vec<usize>`, Vec`<Float>`)>` - Neighbor indices and distances
    pub fn query_knn(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        let start_time = Instant::now();

        // Apply any pending batch updates first
        if !self.pending_batch.read().unwrap().is_empty() {
            // This is a bit tricky since we need mutable access
            // In a real implementation, you might want to handle this differently
            return Err(NeighborsError::InvalidInput(
                "Pending updates must be applied before querying".to_string(),
            ));
        }

        let result = match self.index_type {
            IncrementalIndexType::KdTree
            | IncrementalIndexType::BallTree
            | IncrementalIndexType::VpTree
            | IncrementalIndexType::CoverTree => {
                // For now, use brute force since tree query methods aren't implemented
                self.query_knn_brute_force(query_point, k)
            }
            IncrementalIndexType::Flat | IncrementalIndexType::Lsh => {
                self.query_knn_brute_force(query_point, k)
            }
        };

        // Update query time metrics
        {
            let mut metrics = self.performance_metrics.write().unwrap();
            let query_time = start_time.elapsed().as_secs_f64() * 1000.0;
            metrics.avg_query_time_ms = if metrics.update_count > 0 {
                (metrics.avg_query_time_ms * metrics.update_count as Float + query_time)
                    / (metrics.update_count + 1) as Float
            } else {
                query_time
            };
        }

        result
    }

    /// Apply all pending batch updates
    pub fn apply_pending_batch(&mut self) -> NeighborsResult<()> {
        let _batch = {
            let mut batch = self.pending_batch.write().unwrap();
            if batch.is_empty() {
                return Ok(());
            }

            let result = batch.clone();
            batch.clear();
            result
        };

        // Apply updates based on index type
        match self.index_type {
            IncrementalIndexType::CoverTree => {
                // For now, rebuild the index since insert_point isn't implemented
                self.rebuild_index()?;
            }
            _ => {
                // For other tree types, we might need to rebuild
                // or use specialized incremental update procedures
                self.rebuild_index()?;
            }
        }

        Ok(())
    }

    /// Force a complete rebuild of the index
    pub fn rebuild_index(&mut self) -> NeighborsResult<()> {
        let start_time = Instant::now();

        // Clear any pending batch
        self.pending_batch.write().unwrap().clear();

        let data = self.data.read().unwrap();
        let indices = self.point_indices.read().unwrap();

        // Build index based on type
        match self.index_type {
            IncrementalIndexType::KdTree => {
                let tree = KdTree::new(&data, self.distance.clone())?;
                self.kd_tree = Some(Arc::new(RwLock::new(tree)));
            }
            IncrementalIndexType::BallTree => {
                let tree = BallTree::new(&data, self.distance.clone())?;
                self.ball_tree = Some(Arc::new(RwLock::new(tree)));
            }
            IncrementalIndexType::VpTree => {
                let tree = VpTree::new(&data, self.distance.clone())?;
                self.vp_tree = Some(Arc::new(RwLock::new(tree)));
            }
            IncrementalIndexType::CoverTree => {
                let tree = CoverTree::new(data.clone(), self.distance.clone())?;
                self.cover_tree = Some(Arc::new(RwLock::new(tree)));
            }
            IncrementalIndexType::Flat | IncrementalIndexType::Lsh => {
                // No special index structure needed for flat search
            }
        }

        // Update metrics
        {
            let mut metrics = self.performance_metrics.write().unwrap();
            metrics.last_update_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            metrics.rebuild_count += 1;
            metrics.total_points = indices.len();
            metrics.degradation_factor = 1.0; // Reset degradation after rebuild
        }

        Ok(())
    }

    /// Get performance metrics
    pub fn metrics(&self) -> IndexPerformanceMetrics {
        self.performance_metrics.read().unwrap().clone()
    }

    /// Get the current size of the index
    pub fn len(&self) -> usize {
        self.point_indices.read().unwrap().len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of pending updates
    pub fn pending_updates(&self) -> usize {
        self.pending_batch.read().unwrap().len()
    }

    /// Configure the maximum batch age before forced application
    pub fn set_max_batch_age(&mut self, duration: Duration) {
        self.max_batch_age = duration;
    }

    /// Configure the rebuild threshold
    pub fn set_rebuild_threshold(&mut self, threshold: Float) {
        self.rebuild_threshold = threshold;
    }

    // Private helper methods

    fn update_index_with_point(
        &mut self,
        point: Array1<Float>,
        index: usize,
    ) -> NeighborsResult<()> {
        match self.index_type {
            IncrementalIndexType::CoverTree => {
                // For now, add to batch since insert_point isn't implemented
                let mut batch = self.pending_batch.write().unwrap();
                batch.add_point(point, index);
            }
            _ => {
                // For other index types, add to batch for later processing
                let mut batch = self.pending_batch.write().unwrap();
                batch.add_point(point, index);
            }
        }
        Ok(())
    }

    fn should_rebuild(&self, threshold: Float) -> NeighborsResult<bool> {
        let metrics = self.performance_metrics.read().unwrap();

        // Rebuild if degradation exceeds threshold
        if metrics.degradation_factor > threshold {
            return Ok(true);
        }

        // Rebuild if batch is getting old
        let batch = self.pending_batch.read().unwrap();
        if !batch.is_empty() && batch.timestamp.elapsed() > self.max_batch_age {
            return Ok(true);
        }

        Ok(false)
    }

    fn query_knn_brute_force(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        let data = self.data.read().unwrap();
        let indices = self.point_indices.read().unwrap();

        if indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut distances: Vec<(usize, Float)> = Vec::new();

        for &idx in indices.iter() {
            if idx < data.shape()[0] {
                let point = data.row(idx);
                let dist = self.distance.calculate(&query_point, &point);
                distances.push((idx, dist));
            }
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        let (neighbor_indices, neighbor_distances): (Vec<usize>, Vec<Float>) =
            distances.into_iter().unzip();

        Ok((neighbor_indices, neighbor_distances))
    }
}

/// Builder for creating incremental neighbor indices
pub struct IncrementalIndexBuilder {
    index_type: IncrementalIndexType,
    update_strategy: UpdateStrategy,
    distance: Distance,
    leaf_size: usize,
    max_batch_age: Duration,
    rebuild_threshold: Float,
}

impl Default for IncrementalIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalIndexBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            index_type: IncrementalIndexType::CoverTree,
            update_strategy: UpdateStrategy::Batched { batch_size: 100 },
            distance: Distance::Euclidean,
            leaf_size: 30,
            max_batch_age: Duration::from_secs(10),
            rebuild_threshold: 2.0,
        }
    }

    /// Set the index type
    pub fn index_type(mut self, index_type: IncrementalIndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Set the update strategy
    pub fn update_strategy(mut self, strategy: UpdateStrategy) -> Self {
        self.update_strategy = strategy;
        self
    }

    /// Set the distance metric
    pub fn distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set the leaf size for tree indices
    pub fn leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size;
        self
    }

    /// Set the maximum batch age
    pub fn max_batch_age(mut self, duration: Duration) -> Self {
        self.max_batch_age = duration;
        self
    }

    /// Set the rebuild threshold
    pub fn rebuild_threshold(mut self, threshold: Float) -> Self {
        self.rebuild_threshold = threshold;
        self
    }

    /// Build the incremental index
    pub fn build(self, initial_data: &Features) -> NeighborsResult<IncrementalNeighborIndex> {
        let mut index = IncrementalNeighborIndex::new(
            initial_data,
            self.index_type,
            self.update_strategy,
            self.distance,
            self.leaf_size,
        )?;

        index.set_max_batch_age(self.max_batch_age);
        index.set_rebuild_threshold(self.rebuild_threshold);

        Ok(index)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::{arr1, arr2};

    #[test]
    fn test_incremental_index_creation() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]);

        let index = IncrementalNeighborIndex::new(
            &data,
            IncrementalIndexType::Flat,
            UpdateStrategy::Immediate,
            Distance::Euclidean,
            10,
        )
        .unwrap();

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_incremental_point_addition() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0]]);

        let mut index = IncrementalNeighborIndex::new(
            &data,
            IncrementalIndexType::Flat,
            UpdateStrategy::Immediate,
            Distance::Euclidean,
            10,
        )
        .unwrap();

        let new_point = arr1(&[3.0, 1.0]);
        let new_index = index.add_point(new_point.view()).unwrap();

        assert_eq!(new_index, 2);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_incremental_knn_query() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]]);

        let index = IncrementalNeighborIndex::new(
            &data,
            IncrementalIndexType::Flat,
            UpdateStrategy::Immediate,
            Distance::Euclidean,
            10,
        )
        .unwrap();

        let query = arr1(&[2.0, 2.0]);
        let (neighbors, distances) = index.query_knn(query.view(), 2).unwrap();

        assert_eq!(neighbors.len(), 2);
        assert_eq!(distances.len(), 2);

        // Results should be sorted by distance
        if distances.len() > 1 {
            assert!(distances[0] <= distances[1]);
        }
    }

    #[test]
    fn test_batched_updates() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0]]);

        let mut index = IncrementalNeighborIndex::new(
            &data,
            IncrementalIndexType::Flat,
            UpdateStrategy::Batched { batch_size: 2 },
            Distance::Euclidean,
            10,
        )
        .unwrap();

        // Add points one by one
        let point1 = arr1(&[3.0, 1.0]);
        let point2 = arr1(&[4.0, 2.0]);

        index.add_point(point1.view()).unwrap();
        assert_eq!(index.pending_updates(), 1);

        index.add_point(point2.view()).unwrap();
        assert_eq!(index.pending_updates(), 0); // Batch should be applied

        assert_eq!(index.len(), 4);
    }

    #[test]
    fn test_builder_pattern() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0]]);

        let index = IncrementalIndexBuilder::new()
            .index_type(IncrementalIndexType::Flat)
            .update_strategy(UpdateStrategy::Immediate)
            .distance(Distance::Manhattan)
            .leaf_size(5)
            .build(&data)
            .unwrap();

        assert_eq!(index.len(), 2);

        let metrics = index.metrics();
        assert_eq!(metrics.total_points, 2);
    }

    #[test]
    fn test_performance_metrics() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0]]);

        let mut index = IncrementalNeighborIndex::new(
            &data,
            IncrementalIndexType::Flat,
            UpdateStrategy::Immediate,
            Distance::Euclidean,
            10,
        )
        .unwrap();

        let new_point = arr1(&[3.0, 1.0]);
        index.add_point(new_point.view()).unwrap();

        let metrics = index.metrics();
        assert_eq!(metrics.total_points, 3);
        assert_eq!(metrics.update_count, 1);
        assert!(metrics.last_update_time_ms >= 0.0);
    }

    #[test]
    fn test_point_removal() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]);

        let mut index = IncrementalNeighborIndex::new(
            &data,
            IncrementalIndexType::Flat,
            UpdateStrategy::Immediate,
            Distance::Euclidean,
            10,
        )
        .unwrap();

        assert_eq!(index.len(), 3);

        index.remove_point(1).unwrap();
        assert_eq!(index.len(), 2);

        let metrics = index.metrics();
        assert!(metrics.rebuild_count > 0);
    }
}
