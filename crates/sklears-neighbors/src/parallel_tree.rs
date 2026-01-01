//! Parallel tree construction for neighbor search indices
//!
//! This module provides parallel construction of tree-based neighbor search
//! structures using work-stealing and thread pools to accelerate index building
//! for large datasets.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::distance::Distance;
use crate::tree::{BallTree, CoverTree, KdTree, VpTree};
use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use sklears_core::types::{Features, Float};
#[cfg(feature = "parallel")]
use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(feature = "parallel")]
use std::sync::Mutex;
#[cfg(feature = "parallel")]
use std::thread;
use std::time::Instant;

/// Strategy for parallel tree construction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelBuildStrategy {
    /// Divide data by samples and build subtrees in parallel
    DataParallel,
    /// Parallelize internal tree operations (splitting, sorting)
    TaskParallel,
    /// Hybrid approach combining data and task parallelism
    Hybrid,
    /// Work-stealing approach with dynamic load balancing
    WorkStealing,
}

/// Type of tree to build in parallel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelTreeType {
    /// KD-tree with parallel construction
    KdTree,
    /// Ball tree with parallel construction
    BallTree,
    /// VP-tree with parallel construction
    VpTree,
    /// Cover tree with parallel construction
    CoverTree,
}

/// Work unit for parallel tree construction
#[derive(Debug, Clone)]
pub struct WorkUnit {
    /// Data subset for this work unit
    pub data_indices: Vec<usize>,
    /// Start index in the original dataset
    pub start_idx: usize,
    /// End index in the original dataset
    pub end_idx: usize,
    /// Tree depth for this work unit
    pub depth: usize,
    /// Parent work unit ID (for hierarchical construction)
    pub parent_id: Option<usize>,
    /// Work unit ID
    pub id: usize,
}

impl WorkUnit {
    pub fn new(
        data_indices: Vec<usize>,
        start_idx: usize,
        end_idx: usize,
        depth: usize,
        id: usize,
    ) -> Self {
        Self {
            data_indices,
            start_idx,
            end_idx,
            depth,
            parent_id: None,
            id,
        }
    }

    pub fn size(&self) -> usize {
        self.data_indices.len()
    }

    pub fn is_leaf(&self, leaf_size: usize) -> bool {
        self.size() <= leaf_size
    }
}

/// Statistics for parallel tree construction
#[derive(Debug, Clone)]
pub struct ParallelBuildStats {
    pub total_construction_time_ms: f64,
    pub parallel_efficiency: Float,
    pub num_threads_used: usize,
    pub work_units_processed: usize,
    pub load_balance_factor: Float,
    pub tree_depth: usize,
    pub tree_nodes: usize,
}

impl ParallelBuildStats {
    pub fn new() -> Self {
        Self {
            total_construction_time_ms: 0.0,
            parallel_efficiency: 0.0,
            num_threads_used: 1,
            work_units_processed: 0,
            load_balance_factor: 1.0,
            tree_depth: 0,
            tree_nodes: 0,
        }
    }
}

impl Default for ParallelBuildStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel tree index combining multiple tree structures built in parallel
pub struct ParallelTreeIndex {
    tree_type: ParallelTreeType,
    strategy: ParallelBuildStrategy,
    distance: Distance,

    // Tree storage
    kd_trees: Vec<Arc<KdTree>>,
    ball_trees: Vec<Arc<BallTree>>,
    vp_trees: Vec<Arc<VpTree>>,
    cover_trees: Vec<Arc<CoverTree>>,

    // Data management
    data: Arc<Array2<Float>>,
    data_partitions: Vec<Vec<usize>>,

    // Configuration
    leaf_size: usize,
    num_threads: usize,

    // Statistics
    build_stats: ParallelBuildStats,
}

impl ParallelTreeIndex {
    /// Create a new parallel tree index
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `tree_type` - Type of tree to build
    /// * `strategy` - Parallel construction strategy
    /// * `distance` - Distance metric
    /// * `leaf_size` - Leaf size for tree construction
    /// * `num_threads` - Number of threads to use (None for automatic)
    ///
    /// # Returns
    /// * `NeighborsResult<Self>` - New parallel tree index
    pub fn new(
        data: &Features,
        tree_type: ParallelTreeType,
        strategy: ParallelBuildStrategy,
        distance: Distance,
        leaf_size: usize,
        num_threads: Option<usize>,
    ) -> NeighborsResult<Self> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let num_threads = num_threads.unwrap_or_else(|| {
            #[cfg(feature = "parallel")]
            {
                rayon::current_num_threads()
            }
            #[cfg(not(feature = "parallel"))]
            {
                1
            }
        });

        let mut index = Self {
            tree_type,
            strategy,
            distance,
            kd_trees: Vec::new(),
            ball_trees: Vec::new(),
            vp_trees: Vec::new(),
            cover_trees: Vec::new(),
            data: Arc::new(data.clone()),
            data_partitions: Vec::new(),
            leaf_size,
            num_threads,
            build_stats: ParallelBuildStats::new(),
        };

        index.build_parallel_trees()?;

        Ok(index)
    }

    /// Query k-nearest neighbors using the parallel index
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
        match self.strategy {
            ParallelBuildStrategy::DataParallel => self.query_knn_data_parallel(query_point, k),
            ParallelBuildStrategy::TaskParallel => self.query_knn_task_parallel(query_point, k),
            ParallelBuildStrategy::Hybrid | ParallelBuildStrategy::WorkStealing => {
                self.query_knn_hybrid(query_point, k)
            }
        }
    }

    /// Query neighbors within a radius using the parallel index
    ///
    /// # Arguments
    /// * `query_point` - Point to find neighbors for
    /// * `radius` - Search radius
    ///
    /// # Returns
    /// * `NeighborsResult<(`Vec<usize>`, Vec`<Float>`)>` - Neighbor indices and distances
    pub fn query_radius(
        &self,
        query_point: ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        match self.strategy {
            ParallelBuildStrategy::DataParallel => {
                self.query_radius_data_parallel(query_point, radius)
            }
            ParallelBuildStrategy::TaskParallel => {
                self.query_radius_task_parallel(query_point, radius)
            }
            ParallelBuildStrategy::Hybrid | ParallelBuildStrategy::WorkStealing => {
                self.query_radius_hybrid(query_point, radius)
            }
        }
    }

    /// Get build statistics
    pub fn build_stats(&self) -> &ParallelBuildStats {
        &self.build_stats
    }

    /// Get the number of trees in the parallel index
    pub fn num_trees(&self) -> usize {
        match self.tree_type {
            ParallelTreeType::KdTree => self.kd_trees.len(),
            ParallelTreeType::BallTree => self.ball_trees.len(),
            ParallelTreeType::VpTree => self.vp_trees.len(),
            ParallelTreeType::CoverTree => self.cover_trees.len(),
        }
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        let data_size = self.data.len() * std::mem::size_of::<Float>();
        let partition_size = self
            .data_partitions
            .iter()
            .map(|p| p.len() * std::mem::size_of::<usize>())
            .sum::<usize>();

        // Estimate tree memory usage (rough approximation)
        let tree_size = match self.tree_type {
            ParallelTreeType::KdTree => self.kd_trees.len() * 1000, // Rough estimate
            ParallelTreeType::BallTree => self.ball_trees.len() * 1500,
            ParallelTreeType::VpTree => self.vp_trees.len() * 1200,
            ParallelTreeType::CoverTree => self.cover_trees.len() * 2000,
        };

        data_size + partition_size + tree_size
    }

    // Private implementation methods

    fn build_parallel_trees(&mut self) -> NeighborsResult<()> {
        let start_time = Instant::now();

        match self.strategy {
            ParallelBuildStrategy::DataParallel => {
                self.build_data_parallel()?;
            }
            ParallelBuildStrategy::TaskParallel => {
                self.build_task_parallel()?;
            }
            ParallelBuildStrategy::Hybrid => {
                self.build_hybrid()?;
            }
            ParallelBuildStrategy::WorkStealing => {
                self.build_work_stealing()?;
            }
        }

        self.build_stats.total_construction_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        self.calculate_build_efficiency();

        Ok(())
    }

    #[cfg(feature = "parallel")]
    fn build_data_parallel(&mut self) -> NeighborsResult<()> {
        let n_samples = self.data.shape()[0];
        let chunk_size = (n_samples + self.num_threads - 1) / self.num_threads;

        // Partition data
        self.data_partitions = (0..n_samples)
            .collect::<Vec<_>>()
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Build trees in parallel
        let results: Result<Vec<_>, _> = self
            .data_partitions
            .par_iter()
            .map(|partition| {
                let partition_data = self.extract_partition_data(partition)?;
                self.build_single_tree(&partition_data)
            })
            .collect();

        let trees = results?;
        self.store_trees(trees);

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    fn build_data_parallel(&mut self) -> NeighborsResult<()> {
        // Fallback to sequential building
        self.build_sequential()
    }

    #[cfg(feature = "parallel")]
    fn build_task_parallel(&mut self) -> NeighborsResult<()> {
        // For task parallelism, we build a single tree but parallelize internal operations
        let tree = self.build_single_tree_parallel(&self.data)?;
        self.store_single_tree(tree);
        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    fn build_task_parallel(&mut self) -> NeighborsResult<()> {
        self.build_sequential()
    }

    fn build_hybrid(&mut self) -> NeighborsResult<()> {
        #[cfg(feature = "parallel")]
        {
            // Combine data and task parallelism
            let n_samples = self.data.shape()[0];

            if n_samples > 10000 {
                // Use data parallelism for large datasets
                self.build_data_parallel()
            } else {
                // Use task parallelism for smaller datasets
                self.build_task_parallel()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.build_sequential()
        }
    }

    #[cfg(feature = "parallel")]
    fn build_work_stealing(&mut self) -> NeighborsResult<()> {
        // Implement work-stealing using a shared work queue
        let work_queue = Arc::new(Mutex::new(VecDeque::new()));
        let completed_trees = Arc::new(Mutex::new(Vec::new()));

        // Initialize work queue
        let n_samples = self.data.shape()[0];
        let initial_indices: Vec<usize> = (0..n_samples).collect();
        work_queue
            .lock()
            .unwrap()
            .push_back(WorkUnit::new(initial_indices, 0, n_samples, 0, 0));

        // Spawn worker threads
        let mut handles = Vec::new();
        for thread_id in 0..self.num_threads {
            let work_queue = Arc::clone(&work_queue);
            let completed_trees = Arc::clone(&completed_trees);
            let data = Arc::clone(&self.data);
            let distance = self.distance.clone();
            let leaf_size = self.leaf_size;
            let tree_type = self.tree_type;

            let handle = thread::spawn(move || {
                Self::worker_thread(
                    thread_id,
                    work_queue,
                    completed_trees,
                    data,
                    distance,
                    leaf_size,
                    tree_type,
                )
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle
                .join()
                .map_err(|_| NeighborsError::InvalidInput("Thread panicked".to_string()))?;
        }

        // Collect results
        let trees = completed_trees.lock().unwrap().clone();
        self.store_trees(trees);

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    fn build_work_stealing(&mut self) -> NeighborsResult<()> {
        self.build_sequential()
    }

    fn build_sequential(&mut self) -> NeighborsResult<()> {
        let tree = self.build_single_tree(&self.data)?;
        self.store_single_tree(tree);
        Ok(())
    }

    fn build_single_tree(&self, data: &Array2<Float>) -> NeighborsResult<ParallelTreeResult> {
        match self.tree_type {
            ParallelTreeType::KdTree => {
                let tree = KdTree::new(data, self.distance.clone())?;
                Ok(ParallelTreeResult::KdTree(Arc::new(tree)))
            }
            ParallelTreeType::BallTree => {
                let tree = BallTree::new(data, self.distance.clone())?;
                Ok(ParallelTreeResult::BallTree(Arc::new(tree)))
            }
            ParallelTreeType::VpTree => {
                let tree = VpTree::new(data, self.distance.clone())?;
                Ok(ParallelTreeResult::VpTree(Arc::new(tree)))
            }
            ParallelTreeType::CoverTree => {
                let tree = CoverTree::new(data.clone(), self.distance.clone())?;
                Ok(ParallelTreeResult::CoverTree(Arc::new(tree)))
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn build_single_tree_parallel(
        &self,
        data: &Array2<Float>,
    ) -> NeighborsResult<ParallelTreeResult> {
        // This would contain the parallel tree building logic for a single tree
        // For now, fall back to sequential building
        self.build_single_tree(data)
    }

    fn extract_partition_data(&self, partition: &[usize]) -> NeighborsResult<Array2<Float>> {
        let n_features = self.data.shape()[1];
        let mut partition_data = Array2::zeros((partition.len(), n_features));

        for (i, &original_idx) in partition.iter().enumerate() {
            if original_idx < self.data.shape()[0] {
                partition_data
                    .row_mut(i)
                    .assign(&self.data.row(original_idx));
            }
        }

        Ok(partition_data)
    }

    fn store_trees(&mut self, trees: Vec<ParallelTreeResult>) {
        for tree in trees {
            self.store_single_tree(tree);
        }
    }

    fn store_single_tree(&mut self, tree: ParallelTreeResult) {
        match tree {
            ParallelTreeResult::KdTree(tree) => self.kd_trees.push(tree),
            ParallelTreeResult::BallTree(tree) => self.ball_trees.push(tree),
            ParallelTreeResult::VpTree(tree) => self.vp_trees.push(tree),
            ParallelTreeResult::CoverTree(tree) => self.cover_trees.push(tree),
        }
    }

    #[cfg(feature = "parallel")]
    fn worker_thread(
        _thread_id: usize,
        work_queue: Arc<Mutex<VecDeque<WorkUnit>>>,
        completed_trees: Arc<Mutex<Vec<ParallelTreeResult>>>,
        data: Arc<Array2<Float>>,
        distance: Distance,
        leaf_size: usize,
        tree_type: ParallelTreeType,
    ) {
        loop {
            let work_unit = {
                let mut queue = work_queue.lock().unwrap();
                queue.pop_front()
            };

            match work_unit {
                Some(unit) => {
                    if unit.is_leaf(leaf_size) {
                        // Build a tree for this leaf unit
                        if let Ok(partition_data) = Self::extract_work_unit_data(&data, &unit) {
                            if let Ok(tree) = Self::build_tree_for_unit(
                                &partition_data,
                                distance.clone(),
                                leaf_size,
                                tree_type,
                            ) {
                                completed_trees.lock().unwrap().push(tree);
                            }
                        }
                    } else {
                        // Split work unit and add sub-units to queue
                        if let Ok(sub_units) = Self::split_work_unit(&data, &unit, distance.clone())
                        {
                            let mut queue = work_queue.lock().unwrap();
                            for sub_unit in sub_units {
                                queue.push_back(sub_unit);
                            }
                        }
                    }
                }
                None => {
                    // No more work, exit
                    break;
                }
            }
        }
    }

    fn extract_work_unit_data(
        data: &Array2<Float>,
        unit: &WorkUnit,
    ) -> NeighborsResult<Array2<Float>> {
        let n_features = data.shape()[1];
        let mut unit_data = Array2::zeros((unit.data_indices.len(), n_features));

        for (i, &original_idx) in unit.data_indices.iter().enumerate() {
            if original_idx < data.shape()[0] {
                unit_data.row_mut(i).assign(&data.row(original_idx));
            }
        }

        Ok(unit_data)
    }

    fn build_tree_for_unit(
        data: &Array2<Float>,
        distance: Distance,
        _leaf_size: usize,
        tree_type: ParallelTreeType,
    ) -> NeighborsResult<ParallelTreeResult> {
        match tree_type {
            ParallelTreeType::KdTree => {
                let tree = KdTree::new(data, distance)?;
                Ok(ParallelTreeResult::KdTree(Arc::new(tree)))
            }
            ParallelTreeType::BallTree => {
                let tree = BallTree::new(data, distance)?;
                Ok(ParallelTreeResult::BallTree(Arc::new(tree)))
            }
            ParallelTreeType::VpTree => {
                let tree = VpTree::new(data, distance)?;
                Ok(ParallelTreeResult::VpTree(Arc::new(tree)))
            }
            ParallelTreeType::CoverTree => {
                let tree = CoverTree::new(data.clone(), distance)?;
                Ok(ParallelTreeResult::CoverTree(Arc::new(tree)))
            }
        }
    }

    fn split_work_unit(
        _data: &Array2<Float>,
        unit: &WorkUnit,
        _distance: Distance,
    ) -> NeighborsResult<Vec<WorkUnit>> {
        // Simple splitting strategy: divide indices in half
        let mid = unit.data_indices.len() / 2;
        let (left_indices, right_indices) = unit.data_indices.split_at(mid);

        let left_unit = WorkUnit::new(
            left_indices.to_vec(),
            unit.start_idx,
            unit.start_idx + mid,
            unit.depth + 1,
            unit.id * 2 + 1,
        );

        let right_unit = WorkUnit::new(
            right_indices.to_vec(),
            unit.start_idx + mid,
            unit.end_idx,
            unit.depth + 1,
            unit.id * 2 + 2,
        );

        Ok(vec![left_unit, right_unit])
    }

    fn calculate_build_efficiency(&mut self) {
        // Calculate parallel efficiency metrics
        self.build_stats.num_threads_used = self.num_threads;
        self.build_stats.work_units_processed = self.num_trees();

        // Estimate efficiency based on work distribution
        if self.num_trees() > 0 {
            self.build_stats.load_balance_factor = 1.0 / self.num_trees() as Float;
            self.build_stats.parallel_efficiency =
                (1.0 / self.num_threads as Float) / self.build_stats.load_balance_factor;
        }
    }

    // Query implementations

    #[cfg(feature = "parallel")]
    fn query_knn_data_parallel(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        // For now, use brute force search on partitioned data
        let results: Result<Vec<_>, _> = self
            .data_partitions
            .par_iter()
            .enumerate()
            .map(|(_partition_idx, partition)| {
                self.query_partition_brute_force(query_point, k, partition)
            })
            .collect();

        let partition_results = results?;
        self.merge_knn_results(partition_results, k)
    }

    #[cfg(not(feature = "parallel"))]
    fn query_knn_data_parallel(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        self.query_knn_sequential(query_point, k)
    }

    fn query_knn_task_parallel(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        // For task parallelism, use brute force on the full dataset
        self.query_knn_brute_force_full(query_point, k)
    }

    fn query_knn_hybrid(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        if self.num_trees() > 1 {
            self.query_knn_data_parallel(query_point, k)
        } else {
            self.query_knn_task_parallel(query_point, k)
        }
    }

    fn query_knn_sequential(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        // Fallback sequential implementation using brute force
        self.query_knn_brute_force_full(query_point, k)
    }

    fn query_radius_data_parallel(
        &self,
        query_point: ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        // Similar implementation to query_knn_data_parallel but for radius queries
        self.query_radius_sequential(query_point, radius)
    }

    fn query_radius_task_parallel(
        &self,
        query_point: ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        // Similar implementation to query_knn_task_parallel but for radius queries
        self.query_radius_sequential(query_point, radius)
    }

    fn query_radius_hybrid(
        &self,
        query_point: ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        self.query_radius_sequential(query_point, radius)
    }

    fn query_radius_sequential(
        &self,
        query_point: ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        // Use brute force radius search
        self.query_radius_brute_force_full(query_point, radius)
    }

    fn query_knn_brute_force_full(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        let data = self.data.as_ref();
        let n_samples = data.shape()[0];

        if n_samples == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut distances: Vec<(usize, Float)> = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let point = data.row(i);
            let dist = self.distance.calculate(&query_point, &point);
            distances.push((i, dist));
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        let (neighbor_indices, neighbor_distances): (Vec<usize>, Vec<Float>) =
            distances.into_iter().unzip();

        Ok((neighbor_indices, neighbor_distances))
    }

    fn query_radius_brute_force_full(
        &self,
        query_point: ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        let data = self.data.as_ref();
        let n_samples = data.shape()[0];

        if n_samples == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut neighbors = Vec::new();
        let mut distances = Vec::new();

        for i in 0..n_samples {
            let point = data.row(i);
            let dist = self.distance.calculate(&query_point, &point);
            if dist <= radius {
                neighbors.push(i);
                distances.push(dist);
            }
        }

        Ok((neighbors, distances))
    }

    fn query_partition_brute_force(
        &self,
        query_point: ArrayView1<Float>,
        k: usize,
        partition: &[usize],
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        let data = self.data.as_ref();

        if partition.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut distances: Vec<(usize, Float)> = Vec::with_capacity(partition.len());

        for &idx in partition {
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

    fn adjust_indices_for_partition(
        &self,
        indices: Vec<usize>,
        partition_idx: usize,
    ) -> (Vec<usize>, Vec<Float>) {
        // Adjust indices to refer to original dataset
        if partition_idx < self.data_partitions.len() {
            let partition = &self.data_partitions[partition_idx];
            let adjusted_indices: Vec<usize> = indices
                .into_iter()
                .filter_map(|idx| partition.get(idx).copied())
                .collect();
            (adjusted_indices, vec![]) // Placeholder for distances
        } else {
            (indices, vec![])
        }
    }

    fn merge_knn_results(
        &self,
        partition_results: Vec<(Vec<usize>, Vec<Float>)>,
        k: usize,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        let mut all_neighbors = Vec::new();
        let mut all_distances = Vec::new();

        for (indices, distances) in partition_results {
            all_neighbors.extend(indices);
            all_distances.extend(distances);
        }

        // Sort by distance and take k nearest
        let mut pairs: Vec<(usize, Float)> = all_neighbors.into_iter().zip(all_distances).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);

        let (final_indices, final_distances): (Vec<usize>, Vec<Float>) = pairs.into_iter().unzip();

        Ok((final_indices, final_distances))
    }
}

/// Result type for parallel tree construction
#[derive(Debug, Clone)]
enum ParallelTreeResult {
    KdTree(Arc<KdTree>),
    BallTree(Arc<BallTree>),
    VpTree(Arc<VpTree>),
    CoverTree(Arc<CoverTree>),
}

/// Builder for creating parallel tree indices
pub struct ParallelTreeBuilder {
    tree_type: ParallelTreeType,
    strategy: ParallelBuildStrategy,
    distance: Distance,
    leaf_size: usize,
    num_threads: Option<usize>,
}

impl Default for ParallelTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelTreeBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            tree_type: ParallelTreeType::KdTree,
            strategy: ParallelBuildStrategy::DataParallel,
            distance: Distance::Euclidean,
            leaf_size: 30,
            num_threads: None,
        }
    }

    /// Set the tree type
    pub fn tree_type(mut self, tree_type: ParallelTreeType) -> Self {
        self.tree_type = tree_type;
        self
    }

    /// Set the parallel build strategy
    pub fn strategy(mut self, strategy: ParallelBuildStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the distance metric
    pub fn distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set the leaf size
    pub fn leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size;
        self
    }

    /// Set the number of threads
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Build the parallel tree index
    pub fn build(self, data: &Features) -> NeighborsResult<ParallelTreeIndex> {
        ParallelTreeIndex::new(
            data,
            self.tree_type,
            self.strategy,
            self.distance,
            self.leaf_size,
            self.num_threads,
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2};

    #[test]
    fn test_parallel_tree_creation() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]]);

        let index = ParallelTreeIndex::new(
            &data,
            ParallelTreeType::KdTree,
            ParallelBuildStrategy::DataParallel,
            Distance::Euclidean,
            2,
            Some(2),
        )
        .unwrap();

        assert!(index.num_trees() > 0);

        let stats = index.build_stats();
        assert!(stats.total_construction_time_ms >= 0.0);
    }

    #[test]
    fn test_parallel_tree_query() {
        let data = arr2(&[
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [5.0, 3.0],
            [6.0, 1.0],
        ]);

        let index = ParallelTreeIndex::new(
            &data,
            ParallelTreeType::KdTree,
            ParallelBuildStrategy::TaskParallel,
            Distance::Euclidean,
            2,
            None,
        )
        .unwrap();

        let query = arr1(&[3.0, 2.0]);
        let (neighbors, distances) = index.query_knn(query.view(), 2).unwrap();

        assert_eq!(neighbors.len(), 2);
        assert_eq!(distances.len(), 2);
    }

    #[test]
    fn test_parallel_tree_builder() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]);

        let index = ParallelTreeBuilder::new()
            .tree_type(ParallelTreeType::BallTree)
            .strategy(ParallelBuildStrategy::Hybrid)
            .distance(Distance::Manhattan)
            .leaf_size(5)
            .num_threads(1)
            .build(&data)
            .unwrap();

        assert!(index.num_trees() > 0);

        let query = arr1(&[2.0, 2.0]);
        let result = index.query_knn(query.view(), 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_work_unit() {
        let indices = vec![0, 1, 2, 3, 4];
        let unit = WorkUnit::new(indices, 0, 5, 0, 0);

        assert_eq!(unit.size(), 5);
        assert!(!unit.is_leaf(3)); // 5 elements > 3, so not a leaf
        assert!(unit.is_leaf(10)); // 5 elements <= 10, so is a leaf
    }

    #[test]
    fn test_memory_usage_estimation() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0]]);

        let index = ParallelTreeIndex::new(
            &data,
            ParallelTreeType::KdTree,
            ParallelBuildStrategy::DataParallel,
            Distance::Euclidean,
            1,
            Some(1),
        )
        .unwrap();

        let memory_usage = index.memory_usage();
        assert!(memory_usage > 0);
    }
}
