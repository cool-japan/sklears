//! MapReduce-style neighbor search for distributed computing

use crate::{Distance, NeighborsError, NeighborsResult};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::thread_rng;
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

/// Partition assignment strategy for MapReduce
#[derive(Debug, Clone, Copy)]
pub enum PartitionStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Hash-based partitioning
    Hash,
    /// Range-based partitioning (sorted by first feature)
    Range,
    /// Random partitioning
    Random,
}

/// Reducer strategy for combining results
#[derive(Debug, Clone, Copy)]
pub enum ReduceStrategy {
    /// Take k nearest neighbors globally
    Global,
    /// Take k neighbors from each partition and re-rank
    PartitionedThenMerge,
    /// Weighted combination based on partition sizes
    Weighted,
}

/// MapReduce job configuration
#[derive(Debug, Clone)]
pub struct MapReduceConfig {
    /// Number of partitions/reducers to use
    pub num_partitions: usize,
    /// Strategy for partitioning data
    pub partition_strategy: PartitionStrategy,
    /// Strategy for reducing results
    pub reduce_strategy: ReduceStrategy,
    /// Maximum number of neighbors to find per partition
    pub k_per_partition: usize,
    /// Distance metric to use
    pub distance: Distance,
}

impl Default for MapReduceConfig {
    fn default() -> Self {
        Self {
            num_partitions: 4,
            partition_strategy: PartitionStrategy::RoundRobin,
            reduce_strategy: ReduceStrategy::Global,
            k_per_partition: 10,
            distance: Distance::default(),
        }
    }
}

/// MapReduce-style distributed nearest neighbor search
#[derive(Debug, Clone)]
pub struct MapReduceNeighborSearch {
    config: MapReduceConfig,
    training_data: Option<Array2<Float>>,
    training_labels: Option<Array1<Int>>,
    partitions: Vec<(Array2<Float>, Array1<Int>)>,
}

impl MapReduceNeighborSearch {
    /// Create a new MapReduce neighbor search
    pub fn new(config: MapReduceConfig) -> Self {
        Self {
            config,
            training_data: None,
            training_labels: None,
            partitions: Vec::new(),
        }
    }

    /// Fit the model with training data
    pub fn fit(&mut self, x: &Features, y: &Array1<Int>) -> NeighborsResult<()> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            });
        }

        self.training_data = Some(x.clone());
        self.training_labels = Some(y.clone());

        // Create partitions
        self.create_partitions(x, y)?;

        Ok(())
    }

    /// Create data partitions based on the partition strategy
    fn create_partitions(&mut self, x: &Features, y: &Array1<Int>) -> NeighborsResult<()> {
        let n_samples = x.nrows();
        let partition_size =
            (n_samples + self.config.num_partitions - 1) / self.config.num_partitions;

        self.partitions.clear();

        match self.config.partition_strategy {
            PartitionStrategy::RoundRobin => {
                // Create empty partitions
                for _ in 0..self.config.num_partitions {
                    self.partitions
                        .push((Array2::zeros((0, x.ncols())), Array1::zeros(0)));
                }

                // Distribute samples round-robin
                for (i, (row, &label)) in x.axis_iter(Axis(0)).zip(y.iter()).enumerate() {
                    let partition_idx = i % self.config.num_partitions;

                    // Append to partition
                    let (ref mut part_x, ref mut part_y) = &mut self.partitions[partition_idx];

                    // Create new arrays with one more row
                    let new_x = if part_x.nrows() == 0 {
                        Array2::from_shape_vec((1, x.ncols()), row.to_vec())?
                    } else {
                        let mut new_x = Array2::zeros((part_x.nrows() + 1, x.ncols()));
                        new_x.slice_mut(s![..part_x.nrows(), ..]).assign(part_x);
                        new_x.slice_mut(s![part_x.nrows(), ..]).assign(&row);
                        new_x
                    };

                    let mut new_y = Array1::zeros(part_y.len() + 1);
                    if !part_y.is_empty() {
                        new_y.slice_mut(s![..part_y.len()]).assign(part_y);
                    }
                    new_y[part_y.len()] = label;

                    *part_x = new_x;
                    *part_y = new_y;
                }
            }
            PartitionStrategy::Hash => {
                // Create empty partitions
                for _ in 0..self.config.num_partitions {
                    self.partitions
                        .push((Array2::zeros((0, x.ncols())), Array1::zeros(0)));
                }

                // Hash-based partitioning using first feature
                for (row, &label) in x.axis_iter(Axis(0)).zip(y.iter()) {
                    let hash = (row[0] * 1000.0) as usize; // Simple hash
                    let partition_idx = hash % self.config.num_partitions;

                    let (ref mut part_x, ref mut part_y) = &mut self.partitions[partition_idx];

                    let new_x = if part_x.nrows() == 0 {
                        Array2::from_shape_vec((1, x.ncols()), row.to_vec())?
                    } else {
                        let mut new_x = Array2::zeros((part_x.nrows() + 1, x.ncols()));
                        new_x.slice_mut(s![..part_x.nrows(), ..]).assign(part_x);
                        new_x.slice_mut(s![part_x.nrows(), ..]).assign(&row);
                        new_x
                    };

                    let mut new_y = Array1::zeros(part_y.len() + 1);
                    if !part_y.is_empty() {
                        new_y.slice_mut(s![..part_y.len()]).assign(part_y);
                    }
                    new_y[part_y.len()] = label;

                    *part_x = new_x;
                    *part_y = new_y;
                }
            }
            PartitionStrategy::Range => {
                // Sort by first feature and partition
                let mut indexed_data: Vec<(usize, Float)> = x
                    .axis_iter(Axis(0))
                    .enumerate()
                    .map(|(i, row)| (i, row[0]))
                    .collect();
                indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // Create range partitions
                for part_idx in 0..self.config.num_partitions {
                    let start = part_idx * partition_size;
                    let end = std::cmp::min(start + partition_size, n_samples);

                    if start < end {
                        let indices: Vec<usize> =
                            indexed_data[start..end].iter().map(|(i, _)| *i).collect();

                        let part_x = Array2::from_shape_fn((indices.len(), x.ncols()), |(i, j)| {
                            x[[indices[i], j]]
                        });
                        let part_y = Array1::from_shape_fn(indices.len(), |i| y[indices[i]]);

                        self.partitions.push((part_x, part_y));
                    }
                }
            }
            PartitionStrategy::Random => {
                // Random shuffle then partition
                let mut rng = thread_rng();
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut rng);

                // Create random partitions
                for part_idx in 0..self.config.num_partitions {
                    let start = part_idx * partition_size;
                    let end = std::cmp::min(start + partition_size, n_samples);

                    if start < end {
                        let part_indices = &indices[start..end];

                        let part_x =
                            Array2::from_shape_fn((part_indices.len(), x.ncols()), |(i, j)| {
                                x[[part_indices[i], j]]
                            });
                        let part_y =
                            Array1::from_shape_fn(part_indices.len(), |i| y[part_indices[i]]);

                        self.partitions.push((part_x, part_y));
                    }
                }
            }
        }

        Ok(())
    }

    /// Find k-nearest neighbors using MapReduce approach
    pub fn kneighbors(
        &self,
        x: &Features,
        k: usize,
    ) -> NeighborsResult<(Array2<usize>, Array2<Float>)> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        if self.partitions.is_empty() {
            return Err(NeighborsError::InvalidInput("Model not fitted".to_string()));
        }

        let n_queries = x.nrows();
        let mut all_indices = Array2::zeros((n_queries, k));
        let mut all_distances = Array2::zeros((n_queries, k));

        // Process each query point
        for (query_idx, query_point) in x.axis_iter(Axis(0)).enumerate() {
            let (indices, distances) = self.find_neighbors_for_query(&query_point, k)?;
            all_indices.row_mut(query_idx).assign(&indices);
            all_distances.row_mut(query_idx).assign(&distances);
        }

        Ok((all_indices, all_distances))
    }

    /// Find neighbors for a single query point
    fn find_neighbors_for_query(
        &self,
        query: &ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Array1<usize>, Array1<Float>)> {
        // Map phase: Find neighbors in each partition
        #[cfg(feature = "parallel")]
        let partition_results: Vec<Vec<(Float, usize)>> = self
            .partitions
            .par_iter()
            .enumerate()
            .map(|(partition_idx, (part_x, _part_y))| {
                self.map_find_neighbors(query, part_x, partition_idx, self.config.k_per_partition)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let partition_results: Vec<Vec<(Float, usize)>> = self
            .partitions
            .iter()
            .enumerate()
            .map(|(partition_idx, (part_x, _part_y))| {
                self.map_find_neighbors(query, part_x, partition_idx, self.config.k_per_partition)
            })
            .collect();

        // Reduce phase: Combine results from all partitions
        self.reduce_neighbors(partition_results, k)
    }

    /// Map phase: Find neighbors in a single partition
    fn map_find_neighbors(
        &self,
        query: &ArrayView1<Float>,
        partition_data: &Array2<Float>,
        partition_idx: usize,
        k_partition: usize,
    ) -> Vec<(Float, usize)> {
        let mut distances: Vec<(Float, usize)> = partition_data
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(local_idx, sample)| {
                let distance = self.config.distance.calculate(query, &sample);
                // Convert local index to global index
                let global_idx = self.local_to_global_index(partition_idx, local_idx);
                (distance, global_idx)
            })
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(k_partition);
        distances
    }

    /// Reduce phase: Combine neighbor results from all partitions
    fn reduce_neighbors(
        &self,
        partition_results: Vec<Vec<(Float, usize)>>,
        k: usize,
    ) -> NeighborsResult<(Array1<usize>, Array1<Float>)> {
        match self.config.reduce_strategy {
            ReduceStrategy::Global => {
                // Combine all neighbors and take global k nearest
                let mut all_neighbors: Vec<(Float, usize)> =
                    partition_results.into_iter().flatten().collect();

                all_neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                all_neighbors.truncate(k);

                let distances = Array1::from_iter(all_neighbors.iter().map(|(d, _)| *d));
                let indices = Array1::from_iter(all_neighbors.iter().map(|(_, i)| *i));

                Ok((indices, distances))
            }
            ReduceStrategy::PartitionedThenMerge => {
                // Take best neighbors from each partition, then merge
                let mut selected_neighbors = Vec::new();

                for partition_neighbors in partition_results {
                    let take_count = std::cmp::min(
                        k / self.config.num_partitions + 1,
                        partition_neighbors.len(),
                    );
                    selected_neighbors.extend_from_slice(&partition_neighbors[..take_count]);
                }

                selected_neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                selected_neighbors.truncate(k);

                let distances = Array1::from_iter(selected_neighbors.iter().map(|(d, _)| *d));
                let indices = Array1::from_iter(selected_neighbors.iter().map(|(_, i)| *i));

                Ok((indices, distances))
            }
            ReduceStrategy::Weighted => {
                // Weight neighbors by partition size
                let mut weighted_neighbors = Vec::new();

                for (partition_idx, partition_neighbors) in
                    partition_results.into_iter().enumerate()
                {
                    let partition_weight = self.partitions[partition_idx].0.nrows() as Float
                        / self.training_data.as_ref().unwrap().nrows() as Float;

                    for (distance, idx) in partition_neighbors {
                        weighted_neighbors.push((distance * partition_weight, idx));
                    }
                }

                weighted_neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                weighted_neighbors.truncate(k);

                let distances = Array1::from_iter(weighted_neighbors.iter().map(|(d, _)| *d));
                let indices = Array1::from_iter(weighted_neighbors.iter().map(|(_, i)| *i));

                Ok((indices, distances))
            }
        }
    }

    /// Convert local partition index to global index
    fn local_to_global_index(&self, partition_idx: usize, local_idx: usize) -> usize {
        // This is a simplified mapping - in practice, you'd need to maintain
        // a mapping from local to global indices
        let mut global_idx = 0;
        for i in 0..partition_idx {
            global_idx += self.partitions[i].0.nrows();
        }
        global_idx + local_idx
    }

    /// Get statistics about the partitions
    pub fn partition_stats(&self) -> HashMap<String, Float> {
        let mut stats = HashMap::new();

        if self.partitions.is_empty() {
            return stats;
        }

        let partition_sizes: Vec<usize> = self.partitions.iter().map(|(x, _)| x.nrows()).collect();
        let total_samples: usize = partition_sizes.iter().sum();
        let mean_size = total_samples as Float / self.partitions.len() as Float;

        // Calculate variance in partition sizes
        let variance = partition_sizes
            .iter()
            .map(|&size| (size as Float - mean_size).powi(2))
            .sum::<Float>()
            / self.partitions.len() as Float;

        stats.insert("num_partitions".to_string(), self.partitions.len() as Float);
        stats.insert("total_samples".to_string(), total_samples as Float);
        stats.insert("mean_partition_size".to_string(), mean_size);
        stats.insert("partition_size_variance".to_string(), variance);
        stats.insert(
            "min_partition_size".to_string(),
            *partition_sizes.iter().min().unwrap() as Float,
        );
        stats.insert(
            "max_partition_size".to_string(),
            *partition_sizes.iter().max().unwrap() as Float,
        );

        stats
    }
}

/// Trait for distributing neighbor search across multiple machines
pub trait DistributedMapReduce {
    /// Execute map phase on distributed nodes
    fn distributed_map(
        &self,
        query: &ArrayView1<Float>,
        node_assignments: &[usize],
    ) -> Vec<Vec<(Float, usize)>>;

    /// Execute reduce phase to combine results
    fn distributed_reduce(
        &self,
        partition_results: Vec<Vec<(Float, usize)>>,
        k: usize,
    ) -> NeighborsResult<(Array1<usize>, Array1<Float>)>;
}

impl DistributedMapReduce for MapReduceNeighborSearch {
    fn distributed_map(
        &self,
        query: &ArrayView1<Float>,
        node_assignments: &[usize],
    ) -> Vec<Vec<(Float, usize)>> {
        // This would interface with a distributed computing framework
        // For now, simulate with parallel processing
        #[cfg(feature = "parallel")]
        {
            self.partitions
                .par_iter()
                .enumerate()
                .map(|(partition_idx, (part_x, _part_y))| {
                    self.map_find_neighbors(
                        query,
                        part_x,
                        partition_idx,
                        self.config.k_per_partition,
                    )
                })
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.partitions
                .iter()
                .enumerate()
                .map(|(partition_idx, (part_x, _part_y))| {
                    self.map_find_neighbors(
                        query,
                        part_x,
                        partition_idx,
                        self.config.k_per_partition,
                    )
                })
                .collect()
        }
    }

    fn distributed_reduce(
        &self,
        partition_results: Vec<Vec<(Float, usize)>>,
        k: usize,
    ) -> NeighborsResult<(Array1<usize>, Array1<Float>)> {
        self.reduce_neighbors(partition_results, k)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2, Axis};

    #[test]
    fn test_mapreduce_neighbor_search_basic() {
        let x_train = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1, 5.0, 5.0, 5.1, 5.1, 6.0, 6.0, 6.1, 6.1,
            ],
        )
        .unwrap();
        let y_train = array![0, 0, 0, 0, 1, 1, 1, 1];

        let config = MapReduceConfig {
            num_partitions: 2,
            partition_strategy: PartitionStrategy::RoundRobin,
            reduce_strategy: ReduceStrategy::Global,
            k_per_partition: 3,
            distance: Distance::default(),
        };

        let mut search = MapReduceNeighborSearch::new(config);
        search.fit(&x_train, &y_train).unwrap();

        let x_test = Array2::from_shape_vec((2, 2), vec![1.05, 1.05, 5.05, 5.05]).unwrap();
        let (indices, distances) = search.kneighbors(&x_test, 3).unwrap();

        assert_eq!(indices.nrows(), 2);
        assert_eq!(indices.ncols(), 3);
        assert_eq!(distances.nrows(), 2);
        assert_eq!(distances.ncols(), 3);

        // Verify distances are sorted
        for row in distances.axis_iter(Axis(0)) {
            for i in 1..row.len() {
                assert!(row[i - 1] <= row[i]);
            }
        }
    }

    #[test]
    fn test_partition_strategies() {
        let x_train = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        )
        .unwrap();
        let y_train = array![0, 0, 1, 1, 2, 2];

        let strategies = [
            PartitionStrategy::RoundRobin,
            PartitionStrategy::Hash,
            PartitionStrategy::Range,
            PartitionStrategy::Random,
        ];

        for strategy in &strategies {
            let config = MapReduceConfig {
                num_partitions: 2,
                partition_strategy: *strategy,
                reduce_strategy: ReduceStrategy::Global,
                k_per_partition: 2,
                distance: Distance::default(),
            };

            let mut search = MapReduceNeighborSearch::new(config);
            let result = search.fit(&x_train, &y_train);
            assert!(result.is_ok(), "Strategy {:?} failed", strategy);

            let stats = search.partition_stats();
            assert!(stats.get("num_partitions").unwrap() > &0.0);
        }
    }

    #[test]
    fn test_reduce_strategies() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y_train = array![0, 0, 1, 1];

        let strategies = [
            ReduceStrategy::Global,
            ReduceStrategy::PartitionedThenMerge,
            ReduceStrategy::Weighted,
        ];

        for strategy in &strategies {
            let config = MapReduceConfig {
                num_partitions: 2,
                partition_strategy: PartitionStrategy::RoundRobin,
                reduce_strategy: *strategy,
                k_per_partition: 2,
                distance: Distance::default(),
            };

            let mut search = MapReduceNeighborSearch::new(config);
            search.fit(&x_train, &y_train).unwrap();

            let x_test = Array2::from_shape_vec((1, 2), vec![2.5, 2.5]).unwrap();
            let result = search.kneighbors(&x_test, 2);
            assert!(result.is_ok(), "Reduce strategy {:?} failed", strategy);
        }
    }

    #[test]
    fn test_partition_stats() {
        let x_train = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        )
        .unwrap();
        let y_train = array![0, 0, 1, 1, 2, 2];

        let config = MapReduceConfig::default();
        let mut search = MapReduceNeighborSearch::new(config);
        search.fit(&x_train, &y_train).unwrap();

        let stats = search.partition_stats();

        assert!(stats.contains_key("num_partitions"));
        assert!(stats.contains_key("total_samples"));
        assert!(stats.contains_key("mean_partition_size"));
        assert_eq!(stats.get("total_samples").unwrap(), &6.0);
    }

    #[test]
    fn test_mapreduce_errors() {
        let config = MapReduceConfig::default();
        let search = MapReduceNeighborSearch::new(config);

        // Test with empty data
        let empty_x = Array2::<Float>::zeros((0, 2));
        let empty_test = Array2::<Float>::zeros((0, 2));
        let result = search.kneighbors(&empty_test, 1);
        assert!(result.is_err());

        // Test with unfitted model
        let x_test = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let result = search.kneighbors(&x_test, 1);
        assert!(result.is_err());
    }
}
