use crate::distance::Distance;
use crate::{NeighborsError, NeighborsResult};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DistributedConfiguration {
    pub num_partitions: usize,
    pub replication_factor: usize,
    pub load_balance_strategy: LoadBalanceStrategy,
    pub fault_tolerance: bool,
    pub max_retries: usize,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub enum LoadBalanceStrategy {
    RoundRobin,
    Random,
    FeatureHashBased,
    DataAware,
}

#[derive(Debug, Clone)]
pub struct PartitionInfo {
    pub partition_id: usize,
    pub data_range: (usize, usize),
    pub node_id: String,
    pub is_primary: bool,
    pub replicas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DistributedNeighborSearchResult {
    pub neighbors: Vec<Vec<usize>>,
    pub distances: Vec<Vec<Float>>,
    pub partition_stats: Vec<PartitionStats>,
    pub total_time_ms: u128,
    pub network_time_ms: u128,
}

#[derive(Debug, Clone)]
pub struct PartitionStats {
    pub partition_id: usize,
    pub search_time_ms: u128,
    pub data_size: usize,
    pub num_queries: usize,
    pub cache_hit_rate: Float,
}

pub trait DistributedWorker: Send + Sync {
    fn search_partition(
        &self,
        partition_id: usize,
        query_data: ArrayView2<Float>,
        k: usize,
        distance: &Distance,
    ) -> NeighborsResult<(Vec<Vec<usize>>, Vec<Vec<Float>>)>;

    fn get_partition_info(&self, partition_id: usize) -> NeighborsResult<PartitionInfo>;
    fn health_check(&self) -> bool;
    fn get_load_metrics(&self) -> HashMap<String, Float>;
}

pub struct DistributedNeighborSearch {
    config: DistributedConfiguration,
    partitions: Vec<PartitionInfo>,
    workers: HashMap<String, Arc<dyn DistributedWorker>>,
    data_partitioner: DataPartitioner,
}

pub struct DataPartitioner {
    strategy: LoadBalanceStrategy,
    num_partitions: usize,
    feature_hash_weights: Option<Array1<Float>>,
}

impl Default for DistributedConfiguration {
    fn default() -> Self {
        Self {
            num_partitions: 4,
            replication_factor: 1,
            load_balance_strategy: LoadBalanceStrategy::RoundRobin,
            fault_tolerance: true,
            max_retries: 3,
            timeout_ms: 5000,
        }
    }
}

impl DistributedNeighborSearch {
    pub fn new(config: DistributedConfiguration) -> Self {
        let data_partitioner =
            DataPartitioner::new(config.load_balance_strategy.clone(), config.num_partitions);

        Self {
            config,
            partitions: Vec::new(),
            workers: HashMap::new(),
            data_partitioner,
        }
    }

    pub fn builder() -> DistributedNeighborSearchBuilder {
        DistributedNeighborSearchBuilder::new()
    }

    pub fn register_worker(&mut self, node_id: String, worker: Arc<dyn DistributedWorker>) {
        self.workers.insert(node_id, worker);
    }

    pub fn setup_partitions(&mut self, training_data: ArrayView2<Float>) -> NeighborsResult<()> {
        let num_samples = training_data.nrows();
        let samples_per_partition = num_samples / self.config.num_partitions;

        self.partitions.clear();

        for i in 0..self.config.num_partitions {
            let start = i * samples_per_partition;
            let end = if i == self.config.num_partitions - 1 {
                num_samples
            } else {
                (i + 1) * samples_per_partition
            };

            let partition_info = PartitionInfo {
                partition_id: i,
                data_range: (start, end),
                node_id: format!("node_{}", i % self.workers.len()),
                is_primary: true,
                replicas: self.create_replicas(i),
            };

            self.partitions.push(partition_info);
        }

        Ok(())
    }

    pub fn distributed_kneighbors(
        &self,
        query_data: ArrayView2<Float>,
        k: usize,
        distance: &Distance,
    ) -> NeighborsResult<DistributedNeighborSearchResult> {
        let start_time = std::time::Instant::now();

        if self.partitions.is_empty() {
            return Err(NeighborsError::InvalidInput(
                "No partitions configured. Call setup_partitions first.".to_string(),
            ));
        }

        let num_queries = query_data.nrows();
        let mut all_neighbors = Vec::with_capacity(num_queries);
        let mut all_distances = Vec::with_capacity(num_queries);
        let mut partition_stats = Vec::new();
        let mut total_network_time = 0u128;

        // Distribute queries across partitions
        #[cfg(feature = "parallel")]
        let partition_results: Result<Vec<_>, _> = self
            .partitions
            .par_iter()
            .map(|partition| self.search_partition_with_retry(partition, query_data, k, distance))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let partition_results: Result<Vec<_>, _> = self
            .partitions
            .iter()
            .map(|partition| self.search_partition_with_retry(partition, query_data, k, distance))
            .collect();

        let partition_results = partition_results?;

        // Combine results from all partitions
        for query_idx in 0..num_queries {
            let mut query_candidates = Vec::new();

            for (partition_neighbors, partition_distances, _) in &partition_results {
                if query_idx < partition_neighbors.len() {
                    for (neighbor_idx, &neighbor_id) in
                        partition_neighbors[query_idx].iter().enumerate()
                    {
                        if neighbor_idx < partition_distances[query_idx].len() {
                            query_candidates
                                .push((partition_distances[query_idx][neighbor_idx], neighbor_id));
                        }
                    }
                }
            }

            // Sort by distance and take top k
            query_candidates
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            query_candidates.truncate(k);

            let neighbors: Vec<usize> = query_candidates.iter().map(|(_, id)| *id).collect();
            let distances: Vec<Float> = query_candidates.iter().map(|(dist, _)| *dist).collect();

            all_neighbors.push(neighbors);
            all_distances.push(distances);
        }

        // Collect partition statistics
        for (_i, (_, _, stats)) in partition_results.iter().enumerate() {
            partition_stats.push(stats.clone());
            total_network_time += stats.search_time_ms;
        }

        let total_time = start_time.elapsed().as_millis();

        Ok(DistributedNeighborSearchResult {
            neighbors: all_neighbors,
            distances: all_distances,
            partition_stats,
            total_time_ms: total_time,
            network_time_ms: total_network_time,
        })
    }

    fn search_partition_with_retry(
        &self,
        partition: &PartitionInfo,
        query_data: ArrayView2<Float>,
        k: usize,
        distance: &Distance,
    ) -> NeighborsResult<(Vec<Vec<usize>>, Vec<Vec<Float>>, PartitionStats)> {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts < self.config.max_retries {
            attempts += 1;

            // Try primary node first
            if let Some(worker) = self.workers.get(&partition.node_id) {
                let start_time = std::time::Instant::now();

                match worker.search_partition(partition.partition_id, query_data, k, distance) {
                    Ok((neighbors, distances)) => {
                        let search_time = start_time.elapsed().as_millis();

                        let stats = PartitionStats {
                            partition_id: partition.partition_id,
                            search_time_ms: search_time,
                            data_size: partition.data_range.1 - partition.data_range.0,
                            num_queries: query_data.nrows(),
                            cache_hit_rate: 0.0, // Would be computed by worker
                        };

                        return Ok((neighbors, distances, stats));
                    }
                    Err(e) => {
                        last_error = Some(e);

                        // Try replicas if primary fails
                        if self.config.fault_tolerance {
                            for replica_id in &partition.replicas {
                                if let Some(replica_worker) = self.workers.get(replica_id) {
                                    let start_time = std::time::Instant::now();

                                    match replica_worker.search_partition(
                                        partition.partition_id,
                                        query_data,
                                        k,
                                        distance,
                                    ) {
                                        Ok((neighbors, distances)) => {
                                            let search_time = start_time.elapsed().as_millis();

                                            let stats = PartitionStats {
                                                partition_id: partition.partition_id,
                                                search_time_ms: search_time,
                                                data_size: partition.data_range.1
                                                    - partition.data_range.0,
                                                num_queries: query_data.nrows(),
                                                cache_hit_rate: 0.0,
                                            };

                                            return Ok((neighbors, distances, stats));
                                        }
                                        Err(_) => continue,
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Wait before retry
            if attempts < self.config.max_retries {
                std::thread::sleep(std::time::Duration::from_millis(100 * attempts as u64));
            }
        }

        Err(last_error.unwrap_or(NeighborsError::InvalidInput(format!(
            "Failed to search partition {} after {} attempts",
            partition.partition_id, attempts
        ))))
    }

    fn create_replicas(&self, partition_id: usize) -> Vec<String> {
        let mut replicas = Vec::new();
        let worker_nodes: Vec<String> = self.workers.keys().cloned().collect();

        for i in 1..=self.config.replication_factor {
            let replica_idx = (partition_id + i) % worker_nodes.len();
            if replica_idx < worker_nodes.len() {
                replicas.push(worker_nodes[replica_idx].clone());
            }
        }

        replicas
    }

    pub fn get_cluster_status(&self) -> HashMap<String, bool> {
        let mut status = HashMap::new();

        for (node_id, worker) in &self.workers {
            status.insert(node_id.clone(), worker.health_check());
        }

        status
    }

    pub fn get_load_metrics(&self) -> HashMap<String, HashMap<String, Float>> {
        let mut metrics = HashMap::new();

        for (node_id, worker) in &self.workers {
            metrics.insert(node_id.clone(), worker.get_load_metrics());
        }

        metrics
    }
}

impl DataPartitioner {
    pub fn new(strategy: LoadBalanceStrategy, num_partitions: usize) -> Self {
        Self {
            strategy,
            num_partitions,
            feature_hash_weights: None,
        }
    }

    pub fn with_feature_hash_weights(mut self, weights: Array1<Float>) -> Self {
        self.feature_hash_weights = Some(weights);
        self
    }

    pub fn partition_query(&self, query: ArrayView1<Float>, query_id: usize) -> usize {
        match self.strategy {
            LoadBalanceStrategy::RoundRobin => query_id % self.num_partitions,
            LoadBalanceStrategy::Random => {
                // Simple pseudo-random based on query content
                let hash = query
                    .iter()
                    .fold(0u64, |acc, &x| acc.wrapping_add((x * 1000.0) as u64));
                (hash % self.num_partitions as u64) as usize
            }
            LoadBalanceStrategy::FeatureHashBased => {
                if let Some(ref weights) = self.feature_hash_weights {
                    let weighted_sum: Float =
                        query.iter().zip(weights.iter()).map(|(x, w)| x * w).sum();
                    ((weighted_sum * 1000.0) as usize) % self.num_partitions
                } else {
                    query_id % self.num_partitions
                }
            }
            LoadBalanceStrategy::DataAware => {
                // Simple data-aware partitioning based on first feature
                if query.is_empty() {
                    0
                } else {
                    let normalized_value = (query[0] * 100.0) as usize;
                    normalized_value % self.num_partitions
                }
            }
        }
    }
}

pub struct DistributedNeighborSearchBuilder {
    config: DistributedConfiguration,
}

impl Default for DistributedNeighborSearchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedNeighborSearchBuilder {
    pub fn new() -> Self {
        Self {
            config: DistributedConfiguration::default(),
        }
    }

    pub fn num_partitions(mut self, partitions: usize) -> Self {
        self.config.num_partitions = partitions;
        self
    }

    pub fn replication_factor(mut self, factor: usize) -> Self {
        self.config.replication_factor = factor;
        self
    }

    pub fn load_balance_strategy(mut self, strategy: LoadBalanceStrategy) -> Self {
        self.config.load_balance_strategy = strategy;
        self
    }

    pub fn fault_tolerance(mut self, enabled: bool) -> Self {
        self.config.fault_tolerance = enabled;
        self
    }

    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.max_retries = retries;
        self
    }

    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.config.timeout_ms = timeout;
        self
    }

    pub fn build(self) -> DistributedNeighborSearch {
        DistributedNeighborSearch::new(self.config)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use std::sync::Arc;

    // Mock worker for testing
    struct MockWorker {
        node_id: String,
        should_fail: bool,
    }

    impl MockWorker {
        fn new(node_id: String) -> Self {
            Self {
                node_id,
                should_fail: false,
            }
        }

        fn new_failing(node_id: String) -> Self {
            Self {
                node_id,
                should_fail: true,
            }
        }
    }

    impl DistributedWorker for MockWorker {
        fn search_partition(
            &self,
            _partition_id: usize,
            query_data: ArrayView2<Float>,
            k: usize,
            _distance: &Distance,
        ) -> NeighborsResult<(Vec<Vec<usize>>, Vec<Vec<Float>>)> {
            if self.should_fail {
                return Err(NeighborsError::InvalidInput("Mock failure".to_string()));
            }

            let num_queries = query_data.nrows();
            let mut neighbors = Vec::with_capacity(num_queries);
            let mut distances = Vec::with_capacity(num_queries);

            for _ in 0..num_queries {
                let query_neighbors: Vec<usize> = (0..k).collect();
                let query_distances: Vec<Float> = (0..k).map(|i| i as Float).collect();
                neighbors.push(query_neighbors);
                distances.push(query_distances);
            }

            Ok((neighbors, distances))
        }

        fn get_partition_info(&self, partition_id: usize) -> NeighborsResult<PartitionInfo> {
            Ok(PartitionInfo {
                partition_id,
                data_range: (0, 100),
                node_id: self.node_id.clone(),
                is_primary: true,
                replicas: vec![],
            })
        }

        fn health_check(&self) -> bool {
            !self.should_fail
        }

        fn get_load_metrics(&self) -> HashMap<String, Float> {
            let mut metrics = HashMap::new();
            metrics.insert("cpu_usage".to_string(), 0.5);
            metrics.insert("memory_usage".to_string(), 0.3);
            metrics
        }
    }

    #[test]
    fn test_distributed_search_builder() {
        let search = DistributedNeighborSearch::builder()
            .num_partitions(4)
            .replication_factor(2)
            .fault_tolerance(true)
            .max_retries(3)
            .build();

        assert_eq!(search.config.num_partitions, 4);
        assert_eq!(search.config.replication_factor, 2);
        assert!(search.config.fault_tolerance);
        assert_eq!(search.config.max_retries, 3);
    }

    #[test]
    fn test_data_partitioner_round_robin() {
        let partitioner = DataPartitioner::new(LoadBalanceStrategy::RoundRobin, 4);
        let query = array![1.0, 2.0, 3.0];

        assert_eq!(partitioner.partition_query(query.view(), 0), 0);
        assert_eq!(partitioner.partition_query(query.view(), 1), 1);
        assert_eq!(partitioner.partition_query(query.view(), 4), 0);
    }

    #[test]
    fn test_data_partitioner_feature_hash() {
        let weights = array![1.0, 2.0, 3.0];
        let partitioner = DataPartitioner::new(LoadBalanceStrategy::FeatureHashBased, 4)
            .with_feature_hash_weights(weights);

        let query = array![1.0, 2.0, 3.0];
        let partition = partitioner.partition_query(query.view(), 0);

        assert!(partition < 4);
    }

    #[test]
    fn test_worker_registration() {
        let mut search = DistributedNeighborSearch::builder().build();
        let worker = Arc::new(MockWorker::new("test_node".to_string()));

        search.register_worker("test_node".to_string(), worker);

        assert_eq!(search.workers.len(), 1);
        assert!(search.workers.contains_key("test_node"));
    }

    #[test]
    fn test_health_check() {
        let mut search = DistributedNeighborSearch::builder().build();
        let healthy_worker = Arc::new(MockWorker::new("healthy".to_string()));
        let failing_worker = Arc::new(MockWorker::new_failing("failing".to_string()));

        search.register_worker("healthy".to_string(), healthy_worker);
        search.register_worker("failing".to_string(), failing_worker);

        let status = search.get_cluster_status();

        assert_eq!(status.get("healthy"), Some(&true));
        assert_eq!(status.get("failing"), Some(&false));
    }

    #[test]
    fn test_load_metrics() {
        let mut search = DistributedNeighborSearch::builder().build();
        let worker = Arc::new(MockWorker::new("test".to_string()));

        search.register_worker("test".to_string(), worker);

        let metrics = search.get_load_metrics();

        assert!(metrics.contains_key("test"));
        let node_metrics = metrics.get("test").unwrap();
        assert!(node_metrics.contains_key("cpu_usage"));
        assert!(node_metrics.contains_key("memory_usage"));
    }
}
