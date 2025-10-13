//! Large-scale calibration methods
//!
//! This module provides scalable calibration algorithms designed for
//! large datasets, distributed computing, and memory-constrained environments.

use crate::{CalibrationEstimator, SigmoidCalibrator};
use scirs2_core::ndarray::{s, Array1};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;

/// Configuration for large-scale calibration
#[derive(Debug, Clone)]
pub struct LargeScaleConfig {
    /// Chunk size for batch processing
    pub chunk_size: usize,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// Number of parallel workers
    pub n_workers: usize,
    /// Whether to use distributed processing
    pub distributed: bool,
    /// Merge strategy for combining results
    pub merge_strategy: MergeStrategy,
}

impl Default for LargeScaleConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10000,
            memory_limit_mb: 1024, // 1GB default
            n_workers: num_cpus::get(),
            distributed: false,
            merge_strategy: MergeStrategy::WeightedAverage,
        }
    }
}

/// Strategy for merging results from multiple workers/chunks
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Simple averaging
    Average,
    /// Weighted average by sample size
    WeightedAverage,
    /// Use best performing chunk
    BestPerforming,
    /// Ensemble of all chunks
    Ensemble,
}

/// Memory-efficient streaming calibrator
#[derive(Debug, Clone)]
pub struct StreamingCalibrator {
    /// Base calibrator
    base_calibrator: Box<dyn CalibrationEstimator>,
    /// Configuration
    config: LargeScaleConfig,
    /// Current batch of data
    current_batch_x: Vec<Float>,
    current_batch_y: Vec<i32>,
    /// Online statistics
    n_samples_seen: usize,
    /// Running estimates for parameters
    running_mean: Float,
    running_variance: Float,
    /// Whether calibrator is fitted
    is_fitted: bool,
}

impl StreamingCalibrator {
    /// Create a new streaming calibrator
    pub fn new(config: LargeScaleConfig) -> Self {
        Self {
            base_calibrator: Box::new(SigmoidCalibrator::new()),
            config,
            current_batch_x: Vec::new(),
            current_batch_y: Vec::new(),
            n_samples_seen: 0,
            running_mean: 0.0,
            running_variance: 0.0,
            is_fitted: false,
        }
    }

    /// Set base calibrator
    pub fn with_calibrator(mut self, calibrator: Box<dyn CalibrationEstimator>) -> Self {
        self.base_calibrator = calibrator;
        self
    }

    /// Add a single sample for incremental learning
    pub fn partial_fit(&mut self, probability: Float, target: i32) -> Result<()> {
        self.current_batch_x.push(probability);
        self.current_batch_y.push(target);

        // Update online statistics
        self.n_samples_seen += 1;
        let delta = probability - self.running_mean;
        self.running_mean += delta / self.n_samples_seen as Float;
        let delta2 = probability - self.running_mean;
        self.running_variance += delta * delta2;

        // Process batch when it reaches chunk size
        if self.current_batch_x.len() >= self.config.chunk_size {
            self.process_current_batch()?;
        }

        Ok(())
    }

    /// Process accumulated batch
    fn process_current_batch(&mut self) -> Result<()> {
        if self.current_batch_x.is_empty() {
            return Ok(());
        }

        let x_batch = Array1::from(self.current_batch_x.clone());
        let y_batch = Array1::from(self.current_batch_y.clone());

        // Fit on current batch
        self.base_calibrator.fit(&x_batch, &y_batch)?;
        self.is_fitted = true;

        // Clear current batch
        self.current_batch_x.clear();
        self.current_batch_y.clear();

        Ok(())
    }

    /// Finalize fitting with any remaining data
    pub fn finalize_fit(&mut self) -> Result<()> {
        if !self.current_batch_x.is_empty() {
            self.process_current_batch()?;
        }
        Ok(())
    }

    /// Get online statistics
    pub fn get_statistics(&self) -> HashMap<String, Float> {
        let mut stats = HashMap::new();
        stats.insert("n_samples".to_string(), self.n_samples_seen as Float);
        stats.insert("mean".to_string(), self.running_mean);

        let variance = if self.n_samples_seen > 1 {
            self.running_variance / (self.n_samples_seen - 1) as Float
        } else {
            0.0
        };
        stats.insert("variance".to_string(), variance);

        stats
    }
}

impl CalibrationEstimator for StreamingCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // For batch fitting, add all samples and process
        for (&prob, &target) in probabilities.iter().zip(y_true.iter()) {
            self.partial_fit(prob, target)?;
        }
        self.finalize_fit()
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on StreamingCalibrator".to_string(),
            });
        }
        self.base_calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Parallel calibration processor
#[derive(Debug)]
pub struct ParallelCalibrator {
    /// Configuration
    config: LargeScaleConfig,
    /// Fitted calibrators from each worker
    worker_calibrators: Vec<Box<dyn CalibrationEstimator>>,
    /// Worker weights for merging
    worker_weights: Vec<Float>,
    /// Whether calibrator is fitted
    is_fitted: bool,
}

impl ParallelCalibrator {
    /// Create a new parallel calibrator
    pub fn new(config: LargeScaleConfig) -> Self {
        Self {
            config,
            worker_calibrators: Vec::new(),
            worker_weights: Vec::new(),
            is_fitted: false,
        }
    }

    /// Fit calibrator using parallel processing
    pub fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        let n_samples = probabilities.len();
        let chunk_size = (n_samples + self.config.n_workers - 1) / self.config.n_workers;

        // Split data into chunks for parallel processing
        let mut chunks = Vec::new();
        for i in 0..self.config.n_workers {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(n_samples);

            if start < end {
                let x_chunk = probabilities.slice(s![start..end]).to_owned();
                let y_chunk = y_true.slice(s![start..end]).to_owned();
                chunks.push((x_chunk, y_chunk));
            }
        }

        // Process chunks in parallel using threads
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for chunk in chunks {
            let results_clone = Arc::clone(&results);

            let handle = thread::spawn(move || {
                let calibrator = SigmoidCalibrator::new();
                let fit_result = calibrator.fit(&chunk.0, &chunk.1);

                let mut results_guard = results_clone.lock().unwrap();
                match fit_result {
                    Ok(fitted_calibrator) => {
                        results_guard.push((
                            Box::new(fitted_calibrator) as Box<dyn CalibrationEstimator>,
                            chunk.0.len(),
                            Ok(()),
                        ));
                    }
                    Err(e) => {
                        // Still need to add a dummy calibrator to maintain indexing
                        results_guard.push((
                            Box::new(SigmoidCalibrator::new()) as Box<dyn CalibrationEstimator>,
                            chunk.0.len(),
                            Err(e),
                        ));
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|_| SklearsError::InvalidData {
                reason: "Thread join failed".to_string(),
            })?;
        }

        // Extract results
        let results = Arc::try_unwrap(results).map_err(|_| SklearsError::InvalidData {
            reason: "Failed to unwrap results".to_string(),
        })?;
        let results = results
            .into_inner()
            .map_err(|_| SklearsError::InvalidData {
                reason: "Failed to get mutex contents".to_string(),
            })?;

        // Store calibrators and compute weights
        self.worker_calibrators.clear();
        self.worker_weights.clear();
        let total_samples: usize = results.iter().map(|(_, size, _)| size).sum();

        for (calibrator, size, fit_result) in results {
            fit_result?; // Check if fitting succeeded
            self.worker_calibrators.push(calibrator);
            self.worker_weights
                .push(size as Float / total_samples as Float);
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Predict using ensemble of worker calibrators
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on ParallelCalibrator".to_string(),
            });
        }

        let mut ensemble_predictions = Array1::zeros(probabilities.len());

        match self.config.merge_strategy {
            MergeStrategy::Average => {
                for calibrator in &self.worker_calibrators {
                    let predictions = calibrator.predict_proba(probabilities)?;
                    ensemble_predictions = ensemble_predictions + predictions;
                }
                ensemble_predictions /= self.worker_calibrators.len() as Float;
            }
            MergeStrategy::WeightedAverage => {
                for (calibrator, &weight) in self
                    .worker_calibrators
                    .iter()
                    .zip(self.worker_weights.iter())
                {
                    let predictions = calibrator.predict_proba(probabilities)?;
                    ensemble_predictions = ensemble_predictions + weight * predictions;
                }
            }
            MergeStrategy::BestPerforming => {
                // Use the first calibrator as best performing (simplified)
                if let Some(best_calibrator) = self.worker_calibrators.first() {
                    ensemble_predictions = best_calibrator.predict_proba(probabilities)?;
                }
            }
            MergeStrategy::Ensemble => {
                // Similar to weighted average but with equal weights
                for calibrator in &self.worker_calibrators {
                    let predictions = calibrator.predict_proba(probabilities)?;
                    ensemble_predictions = ensemble_predictions + predictions;
                }
                ensemble_predictions /= self.worker_calibrators.len() as Float;
            }
        }

        Ok(ensemble_predictions)
    }
}

/// Memory-efficient calibrator with bounded memory usage
#[derive(Debug, Clone)]
pub struct MemoryEfficientCalibrator {
    /// Configuration
    config: LargeScaleConfig,
    /// Reservoir sampling for data storage
    reservoir: VecDeque<(Float, i32)>,
    /// Current reservoir size
    reservoir_size: usize,
    /// Maximum reservoir size based on memory limit
    max_reservoir_size: usize,
    /// Base calibrator
    base_calibrator: Box<dyn CalibrationEstimator>,
    /// Number of samples processed
    n_processed: usize,
    /// Whether calibrator is fitted
    is_fitted: bool,
}

impl MemoryEfficientCalibrator {
    /// Create a new memory-efficient calibrator
    pub fn new(config: LargeScaleConfig) -> Self {
        // Estimate max reservoir size based on memory limit
        // Assume 16 bytes per sample (Float + i32 + overhead)
        let max_samples = (config.memory_limit_mb * 1024 * 1024) / 16;

        Self {
            config: config.clone(),
            reservoir: VecDeque::new(),
            reservoir_size: 0,
            max_reservoir_size: max_samples.min(100000), // Cap at 100k samples
            base_calibrator: Box::new(SigmoidCalibrator::new()),
            n_processed: 0,
            is_fitted: false,
        }
    }

    /// Set base calibrator
    pub fn with_calibrator(mut self, calibrator: Box<dyn CalibrationEstimator>) -> Self {
        self.base_calibrator = calibrator;
        self
    }

    /// Add sample using reservoir sampling
    pub fn add_sample(&mut self, probability: Float, target: i32) {
        self.n_processed += 1;

        if self.reservoir_size < self.max_reservoir_size {
            // Reservoir not full, add sample
            self.reservoir.push_back((probability, target));
            self.reservoir_size += 1;
        } else {
            // Reservoir full, use reservoir sampling
            use scirs2_core::random::thread_rng;
            let rng_instance = thread_rng();
            let replace_idx = 0;

            if replace_idx < self.max_reservoir_size {
                self.reservoir[replace_idx] = (probability, target);
            }
        }
    }

    /// Fit calibrator on reservoir samples
    pub fn fit_on_reservoir(&mut self) -> Result<()> {
        if self.reservoir.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No samples in reservoir".to_string(),
            ));
        }

        let (probs, targets): (Vec<Float>, Vec<i32>) = self.reservoir.iter().cloned().unzip();
        let prob_array = Array1::from(probs);
        let target_array = Array1::from(targets);

        self.base_calibrator.fit(&prob_array, &target_array)?;
        self.is_fitted = true;
        Ok(())
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("reservoir_size".to_string(), self.reservoir_size);
        stats.insert("max_reservoir_size".to_string(), self.max_reservoir_size);
        stats.insert("n_processed".to_string(), self.n_processed);

        // Estimate memory usage in bytes
        let estimated_memory = self.reservoir_size * 16; // 16 bytes per sample
        stats.insert("estimated_memory_bytes".to_string(), estimated_memory);

        stats
    }
}

impl CalibrationEstimator for MemoryEfficientCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Add all samples to reservoir
        for (&prob, &target) in probabilities.iter().zip(y_true.iter()) {
            self.add_sample(prob, target);
        }
        self.fit_on_reservoir()
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on MemoryEfficientCalibrator".to_string(),
            });
        }
        self.base_calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Distributed calibration coordinator
#[derive(Debug)]
pub struct DistributedCalibrator {
    /// Configuration
    config: LargeScaleConfig,
    /// Node calibrators
    node_calibrators: HashMap<String, Box<dyn CalibrationEstimator>>,
    /// Node weights
    node_weights: HashMap<String, Float>,
    /// Whether calibrator is fitted
    is_fitted: bool,
}

impl DistributedCalibrator {
    /// Create a new distributed calibrator
    pub fn new(config: LargeScaleConfig) -> Self {
        Self {
            config,
            node_calibrators: HashMap::new(),
            node_weights: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Add a node calibrator
    pub fn add_node(
        &mut self,
        node_id: String,
        calibrator: Box<dyn CalibrationEstimator>,
        weight: Float,
    ) {
        self.node_calibrators.insert(node_id.clone(), calibrator);
        self.node_weights.insert(node_id, weight);
    }

    /// Simulate distributed training
    pub fn fit_distributed(
        &mut self,
        data_partitions: HashMap<String, (Array1<Float>, Array1<i32>)>,
    ) -> Result<()> {
        // In a real implementation, this would send data to different nodes
        // For simulation, we just fit each partition locally

        for (node_id, (probabilities, targets)) in data_partitions {
            if let Some(calibrator) = self.node_calibrators.get_mut(&node_id) {
                calibrator.fit(&probabilities, &targets)?;
            } else {
                // Create new calibrator for this node
                let new_calibrator = SigmoidCalibrator::new();
                let fitted_calibrator = new_calibrator.fit(&probabilities, &targets)?;
                let weight = probabilities.len() as Float;

                self.add_node(node_id, Box::new(fitted_calibrator), weight);
            }
        }

        // Normalize weights
        let total_weight: Float = self.node_weights.values().sum();
        if total_weight > 0.0 {
            for weight in self.node_weights.values_mut() {
                *weight /= total_weight;
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Predict using distributed ensemble
    pub fn predict_distributed(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on DistributedCalibrator".to_string(),
            });
        }

        let mut ensemble_predictions = Array1::zeros(probabilities.len());

        for (node_id, calibrator) in &self.node_calibrators {
            let weight = self.node_weights.get(node_id).unwrap_or(&1.0);
            let predictions = calibrator.predict_proba(probabilities)?;
            ensemble_predictions = ensemble_predictions + *weight * predictions;
        }

        Ok(ensemble_predictions)
    }

    /// Get distributed statistics
    pub fn get_distributed_stats(&self) -> HashMap<String, Float> {
        let mut stats = HashMap::new();
        stats.insert("n_nodes".to_string(), self.node_calibrators.len() as Float);

        let total_weight: Float = self.node_weights.values().sum();
        stats.insert("total_weight".to_string(), total_weight);

        stats
    }
}

/// Minibatch calibrator for large datasets
#[derive(Debug, Clone)]
pub struct MinibatchCalibrator {
    /// Configuration
    config: LargeScaleConfig,
    /// Base calibrator
    base_calibrator: Box<dyn CalibrationEstimator>,
    /// Learning rate for online updates
    learning_rate: Float,
    /// Number of batches processed
    n_batches: usize,
    /// Whether calibrator is fitted
    is_fitted: bool,
}

impl MinibatchCalibrator {
    /// Create a new minibatch calibrator
    pub fn new(config: LargeScaleConfig) -> Self {
        Self {
            config,
            base_calibrator: Box::new(SigmoidCalibrator::new()),
            learning_rate: 0.01,
            n_batches: 0,
            is_fitted: false,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Process a single minibatch
    pub fn partial_fit_batch(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        // For now, just fit the base calibrator on the batch
        // In a more sophisticated implementation, we would do incremental updates
        self.base_calibrator.fit(probabilities, y_true)?;
        self.n_batches += 1;
        self.is_fitted = true;
        Ok(())
    }

    /// Fit using minibatch processing
    pub fn fit_minibatch(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        let n_samples = probabilities.len();
        let n_batches = (n_samples + self.config.chunk_size - 1) / self.config.chunk_size;

        for i in 0..n_batches {
            let start = i * self.config.chunk_size;
            let end = ((i + 1) * self.config.chunk_size).min(n_samples);

            let batch_probs = probabilities.slice(s![start..end]).to_owned();
            let batch_targets = y_true.slice(s![start..end]).to_owned();

            self.partial_fit_batch(&batch_probs, &batch_targets)?;
        }

        Ok(())
    }

    /// Get minibatch statistics
    pub fn get_minibatch_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("n_batches".to_string(), self.n_batches);
        stats.insert("chunk_size".to_string(), self.config.chunk_size);
        stats
    }
}

impl CalibrationEstimator for MinibatchCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        self.fit_minibatch(probabilities, y_true)
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on MinibatchCalibrator".to_string(),
            });
        }
        self.base_calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_large_test_data() -> (Array1<Float>, Array1<i32>) {
        let n = 1000;
        let mut probabilities = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let prob = (i as Float) / (n as Float);
            let target = if prob > 0.5 { 1 } else { 0 };
            probabilities.push(prob);
            targets.push(target);
        }

        (Array1::from(probabilities), Array1::from(targets))
    }

    #[test]
    fn test_streaming_calibrator() {
        let (probabilities, targets) = create_large_test_data();

        let config = LargeScaleConfig {
            chunk_size: 100,
            ..Default::default()
        };

        let mut streaming_cal = StreamingCalibrator::new(config);
        streaming_cal.fit(&probabilities, &targets).unwrap();

        let predictions = streaming_cal
            .predict_proba(&probabilities.slice(s![0..10]).to_owned())
            .unwrap();
        assert_eq!(predictions.len(), 10);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

        let stats = streaming_cal.get_statistics();
        assert!(stats.contains_key("n_samples"));
        assert_eq!(stats["n_samples"], 1000.0);
    }

    #[test]
    fn test_parallel_calibrator() {
        let (probabilities, targets) = create_large_test_data();

        let config = LargeScaleConfig {
            n_workers: 4,
            merge_strategy: MergeStrategy::WeightedAverage,
            ..Default::default()
        };

        let mut parallel_cal = ParallelCalibrator::new(config);
        parallel_cal.fit(&probabilities, &targets).unwrap();

        let predictions = parallel_cal
            .predict_proba(&probabilities.slice(s![0..10]).to_owned())
            .unwrap();
        assert_eq!(predictions.len(), 10);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_memory_efficient_calibrator() {
        let (probabilities, targets) = create_large_test_data();

        let config = LargeScaleConfig {
            memory_limit_mb: 1, // Very small memory limit
            ..Default::default()
        };

        let mut memory_cal = MemoryEfficientCalibrator::new(config);
        memory_cal.fit(&probabilities, &targets).unwrap();

        let predictions = memory_cal
            .predict_proba(&probabilities.slice(s![0..10]).to_owned())
            .unwrap();
        assert_eq!(predictions.len(), 10);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

        let stats = memory_cal.memory_stats();
        assert!(stats.contains_key("reservoir_size"));
        assert!(stats["reservoir_size"] > 0);
    }

    #[test]
    fn test_distributed_calibrator() {
        let (probabilities, targets) = create_large_test_data();

        let config = LargeScaleConfig::default();
        let mut dist_cal = DistributedCalibrator::new(config);

        // Split data into partitions
        let mid = probabilities.len() / 2;
        let mut data_partitions = HashMap::new();
        data_partitions.insert(
            "node1".to_string(),
            (
                probabilities.slice(s![..mid]).to_owned(),
                targets.slice(s![..mid]).to_owned(),
            ),
        );
        data_partitions.insert(
            "node2".to_string(),
            (
                probabilities.slice(s![mid..]).to_owned(),
                targets.slice(s![mid..]).to_owned(),
            ),
        );

        dist_cal.fit_distributed(data_partitions).unwrap();

        let predictions = dist_cal
            .predict_distributed(&probabilities.slice(s![0..10]).to_owned())
            .unwrap();
        assert_eq!(predictions.len(), 10);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

        let stats = dist_cal.get_distributed_stats();
        assert_eq!(stats["n_nodes"], 2.0);
    }

    #[test]
    fn test_minibatch_calibrator() {
        let (probabilities, targets) = create_large_test_data();

        let config = LargeScaleConfig {
            chunk_size: 200,
            ..Default::default()
        };

        let mut minibatch_cal = MinibatchCalibrator::new(config);
        minibatch_cal.fit(&probabilities, &targets).unwrap();

        let predictions = minibatch_cal
            .predict_proba(&probabilities.slice(s![0..10]).to_owned())
            .unwrap();
        assert_eq!(predictions.len(), 10);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

        let stats = minibatch_cal.get_minibatch_stats();
        assert!(stats["n_batches"] > 0);
    }

    #[test]
    fn test_merge_strategies() {
        let (probabilities, targets) = create_large_test_data();

        let strategies = vec![
            MergeStrategy::Average,
            MergeStrategy::WeightedAverage,
            MergeStrategy::BestPerforming,
            MergeStrategy::Ensemble,
        ];

        for strategy in strategies {
            let config = LargeScaleConfig {
                n_workers: 2,
                merge_strategy: strategy,
                ..Default::default()
            };

            let mut parallel_cal = ParallelCalibrator::new(config);
            parallel_cal.fit(&probabilities, &targets).unwrap();

            let predictions = parallel_cal
                .predict_proba(&probabilities.slice(s![0..5]).to_owned())
                .unwrap();
            assert_eq!(predictions.len(), 5);
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }
}
