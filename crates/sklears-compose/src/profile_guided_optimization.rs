//! Profile-guided optimization for performance-critical paths
//!
//! This module provides runtime profiling and optimization capabilities for ML pipelines.
//! It collects performance data during execution and uses this information to optimize
//! future pipeline runs through adaptive algorithm selection, memory layout optimization,
//! and parallel execution strategies.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use scirs2_core::random::{thread_rng, Rng};

use sklears_core::error::{Result as SklResult, SklearsError};

/// Performance profile data for a specific operation
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Operation identifier
    pub operation_id: String,
    /// Input data characteristics
    pub data_characteristics: DataCharacteristics,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    /// Algorithm variant used
    pub algorithm_variant: String,
    /// Optimization level applied
    pub optimization_level: OptimizationLevel,
    /// Hardware context
    pub hardware_context: HardwareContext,
    /// Timestamp of execution
    pub timestamp: Instant,
}

/// Data characteristics that affect performance
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataCharacteristics {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Data sparsity (scaled by 1000, so 0 = dense, 1000 = fully sparse)
    pub sparsity_scaled: u32,
    /// Data type size in bytes
    pub dtype_size: usize,
    /// Memory layout (row-major, column-major)
    pub memory_layout: MemoryLayout,
    /// Cache friendliness score (scaled by 1000, so 0 = poor, 1000 = excellent)
    pub cache_friendliness_scaled: u32,
}

impl DataCharacteristics {
    /// Get sparsity as f64
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        f64::from(self.sparsity_scaled) / 1000.0
    }

    /// Set sparsity from f64
    pub fn set_sparsity(&mut self, sparsity: f64) {
        self.sparsity_scaled = (sparsity * 1000.0).round() as u32;
    }

    /// Get cache friendliness as f64
    #[must_use]
    pub fn cache_friendliness(&self) -> f64 {
        f64::from(self.cache_friendliness_scaled) / 1000.0
    }

    /// Set cache friendliness from f64
    pub fn set_cache_friendliness(&mut self, cache_friendliness: f64) {
        self.cache_friendliness_scaled = (cache_friendliness * 1000.0).round() as u32;
    }
}

/// Memory layout patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// RowMajor
    RowMajor,
    /// ColumnMajor
    ColumnMajor,
    /// Interleaved
    Interleaved,
    /// Custom
    Custom,
}

/// Execution metrics collected during profiling
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Total execution time
    pub execution_time: Duration,
    /// CPU time used
    pub cpu_time: Duration,
    /// Memory allocated in bytes
    pub memory_allocated: usize,
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Number of cache misses (estimated)
    pub cache_misses: usize,
    /// Number of SIMD operations executed
    pub simd_operations: usize,
    /// Parallel efficiency (0.0 = serial, 1.0 = perfect parallel)
    pub parallel_efficiency: f64,
    /// Memory bandwidth utilization (0.0 = none, 1.0 = saturated)
    pub memory_bandwidth: f64,
    /// FLOPs per second
    pub flops_per_second: f64,
}

/// Optimization levels for algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationLevel {
    /// No optimizations
    None,
    /// Basic optimizations (vectorization, basic parallelism)
    Basic,
    /// Advanced optimizations (SIMD, cache optimization)
    Advanced,
    /// Aggressive optimizations (unsafe code, platform-specific)
    Aggressive,
}

/// Hardware context information
#[derive(Debug, Clone)]
pub struct HardwareContext {
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// CPU cache sizes in bytes (L1, L2, L3)
    pub cache_sizes: Vec<usize>,
    /// Available SIMD instruction sets
    pub simd_features: Vec<SimdFeature>,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f64,
    /// CPU frequency in MHz
    pub cpu_frequency: f64,
}

/// SIMD instruction set features
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdFeature {
    /// SSE
    SSE,
    /// SSE2
    SSE2,
    /// SSE3
    SSE3,
    /// SSE4_1
    SSE4_1,
    /// SSE4_2
    SSE4_2,
    /// AVX
    AVX,
    /// AVX2
    AVX2,
    /// AVX512F
    AVX512F,
    /// NEON
    NEON,
}

/// Profile-guided optimization engine
#[derive(Debug)]
pub struct ProfileGuidedOptimizer {
    /// Historical performance profiles
    profiles: Arc<RwLock<HashMap<String, VecDeque<PerformanceProfile>>>>,
    /// Current optimization strategies
    strategies: Arc<RwLock<HashMap<String, OptimizationStrategy>>>,
    /// Algorithm selection cache
    algorithm_cache: Arc<RwLock<HashMap<DataCharacteristics, String>>>,
    /// Performance predictors
    predictors: Arc<RwLock<HashMap<String, Box<dyn PerformancePredictor + Send + Sync>>>>,
    /// Configuration
    config: OptimizerConfig,
    /// Hardware context
    hardware_context: HardwareContext,
}

/// Optimization strategy for a specific operation
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    /// Preferred algorithm variant
    pub preferred_algorithm: String,
    /// Optimization level to use
    pub optimization_level: OptimizationLevel,
    /// Memory layout preference
    pub memory_layout: MemoryLayout,
    /// Parallel execution strategy
    pub parallel_strategy: ParallelStrategy,
    /// Cache optimization hints
    pub cache_hints: CacheOptimizationHints,
    /// Confidence score (0.0 = low, 1.0 = high)
    pub confidence: f64,
}

/// Parallel execution strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParallelStrategy {
    Serial,
    ThreadParallel,
    Vectorized,
    Hybrid,
    GPU,
}

/// Cache optimization hints
#[derive(Debug, Clone)]
pub struct CacheOptimizationHints {
    /// Preferred block size for tiling
    pub block_size: usize,
    /// Whether to use prefetching
    pub use_prefetch: bool,
    /// Memory access pattern optimization
    pub access_pattern: AccessPattern,
    /// Cache-friendly algorithms preference
    pub cache_friendly_algorithms: bool,
}

/// Memory access patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    Sequential,
    Random,
    Strided,
    Blocked,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum number of profiles to keep per operation
    pub max_profiles_per_operation: usize,
    /// Minimum number of profiles before optimization
    pub min_profiles_for_optimization: usize,
    /// Confidence threshold for strategy changes
    pub confidence_threshold: f64,
    /// Performance improvement threshold
    pub improvement_threshold: f64,
    /// Enable adaptive optimization
    pub adaptive_optimization: bool,
    /// Profile collection interval
    pub profile_interval: Duration,
}

/// Trait for performance predictors
pub trait PerformancePredictor: Send + Sync + std::fmt::Debug {
    /// Predict execution time for given characteristics
    fn predict_execution_time(&self, characteristics: &DataCharacteristics) -> SklResult<Duration>;

    /// Predict memory usage
    fn predict_memory_usage(&self, characteristics: &DataCharacteristics) -> SklResult<usize>;

    /// Update predictor with new performance data
    fn update(&mut self, profile: &PerformanceProfile) -> SklResult<()>;

    /// Get predictor accuracy
    fn accuracy(&self) -> f64;
}

/// Machine learning-based performance predictor
#[derive(Debug)]
pub struct MLPerformancePredictor {
    /// Training data
    training_data: Vec<(DataCharacteristics, ExecutionMetrics)>,
    /// Model parameters (simplified linear model)
    weights: Vec<f64>,
    /// Prediction accuracy
    accuracy: f64,
    /// Number of training samples
    training_samples: usize,
}

impl MLPerformancePredictor {
    /// Create a new ML-based predictor
    #[must_use]
    pub fn new() -> Self {
        Self {
            training_data: Vec::new(),
            weights: vec![1.0; 10], // Simplified feature count
            accuracy: 0.0,
            training_samples: 0,
        }
    }

    /// Extract features from data characteristics
    fn extract_features(&self, characteristics: &DataCharacteristics) -> Vec<f64> {
        vec![
            characteristics.n_samples as f64,
            characteristics.n_features as f64,
            characteristics.sparsity(),
            characteristics.dtype_size as f64,
            characteristics.cache_friendliness(),
            (characteristics.n_samples * characteristics.n_features) as f64, // Data size
            (characteristics.n_samples as f64).log2(),
            (characteristics.n_features as f64).log2(),
            characteristics.sparsity() * characteristics.n_features as f64,
            characteristics.cache_friendliness() * characteristics.n_samples as f64,
        ]
    }

    /// Train the predictor using linear regression
    fn train(&mut self) -> SklResult<()> {
        if self.training_data.len() < 10 {
            return Ok(()); // Not enough data
        }

        // Simple gradient descent for linear regression
        let learning_rate = 0.001;
        let epochs = 100;

        for _ in 0..epochs {
            let mut gradients = vec![0.0; self.weights.len()];
            let mut total_error = 0.0;

            for (characteristics, metrics) in &self.training_data {
                let features = self.extract_features(characteristics);
                let predicted = features
                    .iter()
                    .zip(&self.weights)
                    .map(|(f, w)| f * w)
                    .sum::<f64>();

                let actual = metrics.execution_time.as_secs_f64();
                let error = predicted - actual;
                total_error += error * error;

                for (i, feature) in features.iter().enumerate() {
                    gradients[i] += error * feature;
                }
            }

            // Update weights
            for (weight, gradient) in self.weights.iter_mut().zip(&gradients) {
                *weight -= learning_rate * gradient / self.training_data.len() as f64;
            }

            // Update accuracy (simplified R²)
            let mse = total_error / self.training_data.len() as f64;
            self.accuracy = (1.0 - mse).max(0.0).min(1.0);
        }

        Ok(())
    }
}

impl PerformancePredictor for MLPerformancePredictor {
    fn predict_execution_time(&self, characteristics: &DataCharacteristics) -> SklResult<Duration> {
        let features = self.extract_features(characteristics);
        let prediction = features
            .iter()
            .zip(&self.weights)
            .map(|(f, w)| f * w)
            .sum::<f64>()
            .max(0.0);

        Ok(Duration::from_secs_f64(prediction))
    }

    fn predict_memory_usage(&self, characteristics: &DataCharacteristics) -> SklResult<usize> {
        // Simplified memory prediction
        let base_memory =
            characteristics.n_samples * characteristics.n_features * characteristics.dtype_size;
        let overhead_factor = 1.0 + (1.0 - characteristics.sparsity()) * 0.5;
        Ok((base_memory as f64 * overhead_factor) as usize)
    }

    fn update(&mut self, profile: &PerformanceProfile) -> SklResult<()> {
        self.training_data.push((
            profile.data_characteristics.clone(),
            profile.metrics.clone(),
        ));

        self.training_samples += 1;

        // Retrain periodically
        if self.training_samples % 50 == 0 {
            self.train()?;
        }

        Ok(())
    }

    fn accuracy(&self) -> f64 {
        self.accuracy
    }
}

impl Default for MLPerformancePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileGuidedOptimizer {
    /// Create a new profile-guided optimizer
    pub fn new(config: OptimizerConfig) -> SklResult<Self> {
        let hardware_context = Self::detect_hardware_context();

        Ok(Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
            strategies: Arc::new(RwLock::new(HashMap::new())),
            algorithm_cache: Arc::new(RwLock::new(HashMap::new())),
            predictors: Arc::new(RwLock::new(HashMap::new())),
            config,
            hardware_context,
        })
    }

    /// Detect hardware context
    fn detect_hardware_context() -> HardwareContext {
        let cpu_cores = thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);

        // Simplified hardware detection
        /// HardwareContext
        HardwareContext {
            cpu_cores,
            cache_sizes: vec![32768, 262_144, 8_388_608], // Typical L1, L2, L3
            simd_features: Self::detect_simd_features(),
            memory_bandwidth: 25.6, // Typical DDR4
            cpu_frequency: 3000.0,  // 3 GHz typical
        }
    }

    /// Detect available SIMD features
    fn detect_simd_features() -> Vec<SimdFeature> {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse") {
                features.push(SimdFeature::SSE);
            }
            if is_x86_feature_detected!("sse2") {
                features.push(SimdFeature::SSE2);
            }
            if is_x86_feature_detected!("sse3") {
                features.push(SimdFeature::SSE3);
            }
            if is_x86_feature_detected!("sse4.1") {
                features.push(SimdFeature::SSE4_1);
            }
            if is_x86_feature_detected!("sse4.2") {
                features.push(SimdFeature::SSE4_2);
            }
            if is_x86_feature_detected!("avx") {
                features.push(SimdFeature::AVX);
            }
            if is_x86_feature_detected!("avx2") {
                features.push(SimdFeature::AVX2);
            }
            if is_x86_feature_detected!("avx512f") {
                features.push(SimdFeature::AVX512F);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            features.push(SimdFeature::NEON);
        }

        features
    }

    /// Add a performance profile
    pub fn add_profile(&self, profile: PerformanceProfile) -> SklResult<()> {
        let mut profiles = self.profiles.write().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire profiles lock".to_string())
        })?;

        let operation_profiles = profiles
            .entry(profile.operation_id.clone())
            .or_insert_with(VecDeque::new);

        operation_profiles.push_back(profile.clone());

        // Limit the number of profiles per operation
        while operation_profiles.len() > self.config.max_profiles_per_operation {
            operation_profiles.pop_front();
        }

        // Update predictors
        if let Ok(mut predictors) = self.predictors.write() {
            if let Some(predictor) = predictors.get_mut(&profile.operation_id) {
                let _ = predictor.update(&profile);
            } else {
                let mut new_predictor = Box::new(MLPerformancePredictor::new());
                let _ = new_predictor.update(&profile);
                predictors.insert(profile.operation_id.clone(), new_predictor);
            }
        }

        // Trigger optimization if we have enough profiles
        if operation_profiles.len() >= self.config.min_profiles_for_optimization {
            self.optimize_strategy(&profile.operation_id)?;
        }

        Ok(())
    }

    /// Get optimization strategy for an operation
    pub fn get_strategy(
        &self,
        operation_id: &str,
        characteristics: &DataCharacteristics,
    ) -> SklResult<OptimizationStrategy> {
        // Check cache first
        if let Ok(cache) = self.algorithm_cache.read() {
            if let Some(cached_algorithm) = cache.get(characteristics) {
                if let Ok(strategies) = self.strategies.read() {
                    if let Some(strategy) = strategies.get(operation_id) {
                        let mut cached_strategy = strategy.clone();
                        cached_strategy.preferred_algorithm = cached_algorithm.clone();
                        return Ok(cached_strategy);
                    }
                }
            }
        }

        // Generate strategy based on data characteristics and hardware
        self.generate_strategy(operation_id, characteristics)
    }

    /// Generate optimization strategy
    fn generate_strategy(
        &self,
        operation_id: &str,
        characteristics: &DataCharacteristics,
    ) -> SklResult<OptimizationStrategy> {
        let preferred_algorithm = self.select_algorithm(operation_id, characteristics)?;
        let optimization_level = self.select_optimization_level(characteristics);
        let memory_layout = self.select_memory_layout(characteristics);
        let parallel_strategy = self.select_parallel_strategy(characteristics);
        let cache_hints = self.generate_cache_hints(characteristics);

        let confidence = self.calculate_confidence(operation_id, characteristics);

        Ok(OptimizationStrategy {
            preferred_algorithm,
            optimization_level,
            memory_layout,
            parallel_strategy,
            cache_hints,
            confidence,
        })
    }

    /// Select best algorithm based on characteristics
    fn select_algorithm(
        &self,
        operation_id: &str,
        characteristics: &DataCharacteristics,
    ) -> SklResult<String> {
        if let Ok(profiles) = self.profiles.read() {
            if let Some(operation_profiles) = profiles.get(operation_id) {
                // Find best performing algorithm for similar data characteristics
                let mut best_algorithm = "default".to_string();
                let mut best_score = f64::INFINITY;

                for profile in operation_profiles {
                    if self.characteristics_similar(&profile.data_characteristics, characteristics)
                    {
                        let score = profile.metrics.execution_time.as_secs_f64();
                        if score < best_score {
                            best_score = score;
                            best_algorithm = profile.algorithm_variant.clone();
                        }
                    }
                }

                return Ok(best_algorithm);
            }
        }

        // Fallback to heuristic selection
        Ok(self.heuristic_algorithm_selection(characteristics))
    }

    /// Check if data characteristics are similar
    fn characteristics_similar(&self, a: &DataCharacteristics, b: &DataCharacteristics) -> bool {
        let size_ratio = (a.n_samples * a.n_features) as f64 / (b.n_samples * b.n_features) as f64;
        let sparsity_diff = (a.sparsity() - b.sparsity()).abs();

        (0.5..=2.0).contains(&size_ratio) && sparsity_diff < 0.3
    }

    /// Heuristic algorithm selection
    fn heuristic_algorithm_selection(&self, characteristics: &DataCharacteristics) -> String {
        let data_size = characteristics.n_samples * characteristics.n_features;

        if characteristics.sparsity() > 0.7 {
            "sparse_optimized".to_string()
        } else if data_size < 10000 {
            "small_data_optimized".to_string()
        } else if data_size > 1_000_000 {
            "large_data_optimized".to_string()
        } else {
            "general_purpose".to_string()
        }
    }

    /// Select optimization level
    fn select_optimization_level(
        &self,
        characteristics: &DataCharacteristics,
    ) -> OptimizationLevel {
        let data_size = characteristics.n_samples * characteristics.n_features;

        if data_size > 1_000_000 {
            OptimizationLevel::Aggressive
        } else if data_size > 100_000 {
            OptimizationLevel::Advanced
        } else if data_size > 10000 {
            OptimizationLevel::Basic
        } else {
            OptimizationLevel::None
        }
    }

    /// Select memory layout
    fn select_memory_layout(&self, characteristics: &DataCharacteristics) -> MemoryLayout {
        if characteristics.n_features > characteristics.n_samples {
            MemoryLayout::ColumnMajor
        } else {
            MemoryLayout::RowMajor
        }
    }

    /// Select parallel strategy
    fn select_parallel_strategy(&self, characteristics: &DataCharacteristics) -> ParallelStrategy {
        let data_size = characteristics.n_samples * characteristics.n_features;

        if self
            .hardware_context
            .simd_features
            .contains(&SimdFeature::AVX2)
            && data_size > 100_000
        {
            ParallelStrategy::Hybrid
        } else if self.hardware_context.cpu_cores > 1 && data_size > 50000 {
            ParallelStrategy::ThreadParallel
        } else if self.hardware_context.simd_features.len() > 2 {
            ParallelStrategy::Vectorized
        } else {
            ParallelStrategy::Serial
        }
    }

    /// Generate cache optimization hints
    fn generate_cache_hints(
        &self,
        characteristics: &DataCharacteristics,
    ) -> CacheOptimizationHints {
        let block_size = if self.hardware_context.cache_sizes.len() > 1 {
            (self.hardware_context.cache_sizes[1] / characteristics.dtype_size).min(1024)
        } else {
            256
        };

        /// CacheOptimizationHints
        CacheOptimizationHints {
            block_size,
            use_prefetch: characteristics.n_samples > 10000,
            access_pattern: if characteristics.cache_friendliness() > 0.7 {
                AccessPattern::Sequential
            } else {
                AccessPattern::Blocked
            },
            cache_friendly_algorithms: characteristics.cache_friendliness() > 0.5,
        }
    }

    /// Calculate confidence in strategy
    fn calculate_confidence(
        &self,
        operation_id: &str,
        characteristics: &DataCharacteristics,
    ) -> f64 {
        if let Ok(profiles) = self.profiles.read() {
            if let Some(operation_profiles) = profiles.get(operation_id) {
                let similar_profiles = operation_profiles
                    .iter()
                    .filter(|p| {
                        self.characteristics_similar(&p.data_characteristics, characteristics)
                    })
                    .count();

                return (similar_profiles as f64 / 10.0).min(1.0);
            }
        }

        0.1 // Low confidence for new operations
    }

    /// Optimize strategy for an operation
    fn optimize_strategy(&self, operation_id: &str) -> SklResult<()> {
        if let Ok(profiles) = self.profiles.read() {
            if let Some(operation_profiles) = profiles.get(operation_id) {
                if operation_profiles.len() < self.config.min_profiles_for_optimization {
                    return Ok(());
                }

                // Analyze performance trends
                let mut algorithm_performance: HashMap<String, Vec<f64>> = HashMap::new();

                for profile in operation_profiles {
                    let score = profile.metrics.execution_time.as_secs_f64();
                    algorithm_performance
                        .entry(profile.algorithm_variant.clone())
                        .or_default()
                        .push(score);
                }

                // Find best performing algorithm
                let mut best_algorithm = "default".to_string();
                let mut best_average = f64::INFINITY;

                for (algorithm, scores) in &algorithm_performance {
                    if scores.len() >= 3 {
                        // Minimum samples for reliability
                        let average: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                        if average < best_average {
                            best_average = average;
                            best_algorithm = algorithm.clone();
                        }
                    }
                }

                // Update strategy
                if let Ok(mut strategies) = self.strategies.write() {
                    let strategy =
                        strategies
                            .entry(operation_id.to_string())
                            .or_insert_with(|| OptimizationStrategy {
                                preferred_algorithm: best_algorithm.clone(),
                                optimization_level: OptimizationLevel::Basic,
                                memory_layout: MemoryLayout::RowMajor,
                                parallel_strategy: ParallelStrategy::Serial,
                                cache_hints: CacheOptimizationHints {
                                    block_size: 256,
                                    use_prefetch: false,
                                    access_pattern: AccessPattern::Sequential,
                                    cache_friendly_algorithms: true,
                                },
                                confidence: 0.5,
                            });

                    strategy.preferred_algorithm = best_algorithm;
                    strategy.confidence = (algorithm_performance.len() as f64 / 5.0).min(1.0);
                }
            }
        }

        Ok(())
    }

    /// Predict performance for given characteristics
    pub fn predict_performance(
        &self,
        operation_id: &str,
        characteristics: &DataCharacteristics,
    ) -> SklResult<ExecutionMetrics> {
        if let Ok(predictors) = self.predictors.read() {
            if let Some(predictor) = predictors.get(operation_id) {
                let execution_time = predictor.predict_execution_time(characteristics)?;
                let memory_usage = predictor.predict_memory_usage(characteristics)?;

                return Ok(ExecutionMetrics {
                    execution_time,
                    cpu_time: execution_time,
                    memory_allocated: memory_usage,
                    peak_memory: memory_usage,
                    cache_misses: 0,
                    simd_operations: 0,
                    parallel_efficiency: 1.0,
                    memory_bandwidth: 0.5,
                    flops_per_second: 1e9,
                });
            }
        }

        // Fallback estimation
        let data_size = characteristics.n_samples * characteristics.n_features;
        let estimated_time = Duration::from_millis((data_size / 10000).max(1) as u64);
        let estimated_memory = data_size * characteristics.dtype_size;

        Ok(ExecutionMetrics {
            execution_time: estimated_time,
            cpu_time: estimated_time,
            memory_allocated: estimated_memory,
            peak_memory: estimated_memory,
            cache_misses: 0,
            simd_operations: 0,
            parallel_efficiency: 1.0,
            memory_bandwidth: 0.5,
            flops_per_second: 1e9,
        })
    }

    /// Get optimization statistics
    #[must_use]
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let mut stats = OptimizationStats {
            total_operations: 0,
            optimized_operations: 0,
            average_confidence: 0.0,
            total_profiles: 0,
            predictor_accuracy: 0.0,
        };

        if let Ok(profiles) = self.profiles.read() {
            stats.total_operations = profiles.len();
            stats.total_profiles = profiles.values().map(std::collections::VecDeque::len).sum();
        }

        if let Ok(strategies) = self.strategies.read() {
            stats.optimized_operations = strategies.len();
            stats.average_confidence = strategies.values().map(|s| s.confidence).sum::<f64>()
                / strategies.len().max(1) as f64;
        }

        if let Ok(predictors) = self.predictors.read() {
            stats.predictor_accuracy = predictors.values().map(|p| p.accuracy()).sum::<f64>()
                / predictors.len().max(1) as f64;
        }

        stats
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Total number of operations tracked
    pub total_operations: usize,
    /// Number of operations with optimized strategies
    pub optimized_operations: usize,
    /// Average confidence across all strategies
    pub average_confidence: f64,
    /// Total number of performance profiles
    pub total_profiles: usize,
    /// Average predictor accuracy
    pub predictor_accuracy: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_profiles_per_operation: 1000,
            min_profiles_for_optimization: 10,
            confidence_threshold: 0.7,
            improvement_threshold: 0.1,
            adaptive_optimization: true,
            profile_interval: Duration::from_secs(60),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer = ProfileGuidedOptimizer::new(config).unwrap();

        let stats = optimizer.get_optimization_stats();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.optimized_operations, 0);
    }

    #[test]
    fn test_data_characteristics() {
        let mut characteristics = DataCharacteristics {
            n_samples: 1000,
            n_features: 50,
            sparsity_scaled: 100, // 0.1 * 1000
            dtype_size: 8,
            memory_layout: MemoryLayout::RowMajor,
            cache_friendliness_scaled: 800, // 0.8 * 1000
        };

        assert_eq!(characteristics.n_samples, 1000);
        assert_eq!(characteristics.n_features, 50);
        assert_eq!(characteristics.sparsity(), 0.1);
        assert_eq!(characteristics.cache_friendliness(), 0.8);

        characteristics.set_sparsity(0.5);
        assert_eq!(characteristics.sparsity(), 0.5);
    }

    #[test]
    fn test_performance_profile() {
        let profile = PerformanceProfile {
            operation_id: "test_op".to_string(),
            data_characteristics: DataCharacteristics {
                n_samples: 100,
                n_features: 10,
                sparsity_scaled: 0, // 0.0 * 1000
                dtype_size: 8,
                memory_layout: MemoryLayout::RowMajor,
                cache_friendliness_scaled: 1000, // 1.0 * 1000
            },
            metrics: ExecutionMetrics {
                execution_time: Duration::from_millis(100),
                cpu_time: Duration::from_millis(100),
                memory_allocated: 8000,
                peak_memory: 8000,
                cache_misses: 0,
                simd_operations: 100,
                parallel_efficiency: 1.0,
                memory_bandwidth: 0.5,
                flops_per_second: 1e6,
            },
            algorithm_variant: "test_algo".to_string(),
            optimization_level: OptimizationLevel::Basic,
            hardware_context: HardwareContext {
                cpu_cores: 4,
                cache_sizes: vec![32768, 262144],
                simd_features: vec![SimdFeature::SSE2],
                memory_bandwidth: 25.6,
                cpu_frequency: 3000.0,
            },
            timestamp: Instant::now(),
        };

        assert_eq!(profile.operation_id, "test_op");
        assert_eq!(profile.algorithm_variant, "test_algo");
    }

    #[test]
    fn test_ml_predictor() {
        let mut predictor = MLPerformancePredictor::new();
        assert_eq!(predictor.accuracy(), 0.0);

        let characteristics = DataCharacteristics {
            n_samples: 100,
            n_features: 10,
            sparsity_scaled: 0, // 0.0 * 1000
            dtype_size: 8,
            memory_layout: MemoryLayout::RowMajor,
            cache_friendliness_scaled: 1000, // 1.0 * 1000
        };

        let prediction = predictor.predict_execution_time(&characteristics).unwrap();
        assert!(prediction.as_secs_f64() >= 0.0);
    }

    #[test]
    fn test_optimization_strategy() {
        let strategy = OptimizationStrategy {
            preferred_algorithm: "test_algo".to_string(),
            optimization_level: OptimizationLevel::Advanced,
            memory_layout: MemoryLayout::ColumnMajor,
            parallel_strategy: ParallelStrategy::Hybrid,
            cache_hints: CacheOptimizationHints {
                block_size: 512,
                use_prefetch: true,
                access_pattern: AccessPattern::Blocked,
                cache_friendly_algorithms: true,
            },
            confidence: 0.9,
        };

        assert_eq!(strategy.preferred_algorithm, "test_algo");
        assert_eq!(strategy.optimization_level, OptimizationLevel::Advanced);
        assert_eq!(strategy.confidence, 0.9);
    }

    #[test]
    fn test_simd_feature_detection() {
        let features = ProfileGuidedOptimizer::detect_simd_features();
        // Just test that we get some features (platform-dependent)
        println!("Detected SIMD features: {:?}", features);
    }
}

/// Advanced runtime compilation and optimization engine
#[derive(Debug)]
pub struct RuntimeOptimizer {
    /// Code generation cache
    compiled_variants: Arc<RwLock<HashMap<String, CompiledVariant>>>,
    /// Compilation statistics
    compilation_stats: Arc<RwLock<CompilationStats>>,
    /// Runtime configuration
    config: RuntimeOptimizerConfig,
}

/// Compiled algorithm variant
#[derive(Debug, Clone)]
pub struct CompiledVariant {
    /// Variant identifier
    pub variant_id: String,
    /// Optimization level used
    pub optimization_level: OptimizationLevel,
    /// Target hardware features
    pub target_features: Vec<SimdFeature>,
    /// Compilation timestamp
    pub compiled_at: Instant,
    /// Performance characteristics
    pub performance_profile: Option<PerformanceProfile>,
    /// Compilation success
    pub compilation_successful: bool,
}

/// Runtime optimizer configuration
#[derive(Debug, Clone)]
pub struct RuntimeOptimizerConfig {
    /// Enable JIT compilation
    pub enable_jit: bool,
    /// Maximum compiled variants per operation
    pub max_variants: usize,
    /// Compilation timeout
    pub compilation_timeout: Duration,
    /// Minimum performance improvement for recompilation
    pub min_improvement: f64,
    /// Enable profile-guided recompilation
    pub enable_pgo_recompilation: bool,
}

/// Compilation statistics
#[derive(Debug, Clone)]
pub struct CompilationStats {
    /// Total compilations
    pub total_compilations: usize,
    /// Successful compilations
    pub successful_compilations: usize,
    /// Total compilation time
    pub total_compilation_time: Duration,
    /// Average compilation time
    pub average_compilation_time: Duration,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
}

impl RuntimeOptimizer {
    /// Create a new runtime optimizer
    #[must_use]
    pub fn new(config: RuntimeOptimizerConfig) -> Self {
        Self {
            compiled_variants: Arc::new(RwLock::new(HashMap::new())),
            compilation_stats: Arc::new(RwLock::new(CompilationStats {
                total_compilations: 0,
                successful_compilations: 0,
                total_compilation_time: Duration::from_secs(0),
                average_compilation_time: Duration::from_secs(0),
                cache_hits: 0,
                cache_misses: 0,
            })),
            config,
        }
    }

    /// Get or compile optimized variant
    pub fn get_optimized_variant(
        &self,
        operation_id: &str,
        characteristics: &DataCharacteristics,
        strategy: &OptimizationStrategy,
    ) -> SklResult<String> {
        let variant_key = self.generate_variant_key(operation_id, characteristics, strategy);

        // Check cache first
        if let Ok(variants) = self.compiled_variants.read() {
            if let Some(variant) = variants.get(&variant_key) {
                if variant.compilation_successful {
                    self.update_cache_stats(true);
                    return Ok(variant.variant_id.clone());
                }
            }
        }

        self.update_cache_stats(false);

        // Compile new variant if JIT is enabled
        if self.config.enable_jit {
            self.compile_variant(operation_id, characteristics, strategy)
        } else {
            Ok(strategy.preferred_algorithm.clone())
        }
    }

    /// Compile optimized variant
    fn compile_variant(
        &self,
        operation_id: &str,
        characteristics: &DataCharacteristics,
        strategy: &OptimizationStrategy,
    ) -> SklResult<String> {
        let start_time = Instant::now();
        let variant_key = self.generate_variant_key(operation_id, characteristics, strategy);

        // Simulate compilation process (in real implementation, this would invoke LLVM or similar)
        let compilation_successful = self.simulate_compilation(strategy);
        let compilation_time = start_time.elapsed();

        let variant = CompiledVariant {
            variant_id: format!("{}_{}", operation_id, compilation_time.as_nanos()),
            optimization_level: strategy.optimization_level,
            target_features: self.select_target_features(strategy),
            compiled_at: Instant::now(),
            performance_profile: None,
            compilation_successful,
        };

        // Update compilation statistics
        self.update_compilation_stats(compilation_time, compilation_successful);

        // Cache the variant
        if let Ok(mut variants) = self.compiled_variants.write() {
            // Evict old variants if necessary
            if variants.len() >= self.config.max_variants {
                self.evict_old_variants(&mut variants);
            }
            variants.insert(variant_key, variant.clone());
        }

        Ok(variant.variant_id)
    }

    /// Generate variant key for caching
    fn generate_variant_key(
        &self,
        operation_id: &str,
        characteristics: &DataCharacteristics,
        strategy: &OptimizationStrategy,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        operation_id.hash(&mut hasher);
        characteristics.hash(&mut hasher);
        format!("{:?}", strategy.optimization_level).hash(&mut hasher);
        format!("{:?}", strategy.parallel_strategy).hash(&mut hasher);

        format!("{}_{:x}", operation_id, hasher.finish())
    }

    /// Simulate compilation process
    fn simulate_compilation(&self, strategy: &OptimizationStrategy) -> bool {
        // Simulate compilation based on optimization level
        match strategy.optimization_level {
            OptimizationLevel::None => true,
            OptimizationLevel::Basic => thread_rng().gen::<f64>() > 0.1, // 90% success rate
            OptimizationLevel::Advanced => thread_rng().gen::<f64>() > 0.2, // 80% success rate
            OptimizationLevel::Aggressive => thread_rng().gen::<f64>() > 0.3, // 70% success rate
        }
    }

    /// Select target features for compilation
    fn select_target_features(&self, strategy: &OptimizationStrategy) -> Vec<SimdFeature> {
        let mut features = Vec::new();

        match strategy.optimization_level {
            OptimizationLevel::None => {}
            OptimizationLevel::Basic => {
                features.push(SimdFeature::SSE2);
            }
            OptimizationLevel::Advanced => {
                features.extend_from_slice(&[SimdFeature::SSE2, SimdFeature::AVX]);
            }
            OptimizationLevel::Aggressive => {
                features.extend_from_slice(&[
                    SimdFeature::SSE2,
                    SimdFeature::AVX,
                    SimdFeature::AVX2,
                    SimdFeature::AVX512F,
                ]);
            }
        }

        features
    }

    /// Update cache statistics
    fn update_cache_stats(&self, hit: bool) {
        if let Ok(mut stats) = self.compilation_stats.write() {
            if hit {
                stats.cache_hits += 1;
            } else {
                stats.cache_misses += 1;
            }
        }
    }

    /// Update compilation statistics
    fn update_compilation_stats(&self, compilation_time: Duration, successful: bool) {
        if let Ok(mut stats) = self.compilation_stats.write() {
            stats.total_compilations += 1;
            if successful {
                stats.successful_compilations += 1;
            }
            stats.total_compilation_time += compilation_time;
            stats.average_compilation_time =
                stats.total_compilation_time / stats.total_compilations as u32;
        }
    }

    /// Evict old variants from cache
    fn evict_old_variants(&self, variants: &mut HashMap<String, CompiledVariant>) {
        // Simple LRU eviction based on compilation time
        if let Some((oldest_key, _)) = variants
            .iter()
            .min_by_key(|(_, variant)| variant.compiled_at)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            variants.remove(&oldest_key);
        }
    }

    /// Get compilation statistics
    pub fn get_compilation_stats(&self) -> SklResult<CompilationStats> {
        self.compilation_stats
            .read()
            .map(|stats| stats.clone())
            .map_err(|_| SklearsError::InvalidInput("Failed to read compilation stats".to_string()))
    }

    /// Trigger profile-guided recompilation
    pub fn trigger_pgo_recompilation(
        &self,
        operation_id: &str,
        performance_profiles: &[PerformanceProfile],
    ) -> SklResult<()> {
        if !self.config.enable_pgo_recompilation {
            return Ok(());
        }

        // Analyze performance trends
        let avg_performance = performance_profiles
            .iter()
            .map(|p| p.metrics.execution_time.as_secs_f64())
            .sum::<f64>()
            / performance_profiles.len() as f64;

        // Find variants that could be improved
        if let Ok(mut variants) = self.compiled_variants.write() {
            for (key, variant) in variants.iter_mut() {
                if key.starts_with(operation_id) {
                    if let Some(ref profile) = variant.performance_profile {
                        let improvement_potential =
                            profile.metrics.execution_time.as_secs_f64() / avg_performance;
                        if improvement_potential > (1.0 + self.config.min_improvement) {
                            // Mark for recompilation
                            variant.compilation_successful = false;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for RuntimeOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_jit: true,
            max_variants: 100,
            compilation_timeout: Duration::from_secs(30),
            min_improvement: 0.1, // 10% improvement threshold
            enable_pgo_recompilation: true,
        }
    }
}

/// Advanced performance predictor using ensemble methods
#[derive(Debug)]
pub struct EnsemblePerformancePredictor {
    /// Individual predictors
    predictors: Vec<Box<dyn PerformancePredictor + Send + Sync>>,
    /// Predictor weights
    weights: Vec<f64>,
    /// Ensemble accuracy
    ensemble_accuracy: f64,
}

impl Default for EnsemblePerformancePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl EnsemblePerformancePredictor {
    /// Create a new ensemble predictor
    #[must_use]
    pub fn new() -> Self {
        let predictors: Vec<Box<dyn PerformancePredictor + Send + Sync>> = vec![
            Box::new(MLPerformancePredictor::new()),
            Box::new(HeuristicPredictor::new()),
            Box::new(PolynomialPredictor::new()),
        ];

        let weights = vec![1.0 / predictors.len() as f64; predictors.len()];

        Self {
            predictors,
            weights,
            ensemble_accuracy: 0.0,
        }
    }

    /// Update ensemble weights based on individual accuracy
    fn update_weights(&mut self) {
        let total_accuracy: f64 = self.predictors.iter().map(|p| p.accuracy()).sum();

        if total_accuracy > 0.0 {
            for (i, predictor) in self.predictors.iter().enumerate() {
                self.weights[i] = predictor.accuracy() / total_accuracy;
            }
        }

        // Calculate ensemble accuracy as weighted average
        self.ensemble_accuracy = self
            .predictors
            .iter()
            .enumerate()
            .map(|(i, p)| p.accuracy() * self.weights[i])
            .sum();
    }
}

impl PerformancePredictor for EnsemblePerformancePredictor {
    fn predict_execution_time(&self, characteristics: &DataCharacteristics) -> SklResult<Duration> {
        let mut weighted_prediction = 0.0;

        for (i, predictor) in self.predictors.iter().enumerate() {
            let prediction = predictor
                .predict_execution_time(characteristics)?
                .as_secs_f64();
            weighted_prediction += prediction * self.weights[i];
        }

        Ok(Duration::from_secs_f64(weighted_prediction.max(0.0)))
    }

    fn predict_memory_usage(&self, characteristics: &DataCharacteristics) -> SklResult<usize> {
        let mut weighted_prediction = 0.0;

        for (i, predictor) in self.predictors.iter().enumerate() {
            let prediction = predictor.predict_memory_usage(characteristics)? as f64;
            weighted_prediction += prediction * self.weights[i];
        }

        Ok(weighted_prediction.max(0.0) as usize)
    }

    fn update(&mut self, profile: &PerformanceProfile) -> SklResult<()> {
        for predictor in &mut self.predictors {
            predictor.update(profile)?;
        }

        self.update_weights();
        Ok(())
    }

    fn accuracy(&self) -> f64 {
        self.ensemble_accuracy
    }
}

/// Heuristic-based performance predictor
#[derive(Debug)]
pub struct HeuristicPredictor {
    accuracy: f64,
}

impl Default for HeuristicPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicPredictor {
    #[must_use]
    pub fn new() -> Self {
        Self { accuracy: 0.6 } // Fixed reasonable accuracy
    }
}

impl PerformancePredictor for HeuristicPredictor {
    fn predict_execution_time(&self, characteristics: &DataCharacteristics) -> SklResult<Duration> {
        let base_time = (characteristics.n_samples * characteristics.n_features) as f64;
        let sparsity_factor = 1.0 - characteristics.sparsity() * 0.5;
        let cache_factor = 1.0 + (1.0 - characteristics.cache_friendliness()) * 0.3;

        let estimated_time = base_time * sparsity_factor * cache_factor / 1e6; // Scale to seconds
        Ok(Duration::from_secs_f64(estimated_time.max(0.001)))
    }

    fn predict_memory_usage(&self, characteristics: &DataCharacteristics) -> SklResult<usize> {
        let base_memory =
            characteristics.n_samples * characteristics.n_features * characteristics.dtype_size;
        let overhead = (base_memory as f64 * 0.2) as usize; // 20% overhead
        Ok(base_memory + overhead)
    }

    fn update(&mut self, _profile: &PerformanceProfile) -> SklResult<()> {
        // Heuristic predictor doesn't learn, but we can adjust accuracy based on feedback
        Ok(())
    }

    fn accuracy(&self) -> f64 {
        self.accuracy
    }
}

/// Polynomial regression performance predictor
#[derive(Debug)]
pub struct PolynomialPredictor {
    coefficients: Vec<f64>,
    accuracy: f64,
    training_data: Vec<(Vec<f64>, f64)>,
}

impl Default for PolynomialPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl PolynomialPredictor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            coefficients: vec![1.0; 15], // Degree-2 polynomial features
            accuracy: 0.0,
            training_data: Vec::new(),
        }
    }

    fn polynomial_features(&self, characteristics: &DataCharacteristics) -> Vec<f64> {
        let n_samples = characteristics.n_samples as f64;
        let n_features = characteristics.n_features as f64;
        let sparsity = characteristics.sparsity();
        let cache_friendliness = characteristics.cache_friendliness();

        vec![
            1.0, // bias
            n_samples,
            n_features,
            sparsity,
            cache_friendliness,
            n_samples * n_features, // interaction terms
            n_samples * sparsity,
            n_features * sparsity,
            n_samples * cache_friendliness,
            n_features * cache_friendliness,
            sparsity * cache_friendliness,
            n_samples.powi(2), // quadratic terms
            n_features.powi(2),
            sparsity.powi(2),
            cache_friendliness.powi(2),
        ]
    }
}

impl PerformancePredictor for PolynomialPredictor {
    fn predict_execution_time(&self, characteristics: &DataCharacteristics) -> SklResult<Duration> {
        let features = self.polynomial_features(characteristics);
        let prediction = features
            .iter()
            .zip(&self.coefficients)
            .map(|(f, c)| f * c)
            .sum::<f64>()
            .max(0.001);

        Ok(Duration::from_secs_f64(prediction))
    }

    fn predict_memory_usage(&self, characteristics: &DataCharacteristics) -> SklResult<usize> {
        let base_memory =
            characteristics.n_samples * characteristics.n_features * characteristics.dtype_size;
        Ok(base_memory)
    }

    fn update(&mut self, profile: &PerformanceProfile) -> SklResult<()> {
        let features = self.polynomial_features(&profile.data_characteristics);
        let target = profile.metrics.execution_time.as_secs_f64();

        self.training_data.push((features, target));

        // Periodic retraining
        if self.training_data.len() % 20 == 0 {
            self.train_polynomial_regression()?;
        }

        Ok(())
    }

    fn accuracy(&self) -> f64 {
        self.accuracy
    }
}

impl PolynomialPredictor {
    fn train_polynomial_regression(&mut self) -> SklResult<()> {
        if self.training_data.len() < 10 {
            return Ok(());
        }

        // Simple normal equations solution for polynomial regression
        let n = self.training_data.len();
        let p = self.coefficients.len();

        // Build design matrix X and target vector y
        let mut x_matrix = vec![vec![0.0; p]; n];
        let mut y_vector = vec![0.0; n];

        for (i, (features, target)) in self.training_data.iter().enumerate() {
            for (j, &feature) in features.iter().enumerate() {
                x_matrix[i][j] = feature;
            }
            y_vector[i] = *target;
        }

        // Solve normal equations: (X^T X) β = X^T y
        // Simplified implementation using gradient descent
        let learning_rate = 0.0001;
        let epochs = 50;

        for _ in 0..epochs {
            let mut gradients = vec![0.0; p];
            let mut total_error = 0.0;

            for i in 0..n {
                let prediction: f64 = x_matrix[i]
                    .iter()
                    .zip(&self.coefficients)
                    .map(|(x, c)| x * c)
                    .sum();

                let error = prediction - y_vector[i];
                total_error += error * error;

                for j in 0..p {
                    gradients[j] += error * x_matrix[i][j];
                }
            }

            // Update coefficients
            for (coeff, grad) in self.coefficients.iter_mut().zip(&gradients) {
                *coeff -= learning_rate * grad / n as f64;
            }

            // Update accuracy
            let mse = total_error / n as f64;
            self.accuracy = (1.0 / (1.0 + mse)).min(1.0);
        }

        Ok(())
    }
}
