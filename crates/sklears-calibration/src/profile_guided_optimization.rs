//! Profile-Guided Optimization for Calibration
//!
//! This module implements profile-guided optimization capabilities for calibration
//! methods, including performance profiling, optimization hints, adaptive algorithms,
//! and runtime optimization based on usage patterns.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result},
    types::Float,
};
use std::sync::{Arc};
use std::time::{Instant};

use crate::{CalibrationEstimator};

/// Performance profile for calibration operations
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Operation timings
    pub timings: HashMap<String, Vec<Duration>>,
    /// Memory usage statistics
    pub memory_usage: HashMap<String, Vec<usize>>,
    /// Algorithm effectiveness scores
    pub effectiveness_scores: HashMap<String, Vec<Float>>,
    /// Input size distribution
    pub input_sizes: Vec<usize>,
    /// Error rates
    pub error_rates: HashMap<String, Float>,
    /// Cache hit rates
    pub cache_hit_rates: HashMap<String, Float>,
}

impl PerformanceProfile {
    /// Create a new performance profile
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory_usage: HashMap::new(),
            effectiveness_scores: HashMap::new(),
            input_sizes: Vec::new(),
            error_rates: HashMap::new(),
            cache_hit_rates: HashMap::new(),
        }
    }

    /// Add timing measurement
    pub fn add_timing(&mut self, operation: &str, duration: Duration) {
        self.timings.entry(operation.to_string()).or_insert_with(Vec::new).push(duration);
    }

    /// Add memory usage measurement
    pub fn add_memory_usage(&mut self, operation: &str, bytes: usize) {
        self.memory_usage.entry(operation.to_string()).or_insert_with(Vec::new).push(bytes);
    }

    /// Add effectiveness score
    pub fn add_effectiveness_score(&mut self, algorithm: &str, score: Float) {
        self.effectiveness_scores.entry(algorithm.to_string()).or_insert_with(Vec::new).push(score);
    }

    /// Add input size
    pub fn add_input_size(&mut self, size: usize) {
        self.input_sizes.push(size);
    }

    /// Set error rate
    pub fn set_error_rate(&mut self, operation: &str, rate: Float) {
        self.error_rates.insert(operation.to_string(), rate);
    }

    /// Set cache hit rate
    pub fn set_cache_hit_rate(&mut self, cache_name: &str, rate: Float) {
        self.cache_hit_rates.insert(cache_name.to_string(), rate);
    }

    /// Get average timing for operation
    pub fn average_timing(&self, operation: &str) -> Option<Duration> {
        let timings = self.timings.get(operation)?;
        if timings.is_empty() {
            return None;
        }
        
        let total_nanos: u64 = timings.iter().map(|d| d.as_nanos() as u64).sum();
        let avg_nanos = total_nanos / timings.len() as u64;
        Some(Duration::from_nanos(avg_nanos))
    }

    /// Get average memory usage for operation
    pub fn average_memory_usage(&self, operation: &str) -> Option<usize> {
        let usage = self.memory_usage.get(operation)?;
        if usage.is_empty() {
            return None;
        }
        
        let total: usize = usage.iter().sum();
        Some(total / usage.len())
    }

    /// Get average effectiveness score for algorithm
    pub fn average_effectiveness(&self, algorithm: &str) -> Option<Float> {
        let scores = self.effectiveness_scores.get(algorithm)?;
        if scores.is_empty() {
            return None;
        }
        
        let total: Float = scores.iter().sum();
        Some(total / scores.len() as Float)
    }

    /// Get input size distribution statistics
    pub fn input_size_stats(&self) -> Option<(usize, usize, Float)> {
        if self.input_sizes.is_empty() {
            return None;
        }
        
        let min_size = *self.input_sizes.iter().min()?;
        let max_size = *self.input_sizes.iter().max()?;
        let avg_size = self.input_sizes.iter().sum::<usize>() as Float / self.input_sizes.len() as Float;
        
        Some((min_size, max_size, avg_size))
    }
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Profile-guided optimizer for calibration methods
#[derive(Debug)]
pub struct ProfileGuidedOptimizer {
    /// Performance profiles
    profiles: RwLock<HashMap<String, PerformanceProfile>>,
    /// Optimization rules
    rules: RwLock<Vec<OptimizationRule>>,
    /// Adaptive algorithms
    adaptive_algorithms: RwLock<HashMap<String, Box<dyn AdaptiveAlgorithm>>>,
    /// Configuration
    config: OptimizerConfig,
}

/// Optimization rule
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    /// Rule name
    pub name: String,
    /// Condition for applying the rule
    pub condition: OptimizationCondition,
    /// Action to take when condition is met
    pub action: OptimizationAction,
    /// Priority (higher values = higher priority)
    pub priority: i32,
    /// Whether the rule is enabled
    pub enabled: bool,
}

/// Optimization condition
#[derive(Debug, Clone)]
pub enum OptimizationCondition {
    /// Timing threshold
    TimingThreshold { operation: String, threshold: Duration },
    /// Memory usage threshold
    MemoryThreshold { operation: String, threshold: usize },
    /// Effectiveness threshold
    EffectivenessThreshold { algorithm: String, threshold: Float },
    /// Input size range
    InputSizeRange { min_size: usize, max_size: usize },
    /// Error rate threshold
    ErrorRateThreshold { operation: String, threshold: Float },
    /// Cache hit rate threshold
    CacheHitRateThreshold { cache_name: String, threshold: Float },
    /// Custom condition
    Custom(String),
}

/// Optimization action
#[derive(Debug, Clone)]
pub enum OptimizationAction {
    /// Switch algorithm
    SwitchAlgorithm { from: String, to: String },
    /// Adjust parameters
    AdjustParameters { algorithm: String, adjustments: HashMap<String, Float> },
    /// Enable caching
    EnableCaching { cache_name: String, capacity: usize },
    /// Increase parallelism
    IncreaseParallelism { threads: usize },
    /// Reduce precision
    ReducePrecision { algorithm: String, precision: u8 },
    /// Precompute values
    PrecomputeValues { algorithm: String, values: Vec<String> },
    /// Custom action
    Custom(String),
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable profiling
    pub enable_profiling: bool,
    /// Profile collection interval
    pub profile_interval: Duration,
    /// Maximum profile history
    pub max_profile_history: usize,
    /// Optimization frequency
    pub optimization_frequency: Duration,
    /// Enable adaptive algorithms
    pub enable_adaptive: bool,
    /// Memory limit for caching
    pub memory_limit: usize,
}

impl ProfileGuidedOptimizer {
    /// Create a new profile-guided optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            profiles: RwLock::new(HashMap::new()),
            rules: RwLock::new(Vec::new()),
            adaptive_algorithms: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Add optimization rule
    pub fn add_rule(&self, rule: OptimizationRule) {
        let mut rules = self.rules.write().unwrap();
        rules.push(rule);
        // Sort by priority (descending)
        rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Register adaptive algorithm
    pub fn register_adaptive_algorithm(&self, name: String, algorithm: Box<dyn AdaptiveAlgorithm>) {
        let mut algorithms = self.adaptive_algorithms.write().unwrap();
        algorithms.insert(name, algorithm);
    }

    /// Update performance profile
    pub fn update_profile(&self, profile_name: &str, profile: PerformanceProfile) {
        let mut profiles = self.profiles.write().unwrap();
        profiles.insert(profile_name.to_string(), profile);
    }

    /// Get performance profile
    pub fn get_profile(&self, profile_name: &str) -> Option<PerformanceProfile> {
        let profiles = self.profiles.read().unwrap();
        profiles.get(profile_name).cloned()
    }

    /// Apply optimizations based on profiles
    pub fn optimize(&self, calibrator_name: &str) -> Result<Vec<OptimizationAction>> {
        let profiles = self.profiles.read().unwrap();
        let rules = self.rules.read().unwrap();
        
        let profile = profiles.get(calibrator_name)
            .ok_or_else(|| SklearsError::InvalidInput(
                format!("No profile found for calibrator '{}'", calibrator_name)
            ))?;

        let mut applied_actions = Vec::new();

        // Check each rule
        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }

            if self.check_condition(&rule.condition, profile) {
                applied_actions.push(rule.action.clone());
            }
        }

        Ok(applied_actions)
    }

    /// Check if optimization condition is met
    fn check_condition(&self, condition: &OptimizationCondition, profile: &PerformanceProfile) -> bool {
        match condition {
            OptimizationCondition::TimingThreshold { operation, threshold } => {
                if let Some(avg_timing) = profile.average_timing(operation) {
                    avg_timing > *threshold
                } else {
                    false
                }
            }
            OptimizationCondition::MemoryThreshold { operation, threshold } => {
                if let Some(avg_memory) = profile.average_memory_usage(operation) {
                    avg_memory > *threshold
                } else {
                    false
                }
            }
            OptimizationCondition::EffectivenessThreshold { algorithm, threshold } => {
                if let Some(avg_effectiveness) = profile.average_effectiveness(algorithm) {
                    avg_effectiveness < *threshold
                } else {
                    false
                }
            }
            OptimizationCondition::InputSizeRange { min_size, max_size } => {
                if let Some((min, max, _)) = profile.input_size_stats() {
                    min >= *min_size && max <= *max_size
                } else {
                    false
                }
            }
            OptimizationCondition::ErrorRateThreshold { operation, threshold } => {
                if let Some(&error_rate) = profile.error_rates.get(operation) {
                    error_rate > *threshold
                } else {
                    false
                }
            }
            OptimizationCondition::CacheHitRateThreshold { cache_name, threshold } => {
                if let Some(&hit_rate) = profile.cache_hit_rates.get(cache_name) {
                    hit_rate < *threshold
                } else {
                    false
                }
            }
            OptimizationCondition::Custom(_) => {
                // Custom conditions would be handled by specific implementations
                false
            }
        }
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self, calibrator_name: &str) -> Result<Vec<String>> {
        let profiles = self.profiles.read().unwrap();
        let profile = profiles.get(calibrator_name)
            .ok_or_else(|| SklearsError::InvalidInput(
                format!("No profile found for calibrator '{}'", calibrator_name)
            ))?;

        let mut recommendations = Vec::new();

        // Check for performance bottlenecks
        for (operation, timings) in &profile.timings {
            if let Some(avg_timing) = profile.average_timing(operation) {
                if avg_timing > Duration::from_millis(100) {
                    recommendations.push(format!(
                        "Consider optimizing '{}' operation (average time: {:?})",
                        operation, avg_timing
                    ));
                }
            }
        }

        // Check for memory usage
        for (operation, _) in &profile.memory_usage {
            if let Some(avg_memory) = profile.average_memory_usage(operation) {
                if avg_memory > 1024 * 1024 { // 1MB threshold
                    recommendations.push(format!(
                        "High memory usage in '{}' operation ({} bytes)",
                        operation, avg_memory
                    ));
                }
            }
        }

        // Check effectiveness scores
        for (algorithm, _) in &profile.effectiveness_scores {
            if let Some(avg_effectiveness) = profile.average_effectiveness(algorithm) {
                if avg_effectiveness < 0.8 {
                    recommendations.push(format!(
                        "Algorithm '{}' has low effectiveness score ({:.3})",
                        algorithm, avg_effectiveness
                    ));
                }
            }
        }

        Ok(recommendations)
    }

    /// Generate optimization report
    pub fn generate_report(&self) -> String {
        let profiles = self.profiles.read().unwrap();
        let mut report = String::new();

        report.push_str("# Profile-Guided Optimization Report\n\n");

        for (name, profile) in profiles.iter() {
            report.push_str(&format!("## Profile: {}\n\n", name));

            // Timing statistics
            report.push_str("### Timing Statistics\n");
            for (operation, _) in &profile.timings {
                if let Some(avg_timing) = profile.average_timing(operation) {
                    report.push_str(&format!("- {}: {:?}\n", operation, avg_timing));
                }
            }
            report.push('\n');

            // Memory usage statistics
            report.push_str("### Memory Usage Statistics\n");
            for (operation, _) in &profile.memory_usage {
                if let Some(avg_memory) = profile.average_memory_usage(operation) {
                    report.push_str(&format!("- {}: {} bytes\n", operation, avg_memory));
                }
            }
            report.push('\n');

            // Effectiveness scores
            report.push_str("### Effectiveness Scores\n");
            for (algorithm, _) in &profile.effectiveness_scores {
                if let Some(avg_effectiveness) = profile.average_effectiveness(algorithm) {
                    report.push_str(&format!("- {}: {:.3}\n", algorithm, avg_effectiveness));
                }
            }
            report.push('\n');

            // Input size statistics
            if let Some((min_size, max_size, avg_size)) = profile.input_size_stats() {
                report.push_str("### Input Size Statistics\n");
                report.push_str(&format!("- Min: {}\n", min_size));
                report.push_str(&format!("- Max: {}\n", max_size));
                report.push_str(&format!("- Average: {:.1}\n\n", avg_size));
            }
        }

        report
    }
}

impl Default for ProfileGuidedOptimizer {
    fn default() -> Self {
        let config = OptimizerConfig {
            enable_profiling: true,
            profile_interval: Duration::from_secs(60),
            max_profile_history: 1000,
            optimization_frequency: Duration::from_secs(300),
            enable_adaptive: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB
        };
        Self::new(config)
    }
}

/// Adaptive algorithm trait
pub trait AdaptiveAlgorithm: Send + Sync {
    /// Algorithm name
    fn name(&self) -> &str;
    
    /// Adapt algorithm based on performance profile
    fn adapt(&mut self, profile: &PerformanceProfile) -> Result<()>;
    
    /// Get current parameters
    fn get_parameters(&self) -> HashMap<String, Float>;
    
    /// Set parameters
    fn set_parameters(&mut self, parameters: HashMap<String, Float>) -> Result<()>;
    
    /// Reset to default parameters
    fn reset(&mut self);
}

/// Profiling wrapper for calibration estimators
#[derive(Debug)]
pub struct ProfilingCalibrator {
    /// Inner calibrator
    inner: Box<dyn CalibrationEstimator>,
    /// Performance profile
    profile: Arc<Mutex<PerformanceProfile>>,
    /// Calibrator name
    name: String,
    /// Enable profiling
    enable_profiling: bool,
}

impl ProfilingCalibrator {
    /// Create a new profiling calibrator
    pub fn new(inner: Box<dyn CalibrationEstimator>, name: String) -> Self {
        Self {
            inner,
            profile: Arc::new(Mutex::new(PerformanceProfile::new())),
            name,
            enable_profiling: true,
        }
    }

    /// Enable or disable profiling
    pub fn set_profiling_enabled(&mut self, enabled: bool) {
        self.enable_profiling = enabled;
    }

    /// Get performance profile
    pub fn get_profile(&self) -> PerformanceProfile {
        let profile = self.profile.lock().unwrap();
        profile.clone()
    }

    /// Clear performance profile
    pub fn clear_profile(&self) {
        let mut profile = self.profile.lock().unwrap();
        *profile = PerformanceProfile::new();
    }

    /// Record timing
    fn record_timing(&self, operation: &str, duration: Duration) {
        if self.enable_profiling {
            let mut profile = self.profile.lock().unwrap();
            profile.add_timing(operation, duration);
        }
    }

    /// Record memory usage
    fn record_memory_usage(&self, operation: &str, bytes: usize) {
        if self.enable_profiling {
            let mut profile = self.profile.lock().unwrap();
            profile.add_memory_usage(operation, bytes);
        }
    }

    /// Record input size
    fn record_input_size(&self, size: usize) {
        if self.enable_profiling {
            let mut profile = self.profile.lock().unwrap();
            profile.add_input_size(size);
        }
    }
}

impl CalibrationEstimator for ProfilingCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        let start_time = Instant::now();
        self.record_input_size(probabilities.len());
        
        let result = self.inner.fit(probabilities, y_true);
        
        let duration = start_time.elapsed();
        self.record_timing("fit", duration);
        
        // Estimate memory usage (simplified)
        let memory_usage = probabilities.len() * std::mem::size_of::<Float>() + 
                          y_true.len() * std::mem::size_of::<i32>();
        self.record_memory_usage("fit", memory_usage);
        
        result
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let start_time = Instant::now();
        self.record_input_size(probabilities.len());
        
        let result = self.inner.predict_proba(probabilities);
        
        let duration = start_time.elapsed();
        self.record_timing("predict_proba", duration);
        
        // Estimate memory usage
        let memory_usage = probabilities.len() * std::mem::size_of::<Float>() * 2; // Input + output
        self.record_memory_usage("predict_proba", memory_usage);
        
        result
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(ProfilingCalibrator::new(
            self.inner.clone_box(),
            self.name.clone(),
        ))
    }
}

/// Optimization cache for storing computed values
#[derive(Debug)]
pub struct OptimizationCache {
    /// Cached values
    cache: RwLock<HashMap<String, CacheEntry>>,
    /// Maximum cache size
    max_size: usize,
    /// Cache hit count
    hits: RwLock<u64>,
    /// Cache miss count
    misses: RwLock<u64>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    value: Array1<Float>,
    timestamp: Instant,
    access_count: u64,
}

impl OptimizationCache {
    /// Create a new optimization cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size,
            hits: RwLock::new(0),
            misses: RwLock::new(0),
        }
    }

    /// Get value from cache
    pub fn get(&self, key: &str) -> Option<Array1<Float>> {
        let mut cache = self.cache.write().unwrap();
        if let Some(entry) = cache.get_mut(key) {
            entry.access_count += 1;
            *self.hits.write().unwrap() += 1;
            Some(entry.value.clone())
        } else {
            *self.misses.write().unwrap() += 1;
            None
        }
    }

    /// Put value in cache
    pub fn put(&self, key: String, value: Array1<Float>) {
        let mut cache = self.cache.write().unwrap();
        
        // Evict if necessary
        if cache.len() >= self.max_size {
            self.evict_lru(&mut cache);
        }
        
        let entry = CacheEntry {
            value,
            timestamp: Instant::now(),
            access_count: 1,
        };
        
        cache.insert(key, entry);
    }

    /// Evict least recently used entry
    fn evict_lru(&self, cache: &mut HashMap<String, CacheEntry>) {
        if let Some((key_to_remove, _)) = cache.iter()
            .min_by_key(|(_, entry)| (entry.access_count, entry.timestamp))
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            cache.remove(&key_to_remove);
        }
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> Float {
        let hits = *self.hits.read().unwrap();
        let misses = *self.misses.read().unwrap();
        let total = hits + misses;
        
        if total > 0 {
            hits as Float / total as Float
        } else {
            0.0
        }
    }

    /// Clear cache
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
        *self.hits.write().unwrap() = 0;
        *self.misses.write().unwrap() = 0;
    }
}

/// Example adaptive algorithm implementation
#[derive(Debug)]
pub struct AdaptiveSigmoidCalibrator {
    name: String,
    learning_rate: Float,
    max_iterations: usize,
    tolerance: Float,
}

impl AdaptiveSigmoidCalibrator {
    pub fn new() -> Self {
        Self {
            name: "adaptive_sigmoid".to_string(),
            learning_rate: 0.01,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl AdaptiveAlgorithm for AdaptiveSigmoidCalibrator {
    fn name(&self) -> &str {
        &self.name
    }

    fn adapt(&mut self, profile: &PerformanceProfile) -> Result<()> {
        // Adapt learning rate based on timing performance
        if let Some(avg_timing) = profile.average_timing("fit") {
            if avg_timing > Duration::from_millis(1000) {
                // If training is too slow, increase learning rate
                self.learning_rate = (self.learning_rate * 1.1).min(0.1);
            } else if avg_timing < Duration::from_millis(100) {
                // If training is fast, we can afford to be more precise
                self.learning_rate = (self.learning_rate * 0.0.9).max(0.001);
            }
        }

        // Adapt iterations based on effectiveness
        if let Some(avg_effectiveness) = profile.average_effectiveness(&self.name) {
            if avg_effectiveness < 0.8 {
                // Increase iterations if effectiveness is low
                self.max_iterations = (self.max_iterations * 2).min(1000);
            } else if avg_effectiveness > 0.95 {
                // Decrease iterations if effectiveness is very high
                self.max_iterations = (self.max_iterations / 2).max(10);
            }
        }

        Ok(())
    }

    fn get_parameters(&self) -> HashMap<String, Float> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("max_iterations".to_string(), self.max_iterations as Float);
        params.insert("tolerance".to_string(), self.tolerance);
        params
    }

    fn set_parameters(&mut self, parameters: HashMap<String, Float>) -> Result<()> {
        if let Some(&lr) = parameters.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&max_iter) = parameters.get("max_iterations") {
            self.max_iterations = max_iter as usize;
        }
        if let Some(&tol) = parameters.get("tolerance") {
            self.tolerance = tol;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.learning_rate = 0.01;
        self.max_iterations = 100;
        self.tolerance = 1e-6;
    }
}

impl Default for AdaptiveSigmoidCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::SigmoidCalibrator;

    #[test]
    fn test_performance_profile() {
        let mut profile = PerformanceProfile::new();
        
        profile.add_timing("fit", Duration::from_millis(100));
        profile.add_timing("fit", Duration::from_millis(120));
        profile.add_memory_usage("fit", 1024);
        profile.add_effectiveness_score("sigmoid", 0.85);
        profile.add_input_size(1000);

        assert_eq!(profile.average_timing("fit"), Some(Duration::from_millis(110)));
        assert_eq!(profile.average_memory_usage("fit"), Some(1024));
        assert_eq!(profile.average_effectiveness("sigmoid"), Some(0.85));
        assert_eq!(profile.input_size_stats(), Some((1000, 1000, 1000.0)));
    }

    #[test]
    fn test_profile_guided_optimizer() {
        let config = OptimizerConfig {
            enable_profiling: true,
            profile_interval: Duration::from_secs(1),
            max_profile_history: 100,
            optimization_frequency: Duration::from_secs(5),
            enable_adaptive: true,
            memory_limit: 1024 * 1024,
        };
        
        let optimizer = ProfileGuidedOptimizer::new(config);
        
        // Add optimization rule
        let rule = OptimizationRule {
            name: "slow_fit_rule".to_string(),
            condition: OptimizationCondition::TimingThreshold {
                operation: "fit".to_string(),
                threshold: Duration::from_millis(50),
            },
            action: OptimizationAction::AdjustParameters {
                algorithm: "sigmoid".to_string(),
                adjustments: {
                    let mut map = HashMap::new();
                    map.insert("learning_rate".to_string(), 0.1);
                    map
                },
            },
            priority: 1,
            enabled: true,
        };
        
        optimizer.add_rule(rule);

        // Create and update profile
        let mut profile = PerformanceProfile::new();
        profile.add_timing("fit", Duration::from_millis(100));
        optimizer.update_profile("test_calibrator", profile);

        // Test optimization
        let actions = optimizer.optimize("test_calibrator").unwrap();
        assert_eq!(actions.len(), 1);
        
        match &actions[0] {
            OptimizationAction::AdjustParameters { algorithm, adjustments } => {
                assert_eq!(algorithm, "sigmoid");
                assert_eq!(adjustments.get("learning_rate"), Some(&0.1));
            }
            _ => panic!("Expected AdjustParameters action"),
        }
    }

    #[test]
    fn test_profiling_calibrator() {
        let inner = Box::new(SigmoidCalibrator::new());
        let mut profiling_calibrator = ProfilingCalibrator::new(inner, "test".to_string());
        
        let probabilities = Array1::from(vec![0.1, 0.3, 0.7, 0.9]);
        let targets = Array1::from(vec![0, 0, 1, 1]);

        // Fit and predict
        profiling_calibrator.fit(&probabilities, &targets).unwrap();
        let predictions = profiling_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());

        // Check profile
        let profile = profiling_calibrator.get_profile();
        assert!(profile.timings.contains_key("fit"));
        assert!(profile.timings.contains_key("predict_proba"));
        assert!(profile.memory_usage.contains_key("fit"));
        assert!(profile.memory_usage.contains_key("predict_proba"));
    }

    #[test]
    fn test_optimization_cache() {
        let cache = OptimizationCache::new(2);
        
        let key1 = "test_key_1".to_string();
        let value1 = Array1::from(vec![0.1, 0.2, 0.3]);
        cache.put(key1.clone(), value1.clone());

        let retrieved = cache.get(&key1).unwrap();
        assert_eq!(retrieved, value1);
        assert!(cache.hit_rate() > 0.0);

        // Test cache eviction
        let key2 = "test_key_2".to_string();
        let value2 = Array1::from(vec![0.4, 0.5, 0.6]);
        cache.put(key2.clone(), value2.clone());

        let key3 = "test_key_3".to_string();
        let value3 = Array1::from(vec![0.7, 0.8, 0.9]);
        cache.put(key3, value3);

        // key1 should be evicted (LRU)
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_some());
    }

    #[test]
    fn test_adaptive_algorithm() {
        let mut adaptive = AdaptiveSigmoidCalibrator::new();
        
        let mut profile = PerformanceProfile::new();
        profile.add_timing("fit", Duration::from_millis(2000)); // Slow training
        profile.add_effectiveness_score("adaptive_sigmoid", 0.7); // Low effectiveness

        adaptive.adapt(&profile).unwrap();

        let params = adaptive.get_parameters();
        assert!(params["learning_rate"] > 0.01); // Should increase learning rate
        assert!(params["max_iterations"] > 100.0); // Should increase iterations
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = OptimizerConfig::default();
        let optimizer = ProfileGuidedOptimizer::new(config);
        
        let mut profile = PerformanceProfile::new();
        profile.add_timing("fit", Duration::from_millis(200)); // Slow operation
        profile.add_memory_usage("fit", 2 * 1024 * 1024); // High memory usage
        profile.add_effectiveness_score("test_algo", 0.6); // Low effectiveness
        
        optimizer.update_profile("test_calibrator", profile);
        
        let recommendations = optimizer.get_recommendations("test_calibrator").unwrap();
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("optimizing 'fit' operation")));
        assert!(recommendations.iter().any(|r| r.contains("High memory usage")));
        assert!(recommendations.iter().any(|r| r.contains("low effectiveness score")));
    }
}