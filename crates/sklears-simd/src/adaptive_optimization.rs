//! Adaptive optimization and runtime algorithm selection
//!
//! Provides dynamic dispatch, auto-tuning capabilities, and machine learning-guided optimization
//! for SIMD operations based on runtime characteristics and performance feedback.
//!
//! ## no-std Compatibility
//!
//! This module supports both std and no-std environments through conditional compilation.
//!
//! ### Dependencies for no-std:
//! - `alloc` crate for collections and Arc
//! - `spin` crate for Mutex (add `spin = "0.9"` to Cargo.toml)
//! - `rand` crate for random number generation
//!
//! ### Features:
//! - Use `std` feature flag to enable std functionality
//! - Without `std` feature, timing functionality uses mock values
//! - SystemTime-based timestamps are disabled in no-std mode

use crate::SimdCapabilities;

// Required for no-std compatibility
#[cfg(feature = "no-std")]
extern crate alloc;

// Conditional imports for std vs no-std
#[cfg(not(feature = "no-std"))]
use std::{
    boxed::Box,
    collections::HashMap,
    fmt,
    string::ToString,
    sync::{Arc, Mutex},
    time::Duration,
};

#[cfg(feature = "no-std")]
use alloc::boxed::Box;
#[cfg(feature = "no-std")]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "no-std")]
use alloc::string::{String, ToString};
#[cfg(feature = "no-std")]
use alloc::sync::Arc;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(feature = "no-std")]
use core::fmt;
#[cfg(feature = "no-std")]
use spin::Mutex;

#[cfg(feature = "no-std")]
use core::time::Duration;
#[cfg(not(feature = "no-std"))]
use std::time::Instant;

// SystemTime is only available in std
#[cfg(not(feature = "no-std"))]
use std::time::SystemTime;

/// Runtime algorithm selector
pub struct AdaptiveOptimizer {
    capabilities: SimdCapabilities,
    performance_cache: Arc<Mutex<HashMap<String, AlgorithmPerformance>>>,
    auto_tuning_enabled: bool,
    learning_rate: f64,
}

/// Algorithm performance record
#[derive(Debug, Clone)]
pub struct AlgorithmPerformance {
    pub algorithm_name: String,
    pub avg_duration: Duration,
    pub sample_count: usize,
    pub data_size_range: (usize, usize),
    pub success_rate: f64,
    #[cfg(not(feature = "no-std"))]
    pub last_updated: SystemTime,
    #[cfg(feature = "no-std")]
    pub last_updated: (),
}

/// Dynamic dispatch strategy
#[derive(Debug, Clone)]
pub enum DispatchStrategy {
    /// Always use the fastest known algorithm
    AlwaysFastest,
    /// Use algorithm with best success rate
    MostReliable,
    /// Balance speed and reliability
    Balanced,
    /// Adapt based on data characteristics
    DataDriven,
    /// Machine learning guided selection
    MLGuided,
}

/// Algorithm variant for dynamic selection
pub trait AlgorithmVariant<T> {
    fn name(&self) -> &str;
    fn execute(&self, input: &T) -> Result<T, AlgorithmError>;
    fn is_applicable(&self, input: &T) -> bool;
    fn estimated_cost(&self, input: &T) -> f64;
}

/// Error type for algorithm execution
#[derive(Debug)]
pub enum AlgorithmError {
    UnsupportedInput,
    InsufficientResources,
    NumericError,
    RuntimeError(String),
}

impl core::fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AlgorithmError::UnsupportedInput => write!(f, "Unsupported input for algorithm"),
            AlgorithmError::InsufficientResources => write!(f, "Insufficient resources"),
            AlgorithmError::NumericError => write!(f, "Numeric computation error"),
            AlgorithmError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl std::error::Error for AlgorithmError {}

#[cfg(feature = "no-std")]
impl core::error::Error for AlgorithmError {}

impl Default for AdaptiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer
    pub fn new() -> Self {
        Self {
            capabilities: SimdCapabilities::detect(),
            performance_cache: Arc::new(Mutex::new(HashMap::new())),
            auto_tuning_enabled: true,
            learning_rate: 0.1,
        }
    }

    /// Helper function to handle mutex locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn lock_cache(&self) -> std::sync::MutexGuard<HashMap<String, AlgorithmPerformance>> {
        self.performance_cache.lock().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn lock_cache(&self) -> spin::MutexGuard<HashMap<String, AlgorithmPerformance>> {
        self.performance_cache.lock()
    }

    /// Helper function to handle mutable mutex locking in both std and no-std environments
    #[cfg(not(feature = "no-std"))]
    fn lock_cache_mut(&self) -> std::sync::MutexGuard<HashMap<String, AlgorithmPerformance>> {
        self.performance_cache.lock().unwrap()
    }

    #[cfg(feature = "no-std")]
    fn lock_cache_mut(&self) -> spin::MutexGuard<HashMap<String, AlgorithmPerformance>> {
        self.performance_cache.lock()
    }

    /// Enable or disable auto-tuning
    pub fn set_auto_tuning(&mut self, enabled: bool) {
        self.auto_tuning_enabled = enabled;
    }

    /// Set learning rate for performance adaptation
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }

    /// Select best algorithm variant based on strategy
    pub fn select_algorithm<'a, T>(
        &self,
        variants: &'a [Box<dyn AlgorithmVariant<T>>],
        input: &T,
        strategy: DispatchStrategy,
    ) -> Option<&'a Box<dyn AlgorithmVariant<T>>> {
        let applicable_variants: Vec<&Box<dyn AlgorithmVariant<T>>> = variants
            .iter()
            .filter(|variant| variant.is_applicable(input))
            .collect();

        if applicable_variants.is_empty() {
            return None;
        }

        match strategy {
            DispatchStrategy::AlwaysFastest => self.select_fastest(&applicable_variants, input),
            DispatchStrategy::MostReliable => self.select_most_reliable(&applicable_variants),
            DispatchStrategy::Balanced => self.select_balanced(&applicable_variants, input),
            DispatchStrategy::DataDriven => self.select_data_driven(&applicable_variants, input),
            DispatchStrategy::MLGuided => self.select_ml_guided(&applicable_variants, input),
        }
    }

    /// Execute algorithm with performance tracking
    pub fn execute_with_tracking<T>(
        &self,
        variant: &Box<dyn AlgorithmVariant<T>>,
        input: &T,
    ) -> Result<T, AlgorithmError> {
        #[cfg(not(feature = "no-std"))]
        let start = Instant::now();

        let result = variant.execute(input);

        #[cfg(not(feature = "no-std"))]
        let duration = start.elapsed();
        #[cfg(feature = "no-std")]
        let duration = Duration::from_millis(1); // Mock duration for no-std

        if self.auto_tuning_enabled {
            self.update_performance_stats(variant.name(), duration, result.is_ok());
        }

        result
    }

    /// Get performance statistics for an algorithm
    pub fn get_performance_stats(&self, algorithm_name: &str) -> Option<AlgorithmPerformance> {
        let cache = self.lock_cache();
        cache.get(algorithm_name).cloned()
    }

    /// Auto-tune parameters for an algorithm
    pub fn auto_tune_parameters<T, P>(
        &self,
        algorithm_factory: impl Fn(P) -> Box<dyn AlgorithmVariant<T>>,
        parameter_ranges: Vec<(P, P)>,
        test_inputs: &[T],
        iterations: usize,
    ) -> P
    where
        P: Clone + PartialOrd + fmt::Debug,
        T: Clone,
    {
        // Simple grid search for parameter tuning
        // In a real implementation, this could use more sophisticated optimization
        let mut best_params = parameter_ranges[0].0.clone();
        let mut best_performance = Duration::from_secs(u64::MAX);

        for _ in 0..iterations {
            for (min_param, _max_param) in &parameter_ranges {
                // For simplicity, just test the midpoint
                // A real implementation would use proper parameter sampling
                let test_param = min_param.clone(); // Simplified

                let algorithm = algorithm_factory(test_param.clone());
                let mut total_duration = Duration::from_nanos(0);
                let mut successful_runs = 0;

                for test_input in test_inputs {
                    #[cfg(not(feature = "no-std"))]
                    let start = Instant::now();

                    if algorithm.execute(test_input).is_ok() {
                        #[cfg(not(feature = "no-std"))]
                        {
                            total_duration += start.elapsed();
                        }
                        #[cfg(feature = "no-std")]
                        {
                            total_duration += Duration::from_millis(1); // Mock duration for no-std
                        }
                        successful_runs += 1;
                    }
                }

                if successful_runs > 0 {
                    let avg_duration = total_duration / successful_runs as u32;
                    if avg_duration < best_performance {
                        best_performance = avg_duration;
                        best_params = test_param;
                    }
                }
            }
        }

        best_params
    }

    fn select_fastest<'a, T>(
        &self,
        variants: &[&'a Box<dyn AlgorithmVariant<T>>],
        _input: &T,
    ) -> Option<&'a Box<dyn AlgorithmVariant<T>>> {
        let cache = self.lock_cache();

        variants
            .iter()
            .min_by_key(|variant| {
                cache
                    .get(variant.name())
                    .map(|perf| perf.avg_duration)
                    .unwrap_or(Duration::from_secs(u64::MAX))
            })
            .copied()
    }

    fn select_most_reliable<'a, T>(
        &self,
        variants: &[&'a Box<dyn AlgorithmVariant<T>>],
    ) -> Option<&'a Box<dyn AlgorithmVariant<T>>> {
        let cache = self.lock_cache();

        variants
            .iter()
            .max_by(|a, b| {
                let a_reliability = cache
                    .get(a.name())
                    .map(|perf| perf.success_rate)
                    .unwrap_or(0.0);
                let b_reliability = cache
                    .get(b.name())
                    .map(|perf| perf.success_rate)
                    .unwrap_or(0.0);
                a_reliability
                    .partial_cmp(&b_reliability)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .copied()
    }

    fn select_balanced<'a, T>(
        &self,
        variants: &[&'a Box<dyn AlgorithmVariant<T>>],
        _input: &T,
    ) -> Option<&'a Box<dyn AlgorithmVariant<T>>> {
        let cache = self.lock_cache();

        variants
            .iter()
            .max_by(|a, b| {
                let a_score = self.calculate_balanced_score(a.name(), &cache);
                let b_score = self.calculate_balanced_score(b.name(), &cache);
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .copied()
    }

    fn select_data_driven<'a, T>(
        &self,
        variants: &[&'a Box<dyn AlgorithmVariant<T>>],
        input: &T,
    ) -> Option<&'a Box<dyn AlgorithmVariant<T>>> {
        // Simple heuristic: prefer algorithms with lower estimated cost
        variants
            .iter()
            .min_by(|a, b| {
                let a_cost = a.estimated_cost(input);
                let b_cost = b.estimated_cost(input);
                a_cost
                    .partial_cmp(&b_cost)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .copied()
    }

    fn select_ml_guided<'a, T>(
        &self,
        variants: &[&'a Box<dyn AlgorithmVariant<T>>],
        input: &T,
    ) -> Option<&'a Box<dyn AlgorithmVariant<T>>> {
        // Simplified ML-guided selection
        // In a real implementation, this would use a trained model
        let cache = self.lock_cache();

        variants
            .iter()
            .max_by(|a, b| {
                let a_prediction = self.predict_performance(a.name(), input, &cache);
                let b_prediction = self.predict_performance(b.name(), input, &cache);
                a_prediction
                    .partial_cmp(&b_prediction)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .copied()
    }

    fn calculate_balanced_score(
        &self,
        algorithm_name: &str,
        cache: &HashMap<String, AlgorithmPerformance>,
    ) -> f64 {
        if let Some(perf) = cache.get(algorithm_name) {
            // Balance speed (1/duration) and reliability
            let speed_score = 1.0 / perf.avg_duration.as_secs_f64();
            let reliability_score = perf.success_rate;
            (speed_score * 0.6) + (reliability_score * 0.4)
        } else {
            0.0
        }
    }

    fn predict_performance<T>(
        &self,
        algorithm_name: &str,
        _input: &T,
        cache: &HashMap<String, AlgorithmPerformance>,
    ) -> f64 {
        // Simplified performance prediction
        // A real implementation would use features from the input and a trained model
        if let Some(perf) = cache.get(algorithm_name) {
            // Simple heuristic: recent performance with some randomness for exploration
            let base_score = 1.0 / perf.avg_duration.as_secs_f64() * perf.success_rate;
            let exploration_factor = {
                use scirs2_core::random::thread_rng;
                use scirs2_core::Rng;
                let mut rng = thread_rng();
                0.1 * rng.random::<f64>()
            };
            base_score + exploration_factor
        } else {
            // Unknown algorithm gets a default score
            0.5
        }
    }

    fn update_performance_stats(&self, algorithm_name: &str, duration: Duration, success: bool) {
        let mut cache = self.lock_cache_mut();

        let updated_perf = if let Some(mut perf) = cache.get(algorithm_name).cloned() {
            // Update existing performance record using exponential moving average
            let new_duration_secs = duration.as_secs_f64();
            let old_duration_secs = perf.avg_duration.as_secs_f64();
            let updated_duration_secs = old_duration_secs * (1.0 - self.learning_rate)
                + new_duration_secs * self.learning_rate;

            perf.avg_duration = Duration::from_secs_f64(updated_duration_secs);
            perf.sample_count += 1;

            let new_success_rate = if success { 1.0 } else { 0.0 };
            perf.success_rate = perf.success_rate * (1.0 - self.learning_rate)
                + new_success_rate * self.learning_rate;

            #[cfg(not(feature = "no-std"))]
            {
                perf.last_updated = SystemTime::now();
            }
            #[cfg(feature = "no-std")]
            {
                perf.last_updated = ();
            }

            perf
        } else {
            // Create new performance record
            AlgorithmPerformance {
                algorithm_name: algorithm_name.to_string(),
                avg_duration: duration,
                sample_count: 1,
                data_size_range: (0, 0), // Would be updated based on input characteristics
                success_rate: if success { 1.0 } else { 0.0 },
                #[cfg(not(feature = "no-std"))]
                last_updated: SystemTime::now(),
                #[cfg(feature = "no-std")]
                last_updated: (),
            }
        };

        cache.insert(algorithm_name.to_string(), updated_perf);
    }
}

/// Performance feedback loop for continuous optimization
pub struct PerformanceFeedbackLoop {
    optimizer: AdaptiveOptimizer,
    feedback_history: Vec<FeedbackRecord>,
    adaptation_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct FeedbackRecord {
    #[cfg(not(feature = "no-std"))]
    pub timestamp: SystemTime,
    #[cfg(feature = "no-std")]
    pub timestamp: (),
    pub algorithm_name: String,
    pub input_characteristics: String,
    pub performance_metric: f64,
    pub context: String,
}

impl Default for PerformanceFeedbackLoop {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceFeedbackLoop {
    /// Create a new feedback loop
    pub fn new() -> Self {
        Self {
            optimizer: AdaptiveOptimizer::new(),
            feedback_history: Vec::new(),
            adaptation_threshold: 0.05, // 5% performance change triggers adaptation
        }
    }

    /// Add performance feedback
    pub fn add_feedback(&mut self, record: FeedbackRecord) {
        self.feedback_history.push(record);

        // Trigger adaptation if we have enough feedback
        if self.feedback_history.len() % 10 == 0 {
            self.adapt_strategies();
        }
    }

    /// Analyze feedback and adapt optimization strategies
    fn adapt_strategies(&mut self) {
        // Analyze recent feedback to identify trends
        let recent_feedback: Vec<&FeedbackRecord> =
            self.feedback_history.iter().rev().take(20).collect();

        // Group by algorithm
        let mut algorithm_groups: HashMap<String, Vec<&FeedbackRecord>> = HashMap::new();
        for record in recent_feedback {
            algorithm_groups
                .entry(record.algorithm_name.clone())
                .or_default()
                .push(record);
        }

        // Adapt learning rate based on performance variance
        for (_algorithm_name, records) in algorithm_groups {
            if records.len() >= 3 {
                let metrics: Vec<f64> = records.iter().map(|r| r.performance_metric).collect();
                let variance = self.calculate_variance(&metrics);

                // If performance is highly variable, increase exploration
                if variance > self.adaptation_threshold {
                    self.optimizer.set_learning_rate(0.2);
                } else {
                    self.optimizer.set_learning_rate(0.05);
                }
            }
        }
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance
    }
}

/// Auto-tuning configuration
#[derive(Debug, Clone)]
pub struct AutoTuningConfig {
    pub enabled: bool,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub exploration_rate: f64,
    pub adaptation_interval: Duration,
}

impl Default for AutoTuningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_iterations: 100,
            convergence_threshold: 0.01,
            exploration_rate: 0.1,
            adaptation_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "no-std"))]
    use std::time::Duration;

    // Mock algorithm variant for testing
    struct MockAlgorithmVariant {
        name: String,
        execution_time: Duration,
        success_rate: f64,
    }

    impl AlgorithmVariant<Vec<f32>> for MockAlgorithmVariant {
        fn name(&self) -> &str {
            &self.name
        }

        fn execute(&self, _input: &Vec<f32>) -> Result<Vec<f32>, AlgorithmError> {
            #[cfg(not(feature = "no-std"))]
            std::thread::sleep(self.execution_time);

            let random_val = {
                use scirs2_core::random::thread_rng;
                use scirs2_core::Rng;
                let mut rng = thread_rng();
                rng.random::<f64>()
            };
            if random_val < self.success_rate {
                Ok(vec![1.0, 2.0, 3.0])
            } else {
                Err(AlgorithmError::RuntimeError("Mock error".to_string()))
            }
        }

        fn is_applicable(&self, input: &Vec<f32>) -> bool {
            !input.is_empty()
        }

        fn estimated_cost(&self, input: &Vec<f32>) -> f64 {
            input.len() as f64 * self.execution_time.as_secs_f64()
        }
    }

    #[test]
    fn test_adaptive_optimizer_creation() {
        let optimizer = AdaptiveOptimizer::new();
        assert!(optimizer.auto_tuning_enabled);
        assert_eq!(optimizer.learning_rate, 0.1);
    }

    #[test]
    fn test_algorithm_selection() {
        let optimizer = AdaptiveOptimizer::new();
        let input = vec![1.0, 2.0, 3.0];

        let variants: Vec<Box<dyn AlgorithmVariant<Vec<f32>>>> = vec![
            Box::new(MockAlgorithmVariant {
                name: "fast_algorithm".to_string(),
                execution_time: Duration::from_millis(10),
                success_rate: 0.9,
            }),
            Box::new(MockAlgorithmVariant {
                name: "slow_algorithm".to_string(),
                execution_time: Duration::from_millis(100),
                success_rate: 0.99,
            }),
        ];

        let selected =
            optimizer.select_algorithm(&variants, &input, DispatchStrategy::AlwaysFastest);

        assert!(selected.is_some());
    }

    #[test]
    fn test_performance_tracking() {
        let optimizer = AdaptiveOptimizer::new();
        let input = vec![1.0, 2.0, 3.0];

        let variant: Box<dyn AlgorithmVariant<Vec<f32>>> = Box::new(MockAlgorithmVariant {
            name: "test_algorithm".to_string(),
            execution_time: Duration::from_millis(1),
            success_rate: 1.0,
        });

        let result = optimizer.execute_with_tracking(&variant, &input);
        assert!(result.is_ok());

        // Check that performance stats were recorded
        let stats = optimizer.get_performance_stats("test_algorithm");
        assert!(stats.is_some());
    }

    #[test]
    fn test_feedback_loop() {
        let mut feedback_loop = PerformanceFeedbackLoop::new();

        let record = FeedbackRecord {
            #[cfg(not(feature = "no-std"))]
            timestamp: SystemTime::now(),
            #[cfg(feature = "no-std")]
            timestamp: (),
            algorithm_name: "test_algo".to_string(),
            input_characteristics: "small_data".to_string(),
            performance_metric: 0.5,
            context: "test".to_string(),
        };

        feedback_loop.add_feedback(record);
        assert_eq!(feedback_loop.feedback_history.len(), 1);
    }

    #[test]
    fn test_auto_tuning_config() {
        let config = AutoTuningConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.convergence_threshold, 0.01);
    }
}
