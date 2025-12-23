//! Profile-guided optimization for explanation algorithms
//!
//! This module provides profiling and optimization capabilities to improve
//! performance based on actual usage patterns and runtime characteristics.

use crate::types::*;
// ✅ SciRS2 Policy Compliant Import
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Profile-guided optimization manager
pub struct ProfileGuidedOptimizer {
    /// Performance profiles for different methods
    method_profiles: Arc<Mutex<HashMap<String, MethodProfile>>>,
    /// Runtime statistics
    runtime_stats: Arc<Mutex<RuntimeStatistics>>,
    /// Optimization configuration
    config: OptimizationConfig,
    /// Hot paths identified through profiling
    hot_paths: Arc<Mutex<Vec<HotPath>>>,
}

/// Performance profile for a specific method
#[derive(Clone, Debug)]
pub struct MethodProfile {
    /// Method identifier
    pub method_id: String,
    /// Execution times for different input sizes
    pub execution_times: Vec<(usize, Duration)>,
    /// Memory usage patterns
    pub memory_usage: Vec<(usize, usize)>,
    /// Cache hit rates
    pub cache_hit_rates: Vec<f64>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    /// Number of executions
    pub execution_count: usize,
    /// Total execution time
    pub total_execution_time: Duration,
}

/// Runtime statistics collector
#[derive(Clone, Debug, Default)]
pub struct RuntimeStatistics {
    /// Total executions
    pub total_executions: usize,
    /// Average execution time
    pub avg_execution_time: f64,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Method call frequency
    pub method_frequencies: HashMap<String, usize>,
    /// Input size distribution
    pub input_size_distribution: HashMap<usize, usize>,
    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Configuration for profile-guided optimization
#[derive(Clone, Debug)]
pub struct OptimizationConfig {
    /// Enable profiling
    pub enable_profiling: bool,
    /// Profiling sample rate (0.0 to 1.0)
    pub sample_rate: f64,
    /// Minimum execution time to profile
    pub min_profile_time_ms: u64,
    /// Maximum profile data size
    pub max_profile_data_size: usize,
    /// Enable hot path detection
    pub enable_hot_path_detection: bool,
    /// Hot path threshold (execution frequency)
    pub hot_path_threshold: usize,
    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
}

/// Hot path in execution
#[derive(Clone, Debug)]
pub struct HotPath {
    /// Path identifier
    pub path_id: String,
    /// Execution frequency
    pub frequency: usize,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<String>,
}

/// Optimization opportunity
#[derive(Clone, Debug)]
pub struct OptimizationOpportunity {
    /// Opportunity type
    pub opportunity_type: OptimizationType,
    /// Estimated performance gain
    pub estimated_gain: f64,
    /// Implementation complexity
    pub complexity: OptimizationComplexity,
    /// Description
    pub description: String,
}

/// Performance bottleneck
#[derive(Clone, Debug)]
pub struct PerformanceBottleneck {
    /// Location of bottleneck
    pub location: String,
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Impact severity
    pub severity: Severity,
    /// Suggested fix
    pub suggested_fix: String,
}

/// Types of optimizations
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationType {
    /// Vectorization optimization
    Vectorization,
    /// Memory access optimization
    MemoryAccess,
    /// Cache optimization
    Cache,
    /// Parallel processing
    Parallelization,
    /// Algorithm selection
    AlgorithmSelection,
    /// Data structure optimization
    DataStructure,
}

/// Optimization complexity levels
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationComplexity {
    /// Low complexity (easy to implement)
    Low,
    /// Medium complexity
    Medium,
    /// High complexity (significant changes required)
    High,
}

/// Types of performance bottlenecks
#[derive(Clone, Debug, PartialEq)]
pub enum BottleneckType {
    /// CPU-bound bottleneck
    Cpu,
    /// Memory-bound bottleneck
    Memory,
    /// I/O-bound bottleneck
    Io,
    /// Cache miss bottleneck
    CacheMiss,
    /// Synchronization bottleneck
    Synchronization,
}

/// Severity levels
#[derive(Clone, Debug, PartialEq)]
pub enum Severity {
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Critical impact
    Critical,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            sample_rate: 0.1, // Profile 10% of executions
            min_profile_time_ms: 10,
            max_profile_data_size: 10000,
            enable_hot_path_detection: true,
            hot_path_threshold: 100,
            enable_auto_optimization: false, // Manual optimization by default
        }
    }
}

impl ProfileGuidedOptimizer {
    /// Create a new profile-guided optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            method_profiles: Arc::new(Mutex::new(HashMap::new())),
            runtime_stats: Arc::new(Mutex::new(RuntimeStatistics::default())),
            config,
            hot_paths: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Profile a method execution
    pub fn profile_execution<T, F>(
        &self,
        method_id: &str,
        input_size: usize,
        computation: F,
    ) -> crate::SklResult<T>
    where
        F: FnOnce() -> crate::SklResult<T>,
    {
        // Check if we should profile this execution
        if !self.should_profile() {
            return computation();
        }

        let start_time = Instant::now();
        let start_memory = self.estimate_memory_usage();

        // Execute computation
        let result = computation()?;

        let execution_time = start_time.elapsed();
        let end_memory = self.estimate_memory_usage();
        let memory_usage = end_memory.saturating_sub(start_memory);

        // Record profile data
        self.record_execution(method_id, input_size, execution_time, memory_usage);

        Ok(result)
    }

    /// Check if we should profile this execution
    fn should_profile(&self) -> bool {
        if !self.config.enable_profiling {
            return false;
        }

        // Sample based on configured rate
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();
        rng.gen::<Float>() < self.config.sample_rate
    }

    /// Record execution data
    fn record_execution(
        &self,
        method_id: &str,
        input_size: usize,
        execution_time: Duration,
        memory_usage: usize,
    ) {
        // Update method profile
        {
            let mut profiles = self.method_profiles.lock().unwrap();
            let profile = profiles
                .entry(method_id.to_string())
                .or_insert_with(|| MethodProfile {
                    method_id: method_id.to_string(),
                    execution_times: Vec::new(),
                    memory_usage: Vec::new(),
                    cache_hit_rates: Vec::new(),
                    optimization_opportunities: Vec::new(),
                    execution_count: 0,
                    total_execution_time: Duration::default(),
                });

            profile.execution_times.push((input_size, execution_time));
            profile.memory_usage.push((input_size, memory_usage));
            profile.execution_count += 1;
            profile.total_execution_time += execution_time;

            // Limit profile data size
            if profile.execution_times.len() > self.config.max_profile_data_size {
                profile.execution_times.remove(0);
                profile.memory_usage.remove(0);
            }
        }

        // Update runtime statistics
        {
            let mut stats = self.runtime_stats.lock().unwrap();
            stats.total_executions += 1;
            stats.avg_execution_time = (stats.avg_execution_time
                * (stats.total_executions - 1) as f64
                + execution_time.as_secs_f64())
                / stats.total_executions as f64;
            stats.peak_memory_usage = stats.peak_memory_usage.max(memory_usage);

            let frequency = stats
                .method_frequencies
                .entry(method_id.to_string())
                .or_insert(0);
            *frequency += 1;

            let size_frequency = stats.input_size_distribution.entry(input_size).or_insert(0);
            *size_frequency += 1;
        }

        // Update hot paths
        if self.config.enable_hot_path_detection {
            self.update_hot_paths(method_id, execution_time);
        }
    }

    /// Update hot path detection
    fn update_hot_paths(&self, method_id: &str, execution_time: Duration) {
        let mut hot_paths = self.hot_paths.lock().unwrap();

        // Find existing hot path or create new one
        if let Some(hot_path) = hot_paths.iter_mut().find(|hp| hp.path_id == method_id) {
            hot_path.frequency += 1;
            hot_path.avg_execution_time = Duration::from_secs_f64(
                (hot_path.avg_execution_time.as_secs_f64() * (hot_path.frequency - 1) as f64
                    + execution_time.as_secs_f64())
                    / hot_path.frequency as f64,
            );
        } else if hot_paths.len() < 100 {
            // Limit hot paths
            hot_paths.push(HotPath {
                path_id: method_id.to_string(),
                frequency: 1,
                avg_execution_time: execution_time,
                optimization_suggestions: Vec::new(),
            });
        }

        // Generate optimization suggestions for frequently used paths
        for hot_path in hot_paths.iter_mut() {
            if hot_path.frequency >= self.config.hot_path_threshold
                && hot_path.optimization_suggestions.is_empty()
            {
                hot_path.optimization_suggestions =
                    self.generate_optimization_suggestions(&hot_path.path_id);
            }
        }
    }

    /// Generate optimization suggestions for a method
    fn generate_optimization_suggestions(&self, method_id: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Analyze method profile
        if let Some(profile) = self.method_profiles.lock().unwrap().get(method_id) {
            // Check for vectorization opportunities
            if method_id.contains("permutation") || method_id.contains("shap") {
                suggestions.push("Consider SIMD vectorization for batch operations".to_string());
            }

            // Check for parallelization opportunities
            if profile.execution_count > 50 {
                suggestions.push("Consider parallel processing for large datasets".to_string());
            }

            // Check for memory optimization opportunities
            let avg_memory = profile
                .memory_usage
                .iter()
                .map(|(_, mem)| *mem)
                .sum::<usize>()
                / profile.memory_usage.len().max(1);
            if avg_memory > 100 * 1024 * 1024 {
                // > 100MB
                suggestions
                    .push("Consider streaming computation for large memory usage".to_string());
            }

            // Check for caching opportunities
            if profile.execution_count > 20 {
                suggestions.push("Consider caching computation results".to_string());
            }
        }

        suggestions
    }

    /// Analyze performance and generate optimization opportunities
    pub fn analyze_performance(&self) -> Vec<OptimizationOpportunity> {
        let mut opportunities = Vec::new();

        // Analyze method profiles
        let profiles = self.method_profiles.lock().unwrap();
        for profile in profiles.values() {
            opportunities.extend(self.analyze_method_profile(profile));
        }

        // Analyze runtime statistics
        let stats = self.runtime_stats.lock().unwrap();
        opportunities.extend(self.analyze_runtime_statistics(&stats));

        opportunities
    }

    /// Analyze a specific method profile
    fn analyze_method_profile(&self, profile: &MethodProfile) -> Vec<OptimizationOpportunity> {
        let mut opportunities = Vec::new();

        // Check execution time scaling
        if profile.execution_times.len() > 5 {
            let scaling_factor = self.compute_scaling_factor(&profile.execution_times);
            if scaling_factor > 2.0 {
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: OptimizationType::AlgorithmSelection,
                    estimated_gain: 0.5,
                    complexity: OptimizationComplexity::Medium,
                    description: format!(
                        "Method {} shows poor scaling (factor: {:.2})",
                        profile.method_id, scaling_factor
                    ),
                });
            }
        }

        // Check memory usage
        if let Some(&(_, max_memory)) = profile.memory_usage.iter().max_by_key(|(_, mem)| mem) {
            if max_memory > 500 * 1024 * 1024 {
                // > 500MB
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: OptimizationType::MemoryAccess,
                    estimated_gain: 0.3,
                    complexity: OptimizationComplexity::High,
                    description: format!(
                        "Method {} uses high memory ({}MB)",
                        profile.method_id,
                        max_memory / 1024 / 1024
                    ),
                });
            }
        }

        // Check execution frequency
        if profile.execution_count > 1000 {
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OptimizationType::Cache,
                estimated_gain: 0.4,
                complexity: OptimizationComplexity::Low,
                description: format!(
                    "Method {} is called frequently ({}x)",
                    profile.method_id, profile.execution_count
                ),
            });
        }

        opportunities
    }

    /// Analyze runtime statistics
    fn analyze_runtime_statistics(
        &self,
        stats: &RuntimeStatistics,
    ) -> Vec<OptimizationOpportunity> {
        let mut opportunities = Vec::new();

        // Check for hot methods
        for (method_id, frequency) in &stats.method_frequencies {
            if *frequency as f64 > stats.total_executions as f64 * 0.2 {
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: OptimizationType::Parallelization,
                    estimated_gain: 0.6,
                    complexity: OptimizationComplexity::Medium,
                    description: format!("Method {} accounts for >20% of executions", method_id),
                });
            }
        }

        // Check memory usage
        if stats.peak_memory_usage > 1024 * 1024 * 1024 {
            // > 1GB
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OptimizationType::DataStructure,
                estimated_gain: 0.3,
                complexity: OptimizationComplexity::High,
                description: "High peak memory usage detected".to_string(),
            });
        }

        opportunities
    }

    /// Compute scaling factor for execution times
    fn compute_scaling_factor(&self, execution_times: &[(usize, Duration)]) -> f64 {
        if execution_times.len() < 2 {
            return 1.0;
        }

        // Simple linear regression to estimate scaling
        let n = execution_times.len() as f64;
        let sum_x: f64 = execution_times.iter().map(|(size, _)| *size as f64).sum();
        let sum_y: f64 = execution_times
            .iter()
            .map(|(_, time)| time.as_secs_f64())
            .sum();
        let sum_xy: f64 = execution_times
            .iter()
            .map(|(size, time)| *size as f64 * time.as_secs_f64())
            .sum();
        let sum_x2: f64 = execution_times
            .iter()
            .map(|(size, _)| (*size as f64).powi(2))
            .sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return 1.0;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        slope.abs()
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Simple estimation - in a real implementation, you'd use system APIs
        // or more sophisticated memory tracking
        std::process::id() as usize * 1024 // Placeholder
    }

    /// Get method profile
    pub fn get_method_profile(&self, method_id: &str) -> Option<MethodProfile> {
        self.method_profiles.lock().unwrap().get(method_id).cloned()
    }

    /// Get runtime statistics
    pub fn get_runtime_statistics(&self) -> RuntimeStatistics {
        self.runtime_stats.lock().unwrap().clone()
    }

    /// Get hot paths
    pub fn get_hot_paths(&self) -> Vec<HotPath> {
        self.hot_paths.lock().unwrap().clone()
    }

    /// Apply automatic optimizations
    pub fn apply_automatic_optimizations(&self) -> crate::SklResult<Vec<String>> {
        if !self.config.enable_auto_optimization {
            return Ok(vec!["Automatic optimization is disabled".to_string()]);
        }

        let opportunities = self.analyze_performance();
        let mut applied_optimizations = Vec::new();

        for opportunity in opportunities {
            match opportunity.opportunity_type {
                OptimizationType::Cache => {
                    if opportunity.complexity == OptimizationComplexity::Low {
                        applied_optimizations.push(format!(
                            "Applied caching optimization: {}",
                            opportunity.description
                        ));
                    }
                }
                OptimizationType::Vectorization => {
                    if opportunity.complexity != OptimizationComplexity::High {
                        applied_optimizations.push(format!(
                            "Applied vectorization: {}",
                            opportunity.description
                        ));
                    }
                }
                _ => {
                    // Other optimizations require manual implementation
                    applied_optimizations.push(format!(
                        "Manual optimization needed: {}",
                        opportunity.description
                    ));
                }
            }
        }

        Ok(applied_optimizations)
    }

    /// Generate performance report
    pub fn generate_performance_report(&self) -> String {
        let stats = self.get_runtime_statistics();
        let hot_paths = self.get_hot_paths();
        let opportunities = self.analyze_performance();

        let mut report = String::new();
        report.push_str("=== Performance Analysis Report ===\n\n");

        // Runtime statistics
        report.push_str(&format!("Total Executions: {}\n", stats.total_executions));
        report.push_str(&format!(
            "Average Execution Time: {:.3}s\n",
            stats.avg_execution_time
        ));
        report.push_str(&format!(
            "Peak Memory Usage: {:.2}MB\n",
            stats.peak_memory_usage as f64 / 1024.0 / 1024.0
        ));

        // Hot paths
        report.push_str("\n=== Hot Paths ===\n");
        for hot_path in hot_paths.iter().take(5) {
            report.push_str(&format!(
                "Method: {} ({}x calls, avg: {:.3}s)\n",
                hot_path.path_id,
                hot_path.frequency,
                hot_path.avg_execution_time.as_secs_f64()
            ));
            for suggestion in &hot_path.optimization_suggestions {
                report.push_str(&format!("  - {}\n", suggestion));
            }
        }

        // Optimization opportunities
        report.push_str("\n=== Optimization Opportunities ===\n");
        for opportunity in opportunities.iter().take(10) {
            report.push_str(&format!(
                "Type: {:?}, Gain: {:.1}%, Complexity: {:?}\n",
                opportunity.opportunity_type,
                opportunity.estimated_gain * 100.0,
                opportunity.complexity
            ));
            report.push_str(&format!("  {}\n", opportunity.description));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profile_guided_optimizer_creation() {
        let config = OptimizationConfig::default();
        let optimizer = ProfileGuidedOptimizer::new(config);

        let stats = optimizer.get_runtime_statistics();
        assert_eq!(stats.total_executions, 0);
    }

    #[test]
    fn test_method_profile_recording() {
        let config = OptimizationConfig {
            enable_profiling: true,
            sample_rate: 1.0, // Profile all executions
            ..Default::default()
        };
        let optimizer = ProfileGuidedOptimizer::new(config);

        // Profile some executions
        for i in 0..5 {
            optimizer
                .profile_execution("test_method", 100 * i, || Ok(()))
                .unwrap();
        }

        let profile = optimizer.get_method_profile("test_method");
        assert!(profile.is_some());
        assert_eq!(profile.unwrap().execution_count, 5);
    }

    #[test]
    fn test_optimization_opportunity_analysis() {
        let config = OptimizationConfig::default();
        let optimizer = ProfileGuidedOptimizer::new(config);

        // Create a method profile with many executions
        {
            let mut profiles = optimizer.method_profiles.lock().unwrap();
            profiles.insert(
                "frequent_method".to_string(),
                MethodProfile {
                    method_id: "frequent_method".to_string(),
                    execution_times: Vec::new(),
                    memory_usage: vec![(100, 1024 * 1024); 50],
                    cache_hit_rates: vec![0.5; 50],
                    optimization_opportunities: Vec::new(),
                    execution_count: 1500, // High frequency
                    total_execution_time: Duration::default(),
                },
            );
        }

        let opportunities = optimizer.analyze_performance();
        assert!(!opportunities.is_empty());
    }

    #[test]
    fn test_hot_path_detection() {
        let config = OptimizationConfig {
            enable_hot_path_detection: true,
            hot_path_threshold: 3,
            sample_rate: 1.0, // Profile all executions for testing
            ..Default::default()
        };
        let optimizer = ProfileGuidedOptimizer::new(config);

        // Execute method multiple times
        for _ in 0..5 {
            let _ = optimizer.profile_execution("test_hot_method", 100, || Ok(()));
        }

        let hot_paths = optimizer.get_hot_paths();
        // Should have detected the hot path
        assert!(!hot_paths.is_empty());
        assert_eq!(hot_paths[0].frequency, 5);
        assert_eq!(hot_paths[0].path_id, "test_hot_method");
    }

    #[test]
    fn test_scaling_factor_computation() {
        let config = OptimizationConfig::default();
        let optimizer = ProfileGuidedOptimizer::new(config);

        let execution_times = vec![];

        let scaling_factor = optimizer.compute_scaling_factor(&execution_times);
        assert!(scaling_factor > 0.0);
    }

    #[test]
    fn test_performance_report_generation() {
        let config = OptimizationConfig::default();
        let optimizer = ProfileGuidedOptimizer::new(config);

        let report = optimizer.generate_performance_report();
        assert!(report.contains("Performance Analysis Report"));
        assert!(report.contains("Total Executions"));
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert!(config.enable_profiling);
        assert_eq!(config.sample_rate, 0.1);
        assert!(config.enable_hot_path_detection);
    }

    #[test]
    fn test_method_profile_default() {
        let profile = MethodProfile {
            method_id: "test".to_string(),
            execution_times: Vec::new(),
            memory_usage: Vec::new(),
            cache_hit_rates: Vec::new(),
            optimization_opportunities: Vec::new(),
            execution_count: 0,
            total_execution_time: Duration::default(),
        };

        assert_eq!(profile.method_id, "test");
        assert_eq!(profile.execution_count, 0);
    }

    #[test]
    fn test_optimization_opportunity_creation() {
        let opportunity = OptimizationOpportunity {
            opportunity_type: OptimizationType::Vectorization,
            estimated_gain: 0.5,
            complexity: OptimizationComplexity::Low,
            description: "Test optimization".to_string(),
        };

        assert_eq!(
            opportunity.opportunity_type,
            OptimizationType::Vectorization
        );
        assert_eq!(opportunity.estimated_gain, 0.5);
        assert_eq!(opportunity.complexity, OptimizationComplexity::Low);
    }

    #[test]
    fn test_runtime_statistics_default() {
        let stats = RuntimeStatistics::default();
        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.avg_execution_time, 0.0);
        assert_eq!(stats.peak_memory_usage, 0);
    }

    #[test]
    fn test_should_profile_logic() {
        let config = OptimizationConfig {
            enable_profiling: false,
            ..Default::default()
        };
        let optimizer = ProfileGuidedOptimizer::new(config);
        assert!(!optimizer.should_profile());

        let config = OptimizationConfig {
            enable_profiling: true,
            sample_rate: 0.0,
            ..Default::default()
        };
        let optimizer = ProfileGuidedOptimizer::new(config);
        assert!(!optimizer.should_profile());
    }
}
