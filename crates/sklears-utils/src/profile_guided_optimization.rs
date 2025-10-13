//! Profile-guided optimization utilities
//!
//! This module provides utilities for profile-guided optimization including performance profiling,
//! hotspot detection, optimization hints, and automatic optimization recommendations for ML workloads.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Performance profile data
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub function_profiles: HashMap<String, FunctionProfile>,
    pub loop_profiles: HashMap<String, LoopProfile>,
    pub memory_access_patterns: HashMap<String, MemoryAccessPattern>,
    pub branch_predictions: HashMap<String, BranchProfile>,
    pub cache_statistics: CacheStatistics,
    pub instruction_mix: InstructionMix,
    pub profiling_duration: Duration,
    pub total_samples: u64,
}

/// Function-level performance profile
#[derive(Debug, Clone)]
pub struct FunctionProfile {
    pub name: String,
    pub total_time: Duration,
    pub self_time: Duration,
    pub call_count: u64,
    pub avg_time_per_call: Duration,
    pub max_time_per_call: Duration,
    pub min_time_per_call: Duration,
    pub cpu_cycles: u64,
    pub cache_misses: u64,
    pub branch_misses: u64,
    pub hotness_score: f64,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Loop-level performance profile
#[derive(Debug, Clone)]
pub struct LoopProfile {
    pub loop_id: String,
    pub location: String,
    pub iteration_count: u64,
    pub total_time: Duration,
    pub avg_time_per_iteration: Duration,
    pub vectorization_efficiency: f64,
    pub dependency_chains: Vec<DependencyChain>,
    pub memory_access_stride: i64,
    pub loop_carried_dependencies: u32,
    pub optimization_potential: f64,
}

/// Memory access pattern analysis
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    pub function_name: String,
    pub access_type: MemoryAccessType,
    pub access_frequency: u64,
    pub cache_hit_rate: f64,
    pub average_latency: Duration,
    pub stride_pattern: StridePattern,
    pub prefetch_effectiveness: f64,
    pub numa_locality: f64,
}

/// Branch prediction profile
#[derive(Debug, Clone)]
pub struct BranchProfile {
    pub branch_id: String,
    pub location: String,
    pub taken_count: u64,
    pub not_taken_count: u64,
    pub prediction_accuracy: f64,
    pub misprediction_penalty: Duration,
    pub branch_type: BranchType,
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub tlb_hit_rate: f64,
    pub cache_line_utilization: f64,
    pub false_sharing_incidents: u64,
    pub prefetch_accuracy: f64,
}

/// Instruction mix analysis
#[derive(Debug, Clone)]
pub struct InstructionMix {
    pub integer_ops: u64,
    pub floating_point_ops: u64,
    pub vector_ops: u64,
    pub memory_ops: u64,
    pub branch_ops: u64,
    pub simd_utilization: f64,
    pub parallel_efficiency: f64,
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OptimizationType,
    pub description: String,
    pub potential_speedup: f64,
    pub implementation_effort: ImplementationEffort,
    pub confidence: f64,
    pub code_location: String,
    pub suggested_actions: Vec<String>,
}

/// Dependency chain in loops
#[derive(Debug, Clone)]
pub struct DependencyChain {
    pub chain_id: String,
    pub length: u32,
    pub critical_path_time: Duration,
    pub parallelization_potential: f64,
}

/// Memory access types
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccessType {
    Sequential,
    Random,
    Strided,
    Gather,
    Scatter,
}

/// Stride patterns
#[derive(Debug, Clone)]
pub struct StridePattern {
    pub primary_stride: i64,
    pub secondary_stride: Option<i64>,
    pub regularity: f64,
    pub predictability: f64,
}

/// Branch types
#[derive(Debug, Clone, PartialEq)]
pub enum BranchType {
    Conditional,
    Indirect,
    Return,
    Call,
    Loop,
}

/// Optimization types
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    Vectorization,
    LoopUnrolling,
    FunctionInlining,
    MemoryPrefetching,
    BranchElimination,
    CacheOptimization,
    Parallelization,
    AlgorithmicImprovement,
}

/// Implementation effort levels
#[derive(Debug, Clone, PartialEq)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Profile-guided optimizer
pub struct ProfileGuidedOptimizer {
    profiles: Arc<RwLock<HashMap<String, PerformanceProfile>>>,
    optimization_rules: Vec<OptimizationRule>,
    #[allow(dead_code)]
    performance_targets: PerformanceTargets,
    profiler_config: ProfilerConfig,
    optimization_history: Arc<Mutex<Vec<OptimizationApplication>>>,
}

/// Optimization rule
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    pub name: String,
    pub trigger_condition: TriggerCondition,
    pub optimization_type: OptimizationType,
    pub implementation: String,
    pub expected_benefit: f64,
    pub risk_level: RiskLevel,
}

/// Trigger condition for optimization
#[derive(Debug, Clone)]
pub struct TriggerCondition {
    pub min_hotness_score: f64,
    pub min_call_frequency: u64,
    pub max_cache_miss_rate: f64,
    pub min_loop_iterations: u64,
    pub function_name_patterns: Vec<String>,
}

/// Performance targets
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    pub target_speedup: f64,
    pub max_memory_increase: f64,
    pub max_compilation_time: Duration,
    pub stability_requirement: f64,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    pub sampling_frequency: u64,
    pub enable_instruction_profiling: bool,
    pub enable_memory_profiling: bool,
    pub enable_cache_profiling: bool,
    pub enable_branch_profiling: bool,
    pub profiling_duration: Duration,
}

/// Optimization application record
#[derive(Debug, Clone)]
pub struct OptimizationApplication {
    pub timestamp: Instant,
    pub rule_name: String,
    pub target_function: String,
    pub optimization_type: OptimizationType,
    pub measured_speedup: Option<f64>,
    pub success: bool,
    pub notes: String,
}

/// Risk levels for optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Experimental,
}

impl ProfileGuidedOptimizer {
    /// Create new profile-guided optimizer
    pub fn new(config: ProfilerConfig, targets: PerformanceTargets) -> Self {
        let mut optimizer = Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
            optimization_rules: Vec::new(),
            performance_targets: targets,
            profiler_config: config,
            optimization_history: Arc::new(Mutex::new(Vec::new())),
        };

        optimizer.initialize_default_rules();
        optimizer
    }

    /// Initialize default optimization rules
    fn initialize_default_rules(&mut self) {
        self.optimization_rules = vec![
            OptimizationRule {
                name: "Hot Function Inlining".to_string(),
                trigger_condition: TriggerCondition {
                    min_hotness_score: 0.8,
                    min_call_frequency: 1000,
                    max_cache_miss_rate: 1.0,
                    min_loop_iterations: 0,
                    function_name_patterns: vec![".*_hot.*".to_string()],
                },
                optimization_type: OptimizationType::FunctionInlining,
                implementation: "#[inline(always)]".to_string(),
                expected_benefit: 1.15,
                risk_level: RiskLevel::Low,
            },
            OptimizationRule {
                name: "Loop Vectorization".to_string(),
                trigger_condition: TriggerCondition {
                    min_hotness_score: 0.6,
                    min_call_frequency: 0,
                    max_cache_miss_rate: 1.0,
                    min_loop_iterations: 100,
                    function_name_patterns: vec![".*_vectorizable.*".to_string()],
                },
                optimization_type: OptimizationType::Vectorization,
                implementation: "SIMD optimization".to_string(),
                expected_benefit: 2.0,
                risk_level: RiskLevel::Medium,
            },
            OptimizationRule {
                name: "Memory Prefetching".to_string(),
                trigger_condition: TriggerCondition {
                    min_hotness_score: 0.5,
                    min_call_frequency: 0,
                    max_cache_miss_rate: 0.1,
                    min_loop_iterations: 0,
                    function_name_patterns: vec![".*_memory_intensive.*".to_string()],
                },
                optimization_type: OptimizationType::MemoryPrefetching,
                implementation: "Software prefetching".to_string(),
                expected_benefit: 1.3,
                risk_level: RiskLevel::Medium,
            },
            OptimizationRule {
                name: "Loop Unrolling".to_string(),
                trigger_condition: TriggerCondition {
                    min_hotness_score: 0.7,
                    min_call_frequency: 0,
                    max_cache_miss_rate: 1.0,
                    min_loop_iterations: 10,
                    function_name_patterns: vec![".*_tight_loop.*".to_string()],
                },
                optimization_type: OptimizationType::LoopUnrolling,
                implementation: "Unroll factor 4".to_string(),
                expected_benefit: 1.25,
                risk_level: RiskLevel::Low,
            },
        ];
    }

    /// Add custom optimization rule
    pub fn add_optimization_rule(&mut self, rule: OptimizationRule) {
        self.optimization_rules.push(rule);
    }

    /// Collect performance profile
    pub fn collect_profile(&self, program_name: &str) -> Result<PerformanceProfile, ProfileError> {
        // Mock profile collection (in real implementation, this would use hardware counters)
        let mock_profile = PerformanceProfile {
            function_profiles: self.generate_mock_function_profiles(),
            loop_profiles: self.generate_mock_loop_profiles(),
            memory_access_patterns: self.generate_mock_memory_patterns(),
            branch_predictions: self.generate_mock_branch_profiles(),
            cache_statistics: CacheStatistics {
                l1_hit_rate: 0.95,
                l2_hit_rate: 0.85,
                l3_hit_rate: 0.70,
                tlb_hit_rate: 0.98,
                cache_line_utilization: 0.75,
                false_sharing_incidents: 5,
                prefetch_accuracy: 0.80,
            },
            instruction_mix: InstructionMix {
                integer_ops: 1_000_000,
                floating_point_ops: 500_000,
                vector_ops: 100_000,
                memory_ops: 200_000,
                branch_ops: 150_000,
                simd_utilization: 0.60,
                parallel_efficiency: 0.75,
            },
            profiling_duration: self.profiler_config.profiling_duration,
            total_samples: 1_000_000,
        };

        self.profiles
            .write()
            .unwrap()
            .insert(program_name.to_string(), mock_profile.clone());
        Ok(mock_profile)
    }

    /// Generate mock function profiles
    fn generate_mock_function_profiles(&self) -> HashMap<String, FunctionProfile> {
        let mut profiles = HashMap::new();

        profiles.insert(
            "matrix_multiply".to_string(),
            FunctionProfile {
                name: "matrix_multiply".to_string(),
                total_time: Duration::from_millis(500),
                self_time: Duration::from_millis(450),
                call_count: 1000,
                avg_time_per_call: Duration::from_micros(500),
                max_time_per_call: Duration::from_millis(2),
                min_time_per_call: Duration::from_micros(100),
                cpu_cycles: 1_000_000,
                cache_misses: 5000,
                branch_misses: 100,
                hotness_score: 0.9,
                optimization_opportunities: vec![OptimizationOpportunity {
                    opportunity_type: OptimizationType::Vectorization,
                    description: "Loop can be vectorized for SIMD".to_string(),
                    potential_speedup: 2.5,
                    implementation_effort: ImplementationEffort::Medium,
                    confidence: 0.85,
                    code_location: "matrix_multiply.rs:45".to_string(),
                    suggested_actions: vec![
                        "Use SIMD intrinsics".to_string(),
                        "Enable auto-vectorization".to_string(),
                    ],
                }],
            },
        );

        profiles.insert(
            "activation_function".to_string(),
            FunctionProfile {
                name: "activation_function".to_string(),
                total_time: Duration::from_millis(200),
                self_time: Duration::from_millis(180),
                call_count: 10_000,
                avg_time_per_call: Duration::from_micros(20),
                max_time_per_call: Duration::from_micros(100),
                min_time_per_call: Duration::from_micros(5),
                cpu_cycles: 400_000,
                cache_misses: 1000,
                branch_misses: 50,
                hotness_score: 0.7,
                optimization_opportunities: vec![OptimizationOpportunity {
                    opportunity_type: OptimizationType::FunctionInlining,
                    description: "Small function called frequently".to_string(),
                    potential_speedup: 1.15,
                    implementation_effort: ImplementationEffort::Low,
                    confidence: 0.95,
                    code_location: "activation.rs:12".to_string(),
                    suggested_actions: vec!["Add inline attribute".to_string()],
                }],
            },
        );

        profiles
    }

    /// Generate mock loop profiles
    fn generate_mock_loop_profiles(&self) -> HashMap<String, LoopProfile> {
        let mut profiles = HashMap::new();

        profiles.insert(
            "training_loop".to_string(),
            LoopProfile {
                loop_id: "training_loop".to_string(),
                location: "train.rs:100".to_string(),
                iteration_count: 1000,
                total_time: Duration::from_millis(1000),
                avg_time_per_iteration: Duration::from_millis(1),
                vectorization_efficiency: 0.4,
                dependency_chains: vec![DependencyChain {
                    chain_id: "weight_update".to_string(),
                    length: 3,
                    critical_path_time: Duration::from_micros(100),
                    parallelization_potential: 0.8,
                }],
                memory_access_stride: 8,
                loop_carried_dependencies: 1,
                optimization_potential: 0.6,
            },
        );

        profiles
    }

    /// Generate mock memory access patterns
    fn generate_mock_memory_patterns(&self) -> HashMap<String, MemoryAccessPattern> {
        let mut patterns = HashMap::new();

        patterns.insert(
            "data_loading".to_string(),
            MemoryAccessPattern {
                function_name: "data_loading".to_string(),
                access_type: MemoryAccessType::Sequential,
                access_frequency: 10_000,
                cache_hit_rate: 0.85,
                average_latency: Duration::from_nanos(50),
                stride_pattern: StridePattern {
                    primary_stride: 8,
                    secondary_stride: None,
                    regularity: 0.95,
                    predictability: 0.90,
                },
                prefetch_effectiveness: 0.75,
                numa_locality: 0.80,
            },
        );

        patterns
    }

    /// Generate mock branch profiles
    fn generate_mock_branch_profiles(&self) -> HashMap<String, BranchProfile> {
        let mut profiles = HashMap::new();

        profiles.insert(
            "convergence_check".to_string(),
            BranchProfile {
                branch_id: "convergence_check".to_string(),
                location: "optimizer.rs:200".to_string(),
                taken_count: 950,
                not_taken_count: 50,
                prediction_accuracy: 0.95,
                misprediction_penalty: Duration::from_nanos(20),
                branch_type: BranchType::Conditional,
            },
        );

        profiles
    }

    /// Analyze profiles and generate optimization recommendations
    pub fn analyze_and_recommend(
        &self,
        program_name: &str,
    ) -> Result<Vec<OptimizationRecommendation>, ProfileError> {
        let profiles = self.profiles.read().unwrap();
        let profile = profiles
            .get(program_name)
            .ok_or(ProfileError::ProfileNotFound)?;

        let mut recommendations = Vec::new();

        // Analyze function profiles
        for func_profile in profile.function_profiles.values() {
            for rule in &self.optimization_rules {
                if self.matches_trigger_condition(&rule.trigger_condition, func_profile) {
                    recommendations.push(OptimizationRecommendation {
                        rule_name: rule.name.clone(),
                        target_function: func_profile.name.clone(),
                        optimization_type: rule.optimization_type.clone(),
                        expected_speedup: rule.expected_benefit,
                        risk_level: rule.risk_level.clone(),
                        implementation: rule.implementation.clone(),
                        priority: self.calculate_priority(func_profile, rule),
                        estimated_effort: ImplementationEffort::Medium,
                        confidence: 0.8,
                    });
                }
            }
        }

        // Sort by priority
        recommendations.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(recommendations)
    }

    /// Check if function matches trigger condition
    fn matches_trigger_condition(
        &self,
        condition: &TriggerCondition,
        profile: &FunctionProfile,
    ) -> bool {
        profile.hotness_score >= condition.min_hotness_score
            && profile.call_count >= condition.min_call_frequency
            && (profile.cache_misses as f64 / profile.call_count as f64)
                <= condition.max_cache_miss_rate
    }

    /// Calculate optimization priority
    fn calculate_priority(&self, profile: &FunctionProfile, rule: &OptimizationRule) -> f64 {
        let hotness_factor = profile.hotness_score;
        let benefit_factor = rule.expected_benefit - 1.0; // Convert to gain
        let risk_factor = match rule.risk_level {
            RiskLevel::Low => 1.0,
            RiskLevel::Medium => 0.8,
            RiskLevel::High => 0.6,
            RiskLevel::Experimental => 0.4,
        };

        hotness_factor * benefit_factor * risk_factor
    }

    /// Apply optimization recommendation
    pub fn apply_optimization(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<(), ProfileError> {
        // Mock optimization application
        let application = OptimizationApplication {
            timestamp: Instant::now(),
            rule_name: recommendation.rule_name.clone(),
            target_function: recommendation.target_function.clone(),
            optimization_type: recommendation.optimization_type.clone(),
            measured_speedup: Some(recommendation.expected_speedup * 0.9), // Slightly lower than expected
            success: true,
            notes: format!(
                "Applied {} to {}",
                recommendation.implementation, recommendation.target_function
            ),
        };

        self.optimization_history.lock().unwrap().push(application);
        Ok(())
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> Vec<OptimizationApplication> {
        self.optimization_history.lock().unwrap().clone()
    }

    /// Calculate overall performance gain
    pub fn calculate_performance_gain(&self) -> f64 {
        let history = self.optimization_history.lock().unwrap();
        let successful_optimizations: Vec<_> = history
            .iter()
            .filter(|app| app.success && app.measured_speedup.is_some())
            .collect();

        if successful_optimizations.is_empty() {
            return 1.0;
        }

        // Compound the speedups
        successful_optimizations
            .iter()
            .map(|app| app.measured_speedup.unwrap())
            .fold(1.0, |acc, speedup| acc * speedup)
    }

    /// Generate optimization report
    pub fn generate_report(&self, program_name: &str) -> Result<OptimizationReport, ProfileError> {
        let profiles = self.profiles.read().unwrap();
        let profile = profiles
            .get(program_name)
            .ok_or(ProfileError::ProfileNotFound)?;

        let recommendations = self.analyze_and_recommend(program_name)?;
        let history = self.get_optimization_history();
        let performance_gain = self.calculate_performance_gain();

        let potential_further_gains = recommendations
            .iter()
            .map(|r| r.expected_speedup - 1.0)
            .sum::<f64>();

        Ok(OptimizationReport {
            program_name: program_name.to_string(),
            profile_summary: ProfileSummary {
                total_functions: profile.function_profiles.len(),
                hot_functions: profile
                    .function_profiles
                    .values()
                    .filter(|f| f.hotness_score > 0.5)
                    .count(),
                total_loops: profile.loop_profiles.len(),
                vectorizable_loops: profile
                    .loop_profiles
                    .values()
                    .filter(|l| l.vectorization_efficiency < 0.5)
                    .count(),
                cache_efficiency: profile.cache_statistics.l1_hit_rate,
                simd_utilization: profile.instruction_mix.simd_utilization,
            },
            recommendations,
            applied_optimizations: history,
            overall_performance_gain: performance_gain,
            potential_further_gains,
            report_timestamp: Instant::now(),
        })
    }

    /// Reset profiling data
    pub fn reset(&self) {
        self.profiles.write().unwrap().clear();
        self.optimization_history.lock().unwrap().clear();
    }
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub rule_name: String,
    pub target_function: String,
    pub optimization_type: OptimizationType,
    pub expected_speedup: f64,
    pub risk_level: RiskLevel,
    pub implementation: String,
    pub priority: f64,
    pub estimated_effort: ImplementationEffort,
    pub confidence: f64,
}

/// Profile summary
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub total_functions: usize,
    pub hot_functions: usize,
    pub total_loops: usize,
    pub vectorizable_loops: usize,
    pub cache_efficiency: f64,
    pub simd_utilization: f64,
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub program_name: String,
    pub profile_summary: ProfileSummary,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub applied_optimizations: Vec<OptimizationApplication>,
    pub overall_performance_gain: f64,
    pub potential_further_gains: f64,
    pub report_timestamp: Instant,
}

/// Profile-guided optimization errors
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("Profile not found")]
    ProfileNotFound,
    #[error("Profiling failed: {0}")]
    ProfilingFailed(String),
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sampling_frequency: 1000,
            enable_instruction_profiling: true,
            enable_memory_profiling: true,
            enable_cache_profiling: true,
            enable_branch_profiling: true,
            profiling_duration: Duration::from_secs(10),
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_speedup: 1.5,
            max_memory_increase: 0.1,
            max_compilation_time: Duration::from_secs(60),
            stability_requirement: 0.95,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let optimizer = ProfileGuidedOptimizer::new(config, targets);

        assert!(!optimizer.optimization_rules.is_empty());
    }

    #[test]
    fn test_profile_collection() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let optimizer = ProfileGuidedOptimizer::new(config, targets);

        let profile = optimizer.collect_profile("test_program").unwrap();
        assert!(!profile.function_profiles.is_empty());
        assert!(profile.total_samples > 0);
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let optimizer = ProfileGuidedOptimizer::new(config, targets);

        optimizer.collect_profile("test_program").unwrap();
        let recommendations = optimizer.analyze_and_recommend("test_program").unwrap();

        assert!(!recommendations.is_empty());
        assert!(recommendations[0].priority > 0.0);
    }

    #[test]
    fn test_optimization_application() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let optimizer = ProfileGuidedOptimizer::new(config, targets);

        optimizer.collect_profile("test_program").unwrap();
        let recommendations = optimizer.analyze_and_recommend("test_program").unwrap();

        if let Some(recommendation) = recommendations.first() {
            assert!(optimizer.apply_optimization(recommendation).is_ok());

            let history = optimizer.get_optimization_history();
            assert!(!history.is_empty());
            assert!(history[0].success);
        }
    }

    #[test]
    fn test_performance_gain_calculation() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let optimizer = ProfileGuidedOptimizer::new(config, targets);

        optimizer.collect_profile("test_program").unwrap();
        let recommendations = optimizer.analyze_and_recommend("test_program").unwrap();

        for recommendation in recommendations.iter().take(2) {
            optimizer.apply_optimization(recommendation).unwrap();
        }

        let gain = optimizer.calculate_performance_gain();
        assert!(gain >= 1.0);
    }

    #[test]
    fn test_optimization_report() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let optimizer = ProfileGuidedOptimizer::new(config, targets);

        optimizer.collect_profile("test_program").unwrap();
        let report = optimizer.generate_report("test_program").unwrap();

        assert_eq!(report.program_name, "test_program");
        assert!(report.profile_summary.total_functions > 0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_custom_optimization_rule() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let mut optimizer = ProfileGuidedOptimizer::new(config, targets);

        let custom_rule = OptimizationRule {
            name: "Custom Parallel".to_string(),
            trigger_condition: TriggerCondition {
                min_hotness_score: 0.9,
                min_call_frequency: 1000,
                max_cache_miss_rate: 0.05,
                min_loop_iterations: 1000,
                function_name_patterns: vec!["parallel_*".to_string()],
            },
            optimization_type: OptimizationType::Parallelization,
            implementation: "Use rayon parallel iterator".to_string(),
            expected_benefit: 3.0,
            risk_level: RiskLevel::Medium,
        };

        let initial_rules = optimizer.optimization_rules.len();
        optimizer.add_optimization_rule(custom_rule);
        assert_eq!(optimizer.optimization_rules.len(), initial_rules + 1);
    }

    #[test]
    fn test_trigger_condition_matching() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let optimizer = ProfileGuidedOptimizer::new(config, targets);

        let condition = TriggerCondition {
            min_hotness_score: 0.5,
            min_call_frequency: 100,
            max_cache_miss_rate: 0.1,
            min_loop_iterations: 0,
            function_name_patterns: vec![],
        };

        let profile = FunctionProfile {
            name: "test_function".to_string(),
            total_time: Duration::from_millis(100),
            self_time: Duration::from_millis(90),
            call_count: 1000,
            avg_time_per_call: Duration::from_micros(100),
            max_time_per_call: Duration::from_millis(1),
            min_time_per_call: Duration::from_micros(50),
            cpu_cycles: 200_000,
            cache_misses: 50, // 0.05 miss rate
            branch_misses: 10,
            hotness_score: 0.8,
            optimization_opportunities: vec![],
        };

        assert!(optimizer.matches_trigger_condition(&condition, &profile));
    }

    #[test]
    fn test_priority_calculation() {
        let config = ProfilerConfig::default();
        let targets = PerformanceTargets::default();
        let optimizer = ProfileGuidedOptimizer::new(config, targets);

        let profile = FunctionProfile {
            name: "test_function".to_string(),
            total_time: Duration::from_millis(100),
            self_time: Duration::from_millis(90),
            call_count: 1000,
            avg_time_per_call: Duration::from_micros(100),
            max_time_per_call: Duration::from_millis(1),
            min_time_per_call: Duration::from_micros(50),
            cpu_cycles: 200_000,
            cache_misses: 50,
            branch_misses: 10,
            hotness_score: 0.8,
            optimization_opportunities: vec![],
        };

        let rule = OptimizationRule {
            name: "Test Rule".to_string(),
            trigger_condition: TriggerCondition {
                min_hotness_score: 0.5,
                min_call_frequency: 100,
                max_cache_miss_rate: 0.1,
                min_loop_iterations: 0,
                function_name_patterns: vec![],
            },
            optimization_type: OptimizationType::FunctionInlining,
            implementation: "inline".to_string(),
            expected_benefit: 1.5,
            risk_level: RiskLevel::Low,
        };

        let priority = optimizer.calculate_priority(&profile, &rule);
        assert!(priority > 0.0);
    }
}
