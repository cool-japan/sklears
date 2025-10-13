//! Gradient-Based Optimization Framework - Refactored Module Coordinator
//!
//! This module provides a unified interface to the comprehensive gradient-based optimization
//! framework, coordinating all specialized modules and providing factory patterns for
//! easy instantiation and configuration. This is a refactored version that enforces
//! the 2000-line policy by breaking down functionality into focused modules.
//!
//! # Architecture
//!
//! The framework is organized into 8 specialized modules:
//! - **factory_core**: Core factory infrastructure and component registry
//! - **configuration_management**: Comprehensive configuration system and templates
//! - **performance_tracking**: Performance monitoring and analysis
//! - **gradient_core**: Core optimization algorithms and traits (existing)
//! - **gradient_computation**: Gradient and Hessian computation systems (existing)
//! - **algorithm_selection**: Algorithm selection and adaptive control (existing)
//! - **problem_analysis**: Mathematical problem analysis (existing)
//! - **simd_acceleration**: SIMD and hardware optimization (existing)
//!
//! # Usage
//!
//! ```rust
//! use gradient_optimization::{GradientOptimizationFactory, OptimizationConfig};
//!
//! // Create a complete optimization system
//! let factory = GradientOptimizationFactory::new();
//! let optimizer = factory.create_optimizer(OptimizationConfig::default())?;
//!
//! // Or create individual components
//! let core_optimizer = factory.create_core_optimizer()?;
//! let problem_analyzer = factory.create_problem_analyzer()?;
//! ```

// Module declarations - focused modules from refactoring
pub mod factory_core;
pub mod configuration_management;
pub mod performance_tracking;

// Existing modules preserved from original structure
pub mod gradient_core;
pub mod gradient_computation;
pub mod algorithm_selection;
pub mod problem_analysis;
pub mod simd_acceleration;

// Re-exports from factory_core
pub use factory_core::{
    GradientOptimizationFactory,
    FactoryConfiguration,
    ComponentRegistry,
    TemplateRegistry,
    CreationStatistics,
    ValidationRules,
    FactoryStatistics,
};

// Re-exports from configuration_management
pub use configuration_management::{
    OptimizationConfig,
    OptimizationConfigBuilder,
    AlgorithmConfiguration,
    AlgorithmConfigurationBuilder,
    MonitoringConfiguration,
    AlgorithmType,
    StepSizeConfig,
    GradientConfig,
    HessianConfig,
    LineSearchConfig,
    TrustRegionConfig,
    ConvergenceCriteria,
    PrecisionSettings,
    PerformanceSettings,
    DebugConfig,
    ValidationResult,
    ValidationStatus,
    MemoryEstimate,
    ComplexityEstimate,
    ComplexityClass,
};

// Re-exports from performance_tracking
pub use performance_tracking::{
    PerformanceTracker,
    PerformanceConfig,
    PerformanceConfigBuilder,
    IterationMetrics,
    OptimizationSession,
    TimingMetrics,
    MemoryMetrics,
    ConvergenceMetrics,
    PerformanceReport,
    PerformanceSummary,
    PerformanceAnalyzer,
    MonitoringFrequency,
    AnalysisFrequency,
    ProfilingLevel,
    AlertSystem,
    ExportConfiguration,
};

// Re-exports from core modules (preserved from original)
pub use gradient_core::{
    GradientBasedOptimizer,
    OptimizationProblem,
    Solution,
    OptimizationState,
    ConvergenceStatus,
    GradientDescentAlgorithm,
    QuasiNewtonMethod,
    TrustRegionMethod,
    LineSearchMethod,
    ConjugateGradientMethod,
    SecondOrderMethod,
    GradientDescentState,
    AdaptiveGradientState,
    QuasiNewtonParameters,
    TrustRegionParameters,
    LineSearchParameters,
};

pub use gradient_computation::{
    GradientEstimator,
    HessianEstimator,
    GradientComputationMethod,
    HessianComputationMethod,
    FiniteDifferenceParameters,
    AutoDiffConfiguration,
    GradientVerificationSettings,
    BFGSHessianApproximation,
    LBFGSHessianApproximation,
    SR1HessianApproximation,
};

pub use algorithm_selection::{
    AlgorithmSelector,
    StepSizeController,
    ConvergenceAnalyzer,
    GradientAlgorithmType,
    SelectionStrategy,
    AlgorithmSelection,
    ProblemCharacteristics,
    PerformanceBasedSelector,
    MLAlgorithmRecommender,
    StepSizeAdaptationStrategy,
    ConvergenceCriteriaConfig,
};

pub use problem_analysis::{
    ProblemAnalysisEngine,
    DimensionalAnalyzer,
    ConditioningAnalyzer,
    ConvexityAnalyzer,
    SparsityAnalyzer,
    SmoothnessAnalyzer,
    NoiseAnalyzer,
    ComprehensiveAnalysisResult,
    DimensionalAnalysisResult,
    ConditioningAnalysisResult,
    ProblemComplexityClass,
    AnalysisConfiguration,
};

pub use simd_acceleration::{
    SimdConfiguration,
    SimdInstructionSet,
    SimdOptimizationLevel,
};

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

// SciRS2 Core Dependencies
use scirs2_core::ndarray::{Array1, Array2};

// Use standard Rust Result type
type SklResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Central unified factory for the entire gradient optimization framework
///
/// This factory provides a single entry point for creating all optimization
/// components while ensuring proper configuration and integration between modules.
/// It builds upon the specialized factories from each module.
#[derive(Debug)]
pub struct UnifiedGradientOptimizationFactory {
    /// Core factory for basic components
    core_factory: Arc<GradientOptimizationFactory>,

    /// Performance tracker for monitoring
    performance_tracker: Arc<RwLock<PerformanceTracker>>,

    /// Default configurations
    default_configs: DefaultConfigurations,

    /// Factory statistics
    statistics: Arc<RwLock<UnifiedFactoryStatistics>>,
}

impl UnifiedGradientOptimizationFactory {
    /// Creates a new unified factory with default configuration.
    pub fn new() -> Self {
        Self::with_factory_config(FactoryConfiguration::default())
    }

    /// Creates a new unified factory with custom factory configuration.
    pub fn with_factory_config(config: FactoryConfiguration) -> Self {
        let core_factory = Arc::new(GradientOptimizationFactory::with_config(config));
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker::new()));

        Self {
            core_factory,
            performance_tracker,
            default_configs: DefaultConfigurations::new(),
            statistics: Arc::new(RwLock::new(UnifiedFactoryStatistics::new())),
        }
    }

    /// Creates a complete optimization system with default configuration.
    pub fn create_optimization_system(&self) -> SklResult<OptimizationSystem> {
        self.create_optimization_system_with_config(OptimizationConfig::default())
    }

    /// Creates a complete optimization system with custom configuration.
    pub fn create_optimization_system_with_config(
        &self,
        config: OptimizationConfig,
    ) -> SklResult<OptimizationSystem> {
        // Create core components
        let optimizer = self.core_factory.create_core_optimizer()?;
        let gradient_estimator = self.core_factory.create_gradient_estimator(
            config.gradient_config.computation_method.clone()
        )?;
        let algorithm_selector = self.core_factory.create_algorithm_selector(
            SelectionStrategy::PerformanceBased
        )?;
        let problem_analyzer = self.core_factory.create_problem_analyzer(
            AnalysisConfiguration::default()
        )?;

        // Create performance tracker instance
        let perf_config = PerformanceConfig::builder()
            .enable_memory_tracking(config.performance_settings.enable_parallel)
            .monitoring_frequency(MonitoringFrequency::EveryIteration)
            .build();
        let performance_tracker = PerformanceTracker::with_config(perf_config);

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.systems_created += 1;
        }

        Ok(OptimizationSystem {
            optimizer,
            gradient_estimator,
            algorithm_selector,
            problem_analyzer,
            performance_tracker,
            configuration: config,
        })
    }

    /// Creates an optimizer with automatic algorithm selection.
    pub fn create_adaptive_optimizer(
        &self,
        problem_characteristics: &ProblemCharacteristics,
    ) -> SklResult<Arc<GradientBasedOptimizer>> {
        // Use algorithm selector to choose best algorithm
        let selector = self.core_factory.create_algorithm_selector(
            SelectionStrategy::ProblemAware
        )?;

        // Get recommended configuration
        let selection = selector.select_algorithm(problem_characteristics)?;

        // Create optimizer with recommended configuration
        let config = self.create_config_from_selection(&selection)?;
        self.core_factory.create_core_optimizer_with_config(config)
    }

    /// Creates a high-performance optimizer configuration.
    pub fn create_high_performance_system(&self) -> SklResult<OptimizationSystem> {
        let config = OptimizationConfig::builder()
            .algorithm_type(AlgorithmType::LBFGS(
                configuration_management::LBFGSConfig::default()
            ))
            .max_iterations(10000)
            .tolerance(1e-8)
            .performance_settings(PerformanceSettings {
                enable_parallel: true,
                num_threads: Some(num_cpus::get()),
                enable_simd: true,
                memory_strategy: configuration_management::MemoryStrategy::Aggressive,
                cache_config: configuration_management::CacheConfig::default(),
            })
            .build();

        self.create_optimization_system_with_config(config)
    }

    /// Creates a memory-efficient optimizer configuration.
    pub fn create_memory_efficient_system(&self) -> SklResult<OptimizationSystem> {
        let config = OptimizationConfig::builder()
            .algorithm_type(AlgorithmType::ConjugateGradient)
            .performance_settings(PerformanceSettings {
                enable_parallel: false,
                num_threads: Some(1),
                enable_simd: false,
                memory_strategy: configuration_management::MemoryStrategy::Conservative,
                cache_config: configuration_management::CacheConfig {
                    enable_function_cache: false,
                    enable_gradient_cache: false,
                    cache_size_limit: 100,
                    eviction_policy: configuration_management::EvictionPolicy::LRU,
                },
            })
            .build();

        self.create_optimization_system_with_config(config)
    }

    /// Gets unified factory statistics.
    pub fn get_statistics(&self) -> SklResult<UnifiedFactoryStatistics> {
        let stats = self.statistics.read().unwrap();
        Ok(stats.clone())
    }

    // Private helper methods

    fn create_config_from_selection(
        &self,
        selection: &AlgorithmSelection,
    ) -> SklResult<configuration_management::OptimizerConfiguration> {
        Ok(configuration_management::OptimizerConfiguration {
            algorithm_type: format!("{:?}", selection.recommended_algorithm),
            parameters: selection.recommended_parameters.clone(),
        })
    }
}

impl Default for UnifiedGradientOptimizationFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete optimization system with all components
#[derive(Debug)]
pub struct OptimizationSystem {
    /// Core optimizer
    pub optimizer: Arc<GradientBasedOptimizer>,

    /// Gradient estimator
    pub gradient_estimator: Arc<GradientEstimator>,

    /// Algorithm selector
    pub algorithm_selector: Arc<AlgorithmSelector>,

    /// Problem analyzer
    pub problem_analyzer: Arc<ProblemAnalysisEngine>,

    /// Performance tracker
    pub performance_tracker: PerformanceTracker,

    /// System configuration
    pub configuration: OptimizationConfig,
}

impl OptimizationSystem {
    /// Optimizes a problem using the complete system.
    pub fn optimize(&mut self, problem: &OptimizationProblem) -> SklResult<OptimizationResult> {
        // Start performance tracking
        let session_id = self.performance_tracker.start_optimization(
            &format!("{:?}", self.configuration.algorithm_type)
        )?;

        // Analyze problem characteristics
        let analysis = self.problem_analyzer.analyze(problem)?;

        // Perform optimization with the core optimizer
        let result = self.optimizer.optimize(problem)?;

        // End performance tracking
        let performance_report = self.performance_tracker.end_optimization()?;

        Ok(OptimizationResult {
            solution: result.solution,
            convergence_info: result.convergence_info,
            performance_report: Some(performance_report),
            problem_analysis: Some(analysis),
            session_id: Some(session_id),
        })
    }

    /// Gets real-time optimization status.
    pub fn get_status(&self) -> SklResult<OptimizationStatus> {
        let performance_summary = self.performance_tracker.get_performance_summary()?;

        Ok(OptimizationStatus {
            session_id: performance_summary.session_id,
            algorithm: performance_summary.algorithm_name,
            elapsed_time: performance_summary.elapsed_time,
            iterations: performance_summary.iterations_completed,
            current_objective: performance_summary.current_function_value,
            convergence_rate: performance_summary.convergence_rate,
            estimated_remaining: performance_summary.estimated_time_remaining,
        })
    }
}

/// Result of optimization including all metadata
#[derive(Debug)]
pub struct OptimizationResult {
    /// The optimization solution
    pub solution: Solution,

    /// Convergence information
    pub convergence_info: ConvergenceStatus,

    /// Performance report
    pub performance_report: Option<PerformanceReport>,

    /// Problem analysis results
    pub problem_analysis: Option<ComprehensiveAnalysisResult>,

    /// Session identifier
    pub session_id: Option<String>,
}

/// Real-time optimization status
#[derive(Debug)]
pub struct OptimizationStatus {
    /// Session identifier
    pub session_id: String,

    /// Algorithm being used
    pub algorithm: String,

    /// Elapsed time
    pub elapsed_time: Duration,

    /// Iterations completed
    pub iterations: usize,

    /// Current objective value
    pub current_objective: f64,

    /// Current convergence rate
    pub convergence_rate: f64,

    /// Estimated time remaining
    pub estimated_remaining: Option<Duration>,
}

/// Default configurations for different scenarios
#[derive(Debug)]
pub struct DefaultConfigurations {
    /// High-performance configuration
    pub high_performance: OptimizationConfig,

    /// Memory-efficient configuration
    pub memory_efficient: OptimizationConfig,

    /// Robust configuration for noisy problems
    pub robust: OptimizationConfig,

    /// Fast configuration for quick results
    pub fast: OptimizationConfig,
}

impl DefaultConfigurations {
    pub fn new() -> Self {
        Self {
            high_performance: Self::create_high_performance_config(),
            memory_efficient: Self::create_memory_efficient_config(),
            robust: Self::create_robust_config(),
            fast: Self::create_fast_config(),
        }
    }

    fn create_high_performance_config() -> OptimizationConfig {
        OptimizationConfig::builder()
            .algorithm_type(AlgorithmType::LBFGS(
                configuration_management::LBFGSConfig::default()
            ))
            .max_iterations(10000)
            .tolerance(1e-8)
            .performance_settings(PerformanceSettings {
                enable_parallel: true,
                num_threads: None,
                enable_simd: true,
                memory_strategy: configuration_management::MemoryStrategy::Aggressive,
                cache_config: configuration_management::CacheConfig::default(),
            })
            .build()
    }

    fn create_memory_efficient_config() -> OptimizationConfig {
        OptimizationConfig::builder()
            .algorithm_type(AlgorithmType::ConjugateGradient)
            .max_iterations(5000)
            .tolerance(1e-6)
            .performance_settings(PerformanceSettings {
                enable_parallel: false,
                num_threads: Some(1),
                enable_simd: false,
                memory_strategy: configuration_management::MemoryStrategy::Conservative,
                cache_config: configuration_management::CacheConfig {
                    enable_function_cache: false,
                    enable_gradient_cache: false,
                    cache_size_limit: 50,
                    eviction_policy: configuration_management::EvictionPolicy::LRU,
                },
            })
            .build()
    }

    fn create_robust_config() -> OptimizationConfig {
        OptimizationConfig::builder()
            .algorithm_type(AlgorithmType::GradientDescent)
            .max_iterations(20000)
            .tolerance(1e-4)
            .step_size_config(StepSizeConfig {
                initial_step_size: 0.001,
                adaptation_method: configuration_management::StepSizeAdaptation::Adaptive,
                min_step_size: 1e-8,
                max_step_size: 0.1,
                reduction_factor: 0.8,
                increase_factor: 1.1,
            })
            .build()
    }

    fn create_fast_config() -> OptimizationConfig {
        OptimizationConfig::builder()
            .algorithm_type(AlgorithmType::GradientDescent)
            .max_iterations(1000)
            .tolerance(1e-3)
            .step_size_config(StepSizeConfig {
                initial_step_size: 0.1,
                adaptation_method: configuration_management::StepSizeAdaptation::Fixed,
                min_step_size: 1e-4,
                max_step_size: 1.0,
                reduction_factor: 0.5,
                increase_factor: 2.0,
            })
            .build()
    }
}

/// Statistics for the unified factory
#[derive(Debug, Clone)]
pub struct UnifiedFactoryStatistics {
    /// Number of optimization systems created
    pub systems_created: usize,

    /// Number of individual components created
    pub components_created: usize,

    /// Total optimization time across all systems
    pub total_optimization_time: Duration,

    /// Average optimization time
    pub average_optimization_time: Duration,

    /// Success rate
    pub success_rate: f64,

    /// Most used algorithm
    pub most_used_algorithm: Option<String>,
}

impl UnifiedFactoryStatistics {
    pub fn new() -> Self {
        Self {
            systems_created: 0,
            components_created: 0,
            total_optimization_time: Duration::new(0, 0),
            average_optimization_time: Duration::new(0, 0),
            success_rate: 0.0,
            most_used_algorithm: None,
        }
    }
}

/// Convenience functions for common optimization scenarios

/// Creates a default optimization system for general use.
pub fn create_default_optimizer() -> SklResult<OptimizationSystem> {
    let factory = UnifiedGradientOptimizationFactory::new();
    factory.create_optimization_system()
}

/// Creates a high-performance optimization system.
pub fn create_high_performance_optimizer() -> SklResult<OptimizationSystem> {
    let factory = UnifiedGradientOptimizationFactory::new();
    factory.create_high_performance_system()
}

/// Creates a memory-efficient optimization system.
pub fn create_memory_efficient_optimizer() -> SklResult<OptimizationSystem> {
    let factory = UnifiedGradientOptimizationFactory::new();
    factory.create_memory_efficient_system()
}

/// Creates an optimizer adapted to specific problem characteristics.
pub fn create_adaptive_optimizer(
    problem_characteristics: &ProblemCharacteristics,
) -> SklResult<Arc<GradientBasedOptimizer>> {
    let factory = UnifiedGradientOptimizationFactory::new();
    factory.create_adaptive_optimizer(problem_characteristics)
}

// External dependencies
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_factory_creation() {
        let factory = UnifiedGradientOptimizationFactory::new();
        let stats = factory.get_statistics().unwrap();

        assert_eq!(stats.systems_created, 0);
        assert_eq!(stats.components_created, 0);
    }

    #[test]
    fn test_default_configurations() {
        let configs = DefaultConfigurations::new();

        // Test that all configurations are valid
        assert!(configs.high_performance.validate().unwrap().status == ValidationStatus::Valid
                || configs.high_performance.validate().unwrap().status == ValidationStatus::ValidWithWarnings);
        assert!(configs.memory_efficient.validate().unwrap().status == ValidationStatus::Valid
                || configs.memory_efficient.validate().unwrap().status == ValidationStatus::ValidWithWarnings);
        assert!(configs.robust.validate().unwrap().status == ValidationStatus::Valid
                || configs.robust.validate().unwrap().status == ValidationStatus::ValidWithWarnings);
        assert!(configs.fast.validate().unwrap().status == ValidationStatus::Valid
                || configs.fast.validate().unwrap().status == ValidationStatus::ValidWithWarnings);
    }

    #[test]
    fn test_convenience_functions() {
        // Test that convenience functions can be called without error
        let _default_optimizer = create_default_optimizer();
        let _high_perf_optimizer = create_high_performance_optimizer();
        let _memory_efficient_optimizer = create_memory_efficient_optimizer();
    }

    #[test]
    fn test_optimization_system_creation() {
        let factory = UnifiedGradientOptimizationFactory::new();
        let result = factory.create_optimization_system();

        assert!(result.is_ok());

        if let Ok(system) = result {
            assert_eq!(system.configuration.algorithm_type, AlgorithmType::GradientDescent);
        }
    }

    #[test]
    fn test_factory_statistics_update() {
        let factory = UnifiedGradientOptimizationFactory::new();

        // Create a system to update statistics
        let _system = factory.create_optimization_system().unwrap();

        let stats = factory.get_statistics().unwrap();
        assert_eq!(stats.systems_created, 1);
    }
}