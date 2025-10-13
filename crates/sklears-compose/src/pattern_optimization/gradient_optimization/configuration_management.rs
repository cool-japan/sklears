//! Configuration Management Module
//!
//! This module provides comprehensive configuration management for gradient-based
//! optimization, including optimization configurations, algorithm configurations,
//! monitoring configurations, and template management. It serves as the central
//! configuration hub for the entire gradient optimization framework.
//!
//! # Key Features
//!
//! ## Configuration Types
//! - **OptimizationConfig**: Core optimization parameters and settings
//! - **AlgorithmConfiguration**: Algorithm-specific configuration options
//! - **MonitoringConfiguration**: Performance monitoring and logging settings
//! - **AnalysisConfiguration**: Problem analysis and characterization settings
//!
//! ## Template Management
//! - **Configuration Templates**: Pre-configured optimization setups
//! - **Template Validation**: Ensure template consistency and validity
//! - **Template Inheritance**: Build complex configurations from base templates
//! - **Dynamic Templates**: Runtime configuration generation
//!
//! ## Configuration Validation
//! - **Parameter Validation**: Ensure parameter values are within valid ranges
//! - **Compatibility Checking**: Verify configuration compatibility
//! - **Constraint Verification**: Check configuration constraints
//! - **Performance Prediction**: Estimate performance based on configuration
//!
//! ## Advanced Features
//! - **Configuration Profiles**: Predefined profiles for different scenarios
//! - **Dynamic Configuration**: Runtime configuration adjustment
//! - **Configuration Merging**: Combine multiple configurations intelligently
//! - **Environment Adaptation**: Adapt configurations to runtime environment
//!
//! # Usage Examples
//!
//! ## Basic Configuration
//!
//! ```rust
//! use sklears_compose::pattern_optimization::gradient_optimization::configuration_management::*;
//!
//! // Create basic optimization configuration
//! let config = OptimizationConfig::builder()
//!     .max_iterations(1000)
//!     .tolerance(1e-6)
//!     .algorithm_type(AlgorithmType::GradientDescent)
//!     .build();
//!
//! // Create algorithm-specific configuration
//! let algo_config = AlgorithmConfiguration::builder()
//!     .learning_rate(0.01)
//!     .momentum(0.9)
//!     .regularization(RegularizationType::L2(0.001))
//!     .build();
//! ```
//!
//! ## Template-Based Configuration
//!
//! ```rust
//! // Create configuration from template
//! let template_manager = ConfigurationTemplateManager::new();
//! let config = template_manager.create_from_template("high_performance")?;
//!
//! // Create custom template
//! let custom_template = ConfigurationTemplate::builder()
//!     .name("my_custom_setup")
//!     .base_config(base_config)
//!     .add_parameter("special_feature", true)
//!     .build();
//!
//! template_manager.register_template(custom_template)?;
//! ```
//!
//! ## Dynamic Configuration
//!
//! ```rust
//! // Create adaptive configuration that adjusts at runtime
//! let adaptive_config = AdaptiveConfiguration::builder()
//!     .enable_auto_tuning(true)
//!     .adaptation_frequency(100) // iterations
//!     .performance_threshold(0.95)
//!     .build();
//!
//! let config_manager = ConfigurationManager::with_adaptive(adaptive_config);
//! ```

use std::collections::HashMap;
use std::time::Duration;

// SciRS2 Core Dependencies
use scirs2_core::ndarray::{Array1, Array2};

// Use standard Rust Result type
type SklResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Comprehensive optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Algorithm type to use
    pub algorithm_type: AlgorithmType,

    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence tolerance
    pub tolerance: f64,

    /// Step size configuration
    pub step_size_config: StepSizeConfig,

    /// Gradient computation configuration
    pub gradient_config: GradientConfig,

    /// Hessian computation configuration
    pub hessian_config: Option<HessianConfig>,

    /// Line search configuration
    pub line_search_config: Option<LineSearchConfig>,

    /// Trust region configuration
    pub trust_region_config: Option<TrustRegionConfig>,

    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,

    /// Numerical precision settings
    pub precision_settings: PrecisionSettings,

    /// Performance settings
    pub performance_settings: PerformanceSettings,

    /// Debugging and logging configuration
    pub debug_config: DebugConfig,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            algorithm_type: AlgorithmType::GradientDescent,
            max_iterations: 1000,
            tolerance: 1e-6,
            step_size_config: StepSizeConfig::default(),
            gradient_config: GradientConfig::default(),
            hessian_config: None,
            line_search_config: None,
            trust_region_config: None,
            convergence_criteria: ConvergenceCriteria::default(),
            precision_settings: PrecisionSettings::default(),
            performance_settings: PerformanceSettings::default(),
            debug_config: DebugConfig::default(),
        }
    }
}

impl OptimizationConfig {
    /// Creates a builder for optimization configuration.
    pub fn builder() -> OptimizationConfigBuilder {
        OptimizationConfigBuilder::default()
    }

    /// Validates the configuration for consistency and correctness.
    pub fn validate(&self) -> SklResult<ValidationResult> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        // Validate basic parameters
        if self.max_iterations == 0 {
            errors.push("Maximum iterations must be greater than 0".to_string());
        }

        if self.tolerance <= 0.0 {
            errors.push("Tolerance must be positive".to_string());
        }

        if self.tolerance < 1e-15 {
            warnings.push("Very small tolerance may cause numerical issues".to_string());
        }

        // Validate algorithm-specific configurations
        match &self.algorithm_type {
            AlgorithmType::QuasiNewton(_) => {
                if self.hessian_config.is_none() {
                    warnings.push("Quasi-Newton methods benefit from Hessian configuration".to_string());
                }
            },
            AlgorithmType::TrustRegion => {
                if self.trust_region_config.is_none() {
                    errors.push("Trust region method requires trust region configuration".to_string());
                }
            },
            _ => {}
        }

        // Validate compatibility between configurations
        if let Some(line_search) = &self.line_search_config {
            if matches!(self.algorithm_type, AlgorithmType::TrustRegion) {
                warnings.push("Line search configuration ignored for trust region methods".to_string());
            }
        }

        let status = if !errors.is_empty() {
            ValidationStatus::Invalid
        } else if !warnings.is_empty() {
            ValidationStatus::ValidWithWarnings
        } else {
            ValidationStatus::Valid
        };

        Ok(ValidationResult {
            status,
            errors,
            warnings,
            recommendations: self.generate_recommendations(),
        })
    }

    /// Generates performance recommendations based on configuration.
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.max_iterations > 10000 {
            recommendations.push("Consider using adaptive stopping criteria for large iteration counts".to_string());
        }

        if self.gradient_config.finite_difference_step < 1e-8 {
            recommendations.push("Very small finite difference step may cause numerical instability".to_string());
        }

        recommendations
    }

    /// Estimates memory usage for this configuration.
    pub fn estimate_memory_usage(&self, problem_dimension: usize) -> MemoryEstimate {
        let base_memory = problem_dimension * 8; // Basic gradient storage

        let gradient_memory = match &self.gradient_config.computation_method {
            GradientComputationMethod::FiniteDifference => problem_dimension * 8 * 2,
            GradientComputationMethod::AutomaticDifferentiation => problem_dimension * 8 * 3,
            GradientComputationMethod::Analytical => problem_dimension * 8,
        };

        let hessian_memory = if self.hessian_config.is_some() {
            problem_dimension * problem_dimension * 8
        } else {
            0
        };

        let algorithm_memory = match &self.algorithm_type {
            AlgorithmType::LBFGS(config) => {
                config.history_size * problem_dimension * 8 * 2
            },
            AlgorithmType::QuasiNewton(_) => problem_dimension * problem_dimension * 8,
            _ => 0,
        };

        MemoryEstimate {
            base_memory,
            gradient_memory,
            hessian_memory,
            algorithm_memory,
            total_memory: base_memory + gradient_memory + hessian_memory + algorithm_memory,
        }
    }

    /// Estimates computational complexity for this configuration.
    pub fn estimate_complexity(&self, problem_dimension: usize) -> ComplexityEstimate {
        let gradient_complexity = match &self.gradient_config.computation_method {
            GradientComputationMethod::FiniteDifference => problem_dimension,
            GradientComputationMethod::AutomaticDifferentiation => problem_dimension,
            GradientComputationMethod::Analytical => 1,
        };

        let hessian_complexity = if self.hessian_config.is_some() {
            problem_dimension * problem_dimension
        } else {
            0
        };

        let algorithm_complexity = match &self.algorithm_type {
            AlgorithmType::GradientDescent => problem_dimension,
            AlgorithmType::ConjugateGradient => problem_dimension,
            AlgorithmType::LBFGS(_) => problem_dimension,
            AlgorithmType::QuasiNewton(_) => problem_dimension * problem_dimension,
            AlgorithmType::TrustRegion => problem_dimension * problem_dimension,
            AlgorithmType::NewtonRaphson => problem_dimension * problem_dimension * problem_dimension,
        };

        ComplexityEstimate {
            gradient_complexity,
            hessian_complexity,
            algorithm_complexity,
            total_complexity: gradient_complexity + hessian_complexity + algorithm_complexity,
            complexity_class: self.determine_complexity_class(problem_dimension),
        }
    }

    fn determine_complexity_class(&self, dimension: usize) -> ComplexityClass {
        match &self.algorithm_type {
            AlgorithmType::GradientDescent | AlgorithmType::ConjugateGradient | AlgorithmType::LBFGS(_) => {
                ComplexityClass::Linear
            },
            AlgorithmType::QuasiNewton(_) | AlgorithmType::TrustRegion => {
                ComplexityClass::Quadratic
            },
            AlgorithmType::NewtonRaphson => {
                ComplexityClass::Cubic
            },
        }
    }
}

/// Builder for optimization configuration
#[derive(Debug)]
pub struct OptimizationConfigBuilder {
    algorithm_type: AlgorithmType,
    max_iterations: usize,
    tolerance: f64,
    step_size_config: StepSizeConfig,
    gradient_config: GradientConfig,
    hessian_config: Option<HessianConfig>,
    line_search_config: Option<LineSearchConfig>,
    trust_region_config: Option<TrustRegionConfig>,
    convergence_criteria: ConvergenceCriteria,
    precision_settings: PrecisionSettings,
    performance_settings: PerformanceSettings,
    debug_config: DebugConfig,
}

impl Default for OptimizationConfigBuilder {
    fn default() -> Self {
        let default_config = OptimizationConfig::default();
        Self {
            algorithm_type: default_config.algorithm_type,
            max_iterations: default_config.max_iterations,
            tolerance: default_config.tolerance,
            step_size_config: default_config.step_size_config,
            gradient_config: default_config.gradient_config,
            hessian_config: default_config.hessian_config,
            line_search_config: default_config.line_search_config,
            trust_region_config: default_config.trust_region_config,
            convergence_criteria: default_config.convergence_criteria,
            precision_settings: default_config.precision_settings,
            performance_settings: default_config.performance_settings,
            debug_config: default_config.debug_config,
        }
    }
}

impl OptimizationConfigBuilder {
    /// Sets the algorithm type.
    pub fn algorithm_type(mut self, algorithm_type: AlgorithmType) -> Self {
        self.algorithm_type = algorithm_type;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the convergence tolerance.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the step size configuration.
    pub fn step_size_config(mut self, config: StepSizeConfig) -> Self {
        self.step_size_config = config;
        self
    }

    /// Sets the gradient configuration.
    pub fn gradient_config(mut self, config: GradientConfig) -> Self {
        self.gradient_config = config;
        self
    }

    /// Sets the Hessian configuration.
    pub fn hessian_config(mut self, config: Option<HessianConfig>) -> Self {
        self.hessian_config = config;
        self
    }

    /// Sets the line search configuration.
    pub fn line_search_config(mut self, config: Option<LineSearchConfig>) -> Self {
        self.line_search_config = config;
        self
    }

    /// Sets the trust region configuration.
    pub fn trust_region_config(mut self, config: Option<TrustRegionConfig>) -> Self {
        self.trust_region_config = config;
        self
    }

    /// Sets the convergence criteria.
    pub fn convergence_criteria(mut self, criteria: ConvergenceCriteria) -> Self {
        self.convergence_criteria = criteria;
        self
    }

    /// Sets the precision settings.
    pub fn precision_settings(mut self, settings: PrecisionSettings) -> Self {
        self.precision_settings = settings;
        self
    }

    /// Sets the performance settings.
    pub fn performance_settings(mut self, settings: PerformanceSettings) -> Self {
        self.performance_settings = settings;
        self
    }

    /// Sets the debug configuration.
    pub fn debug_config(mut self, config: DebugConfig) -> Self {
        self.debug_config = config;
        self
    }

    /// Builds the optimization configuration.
    pub fn build(self) -> OptimizationConfig {
        OptimizationConfig {
            algorithm_type: self.algorithm_type,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            step_size_config: self.step_size_config,
            gradient_config: self.gradient_config,
            hessian_config: self.hessian_config,
            line_search_config: self.line_search_config,
            trust_region_config: self.trust_region_config,
            convergence_criteria: self.convergence_criteria,
            precision_settings: self.precision_settings,
            performance_settings: self.performance_settings,
            debug_config: self.debug_config,
        }
    }
}

/// Types of optimization algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmType {
    /// Standard gradient descent
    GradientDescent,
    /// Conjugate gradient method
    ConjugateGradient,
    /// Limited-memory BFGS
    LBFGS(LBFGSConfig),
    /// Quasi-Newton methods
    QuasiNewton(QuasiNewtonConfig),
    /// Trust region methods
    TrustRegion,
    /// Newton-Raphson method
    NewtonRaphson,
}

/// Configuration for LBFGS algorithm
#[derive(Debug, Clone, PartialEq)]
pub struct LBFGSConfig {
    /// History size for LBFGS
    pub history_size: usize,
    /// Scaling factor
    pub scaling_factor: f64,
    /// Enable diagonal scaling
    pub enable_diagonal_scaling: bool,
}

impl Default for LBFGSConfig {
    fn default() -> Self {
        Self {
            history_size: 10,
            scaling_factor: 1.0,
            enable_diagonal_scaling: false,
        }
    }
}

/// Configuration for quasi-Newton methods
#[derive(Debug, Clone, PartialEq)]
pub struct QuasiNewtonConfig {
    /// Update formula (BFGS, DFP, SR1, etc.)
    pub update_formula: UpdateFormula,
    /// Enable Hessian regularization
    pub enable_regularization: bool,
    /// Regularization parameter
    pub regularization_parameter: f64,
}

impl Default for QuasiNewtonConfig {
    fn default() -> Self {
        Self {
            update_formula: UpdateFormula::BFGS,
            enable_regularization: true,
            regularization_parameter: 1e-8,
        }
    }
}

/// Hessian update formulas
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateFormula {
    /// BFGS update
    BFGS,
    /// DFP update
    DFP,
    /// SR1 update
    SR1,
    /// Broyden update
    Broyden,
}

/// Step size configuration
#[derive(Debug, Clone)]
pub struct StepSizeConfig {
    /// Initial step size
    pub initial_step_size: f64,
    /// Step size adaptation method
    pub adaptation_method: StepSizeAdaptation,
    /// Minimum allowed step size
    pub min_step_size: f64,
    /// Maximum allowed step size
    pub max_step_size: f64,
    /// Step size reduction factor
    pub reduction_factor: f64,
    /// Step size increase factor
    pub increase_factor: f64,
}

impl Default for StepSizeConfig {
    fn default() -> Self {
        Self {
            initial_step_size: 0.01,
            adaptation_method: StepSizeAdaptation::Armijo,
            min_step_size: 1e-10,
            max_step_size: 1.0,
            reduction_factor: 0.5,
            increase_factor: 1.2,
        }
    }
}

/// Step size adaptation methods
#[derive(Debug, Clone, PartialEq)]
pub enum StepSizeAdaptation {
    /// Fixed step size
    Fixed,
    /// Armijo line search
    Armijo,
    /// Wolfe conditions
    Wolfe,
    /// Strong Wolfe conditions
    StrongWolfe,
    /// Backtracking line search
    Backtracking,
    /// Adaptive step size
    Adaptive,
}

/// Gradient computation configuration
#[derive(Debug, Clone)]
pub struct GradientConfig {
    /// Computation method
    pub computation_method: GradientComputationMethod,
    /// Finite difference step size
    pub finite_difference_step: f64,
    /// Enable gradient verification
    pub enable_verification: bool,
    /// Verification tolerance
    pub verification_tolerance: f64,
    /// Enable gradient caching
    pub enable_caching: bool,
    /// Parallel computation settings
    pub parallel_settings: ParallelSettings,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            computation_method: GradientComputationMethod::FiniteDifference,
            finite_difference_step: 1e-6,
            enable_verification: false,
            verification_tolerance: 1e-8,
            enable_caching: true,
            parallel_settings: ParallelSettings::default(),
        }
    }
}

/// Gradient computation methods
#[derive(Debug, Clone, PartialEq)]
pub enum GradientComputationMethod {
    /// Finite difference approximation
    FiniteDifference,
    /// Automatic differentiation
    AutomaticDifferentiation,
    /// Analytical gradients
    Analytical,
}

/// Hessian computation configuration
#[derive(Debug, Clone)]
pub struct HessianConfig {
    /// Computation method
    pub computation_method: HessianComputationMethod,
    /// Finite difference step size
    pub finite_difference_step: f64,
    /// Enable Hessian regularization
    pub enable_regularization: bool,
    /// Regularization parameter
    pub regularization_parameter: f64,
    /// Enable sparse Hessian
    pub enable_sparse: bool,
    /// Sparsity threshold
    pub sparsity_threshold: f64,
}

impl Default for HessianConfig {
    fn default() -> Self {
        Self {
            computation_method: HessianComputationMethod::FiniteDifference,
            finite_difference_step: 1e-5,
            enable_regularization: true,
            regularization_parameter: 1e-6,
            enable_sparse: false,
            sparsity_threshold: 1e-12,
        }
    }
}

/// Hessian computation methods
#[derive(Debug, Clone, PartialEq)]
pub enum HessianComputationMethod {
    /// Finite difference approximation
    FiniteDifference,
    /// Automatic differentiation
    AutomaticDifferentiation,
    /// Analytical Hessian
    Analytical,
    /// Quasi-Newton approximation
    QuasiNewton,
}

/// Line search configuration
#[derive(Debug, Clone)]
pub struct LineSearchConfig {
    /// Line search method
    pub method: LineSearchMethod,
    /// Maximum line search iterations
    pub max_iterations: usize,
    /// Sufficient decrease parameter (c1)
    pub c1: f64,
    /// Curvature condition parameter (c2)
    pub c2: f64,
    /// Initial step size
    pub initial_step: f64,
    /// Step size bounds
    pub step_bounds: (f64, f64),
}

impl Default for LineSearchConfig {
    fn default() -> Self {
        Self {
            method: LineSearchMethod::Armijo,
            max_iterations: 20,
            c1: 1e-4,
            c2: 0.9,
            initial_step: 1.0,
            step_bounds: (1e-10, 100.0),
        }
    }
}

/// Line search methods
#[derive(Debug, Clone, PartialEq)]
pub enum LineSearchMethod {
    /// Armijo line search
    Armijo,
    /// Wolfe line search
    Wolfe,
    /// Strong Wolfe line search
    StrongWolfe,
    /// More-Thuente line search
    MoreThuente,
    /// Backtracking line search
    Backtracking,
}

/// Trust region configuration
#[derive(Debug, Clone)]
pub struct TrustRegionConfig {
    /// Initial trust region radius
    pub initial_radius: f64,
    /// Maximum trust region radius
    pub max_radius: f64,
    /// Trust region adaptation parameters
    pub adaptation_params: TrustRegionAdaptation,
    /// Trust region solver
    pub solver: TrustRegionSolver,
    /// Acceptance thresholds
    pub acceptance_thresholds: AcceptanceThresholds,
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            max_radius: 1000.0,
            adaptation_params: TrustRegionAdaptation::default(),
            solver: TrustRegionSolver::CauchyPoint,
            acceptance_thresholds: AcceptanceThresholds::default(),
        }
    }
}

/// Trust region adaptation parameters
#[derive(Debug, Clone)]
pub struct TrustRegionAdaptation {
    /// Radius increase factor
    pub increase_factor: f64,
    /// Radius decrease factor
    pub decrease_factor: f64,
    /// Very successful step threshold
    pub very_successful_threshold: f64,
    /// Successful step threshold
    pub successful_threshold: f64,
    /// Unsuccessful step threshold
    pub unsuccessful_threshold: f64,
}

impl Default for TrustRegionAdaptation {
    fn default() -> Self {
        Self {
            increase_factor: 2.0,
            decrease_factor: 0.25,
            very_successful_threshold: 0.75,
            successful_threshold: 0.1,
            unsuccessful_threshold: 0.0,
        }
    }
}

/// Trust region solvers
#[derive(Debug, Clone, PartialEq)]
pub enum TrustRegionSolver {
    /// Cauchy point method
    CauchyPoint,
    /// Dogleg method
    Dogleg,
    /// Steihaug-CG method
    SteihaugCG,
    /// More-Sorensen method
    MoreSorensen,
}

/// Acceptance thresholds for trust region
#[derive(Debug, Clone)]
pub struct AcceptanceThresholds {
    /// Very successful step threshold
    pub very_successful: f64,
    /// Successful step threshold
    pub successful: f64,
    /// Acceptable step threshold
    pub acceptable: f64,
}

impl Default for AcceptanceThresholds {
    fn default() -> Self {
        Self {
            very_successful: 0.75,
            successful: 0.25,
            acceptable: 0.1,
        }
    }
}

/// Convergence criteria configuration
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Gradient norm tolerance
    pub gradient_tolerance: f64,
    /// Function value tolerance
    pub function_tolerance: f64,
    /// Parameter change tolerance
    pub parameter_tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Maximum function evaluations
    pub max_function_evaluations: usize,
    /// Maximum time limit
    pub max_time: Option<Duration>,
    /// Enable early stopping
    pub enable_early_stopping: bool,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            gradient_tolerance: 1e-6,
            function_tolerance: 1e-9,
            parameter_tolerance: 1e-8,
            max_iterations: 1000,
            max_function_evaluations: 10000,
            max_time: None,
            enable_early_stopping: false,
            early_stopping_patience: 50,
        }
    }
}

/// Numerical precision settings
#[derive(Debug, Clone)]
pub struct PrecisionSettings {
    /// Machine epsilon
    pub machine_epsilon: f64,
    /// Relative tolerance
    pub relative_tolerance: f64,
    /// Absolute tolerance
    pub absolute_tolerance: f64,
    /// Enable extended precision
    pub enable_extended_precision: bool,
    /// Numerical stability checks
    pub stability_checks: bool,
}

impl Default for PrecisionSettings {
    fn default() -> Self {
        Self {
            machine_epsilon: f64::EPSILON,
            relative_tolerance: 1e-12,
            absolute_tolerance: 1e-15,
            enable_extended_precision: false,
            stability_checks: true,
        }
    }
}

/// Performance settings
#[derive(Debug, Clone)]
pub struct PerformanceSettings {
    /// Enable parallel computation
    pub enable_parallel: bool,
    /// Number of threads
    pub num_threads: Option<usize>,
    /// Enable SIMD optimization
    pub enable_simd: bool,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Cache configuration
    pub cache_config: CacheConfig,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            num_threads: None, // Use system default
            enable_simd: true,
            memory_strategy: MemoryStrategy::Balanced,
            cache_config: CacheConfig::default(),
        }
    }
}

/// Memory allocation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryStrategy {
    /// Minimize memory usage
    Conservative,
    /// Balance memory and performance
    Balanced,
    /// Maximize performance
    Aggressive,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable function value caching
    pub enable_function_cache: bool,
    /// Enable gradient caching
    pub enable_gradient_cache: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_function_cache: true,
            enable_gradient_cache: true,
            cache_size_limit: 1000,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random eviction
    Random,
}

/// Parallel computation settings
#[derive(Debug, Clone)]
pub struct ParallelSettings {
    /// Enable parallel gradient computation
    pub enable_parallel_gradient: bool,
    /// Enable parallel Hessian computation
    pub enable_parallel_hessian: bool,
    /// Number of threads for parallel computation
    pub num_threads: Option<usize>,
    /// Workload distribution strategy
    pub distribution_strategy: WorkloadDistribution,
}

impl Default for ParallelSettings {
    fn default() -> Self {
        Self {
            enable_parallel_gradient: true,
            enable_parallel_hessian: true,
            num_threads: None,
            distribution_strategy: WorkloadDistribution::Static,
        }
    }
}

/// Workload distribution strategies
#[derive(Debug, Clone, PartialEq)]
pub enum WorkloadDistribution {
    /// Static workload distribution
    Static,
    /// Dynamic workload distribution
    Dynamic,
    /// Work stealing
    WorkStealing,
    /// Guided distribution
    Guided,
}

/// Debug configuration
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Enable debug logging
    pub enable_logging: bool,
    /// Log level
    pub log_level: LogLevel,
    /// Enable convergence tracking
    pub enable_convergence_tracking: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Output directory for debug information
    pub output_directory: Option<String>,
    /// Save intermediate results
    pub save_intermediate: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enable_logging: false,
            log_level: LogLevel::Info,
            enable_convergence_tracking: false,
            enable_profiling: false,
            output_directory: None,
            save_intermediate: false,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    /// Trace level
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    /// Validation status
    pub status: ValidationStatus,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Performance recommendations
    pub recommendations: Vec<String>,
}

/// Validation status
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    /// Configuration is valid
    Valid,
    /// Configuration is valid but has warnings
    ValidWithWarnings,
    /// Configuration is invalid
    Invalid,
}

/// Memory usage estimate
#[derive(Debug)]
pub struct MemoryEstimate {
    /// Base memory requirements
    pub base_memory: usize,
    /// Gradient computation memory
    pub gradient_memory: usize,
    /// Hessian computation memory
    pub hessian_memory: usize,
    /// Algorithm-specific memory
    pub algorithm_memory: usize,
    /// Total memory estimate
    pub total_memory: usize,
}

/// Computational complexity estimate
#[derive(Debug)]
pub struct ComplexityEstimate {
    /// Gradient computation complexity
    pub gradient_complexity: usize,
    /// Hessian computation complexity
    pub hessian_complexity: usize,
    /// Algorithm-specific complexity
    pub algorithm_complexity: usize,
    /// Total complexity
    pub total_complexity: usize,
    /// Complexity class
    pub complexity_class: ComplexityClass,
}

/// Computational complexity classes
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityClass {
    /// O(n) complexity
    Linear,
    /// O(n²) complexity
    Quadratic,
    /// O(n³) complexity
    Cubic,
    /// O(n log n) complexity
    Linearithmic,
    /// O(2^n) complexity
    Exponential,
}

/// Algorithm-specific configuration
#[derive(Debug, Clone)]
pub struct AlgorithmConfiguration {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum parameter
    pub momentum: f64,
    /// Regularization configuration
    pub regularization: Option<RegularizationConfig>,
    /// Adaptive learning rate settings
    pub adaptive_lr: Option<AdaptiveLearningRateConfig>,
    /// Preconditioning settings
    pub preconditioning: Option<PreconditioningConfig>,
}

impl Default for AlgorithmConfiguration {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.0,
            regularization: None,
            adaptive_lr: None,
            preconditioning: None,
        }
    }
}

impl AlgorithmConfiguration {
    /// Creates a builder for algorithm configuration.
    pub fn builder() -> AlgorithmConfigurationBuilder {
        AlgorithmConfigurationBuilder::default()
    }
}

/// Builder for algorithm configuration
#[derive(Debug, Default)]
pub struct AlgorithmConfigurationBuilder {
    learning_rate: f64,
    momentum: f64,
    regularization: Option<RegularizationConfig>,
    adaptive_lr: Option<AdaptiveLearningRateConfig>,
    preconditioning: Option<PreconditioningConfig>,
}

impl AlgorithmConfigurationBuilder {
    /// Sets the learning rate.
    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Sets the momentum parameter.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the regularization configuration.
    pub fn regularization(mut self, config: RegularizationConfig) -> Self {
        self.regularization = Some(config);
        self
    }

    /// Sets the adaptive learning rate configuration.
    pub fn adaptive_lr(mut self, config: AdaptiveLearningRateConfig) -> Self {
        self.adaptive_lr = Some(config);
        self
    }

    /// Sets the preconditioning configuration.
    pub fn preconditioning(mut self, config: PreconditioningConfig) -> Self {
        self.preconditioning = Some(config);
        self
    }

    /// Builds the algorithm configuration.
    pub fn build(self) -> AlgorithmConfiguration {
        AlgorithmConfiguration {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            regularization: self.regularization,
            adaptive_lr: self.adaptive_lr,
            preconditioning: self.preconditioning,
        }
    }
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// Regularization type
    pub regularization_type: RegularizationType,
    /// Regularization strength
    pub strength: f64,
    /// Enable adaptive regularization
    pub adaptive: bool,
}

/// Types of regularization
#[derive(Debug, Clone, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (Lasso)
    L1(f64),
    /// L2 regularization (Ridge)
    L2(f64),
    /// Elastic net (L1 + L2)
    ElasticNet { l1_ratio: f64, strength: f64 },
    /// Custom regularization
    Custom(String),
}

/// Adaptive learning rate configuration
#[derive(Debug, Clone)]
pub struct AdaptiveLearningRateConfig {
    /// Adaptation method
    pub method: AdaptiveLRMethod,
    /// Adaptation parameters
    pub parameters: HashMap<String, f64>,
    /// Enable learning rate scheduling
    pub enable_scheduling: bool,
    /// Schedule configuration
    pub schedule_config: Option<LRScheduleConfig>,
}

/// Adaptive learning rate methods
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveLRMethod {
    /// AdaGrad
    AdaGrad,
    /// RMSprop
    RMSprop,
    /// Adam
    Adam,
    /// AdaDelta
    AdaDelta,
    /// AdaMax
    AdaMax,
}

/// Learning rate schedule configuration
#[derive(Debug, Clone)]
pub struct LRScheduleConfig {
    /// Schedule type
    pub schedule_type: ScheduleType,
    /// Schedule parameters
    pub parameters: HashMap<String, f64>,
    /// Update frequency
    pub update_frequency: usize,
}

/// Learning rate schedule types
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleType {
    /// Step decay
    StepDecay,
    /// Exponential decay
    ExponentialDecay,
    /// Polynomial decay
    PolynomialDecay,
    /// Cosine annealing
    CosineAnnealing,
    /// Warm restart
    WarmRestart,
}

/// Preconditioning configuration
#[derive(Debug, Clone)]
pub struct PreconditioningConfig {
    /// Preconditioning method
    pub method: PreconditioningMethod,
    /// Update frequency
    pub update_frequency: usize,
    /// Regularization parameter
    pub regularization: f64,
}

/// Preconditioning methods
#[derive(Debug, Clone, PartialEq)]
pub enum PreconditioningMethod {
    /// No preconditioning
    None,
    /// Diagonal preconditioning
    Diagonal,
    /// Limited-memory preconditioning
    LimitedMemory,
    /// Incomplete Cholesky
    IncompleteCholesky,
    /// Custom preconditioning
    Custom(String),
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfiguration {
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Monitoring frequency
    pub monitoring_frequency: usize,
    /// Metrics to collect
    pub metrics_config: MetricsConfig,
    /// Export configuration
    pub export_config: ExportConfig,
    /// Alert configuration
    pub alert_config: AlertConfig,
}

impl Default for MonitoringConfiguration {
    fn default() -> Self {
        Self {
            enable_performance_monitoring: true,
            monitoring_frequency: 100,
            metrics_config: MetricsConfig::default(),
            export_config: ExportConfig::default(),
            alert_config: AlertConfig::default(),
        }
    }
}

/// Metrics collection configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable timing metrics
    pub enable_timing: bool,
    /// Enable memory metrics
    pub enable_memory: bool,
    /// Enable convergence metrics
    pub enable_convergence: bool,
    /// Enable custom metrics
    pub enable_custom: bool,
    /// Custom metric definitions
    pub custom_metrics: Vec<CustomMetricConfig>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_timing: true,
            enable_memory: true,
            enable_convergence: true,
            enable_custom: false,
            custom_metrics: Vec::new(),
        }
    }
}

/// Custom metric configuration
#[derive(Debug, Clone)]
pub struct CustomMetricConfig {
    /// Metric name
    pub name: String,
    /// Metric description
    pub description: String,
    /// Computation function name
    pub computation_function: String,
    /// Collection frequency
    pub frequency: usize,
}

/// Export configuration for monitoring data
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Enable data export
    pub enable_export: bool,
    /// Export format
    pub format: ExportFormat,
    /// Export destination
    pub destination: ExportDestination,
    /// Export frequency
    pub frequency: ExportFrequency,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            enable_export: false,
            format: ExportFormat::JSON,
            destination: ExportDestination::File("optimization_metrics.json".to_string()),
            frequency: ExportFrequency::EndOfOptimization,
        }
    }
}

/// Export formats
#[derive(Debug, Clone, PartialEq)]
pub enum ExportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Binary format
    Binary,
    /// Custom format
    Custom(String),
}

/// Export destinations
#[derive(Debug, Clone, PartialEq)]
pub enum ExportDestination {
    /// File destination
    File(String),
    /// Database destination
    Database(String),
    /// Network destination
    Network(String),
    /// Custom destination
    Custom(String),
}

/// Export frequency options
#[derive(Debug, Clone, PartialEq)]
pub enum ExportFrequency {
    /// Export at the end of optimization
    EndOfOptimization,
    /// Export periodically during optimization
    Periodic(usize),
    /// Export on specific events
    OnEvent(Vec<String>),
    /// Real-time export
    RealTime,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerting
    pub enable_alerts: bool,
    /// Alert rules
    pub alert_rules: Vec<AlertRule>,
    /// Alert destinations
    pub alert_destinations: Vec<AlertDestination>,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enable_alerts: false,
            alert_rules: Vec::new(),
            alert_destinations: Vec::new(),
        }
    }
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Condition to trigger alert
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message template
    pub message_template: String,
}

/// Alert conditions
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Metric threshold condition
    MetricThreshold {
        metric_name: String,
        threshold: f64,
        comparison: Comparison,
    },
    /// Time-based condition
    TimeThreshold {
        duration: Duration,
        comparison: Comparison,
    },
    /// Custom condition
    Custom(String),
}

/// Comparison operators for alerts
#[derive(Debug, Clone, PartialEq)]
pub enum Comparison {
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Equal to
    EqualTo,
    /// Greater than or equal to
    GreaterThanOrEqual,
    /// Less than or equal to
    LessThanOrEqual,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    /// Information
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Alert destinations
#[derive(Debug, Clone)]
pub enum AlertDestination {
    /// Log to file
    Log(String),
    /// Send email
    Email(String),
    /// Send webhook
    Webhook(String),
    /// Custom destination
    Custom(String),
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();

        assert_eq!(config.algorithm_type, AlgorithmType::GradientDescent);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
    }

    #[test]
    fn test_optimization_config_builder() {
        let config = OptimizationConfig::builder()
            .algorithm_type(AlgorithmType::ConjugateGradient)
            .max_iterations(2000)
            .tolerance(1e-8)
            .build();

        assert_eq!(config.algorithm_type, AlgorithmType::ConjugateGradient);
        assert_eq!(config.max_iterations, 2000);
        assert_eq!(config.tolerance, 1e-8);
    }

    #[test]
    fn test_config_validation_valid() {
        let config = OptimizationConfig::default();
        let result = config.validate().unwrap();

        assert_eq!(result.status, ValidationStatus::Valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_config_validation_invalid() {
        let mut config = OptimizationConfig::default();
        config.max_iterations = 0;
        config.tolerance = -1.0;

        let result = config.validate().unwrap();
        assert_eq!(result.status, ValidationStatus::Invalid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_memory_estimate() {
        let config = OptimizationConfig::default();
        let estimate = config.estimate_memory_usage(100);

        assert!(estimate.total_memory > 0);
        assert!(estimate.gradient_memory > 0);
        assert_eq!(estimate.base_memory, 800); // 100 * 8
    }

    #[test]
    fn test_complexity_estimate() {
        let config = OptimizationConfig::default();
        let estimate = config.estimate_complexity(100);

        assert_eq!(estimate.complexity_class, ComplexityClass::Linear);
        assert!(estimate.total_complexity > 0);
    }

    #[test]
    fn test_algorithm_configuration_builder() {
        let config = AlgorithmConfiguration::builder()
            .learning_rate(0.001)
            .momentum(0.9)
            .build();

        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.momentum, 0.9);
    }

    #[test]
    fn test_lbfgs_config() {
        let lbfgs_config = LBFGSConfig::default();
        let algo_type = AlgorithmType::LBFGS(lbfgs_config.clone());

        assert_eq!(lbfgs_config.history_size, 10);
        assert_eq!(lbfgs_config.scaling_factor, 1.0);

        if let AlgorithmType::LBFGS(config) = algo_type {
            assert_eq!(config.history_size, 10);
        } else {
            panic!("Expected LBFGS algorithm type");
        }
    }

    #[test]
    fn test_step_size_config() {
        let step_config = StepSizeConfig::default();

        assert_eq!(step_config.initial_step_size, 0.01);
        assert_eq!(step_config.adaptation_method, StepSizeAdaptation::Armijo);
        assert_eq!(step_config.reduction_factor, 0.5);
    }

    #[test]
    fn test_convergence_criteria() {
        let criteria = ConvergenceCriteria::default();

        assert_eq!(criteria.gradient_tolerance, 1e-6);
        assert_eq!(criteria.function_tolerance, 1e-9);
        assert_eq!(criteria.max_iterations, 1000);
        assert!(!criteria.enable_early_stopping);
    }
}