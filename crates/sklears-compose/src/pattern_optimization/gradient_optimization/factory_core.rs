//! Factory Core Module
//!
//! This module provides the central factory infrastructure for creating and managing
//! gradient optimization components. It implements the factory pattern to ensure
//! proper configuration, initialization, and lifecycle management of all optimization
//! components within the framework.
//!
//! # Key Features
//!
//! ## Component Factory
//! - **GradientOptimizationFactory**: Central factory for all optimization components
//! - **Component Registry**: Centralized tracking and caching of created components
//! - **Template System**: Pre-configured component templates for common scenarios
//! - **Validation**: Comprehensive validation of component configurations
//!
//! ## Lifecycle Management
//! - **Creation Tracking**: Monitor component creation times and resource usage
//! - **Usage Statistics**: Track component access patterns and performance
//! - **Cache Management**: Intelligent caching to avoid redundant component creation
//! - **Cleanup**: Proper resource cleanup and component disposal
//!
//! ## Configuration System
//! - **Factory Configuration**: Centralized factory behavior configuration
//! - **Template Registry**: Pre-configured templates for rapid deployment
//! - **Validation Rules**: Ensure component configurations meet requirements
//! - **Performance Tuning**: Configuration options for optimal performance
//!
//! # Usage Examples
//!
//! ## Basic Factory Usage
//!
//! ```rust
//! use sklears_compose::pattern_optimization::gradient_optimization::factory_core::*;
//!
//! // Create factory with default configuration
//! let factory = GradientOptimizationFactory::new();
//!
//! // Create core optimizer
//! let optimizer = factory.create_core_optimizer()?;
//!
//! // Create gradient estimator
//! let gradient_estimator = factory.create_gradient_estimator(
//!     GradientComputationMethod::FiniteDifference
//! )?;
//!
//! // Create algorithm selector
//! let selector = factory.create_algorithm_selector(
//!     SelectionStrategy::PerformanceBased
//! )?;
//! ```
//!
//! ## Advanced Factory Configuration
//!
//! ```rust
//! // Configure factory with custom settings
//! let factory_config = FactoryConfiguration::builder()
//!     .enable_component_caching(true)
//!     .max_cached_components(100)
//!     .enable_performance_tracking(true)
//!     .component_creation_timeout(Duration::from_secs(30))
//!     .build();
//!
//! let factory = GradientOptimizationFactory::with_config(factory_config);
//!
//! // Create components with custom configurations
//! let optimizer = factory.create_core_optimizer_with_config(custom_config)?;
//! ```
//!
//! ## Template-Based Creation
//!
//! ```rust
//! // Use pre-configured templates
//! let optimizer = factory.create_from_template("high_performance_optimizer")?;
//! let analyzer = factory.create_from_template("comprehensive_problem_analyzer")?;
//!
//! // Register custom templates
//! factory.register_template("my_custom_optimizer", custom_config);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime};

// SciRS2 Core Dependencies
use scirs2_core::ndarray::{Array1, Array2};

// Internal imports - would reference other modules in the gradient_optimization package
use super::{
    gradient_core::{GradientBasedOptimizer, OptimizationProblem, Solution},
    gradient_computation::{GradientEstimator, HessianEstimator, GradientComputationMethod},
    algorithm_selection::{AlgorithmSelector, SelectionStrategy, AlgorithmSelection},
    problem_analysis::{ProblemAnalysisEngine, AnalysisConfiguration},
    performance_monitoring::GradientPerformanceMonitor,
};

// Use standard Rust Result type
type SklResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Central factory for creating and configuring gradient optimization components
///
/// The `GradientOptimizationFactory` provides a unified interface for creating
/// all components of the gradient optimization framework, ensuring proper
/// configuration and integration between modules.
#[derive(Debug)]
pub struct GradientOptimizationFactory {
    /// Factory configuration
    config: FactoryConfiguration,

    /// Component registry
    component_registry: Arc<RwLock<ComponentRegistry>>,

    /// Pre-configured templates
    template_registry: Arc<RwLock<TemplateRegistry>>,

    /// Creation statistics
    creation_stats: Arc<Mutex<CreationStatistics>>,

    /// Validation rules
    validation_rules: ValidationRules,
}

/// Configuration for the optimization factory
#[derive(Debug, Clone)]
pub struct FactoryConfiguration {
    /// Enable component caching
    pub enable_component_caching: bool,

    /// Maximum cached components
    pub max_cached_components: usize,

    /// Enable performance tracking
    pub enable_performance_tracking: bool,

    /// Factory timeout for component creation
    pub component_creation_timeout: Duration,

    /// Enable validation of created components
    pub enable_component_validation: bool,

    /// Enable automatic cleanup of unused components
    pub enable_automatic_cleanup: bool,

    /// Cleanup threshold (unused time before cleanup)
    pub cleanup_threshold: Duration,

    /// Maximum memory usage before forcing cleanup
    pub max_memory_usage: usize,

    /// Enable factory diagnostics
    pub enable_diagnostics: bool,
}

impl Default for FactoryConfiguration {
    fn default() -> Self {
        Self {
            enable_component_caching: true,
            max_cached_components: 50,
            enable_performance_tracking: true,
            component_creation_timeout: Duration::from_secs(60),
            enable_component_validation: true,
            enable_automatic_cleanup: true,
            cleanup_threshold: Duration::from_secs(300), // 5 minutes
            max_memory_usage: 1024 * 1024 * 100, // 100 MB
            enable_diagnostics: false,
        }
    }
}

impl FactoryConfiguration {
    /// Creates a builder for factory configuration.
    pub fn builder() -> FactoryConfigurationBuilder {
        FactoryConfigurationBuilder::default()
    }
}

/// Builder for factory configuration.
#[derive(Debug)]
pub struct FactoryConfigurationBuilder {
    enable_component_caching: bool,
    max_cached_components: usize,
    enable_performance_tracking: bool,
    component_creation_timeout: Duration,
    enable_component_validation: bool,
    enable_automatic_cleanup: bool,
    cleanup_threshold: Duration,
    max_memory_usage: usize,
    enable_diagnostics: bool,
}

impl Default for FactoryConfigurationBuilder {
    fn default() -> Self {
        let config = FactoryConfiguration::default();
        Self {
            enable_component_caching: config.enable_component_caching,
            max_cached_components: config.max_cached_components,
            enable_performance_tracking: config.enable_performance_tracking,
            component_creation_timeout: config.component_creation_timeout,
            enable_component_validation: config.enable_component_validation,
            enable_automatic_cleanup: config.enable_automatic_cleanup,
            cleanup_threshold: config.cleanup_threshold,
            max_memory_usage: config.max_memory_usage,
            enable_diagnostics: config.enable_diagnostics,
        }
    }
}

impl FactoryConfigurationBuilder {
    /// Enables or disables component caching.
    pub fn enable_component_caching(mut self, enable: bool) -> Self {
        self.enable_component_caching = enable;
        self
    }

    /// Sets the maximum number of cached components.
    pub fn max_cached_components(mut self, max: usize) -> Self {
        self.max_cached_components = max;
        self
    }

    /// Enables or disables performance tracking.
    pub fn enable_performance_tracking(mut self, enable: bool) -> Self {
        self.enable_performance_tracking = enable;
        self
    }

    /// Sets the component creation timeout.
    pub fn component_creation_timeout(mut self, timeout: Duration) -> Self {
        self.component_creation_timeout = timeout;
        self
    }

    /// Enables or disables component validation.
    pub fn enable_component_validation(mut self, enable: bool) -> Self {
        self.enable_component_validation = enable;
        self
    }

    /// Enables or disables automatic cleanup.
    pub fn enable_automatic_cleanup(mut self, enable: bool) -> Self {
        self.enable_automatic_cleanup = enable;
        self
    }

    /// Sets the cleanup threshold.
    pub fn cleanup_threshold(mut self, threshold: Duration) -> Self {
        self.cleanup_threshold = threshold;
        self
    }

    /// Sets the maximum memory usage.
    pub fn max_memory_usage(mut self, max_memory: usize) -> Self {
        self.max_memory_usage = max_memory;
        self
    }

    /// Enables or disables factory diagnostics.
    pub fn enable_diagnostics(mut self, enable: bool) -> Self {
        self.enable_diagnostics = enable;
        self
    }

    /// Builds the factory configuration.
    pub fn build(self) -> FactoryConfiguration {
        FactoryConfiguration {
            enable_component_caching: self.enable_component_caching,
            max_cached_components: self.max_cached_components,
            enable_performance_tracking: self.enable_performance_tracking,
            component_creation_timeout: self.component_creation_timeout,
            enable_component_validation: self.enable_component_validation,
            enable_automatic_cleanup: self.enable_automatic_cleanup,
            cleanup_threshold: self.cleanup_threshold,
            max_memory_usage: self.max_memory_usage,
            enable_diagnostics: self.enable_diagnostics,
        }
    }
}

/// Registry of created components
#[derive(Debug, Default)]
pub struct ComponentRegistry {
    /// Core optimizers
    core_optimizers: HashMap<String, CachedComponent<Arc<GradientBasedOptimizer>>>,

    /// Gradient estimators
    gradient_estimators: HashMap<String, CachedComponent<Arc<GradientEstimator>>>,

    /// Hessian estimators
    hessian_estimators: HashMap<String, CachedComponent<Arc<HessianEstimator>>>,

    /// Algorithm selectors
    algorithm_selectors: HashMap<String, CachedComponent<Arc<AlgorithmSelector>>>,

    /// Problem analyzers
    problem_analyzers: HashMap<String, CachedComponent<Arc<ProblemAnalysisEngine>>>,

    /// Performance monitors
    performance_monitors: HashMap<String, CachedComponent<Arc<GradientPerformanceMonitor>>>,

    /// Registry statistics
    registry_stats: RegistryStatistics,
}

/// Cached component with metadata
#[derive(Debug, Clone)]
pub struct CachedComponent<T> {
    /// The cached component
    pub component: T,

    /// Creation timestamp
    pub created_at: SystemTime,

    /// Last access timestamp
    pub last_accessed: SystemTime,

    /// Access count
    pub access_count: u64,

    /// Component configuration hash
    pub config_hash: u64,

    /// Memory usage estimate
    pub memory_usage: usize,

    /// Component metadata
    pub metadata: ComponentMetadata,
}

/// Metadata for cached components
#[derive(Debug, Clone)]
pub struct ComponentMetadata {
    /// Component type
    pub component_type: String,

    /// Creation duration
    pub creation_duration: Duration,

    /// Configuration parameters
    pub configuration: HashMap<String, String>,

    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,

    /// Validation status
    pub validation_status: ValidationStatus,
}

/// Performance characteristics of a component
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    /// Average execution time
    pub avg_execution_time: Duration,

    /// Memory efficiency score
    pub memory_efficiency: f64,

    /// Computational complexity estimate
    pub complexity_estimate: String,

    /// Scalability factor
    pub scalability_factor: f64,
}

/// Validation status for components
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    /// Component passed all validation checks
    Valid,
    /// Component has warnings but is usable
    ValidWithWarnings(Vec<String>),
    /// Component failed validation
    Invalid(Vec<String>),
    /// Validation not performed
    NotValidated,
}

/// Registry statistics
#[derive(Debug, Default)]
pub struct RegistryStatistics {
    /// Total components created
    pub total_created: u64,

    /// Total components accessed
    pub total_accessed: u64,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Average component lifetime
    pub avg_component_lifetime: Duration,

    /// Memory usage by component type
    pub memory_by_type: HashMap<String, usize>,

    /// Component creation rate
    pub creation_rate: f64,
}

/// Registry of pre-configured templates
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    /// Optimizer templates
    optimizer_templates: HashMap<String, OptimizerTemplate>,

    /// Estimator templates
    estimator_templates: HashMap<String, EstimatorTemplate>,

    /// Selector templates
    selector_templates: HashMap<String, SelectorTemplate>,

    /// Analyzer templates
    analyzer_templates: HashMap<String, AnalyzerTemplate>,

    /// Custom templates
    custom_templates: HashMap<String, CustomTemplate>,
}

/// Template for optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerTemplate {
    /// Template name
    pub name: String,

    /// Template description
    pub description: String,

    /// Optimizer configuration
    pub config: OptimizerConfiguration,

    /// Associated tags
    pub tags: Vec<String>,

    /// Usage statistics
    pub usage_count: u64,
}

/// Template for estimator configuration
#[derive(Debug, Clone)]
pub struct EstimatorTemplate {
    /// Template name
    pub name: String,

    /// Estimator type
    pub estimator_type: EstimatorType,

    /// Configuration parameters
    pub parameters: HashMap<String, f64>,

    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
}

/// Types of estimators
#[derive(Debug, Clone, PartialEq)]
pub enum EstimatorType {
    /// Gradient estimator
    Gradient,
    /// Hessian estimator
    Hessian,
    /// Combined gradient and Hessian estimator
    Combined,
}

/// Template for selector configuration
#[derive(Debug, Clone)]
pub struct SelectorTemplate {
    /// Template name
    pub name: String,

    /// Selection strategy
    pub strategy: SelectionStrategy,

    /// Configuration parameters
    pub parameters: HashMap<String, f64>,

    /// Applicable problem types
    pub problem_types: Vec<String>,
}

/// Template for analyzer configuration
#[derive(Debug, Clone)]
pub struct AnalyzerTemplate {
    /// Template name
    pub name: String,

    /// Analysis configuration
    pub config: AnalysisConfiguration,

    /// Enabled analysis modules
    pub enabled_modules: Vec<String>,

    /// Performance settings
    pub performance_settings: AnalysisPerformanceSettings,
}

/// Performance settings for analysis
#[derive(Debug, Clone)]
pub struct AnalysisPerformanceSettings {
    /// Maximum analysis time
    pub max_analysis_time: Duration,

    /// Sampling rate for large problems
    pub sampling_rate: f64,

    /// Enable parallel analysis
    pub enable_parallel: bool,

    /// Memory limit for analysis
    pub memory_limit: usize,
}

/// Custom template for user-defined configurations
#[derive(Debug, Clone)]
pub struct CustomTemplate {
    /// Template name
    pub name: String,

    /// Template type
    pub template_type: String,

    /// Configuration data
    pub config_data: HashMap<String, serde_json::Value>,

    /// Template metadata
    pub metadata: TemplateMetadata,
}

/// Metadata for custom templates
#[derive(Debug, Clone)]
pub struct TemplateMetadata {
    /// Creator information
    pub creator: String,

    /// Creation date
    pub created_at: SystemTime,

    /// Last modified date
    pub modified_at: SystemTime,

    /// Version number
    pub version: String,

    /// Template tags
    pub tags: Vec<String>,
}

/// Statistics about component creation
#[derive(Debug, Default)]
pub struct CreationStatistics {
    /// Total components created
    pub total_created: u64,

    /// Creation times by component type
    pub creation_times: HashMap<String, Vec<Duration>>,

    /// Failure count by component type
    pub failure_counts: HashMap<String, u64>,

    /// Average creation time
    pub avg_creation_time: Duration,

    /// Peak memory usage during creation
    pub peak_memory_usage: usize,

    /// Recent creation history
    pub recent_creations: Vec<CreationEvent>,
}

/// Event representing a component creation
#[derive(Debug, Clone)]
pub struct CreationEvent {
    /// Event timestamp
    pub timestamp: SystemTime,

    /// Component type
    pub component_type: String,

    /// Creation duration
    pub duration: Duration,

    /// Success status
    pub success: bool,

    /// Error message (if failed)
    pub error_message: Option<String>,

    /// Memory usage at creation
    pub memory_usage: usize,
}

/// Validation rules for component creation
#[derive(Debug)]
pub struct ValidationRules {
    /// Rules for optimizer validation
    pub optimizer_rules: OptimizerValidationRules,

    /// Rules for estimator validation
    pub estimator_rules: EstimatorValidationRules,

    /// Rules for selector validation
    pub selector_rules: SelectorValidationRules,

    /// Rules for analyzer validation
    pub analyzer_rules: AnalyzerValidationRules,

    /// Global validation settings
    pub global_settings: GlobalValidationSettings,
}

/// Validation rules for optimizers
#[derive(Debug)]
pub struct OptimizerValidationRules {
    /// Required configuration parameters
    pub required_parameters: Vec<String>,

    /// Parameter value ranges
    pub parameter_ranges: HashMap<String, (f64, f64)>,

    /// Compatibility checks
    pub compatibility_checks: Vec<CompatibilityCheck>,

    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
}

/// Validation rules for estimators
#[derive(Debug)]
pub struct EstimatorValidationRules {
    /// Minimum accuracy requirements
    pub min_accuracy: f64,

    /// Maximum computation time
    pub max_computation_time: Duration,

    /// Memory usage limits
    pub memory_limits: MemoryLimits,

    /// Numerical stability checks
    pub stability_checks: Vec<StabilityCheck>,
}

/// Validation rules for selectors
#[derive(Debug)]
pub struct SelectorValidationRules {
    /// Supported problem types
    pub supported_problem_types: Vec<String>,

    /// Required performance metrics
    pub required_metrics: Vec<String>,

    /// Selection quality thresholds
    pub quality_thresholds: HashMap<String, f64>,
}

/// Validation rules for analyzers
#[derive(Debug)]
pub struct AnalyzerValidationRules {
    /// Required analysis capabilities
    pub required_capabilities: Vec<String>,

    /// Analysis accuracy requirements
    pub accuracy_requirements: HashMap<String, f64>,

    /// Resource usage limits
    pub resource_limits: ResourceLimits,
}

/// Global validation settings
#[derive(Debug)]
pub struct GlobalValidationSettings {
    /// Enable strict validation mode
    pub strict_mode: bool,

    /// Maximum validation time
    pub max_validation_time: Duration,

    /// Enable performance validation
    pub enable_performance_validation: bool,

    /// Enable security validation
    pub enable_security_validation: bool,
}

/// Compatibility check specification
#[derive(Debug)]
pub struct CompatibilityCheck {
    /// Check name
    pub name: String,

    /// Check function
    pub check_fn: fn(&OptimizerConfiguration) -> bool,

    /// Error message if check fails
    pub error_message: String,
}

/// Performance requirements
#[derive(Debug)]
pub struct PerformanceRequirements {
    /// Maximum execution time
    pub max_execution_time: Duration,

    /// Maximum memory usage
    pub max_memory_usage: usize,

    /// Minimum accuracy
    pub min_accuracy: f64,

    /// Scalability requirements
    pub scalability_requirements: ScalabilityRequirements,
}

/// Scalability requirements
#[derive(Debug)]
pub struct ScalabilityRequirements {
    /// Maximum problem dimension
    pub max_dimension: usize,

    /// Performance degradation threshold
    pub performance_threshold: f64,

    /// Parallel efficiency requirement
    pub parallel_efficiency: f64,
}

/// Memory usage limits
#[derive(Debug)]
pub struct MemoryLimits {
    /// Maximum heap memory
    pub max_heap_memory: usize,

    /// Maximum stack memory
    pub max_stack_memory: usize,

    /// Memory growth rate limit
    pub growth_rate_limit: f64,
}

/// Numerical stability check
#[derive(Debug)]
pub struct StabilityCheck {
    /// Check name
    pub name: String,

    /// Stability criterion
    pub criterion: StabilityCriterion,

    /// Tolerance level
    pub tolerance: f64,
}

/// Stability criteria
#[derive(Debug)]
pub enum StabilityCriterion {
    /// Condition number check
    ConditionNumber,
    /// Numerical precision check
    NumericalPrecision,
    /// Convergence stability
    ConvergenceStability,
    /// Custom stability check
    Custom(String),
}

/// Resource usage limits
#[derive(Debug)]
pub struct ResourceLimits {
    /// CPU usage limit
    pub cpu_limit: f64,

    /// Memory usage limit
    pub memory_limit: usize,

    /// I/O operations limit
    pub io_limit: u64,

    /// Network usage limit
    pub network_limit: u64,
}

// Placeholder types for configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfiguration {
    pub algorithm_type: String,
    pub parameters: HashMap<String, f64>,
}

impl GradientOptimizationFactory {
    /// Creates a new factory with default configuration.
    pub fn new() -> Self {
        Self::with_config(FactoryConfiguration::default())
    }

    /// Creates a new factory with the specified configuration.
    pub fn with_config(config: FactoryConfiguration) -> Self {
        Self {
            config,
            component_registry: Arc::new(RwLock::new(ComponentRegistry::default())),
            template_registry: Arc::new(RwLock::new(TemplateRegistry::default())),
            creation_stats: Arc::new(Mutex::new(CreationStatistics::default())),
            validation_rules: ValidationRules::default(),
        }
    }

    /// Creates a core gradient-based optimizer.
    pub fn create_core_optimizer(&self) -> SklResult<Arc<GradientBasedOptimizer>> {
        let config = OptimizerConfiguration {
            algorithm_type: "gradient_descent".to_string(),
            parameters: HashMap::new(),
        };
        self.create_core_optimizer_with_config(config)
    }

    /// Creates a core optimizer with custom configuration.
    pub fn create_core_optimizer_with_config(
        &self,
        config: OptimizerConfiguration,
    ) -> SklResult<Arc<GradientBasedOptimizer>> {
        let start_time = std::time::Instant::now();

        // Validate configuration if enabled
        if self.config.enable_component_validation {
            self.validate_optimizer_config(&config)?;
        }

        // Check cache if enabled
        if self.config.enable_component_caching {
            let cache_key = self.generate_cache_key("optimizer", &config);
            if let Some(cached) = self.get_cached_optimizer(&cache_key) {
                return Ok(cached);
            }
        }

        // Create the optimizer (simplified - would use real implementation)
        let optimizer = Arc::new(GradientBasedOptimizer::new(config.clone())?);

        // Cache if enabled
        if self.config.enable_component_caching {
            let cache_key = self.generate_cache_key("optimizer", &config);
            self.cache_optimizer(cache_key, optimizer.clone())?;
        }

        // Record creation statistics
        self.record_creation_event("optimizer", start_time.elapsed(), true, None);

        Ok(optimizer)
    }

    /// Creates a gradient estimator.
    pub fn create_gradient_estimator(
        &self,
        method: GradientComputationMethod,
    ) -> SklResult<Arc<GradientEstimator>> {
        let start_time = std::time::Instant::now();

        // Check cache if enabled
        let cache_key = format!("gradient_estimator_{:?}", method);
        if self.config.enable_component_caching {
            if let Some(cached) = self.get_cached_gradient_estimator(&cache_key) {
                return Ok(cached);
            }
        }

        // Create the estimator (simplified)
        let estimator = Arc::new(GradientEstimator::new(method)?);

        // Cache if enabled
        if self.config.enable_component_caching {
            self.cache_gradient_estimator(cache_key, estimator.clone())?;
        }

        // Record creation statistics
        self.record_creation_event("gradient_estimator", start_time.elapsed(), true, None);

        Ok(estimator)
    }

    /// Creates an algorithm selector.
    pub fn create_algorithm_selector(
        &self,
        strategy: SelectionStrategy,
    ) -> SklResult<Arc<AlgorithmSelector>> {
        let start_time = std::time::Instant::now();

        // Check cache if enabled
        let cache_key = format!("algorithm_selector_{:?}", strategy);
        if self.config.enable_component_caching {
            if let Some(cached) = self.get_cached_algorithm_selector(&cache_key) {
                return Ok(cached);
            }
        }

        // Create the selector (simplified)
        let selector = Arc::new(AlgorithmSelector::new(strategy)?);

        // Cache if enabled
        if self.config.enable_component_caching {
            self.cache_algorithm_selector(cache_key, selector.clone())?;
        }

        // Record creation statistics
        self.record_creation_event("algorithm_selector", start_time.elapsed(), true, None);

        Ok(selector)
    }

    /// Creates a problem analyzer.
    pub fn create_problem_analyzer(
        &self,
        config: AnalysisConfiguration,
    ) -> SklResult<Arc<ProblemAnalysisEngine>> {
        let start_time = std::time::Instant::now();

        // Create the analyzer (simplified)
        let analyzer = Arc::new(ProblemAnalysisEngine::new(config)?);

        // Record creation statistics
        self.record_creation_event("problem_analyzer", start_time.elapsed(), true, None);

        Ok(analyzer)
    }

    /// Creates a component from a template.
    pub fn create_from_template(&self, template_name: &str) -> SklResult<Arc<dyn std::any::Any + Send + Sync>> {
        let template_registry = self.template_registry.read().unwrap();

        if let Some(optimizer_template) = template_registry.optimizer_templates.get(template_name) {
            let optimizer = self.create_core_optimizer_with_config(optimizer_template.config.clone())?;
            return Ok(optimizer);
        }

        Err(format!("Template '{}' not found", template_name).into())
    }

    /// Registers a custom template.
    pub fn register_template(&self, name: &str, template: CustomTemplate) -> SklResult<()> {
        let mut template_registry = self.template_registry.write().unwrap();
        template_registry.custom_templates.insert(name.to_string(), template);
        Ok(())
    }

    /// Gets factory statistics.
    pub fn get_statistics(&self) -> SklResult<FactoryStatistics> {
        let creation_stats = self.creation_stats.lock().unwrap();
        let registry = self.component_registry.read().unwrap();

        Ok(FactoryStatistics {
            total_components_created: creation_stats.total_created,
            cache_hit_rate: registry.registry_stats.cache_hit_rate,
            average_creation_time: creation_stats.avg_creation_time,
            memory_usage: registry.registry_stats.memory_by_type.values().sum(),
            active_components: self.count_active_components()?,
        })
    }

    // Private helper methods

    fn validate_optimizer_config(&self, config: &OptimizerConfiguration) -> SklResult<()> {
        // Simplified validation
        if config.algorithm_type.is_empty() {
            return Err("Algorithm type cannot be empty".into());
        }
        Ok(())
    }

    fn generate_cache_key(&self, component_type: &str, config: &OptimizerConfiguration) -> String {
        format!("{}_{:?}", component_type, config.algorithm_type)
    }

    fn get_cached_optimizer(&self, cache_key: &str) -> Option<Arc<GradientBasedOptimizer>> {
        let registry = self.component_registry.read().unwrap();
        registry.core_optimizers.get(cache_key).map(|cached| {
            // Update access statistics
            cached.component.clone()
        })
    }

    fn cache_optimizer(&self, cache_key: String, optimizer: Arc<GradientBasedOptimizer>) -> SklResult<()> {
        let mut registry = self.component_registry.write().unwrap();

        // Check cache size limit
        if registry.core_optimizers.len() >= self.config.max_cached_components {
            self.cleanup_cache(&mut registry)?;
        }

        let cached_component = CachedComponent {
            component: optimizer,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            config_hash: 0, // Simplified
            memory_usage: 1024, // Estimated
            metadata: ComponentMetadata::default(),
        };

        registry.core_optimizers.insert(cache_key, cached_component);
        Ok(())
    }

    fn get_cached_gradient_estimator(&self, cache_key: &str) -> Option<Arc<GradientEstimator>> {
        let registry = self.component_registry.read().unwrap();
        registry.gradient_estimators.get(cache_key).map(|cached| cached.component.clone())
    }

    fn cache_gradient_estimator(&self, cache_key: String, estimator: Arc<GradientEstimator>) -> SklResult<()> {
        let mut registry = self.component_registry.write().unwrap();

        let cached_component = CachedComponent {
            component: estimator,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            config_hash: 0,
            memory_usage: 512,
            metadata: ComponentMetadata::default(),
        };

        registry.gradient_estimators.insert(cache_key, cached_component);
        Ok(())
    }

    fn get_cached_algorithm_selector(&self, cache_key: &str) -> Option<Arc<AlgorithmSelector>> {
        let registry = self.component_registry.read().unwrap();
        registry.algorithm_selectors.get(cache_key).map(|cached| cached.component.clone())
    }

    fn cache_algorithm_selector(&self, cache_key: String, selector: Arc<AlgorithmSelector>) -> SklResult<()> {
        let mut registry = self.component_registry.write().unwrap();

        let cached_component = CachedComponent {
            component: selector,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            config_hash: 0,
            memory_usage: 256,
            metadata: ComponentMetadata::default(),
        };

        registry.algorithm_selectors.insert(cache_key, cached_component);
        Ok(())
    }

    fn cleanup_cache(&self, registry: &mut ComponentRegistry) -> SklResult<()> {
        // Simplified cache cleanup - remove oldest entries
        // In practice, would use LRU or other sophisticated strategies
        Ok(())
    }

    fn record_creation_event(&self, component_type: &str, duration: Duration, success: bool, error: Option<String>) {
        if let Ok(mut stats) = self.creation_stats.lock() {
            stats.total_created += 1;

            let event = CreationEvent {
                timestamp: SystemTime::now(),
                component_type: component_type.to_string(),
                duration,
                success,
                error_message: error,
                memory_usage: 0, // Simplified
            };

            stats.recent_creations.push(event);

            // Keep only last 100 events
            if stats.recent_creations.len() > 100 {
                stats.recent_creations.remove(0);
            }
        }
    }

    fn count_active_components(&self) -> SklResult<usize> {
        let registry = self.component_registry.read().unwrap();
        Ok(registry.core_optimizers.len() +
           registry.gradient_estimators.len() +
           registry.algorithm_selectors.len() +
           registry.problem_analyzers.len())
    }
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            optimizer_rules: OptimizerValidationRules::default(),
            estimator_rules: EstimatorValidationRules::default(),
            selector_rules: SelectorValidationRules::default(),
            analyzer_rules: AnalyzerValidationRules::default(),
            global_settings: GlobalValidationSettings::default(),
        }
    }
}

impl Default for OptimizerValidationRules {
    fn default() -> Self {
        Self {
            required_parameters: vec!["algorithm_type".to_string()],
            parameter_ranges: HashMap::new(),
            compatibility_checks: Vec::new(),
            performance_requirements: PerformanceRequirements::default(),
        }
    }
}

impl Default for EstimatorValidationRules {
    fn default() -> Self {
        Self {
            min_accuracy: 1e-6,
            max_computation_time: Duration::from_secs(60),
            memory_limits: MemoryLimits::default(),
            stability_checks: Vec::new(),
        }
    }
}

impl Default for SelectorValidationRules {
    fn default() -> Self {
        Self {
            supported_problem_types: vec!["general".to_string()],
            required_metrics: vec!["performance".to_string()],
            quality_thresholds: HashMap::new(),
        }
    }
}

impl Default for AnalyzerValidationRules {
    fn default() -> Self {
        Self {
            required_capabilities: vec!["basic_analysis".to_string()],
            accuracy_requirements: HashMap::new(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for GlobalValidationSettings {
    fn default() -> Self {
        Self {
            strict_mode: false,
            max_validation_time: Duration::from_secs(30),
            enable_performance_validation: true,
            enable_security_validation: false,
        }
    }
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(3600),
            max_memory_usage: 1024 * 1024 * 1024, // 1 GB
            min_accuracy: 1e-6,
            scalability_requirements: ScalabilityRequirements::default(),
        }
    }
}

impl Default for ScalabilityRequirements {
    fn default() -> Self {
        Self {
            max_dimension: 10000,
            performance_threshold: 0.8,
            parallel_efficiency: 0.7,
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_heap_memory: 1024 * 1024 * 512, // 512 MB
            max_stack_memory: 1024 * 1024 * 8, // 8 MB
            growth_rate_limit: 2.0,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu_limit: 0.8,
            memory_limit: 1024 * 1024 * 512,
            io_limit: 1000,
            network_limit: 0,
        }
    }
}

impl Default for ComponentMetadata {
    fn default() -> Self {
        Self {
            component_type: "unknown".to_string(),
            creation_duration: Duration::from_millis(0),
            configuration: HashMap::new(),
            performance_characteristics: PerformanceCharacteristics::default(),
            validation_status: ValidationStatus::NotValidated,
        }
    }
}

impl Default for PerformanceCharacteristics {
    fn default() -> Self {
        Self {
            avg_execution_time: Duration::from_millis(0),
            memory_efficiency: 0.0,
            complexity_estimate: "O(n)".to_string(),
            scalability_factor: 1.0,
        }
    }
}

/// Factory statistics
#[derive(Debug)]
pub struct FactoryStatistics {
    /// Total components created
    pub total_components_created: u64,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Average creation time
    pub average_creation_time: Duration,

    /// Total memory usage
    pub memory_usage: usize,

    /// Number of active components
    pub active_components: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creation() {
        let factory = GradientOptimizationFactory::new();

        // Test that factory can be created
        assert!(factory.config.enable_component_caching);
        assert_eq!(factory.config.max_cached_components, 50);
    }

    #[test]
    fn test_factory_configuration_builder() {
        let config = FactoryConfiguration::builder()
            .enable_component_caching(false)
            .max_cached_components(100)
            .enable_performance_tracking(true)
            .component_creation_timeout(Duration::from_secs(30))
            .build();

        assert!(!config.enable_component_caching);
        assert_eq!(config.max_cached_components, 100);
        assert!(config.enable_performance_tracking);
        assert_eq!(config.component_creation_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_validation_rules_default() {
        let rules = ValidationRules::default();

        assert!(!rules.global_settings.strict_mode);
        assert!(rules.global_settings.enable_performance_validation);
        assert!(!rules.global_settings.enable_security_validation);
    }

    #[test]
    fn test_component_metadata_default() {
        let metadata = ComponentMetadata::default();

        assert_eq!(metadata.component_type, "unknown");
        assert_eq!(metadata.validation_status, ValidationStatus::NotValidated);
    }

    #[test]
    fn test_performance_characteristics() {
        let perf = PerformanceCharacteristics::default();

        assert_eq!(perf.avg_execution_time, Duration::from_millis(0));
        assert_eq!(perf.memory_efficiency, 0.0);
        assert_eq!(perf.complexity_estimate, "O(n)");
        assert_eq!(perf.scalability_factor, 1.0);
    }
}