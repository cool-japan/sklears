//! Plugin Architecture for Custom Covariance Estimators
//!
//! This module provides a plugin system for registering and using custom
//! covariance estimators, hooks for estimation callbacks, custom regularization
//! registration, and middleware for estimation pipelines.

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use sklears_core::error::SklearsError;
use sklears_core::traits::Estimator;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};

/// Plugin registry for custom covariance estimators
pub struct CovariancePluginRegistry {
    /// Registered estimator factories
    estimator_factories: RwLock<HashMap<String, Box<dyn EstimatorFactory>>>,
    /// Registered regularization functions
    regularization_functions: RwLock<HashMap<String, Box<dyn RegularizationFunction>>>,
    /// Registered hooks
    hooks: RwLock<HashMap<HookType, Vec<Box<dyn Hook>>>>,
    /// Registered middleware
    middleware: RwLock<Vec<Box<dyn Middleware>>>,
}

/// Factory trait for creating custom estimators
pub trait EstimatorFactory: Send + Sync + Debug {
    /// Create a new estimator instance
    fn create(&self) -> Result<Box<dyn CustomCovarianceEstimator>, SklearsError>;

    /// Get estimator name
    fn name(&self) -> &str;

    /// Get estimator description
    fn description(&self) -> &str;

    /// Get required parameters
    fn required_parameters(&self) -> Vec<&str>;
}

/// Trait for custom covariance estimators
pub trait CustomCovarianceEstimator: Send + Sync + Debug {
    /// Fit the estimator to data
    fn fit(&mut self, X: ArrayView2<f64>) -> Result<(), SklearsError>;

    /// Get the estimated covariance matrix
    fn covariance(&self) -> Result<Array2<f64>, SklearsError>;

    /// Set parameter
    fn set_parameter(&mut self, name: &str, value: Box<dyn Any + Send>)
        -> Result<(), SklearsError>;

    /// Get parameter
    fn get_parameter(&self, name: &str) -> Result<Box<dyn Any + Send>, SklearsError>;

    /// Get estimator metadata
    fn metadata(&self) -> EstimatorMetadata;
}

/// Metadata for custom estimators
#[derive(Debug, Clone)]
pub struct EstimatorMetadata {
    /// Estimator name
    pub name: String,
    /// Version
    pub version: String,
    /// Author
    pub author: String,
    /// Description
    pub description: String,
    /// Capabilities
    pub capabilities: Vec<String>,
    /// Parameter schema
    pub parameters: HashMap<String, ParameterSpec>,
}

/// Parameter specification
#[derive(Debug, Clone)]
pub struct ParameterSpec {
    /// Parameter type
    pub param_type: String,
    /// Default value description
    pub default: Option<String>,
    /// Parameter description
    pub description: String,
    /// Valid range or values
    pub constraints: Option<String>,
}

/// Regularization function trait
pub trait RegularizationFunction: Send + Sync + Debug {
    /// Apply regularization to covariance matrix
    fn apply(&self, covariance: &mut Array2<f64>, lambda: f64) -> Result<(), SklearsError>;

    /// Get regularization name
    fn name(&self) -> &str;

    /// Get gradient (if applicable)
    fn gradient(&self, covariance: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, SklearsError>;
}

/// Hook types for different estimation phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookType {
    /// Before fitting starts
    PreFit,
    /// After fitting completes
    PostFit,
    /// During each iteration
    IterationHook,
    /// On convergence
    ConvergenceHook,
    /// On error
    ErrorHook,
    /// Custom hook type
    Custom(&'static str),
}

/// Hook trait for callback functions
pub trait Hook: Send + Sync + Debug {
    /// Execute the hook
    fn execute(&self, context: &HookContext<'_>) -> Result<(), SklearsError>;

    /// Get hook priority (higher values execute first)
    fn priority(&self) -> i32 {
        0
    }

    /// Whether this hook can modify the context
    fn is_modifying(&self) -> bool {
        false
    }
}

/// Context passed to hooks
#[derive(Debug)]
pub struct HookContext<'a> {
    /// Current estimator state
    pub estimator_state: EstimatorState,
    /// Current data
    pub data: Option<ArrayView2<'a, f64>>,
    /// Current covariance matrix
    pub covariance: Option<Array2<f64>>,
    /// Iteration number (for iteration hooks)
    pub iteration: Option<usize>,
    /// Error information (for error hooks)
    pub error: Option<SklearsError>,
    /// Custom metadata
    pub metadata: HashMap<String, Box<dyn Any + Send>>,
}

/// Estimator state for hooks
#[derive(Debug, Clone)]
pub enum EstimatorState {
    /// Before fitting
    Uninitialized,
    /// Currently fitting
    Fitting,
    /// Fitting completed successfully
    Fitted,
    /// Error occurred during fitting
    Error,
}

/// Middleware trait for estimation pipelines
pub trait Middleware: Send + Sync + Debug {
    /// Process request before estimation
    fn before_estimation(&self, request: &mut EstimationRequest) -> Result<(), SklearsError>;

    /// Process response after estimation
    fn after_estimation(&self, response: &mut EstimationResponse) -> Result<(), SklearsError>;

    /// Get middleware name
    fn name(&self) -> &str;

    /// Get middleware priority (higher values execute first)
    fn priority(&self) -> i32 {
        0
    }
}

/// Estimation request
#[derive(Debug)]
pub struct EstimationRequest {
    /// Input data
    pub data: Array2<f64>,
    /// Estimator name
    pub estimator_name: String,
    /// Parameters
    pub parameters: HashMap<String, Box<dyn Any + Send>>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Estimation response
#[derive(Debug)]
pub struct EstimationResponse {
    /// Estimated covariance matrix
    pub covariance: Array2<f64>,
    /// Estimation metadata
    pub metadata: HashMap<String, Box<dyn Any + Send>>,
    /// Performance metrics
    pub performance: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Fitting time in milliseconds
    pub fit_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of iterations
    pub iterations: Option<usize>,
    /// Convergence flag
    pub converged: bool,
}

impl CovariancePluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            estimator_factories: RwLock::new(HashMap::new()),
            regularization_functions: RwLock::new(HashMap::new()),
            hooks: RwLock::new(HashMap::new()),
            middleware: RwLock::new(Vec::new()),
        }
    }

    /// Register a custom estimator factory
    pub fn register_estimator_factory(
        &self,
        factory: Box<dyn EstimatorFactory>,
    ) -> Result<(), SklearsError> {
        let mut factories = self
            .estimator_factories
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;

        let name = factory.name().to_string();
        factories.insert(name, factory);
        Ok(())
    }

    /// Register a regularization function
    pub fn register_regularization(
        &self,
        function: Box<dyn RegularizationFunction>,
    ) -> Result<(), SklearsError> {
        let mut functions = self
            .regularization_functions
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;

        let name = function.name().to_string();
        functions.insert(name, function);
        Ok(())
    }

    /// Register a hook
    pub fn register_hook(
        &self,
        hook_type: HookType,
        hook: Box<dyn Hook>,
    ) -> Result<(), SklearsError> {
        let mut hooks = self
            .hooks
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;

        hooks.entry(hook_type).or_insert_with(Vec::new).push(hook);

        // Sort by priority (descending)
        if let Some(hook_list) = hooks.get_mut(&hook_type) {
            hook_list.sort_by(|a, b| b.priority().cmp(&a.priority()));
        }

        Ok(())
    }

    /// Register middleware
    pub fn register_middleware(&self, middleware: Box<dyn Middleware>) -> Result<(), SklearsError> {
        let mut middleware_list = self
            .middleware
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;

        middleware_list.push(middleware);

        // Sort by priority (descending)
        middleware_list.sort_by(|a, b| b.priority().cmp(&a.priority()));

        Ok(())
    }

    /// Create estimator by name
    pub fn create_estimator(
        &self,
        name: &str,
    ) -> Result<Box<dyn CustomCovarianceEstimator>, SklearsError> {
        let factories = self
            .estimator_factories
            .read()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire read lock".to_string()))?;

        factories
            .get(name)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Estimator '{}' not found", name)))?
            .create()
    }

    /// Get regularization function by name
    pub fn get_regularization(
        &self,
        name: &str,
    ) -> Result<Box<dyn RegularizationFunction>, SklearsError> {
        let functions = self
            .regularization_functions
            .read()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire read lock".to_string()))?;

        // This is a simplified approach - in practice, you'd want to clone or use Arc
        Err(SklearsError::InvalidInput(format!(
            "Regularization '{}' access not implemented",
            name
        )))
    }

    /// Execute hooks for a given type
    pub fn execute_hooks(
        &self,
        hook_type: HookType,
        context: &HookContext<'_>,
    ) -> Result<(), SklearsError> {
        let hooks = self
            .hooks
            .read()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire read lock".to_string()))?;

        if let Some(hook_list) = hooks.get(&hook_type) {
            for hook in hook_list {
                hook.execute(context)?;
            }
        }

        Ok(())
    }

    /// Process estimation with middleware
    pub fn process_estimation(
        &self,
        mut request: EstimationRequest,
    ) -> Result<EstimationResponse, SklearsError> {
        let middleware_list = self
            .middleware
            .read()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire read lock".to_string()))?;

        // Before middleware
        for middleware in middleware_list.iter() {
            middleware.before_estimation(&mut request)?;
        }

        // Create and fit estimator
        let start_time = std::time::Instant::now();
        let mut estimator = self.create_estimator(&request.estimator_name)?;

        // Set parameters
        for (name, value) in request.parameters {
            estimator.set_parameter(&name, value)?;
        }

        // Execute pre-fit hooks
        let hook_context = HookContext {
            estimator_state: EstimatorState::Uninitialized,
            data: Some(request.data.view()),
            covariance: None,
            iteration: None,
            error: None,
            metadata: HashMap::new(),
        };
        self.execute_hooks(HookType::PreFit, &hook_context)?;

        // Fit the estimator
        let fit_result = estimator.fit(request.data.view());
        let fit_time = start_time.elapsed().as_millis() as f64;

        let mut response = match fit_result {
            Ok(()) => {
                let covariance = estimator.covariance()?;

                // Execute post-fit hooks
                let hook_context = HookContext {
                    estimator_state: EstimatorState::Fitted,
                    data: Some(request.data.view()),
                    covariance: Some(covariance.clone()),
                    iteration: None,
                    error: None,
                    metadata: HashMap::new(),
                };
                self.execute_hooks(HookType::PostFit, &hook_context)?;

                EstimationResponse {
                    covariance,
                    metadata: HashMap::new(),
                    performance: PerformanceMetrics {
                        fit_time_ms: fit_time,
                        memory_usage_bytes: 0, // Simplified
                        iterations: None,
                        converged: true,
                    },
                }
            }
            Err(error) => {
                // Execute error hooks
                let hook_context = HookContext {
                    estimator_state: EstimatorState::Error,
                    data: Some(request.data.view()),
                    covariance: None,
                    iteration: None,
                    error: Some(error.clone()),
                    metadata: HashMap::new(),
                };
                self.execute_hooks(HookType::ErrorHook, &hook_context)?;

                return Err(error);
            }
        };

        // After middleware
        for middleware in middleware_list.iter().rev() {
            middleware.after_estimation(&mut response)?;
        }

        Ok(response)
    }

    /// List available estimators
    pub fn list_estimators(&self) -> Result<Vec<String>, SklearsError> {
        let factories = self
            .estimator_factories
            .read()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire read lock".to_string()))?;

        Ok(factories.keys().cloned().collect())
    }

    /// List available regularization functions
    pub fn list_regularizations(&self) -> Result<Vec<String>, SklearsError> {
        let functions = self
            .regularization_functions
            .read()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire read lock".to_string()))?;

        Ok(functions.keys().cloned().collect())
    }
}

/// Global plugin registry instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_REGISTRY: CovariancePluginRegistry = CovariancePluginRegistry::new();
}

/// Convenience macros for plugin registration
#[macro_export]
macro_rules! register_covariance_estimator {
    ($factory:expr) => {
        $crate::plugin_architecture::GLOBAL_REGISTRY
            .register_estimator_factory(Box::new($factory))
            .expect("Failed to register estimator");
    };
}

#[macro_export]
macro_rules! register_regularization {
    ($function:expr) => {
        $crate::plugin_architecture::GLOBAL_REGISTRY
            .register_regularization(Box::new($function))
            .expect("Failed to register regularization");
    };
}

#[macro_export]
macro_rules! register_hook {
    ($hook_type:expr, $hook:expr) => {
        $crate::plugin_architecture::GLOBAL_REGISTRY
            .register_hook($hook_type, Box::new($hook))
            .expect("Failed to register hook");
    };
}

/// Example implementations
/// Example custom estimator: Simple empirical covariance
#[derive(Debug)]
pub struct SimpleEmpiricalEstimator {
    covariance: Option<Array2<f64>>,
    parameters: Arc<Mutex<HashMap<String, String>>>, // Simplified to just String values
}

impl SimpleEmpiricalEstimator {
    pub fn new() -> Self {
        Self {
            covariance: None,
            parameters: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl CustomCovarianceEstimator for SimpleEmpiricalEstimator {
    fn fit(&mut self, X: ArrayView2<f64>) -> Result<(), SklearsError> {
        let (n_samples, n_features) = X.dim();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        let mean = X.mean_axis(Axis(0)).unwrap();
        let centered = &X - &mean;
        let covariance = centered.t().dot(&centered) / (n_samples - 1) as f64;

        self.covariance = Some(covariance);
        Ok(())
    }

    fn covariance(&self) -> Result<Array2<f64>, SklearsError> {
        self.covariance
            .clone()
            .ok_or_else(|| SklearsError::InvalidInput("Estimator not fitted".to_string()))
    }

    fn set_parameter(
        &mut self,
        name: &str,
        value: Box<dyn Any + Send>,
    ) -> Result<(), SklearsError> {
        // For simplicity, convert to string representation
        let string_value = format!("{:?}", value);
        if let Ok(mut params) = self.parameters.lock() {
            params.insert(name.to_string(), string_value);
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(
                "Failed to acquire parameter lock".to_string(),
            ))
        }
    }

    fn get_parameter(&self, name: &str) -> Result<Box<dyn Any + Send>, SklearsError> {
        if let Ok(params) = self.parameters.lock() {
            if let Some(value) = params.get(name) {
                Ok(Box::new(value.clone()) as Box<dyn Any + Send>)
            } else {
                Err(SklearsError::InvalidInput(format!(
                    "Parameter '{}' not found",
                    name
                )))
            }
        } else {
            Err(SklearsError::InvalidInput(
                "Failed to acquire parameter lock".to_string(),
            ))
        }
    }

    fn metadata(&self) -> EstimatorMetadata {
        EstimatorMetadata {
            name: "SimpleEmpirical".to_string(),
            version: "1.0.0".to_string(),
            author: "Example".to_string(),
            description: "Simple empirical covariance estimator".to_string(),
            capabilities: vec!["basic_covariance".to_string()],
            parameters: HashMap::new(),
        }
    }
}

/// Example factory for simple empirical estimator
#[derive(Debug)]
pub struct SimpleEmpiricalFactory;

impl EstimatorFactory for SimpleEmpiricalFactory {
    fn create(&self) -> Result<Box<dyn CustomCovarianceEstimator>, SklearsError> {
        Ok(Box::new(SimpleEmpiricalEstimator::new()))
    }

    fn name(&self) -> &str {
        "simple_empirical"
    }

    fn description(&self) -> &str {
        "Simple empirical covariance estimation"
    }

    fn required_parameters(&self) -> Vec<&str> {
        vec![]
    }
}

/// Example L2 regularization
#[derive(Debug)]
pub struct L2Regularization {
    name: String,
}

impl L2Regularization {
    pub fn new() -> Self {
        Self {
            name: "l2".to_string(),
        }
    }
}

impl RegularizationFunction for L2Regularization {
    fn apply(&self, covariance: &mut Array2<f64>, lambda: f64) -> Result<(), SklearsError> {
        let (n, _) = covariance.dim();
        for i in 0..n {
            covariance[[i, i]] += lambda;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn gradient(&self, covariance: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, SklearsError> {
        let (n, m) = covariance.dim();
        let mut gradient = Array2::zeros((n, m));
        for i in 0..n {
            gradient[[i, i]] = lambda;
        }
        Ok(gradient)
    }
}

/// Example logging hook
#[derive(Debug)]
pub struct LoggingHook {
    name: String,
}

impl LoggingHook {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl Hook for LoggingHook {
    fn execute(&self, context: &HookContext<'_>) -> Result<(), SklearsError> {
        println!(
            "[{}] Hook executed: {:?}",
            self.name, context.estimator_state
        );
        Ok(())
    }
}

/// Example timing middleware
#[derive(Debug)]
pub struct TimingMiddleware;

impl Middleware for TimingMiddleware {
    fn before_estimation(&self, request: &mut EstimationRequest) -> Result<(), SklearsError> {
        request.metadata.insert(
            "start_time".to_string(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
        );
        Ok(())
    }

    fn after_estimation(&self, response: &mut EstimationResponse) -> Result<(), SklearsError> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        response.metadata.insert(
            "end_time".to_string(),
            Box::new(current_time) as Box<dyn Any + Send>,
        );
        Ok(())
    }

    fn name(&self) -> &str {
        "timing"
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::essentials::Normal;
    use scirs2_core::random::thread_rng;
    use scirs2_core::random::Distribution;

    #[test]
    fn test_plugin_registry() {
        let registry = CovariancePluginRegistry::new();

        // Register factory
        let factory = SimpleEmpiricalFactory;
        registry
            .register_estimator_factory(Box::new(factory))
            .unwrap();

        // Register regularization
        let regularization = L2Regularization::new();
        registry
            .register_regularization(Box::new(regularization))
            .unwrap();

        // Register hook
        let hook = LoggingHook::new("test".to_string());
        registry
            .register_hook(HookType::PreFit, Box::new(hook))
            .unwrap();

        // Register middleware
        let middleware = TimingMiddleware;
        registry.register_middleware(Box::new(middleware)).unwrap();

        // Test estimator creation
        let estimator = registry.create_estimator("simple_empirical");
        assert!(estimator.is_ok());

        // Test listing
        let estimators = registry.list_estimators().unwrap();
        assert!(estimators.contains(&"simple_empirical".to_string()));
    }

    #[test]
    fn test_custom_estimator() {
        let mut estimator = SimpleEmpiricalEstimator::new();
        let mut local_rng = thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let X = Array2::from_shape_fn((10, 3), |_| dist.sample(&mut local_rng));

        let result = estimator.fit(X.view());
        assert!(result.is_ok());

        let covariance = estimator.covariance();
        assert!(covariance.is_ok());

        let cov_matrix = covariance.unwrap();
        assert_eq!(cov_matrix.shape(), &[3, 3]);
    }

    #[test]
    fn test_estimation_pipeline() {
        let registry = CovariancePluginRegistry::new();

        // Register components
        registry
            .register_estimator_factory(Box::new(SimpleEmpiricalFactory))
            .unwrap();
        registry
            .register_middleware(Box::new(TimingMiddleware))
            .unwrap();

        let mut local_rng = thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let X = Array2::from_shape_fn((20, 2), |_| dist.sample(&mut local_rng));
        let request = EstimationRequest {
            data: X,
            estimator_name: "simple_empirical".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        };

        let response = registry.process_estimation(request);
        assert!(response.is_ok());

        let result = response.unwrap();
        assert_eq!(result.covariance.shape(), &[2, 2]);
        assert!(result.performance.fit_time_ms >= 0.0);
    }
}
