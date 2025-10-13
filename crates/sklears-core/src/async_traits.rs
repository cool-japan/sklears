/// Enhanced async trait implementations for non-blocking ML operations
///
/// This module provides comprehensive async support for machine learning operations,
/// including streaming data processing, batch operations, and progress tracking.
use crate::error::Result;
use crate::types::FloatBounds;
use futures_core::{Future, Stream};
use std::pin::Pin;
use std::time::Duration;

/// Type alias for async partial fit future
pub type AsyncPartialFitFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<Option<T>>> + Send + 'a>>;

/// Type alias for async predict with confidence future
pub type AsyncPredictConfidenceFuture<'a, Output> =
    Pin<Box<dyn Future<Output = Result<(Output, ConfidenceInterval)>> + Send + 'a>>;

/// Type alias for async score future
pub type AsyncScoreFuture<'a, Score> =
    Pin<Box<dyn Future<Output = Result<Vec<Score>>> + Send + 'a>>;

/// Type alias for async fit future
pub type AsyncFitFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// Type alias for async transform future
pub type AsyncTransformFuture<'a, Output> =
    Pin<Box<dyn Future<Output = Result<Output>> + Send + 'a>>;

/// Type alias for async cross-validation stream
pub type AsyncCVStream<'a, Score> = Pin<Box<dyn Stream<Item = Result<(usize, Score)>> + Send + 'a>>;

/// Type alias for async ensemble fit stream
pub type AsyncEnsembleFitStream<'a, Model> =
    Pin<Box<dyn Stream<Item = Result<(usize, Model)>> + Send + 'a>>;

/// Type alias for async ensemble predict stream
pub type AsyncEnsemblePredictStream<'a, Output> =
    Pin<Box<dyn Stream<Item = Result<(usize, Output)>> + Send + 'a>>;

/// Type alias for async optimization stream
pub type AsyncOptimizationStream<'a, Config, Score> =
    Pin<Box<dyn Stream<Item = Result<OptimizationResult<Config, Score>>> + Send + 'a>>;

/// Type alias for config factory function
pub type ConfigFactory<Config> =
    Box<dyn Fn(&std::collections::HashMap<String, f64>) -> Config + Send + Sync>;

/// Type alias for async unit future
pub type AsyncUnitFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

/// Progress information for long-running operations
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current progress (0.0 to 1.0)
    pub progress: f64,
    /// Current step/iteration
    pub current_step: usize,
    /// Total steps (if known)
    pub total_steps: Option<usize>,
    /// Elapsed time
    pub elapsed: Duration,
    /// Estimated time remaining
    pub eta: Option<Duration>,
    /// Current metric value (e.g., loss, accuracy)
    pub current_metric: Option<f64>,
    /// Additional status message
    pub message: String,
}

impl ProgressInfo {
    /// Create new progress info
    pub fn new(progress: f64, current_step: usize) -> Self {
        Self {
            progress: progress.clamp(0.0, 1.0),
            current_step,
            total_steps: None,
            elapsed: Duration::from_secs(0),
            eta: None,
            current_metric: None,
            message: String::new(),
        }
    }

    /// Set total steps
    pub fn with_total_steps(mut self, total: usize) -> Self {
        self.total_steps = Some(total);
        if total > 0 {
            self.progress = self.current_step as f64 / total as f64;
        }
        self
    }

    /// Set elapsed time
    pub fn with_elapsed(mut self, elapsed: Duration) -> Self {
        self.elapsed = elapsed;
        self
    }

    /// Set estimated time remaining
    pub fn with_eta(mut self, eta: Duration) -> Self {
        self.eta = Some(eta);
        self
    }

    /// Set current metric value
    pub fn with_metric(mut self, metric: f64) -> Self {
        self.current_metric = Some(metric);
        self
    }

    /// Set status message
    pub fn with_message<S: Into<String>>(mut self, message: S) -> Self {
        self.message = message.into();
        self
    }
}

/// Configuration for async operations
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Batch size for streaming operations
    pub batch_size: usize,
    /// Timeout for individual operations
    pub operation_timeout: Option<Duration>,
    /// Whether to report progress
    pub enable_progress: bool,
    /// Progress reporting interval
    pub progress_interval: Duration,
    /// Maximum concurrent operations
    pub max_concurrency: usize,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            operation_timeout: Some(Duration::from_secs(300)), // 5 minutes
            enable_progress: true,
            progress_interval: Duration::from_secs(1),
            max_concurrency: num_cpus::get(),
        }
    }
}

/// Enhanced async fit trait with progress tracking and cancellation
pub trait AsyncFitAdvanced<X, Y, State = crate::traits::Untrained> {
    /// The fitted model type
    type Fitted;

    /// Error type
    type Error: std::error::Error + Send + Sync;

    /// Fit the model asynchronously with progress tracking
    fn fit_async_with_progress<'a>(
        self,
        x: &'a X,
        y: &'a Y,
        config: &'a AsyncConfig,
    ) -> AsyncFitFuture<'a, Self::Fitted>
    where
        Self: 'a;

    /// Fit the model with progress reporting via a stream
    fn fit_async_with_progress_stream<'a>(
        self,
        x: &'a X,
        y: &'a Y,
        config: &'a AsyncConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<ProgressInfo>> + Send + 'a>>
    where
        Self: 'a;

    /// Fit the model with cancellation support
    fn fit_async_cancellable<'a>(
        self,
        x: &'a X,
        y: &'a Y,
        cancel_token: CancellationToken,
    ) -> AsyncPartialFitFuture<'a, Self::Fitted>
    where
        Self: 'a;
}

/// Enhanced async predict trait with batch processing
pub trait AsyncPredictAdvanced<X, Output> {
    /// Error type
    type Error: std::error::Error + Send + Sync;

    /// Make predictions asynchronously with batching
    fn predict_async_batched<'a>(
        &'a self,
        x: &'a X,
        config: &'a AsyncConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Output>> + Send + 'a>>;

    /// Stream predictions for large datasets
    fn predict_stream<'a>(
        &'a self,
        x_stream: Pin<Box<dyn Stream<Item = X> + Send + 'a>>,
        config: &'a AsyncConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<Output>> + Send + 'a>>;

    /// Predict with confidence intervals (if supported)
    fn predict_async_with_uncertainty<'a>(
        &'a self,
        x: &'a X,
        confidence_level: f64,
    ) -> AsyncPredictConfidenceFuture<'a, Output>
    where
        Self: 'a;
}

/// Enhanced async transform trait with streaming support
pub trait AsyncTransformAdvanced<X, Output = X> {
    /// Error type
    type Error: std::error::Error + Send + Sync;

    /// Transform data asynchronously with progress tracking
    fn transform_async_with_progress<'a>(
        &'a self,
        x: &'a X,
        config: &'a AsyncConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Output>> + Send + 'a>>;

    /// Stream data transformation
    fn transform_stream<'a>(
        &'a self,
        x_stream: Pin<Box<dyn Stream<Item = X> + Send + 'a>>,
        config: &'a AsyncConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<Output>> + Send + 'a>>;

    /// Transform with memory-efficient chunking
    fn transform_async_chunked<'a>(
        &'a self,
        x: &'a X,
        chunk_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<Output>> + Send + 'a>>;
}

/// Async partial fit trait for online learning
pub trait AsyncPartialFit<X, Y> {
    /// Error type
    type Error: std::error::Error + Send + Sync;

    /// Perform partial fit asynchronously
    fn partial_fit_async<'a>(
        &'a mut self,
        x: &'a X,
        y: &'a Y,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Stream partial fit for continuous learning
    fn partial_fit_stream<'a>(
        &'a mut self,
        data_stream: Pin<Box<dyn Stream<Item = (X, Y)> + Send + 'a>>,
        config: &'a AsyncConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<ProgressInfo>> + Send + 'a>>;

    /// Adaptive learning with dynamic batch sizing
    fn adaptive_partial_fit<'a>(
        &'a mut self,
        data_stream: Pin<Box<dyn Stream<Item = (X, Y)> + Send + 'a>>,
        adaptation_config: AdaptationConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<AdaptationInfo>> + Send + 'a>>;
}

/// Configuration for adaptive learning
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Initial batch size
    pub initial_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Learning rate adaptation factor
    pub adaptation_rate: f64,
    /// Performance threshold for batch size increase
    pub performance_threshold: f64,
    /// Memory usage threshold (bytes)
    pub memory_threshold: usize,
}

/// Information about adaptive learning progress
#[derive(Debug, Clone)]
pub struct AdaptationInfo {
    /// Current batch size
    pub current_batch_size: usize,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Performance metric
    pub performance_metric: f64,
    /// Memory usage
    pub memory_usage: usize,
    /// General progress information
    pub progress: ProgressInfo,
}

/// Confidence interval for predictions
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    /// Lower bound
    pub lower: f64,
    /// Upper bound
    pub upper: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence_level: f64,
}

/// Cancellation token for async operations
#[derive(Debug, Clone)]
pub struct CancellationToken {
    inner: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Cancel the operation
    pub fn cancel(&self) {
        self.inner.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check if cancellation was requested
    pub fn is_cancelled(&self) -> bool {
        self.inner.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Async cross-validation with parallel fold execution
pub trait AsyncCrossValidation<X, Y> {
    type Score: FloatBounds + Send;
    type Model: Clone + Send + Sync;

    /// Perform k-fold cross-validation asynchronously
    fn cross_validate_async<'a>(
        &'a self,
        model: Self::Model,
        x: &'a X,
        y: &'a Y,
        cv_folds: usize,
        config: &'a AsyncConfig,
    ) -> AsyncScoreFuture<'a, Self::Score>
    where
        X: Clone + Send + Sync,
        Y: Clone + Send + Sync;

    /// Stream cross-validation results as they complete
    fn cross_validate_stream<'a>(
        &'a self,
        model: Self::Model,
        x: &'a X,
        y: &'a Y,
        cv_folds: usize,
        config: &'a AsyncConfig,
    ) -> AsyncCVStream<'a, Self::Score>
    where
        X: Clone + Send + Sync,
        Y: Clone + Send + Sync;
}

/// Async ensemble methods
pub trait AsyncEnsemble<X, Y, Output> {
    type Model: Send + Sync;
    type Error: std::error::Error + Send + Sync;

    /// Train ensemble members asynchronously
    fn fit_ensemble_async<'a>(
        models: Vec<Self::Model>,
        x: &'a X,
        y: &'a Y,
        config: &'a AsyncConfig,
    ) -> AsyncEnsembleFitStream<'a, Self::Model>
    where
        X: Send + Sync,
        Y: Send + Sync,
        Self::Model: 'a;

    /// Make ensemble predictions asynchronously
    fn predict_ensemble_async<'a>(
        models: &'a [Self::Model],
        x: &'a X,
        config: &'a AsyncConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Output>> + Send + 'a>>
    where
        X: Send + Sync;

    /// Stream ensemble predictions with individual model results
    fn predict_ensemble_stream<'a>(
        models: &'a [Self::Model],
        x: &'a X,
        config: &'a AsyncConfig,
    ) -> AsyncEnsemblePredictStream<'a, Output>
    where
        X: Send + Sync;
}

/// Async hyperparameter optimization
pub trait AsyncHyperparameterOptimization<X, Y, Config> {
    type Score: FloatBounds + Send;
    type Error: std::error::Error + Send + Sync;

    /// Optimize hyperparameters asynchronously
    fn optimize_async<'a>(
        &'a self,
        x: &'a X,
        y: &'a Y,
        param_space: ParameterSpace<Config>,
        optimization_config: OptimizationConfig,
    ) -> AsyncOptimizationStream<'a, Config, Self::Score>
    where
        X: Send + Sync,
        Y: Send + Sync,
        Config: Send + Sync;
}

/// Parameter space definition for optimization
pub struct ParameterSpace<Config> {
    /// Parameter ranges and distributions
    pub parameters: std::collections::HashMap<String, ParameterRange>,
    /// Parameter dependencies
    pub dependencies: Vec<ParameterDependency>,
    /// Configuration factory function
    pub config_factory: ConfigFactory<Config>,
}

/// Parameter range definition
#[derive(Debug, Clone)]
pub enum ParameterRange {
    /// Continuous range [min, max]
    Continuous { min: f64, max: f64 },
    /// Discrete choices
    Discrete { values: Vec<f64> },
    /// Log-scale continuous range
    LogContinuous { min: f64, max: f64 },
    /// Integer range [min, max]
    Integer { min: i64, max: i64 },
}

/// Parameter dependency definition
pub struct ParameterDependency {
    /// Dependent parameter name
    pub dependent: String,
    /// Parent parameter name
    pub parent: String,
    /// Condition for dependency
    pub condition: Box<dyn Fn(f64) -> bool + Send + Sync>,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of evaluations
    pub max_evaluations: usize,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Parallel evaluation configuration
    pub parallel_config: AsyncConfig,
}

/// Optimization algorithm selection
#[derive(Debug, Clone)]
pub enum OptimizationAlgorithm {
    /// Random search
    Random,
    /// Bayesian optimization with Gaussian processes
    BayesianOptimization {
        acquisition_function: AcquisitionFunction,
        n_initial_points: usize,
    },
    /// Tree-structured Parzen estimators
    TPE {
        n_startup_trials: usize,
        n_ei_candidates: usize,
    },
    /// Hyperband algorithm
    Hyperband { max_resource: usize, eta: f64 },
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected improvement
    ExpectedImprovement,
    /// Upper confidence bound
    UpperConfidenceBound { kappa: f64 },
    /// Probability of improvement
    ProbabilityOfImprovement,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Patience (iterations without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Direction of optimization (maximize or minimize)
    pub maximize: bool,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult<Config, Score> {
    /// Trial number
    pub trial: usize,
    /// Parameter configuration
    pub config: Config,
    /// Achieved score
    pub score: Score,
    /// Evaluation time
    pub evaluation_time: Duration,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f64>,
}

/// Async model persistence
pub trait AsyncModelPersistence {
    type Error: std::error::Error + Send + Sync;

    /// Save model asynchronously
    fn save_async<'a>(
        &'a self,
        path: &'a std::path::Path,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Load model asynchronously
    fn load_async<'a>(
        path: &'a std::path::Path,
    ) -> Pin<Box<dyn Future<Output = Result<Self>> + Send + 'a>>
    where
        Self: Sized;

    /// Save model with compression
    fn save_compressed_async<'a>(
        &'a self,
        path: &'a std::path::Path,
        compression_level: u32,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_progress_info() {
        let progress = ProgressInfo::new(0.5, 50)
            .with_total_steps(100)
            .with_elapsed(Duration::from_secs(30))
            .with_eta(Duration::from_secs(30))
            .with_metric(0.85)
            .with_message("Training in progress");

        assert_eq!(progress.progress, 0.5);
        assert_eq!(progress.current_step, 50);
        assert_eq!(progress.total_steps, Some(100));
        assert_eq!(progress.elapsed, Duration::from_secs(30));
        assert_eq!(progress.eta, Some(Duration::from_secs(30)));
        assert_eq!(progress.current_metric, Some(0.85));
        assert_eq!(progress.message, "Training in progress");
    }

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_async_config_default() {
        let config = AsyncConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert!(config.enable_progress);
        assert_eq!(config.progress_interval, Duration::from_secs(1));
    }

    #[test]
    fn test_confidence_interval() {
        let ci = ConfidenceInterval {
            lower: 0.1,
            upper: 0.9,
            confidence_level: 0.95,
        };

        assert_eq!(ci.lower, 0.1);
        assert_eq!(ci.upper, 0.9);
        assert_eq!(ci.confidence_level, 0.95);
    }
}
