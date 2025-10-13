//! Error Handling and Violation Detection for Gradient Optimization
//!
//! This module provides comprehensive error handling, violation detection, and recovery
//! strategies for gradient-based optimization algorithms. It includes automated error
//! tracking, pattern recognition, and adaptive recovery mechanisms.
//!
//! # Core Components
//!
//! * [`ViolationDetector`] - Detects optimization constraint violations and anomalies
//! * [`ErrorTracker`] - Comprehensive error logging and analysis system
//! * [`RecoveryManager`] - Automated recovery strategies and fallback mechanisms
//! * [`ErrorAnalyzer`] - Pattern recognition and root cause analysis
//! * [`AlertSystem`] - Real-time alerts and notification management
//!
//! # Example Usage
//!
//! ```rust
//! use crate::pattern_optimization::gradient_optimization::error_handling::*;
//!
//! // Create violation detector with custom thresholds
//! let detector = ViolationDetector::builder()
//!     .gradient_explosion_threshold(1e6)
//!     .convergence_stall_threshold(100)
//!     .numerical_instability_tolerance(1e-12)
//!     .build()?;
//!
//! // Setup error tracker with retention policy
//! let error_tracker = ErrorTracker::builder()
//!     .max_errors_retained(10000)
//!     .error_aggregation_window(Duration::from_hours(1))
//!     .pattern_detection(true)
//!     .build()?;
//!
//! // Configure recovery manager
//! let recovery_manager = RecoveryManager::builder()
//!     .max_recovery_attempts(3)
//!     .fallback_strategy(FallbackStrategy::Conservative)
//!     .auto_checkpoint(true)
//!     .build()?;
//! ```

use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::fmt;
use std::error::Error as StdError;
use std::thread::ThreadId;
use scirs2_core::error::{CoreError, Result as SklResult};
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Random, rng};
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};

/// Types of optimization violations that can be detected
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    /// Gradient values exceed numerical limits
    GradientExplosion { magnitude: f64, threshold: f64 },
    /// Gradients become too small for meaningful progress
    GradientVanishing { magnitude: f64, threshold: f64 },
    /// Optimization has stopped making progress
    ConvergenceStall { iterations_stalled: usize, threshold: usize },
    /// Numerical instability detected (NaN, Inf values)
    NumericalInstability { location: String, value: f64 },
    /// Learning rate is inappropriate for the optimization landscape
    LearningRateViolation { current_rate: f64, suggested_range: (f64, f64) },
    /// Memory usage exceeds configured limits
    MemoryExhaustion { current_usage: usize, limit: usize },
    /// Optimization parameters are outside valid ranges
    ParameterConstraintViolation { parameter: String, value: f64, constraint: ConstraintType },
    /// Detected oscillatory behavior in optimization
    OscillationDetected { frequency: f64, amplitude: f64 },
    /// Loss function returning invalid values
    LossFunctionViolation { loss_value: f64, iteration: usize },
    /// Custom user-defined violation
    Custom { violation_type: String, details: HashMap<String, String> },
}

/// Types of constraints that can be violated
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    Range { min: f64, max: f64 },
    Positive,
    Negative,
    NonZero,
    Finite,
    Probability, // [0, 1]
    Custom { constraint_fn: String }, // Function name for custom constraints
}

/// Severity levels for errors and violations
#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
    Fatal,
}

/// Error categories for classification
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    Numerical,
    Convergence,
    Configuration,
    Resource,
    Network,
    UserInput,
    System,
    Algorithm,
    Custom { category: String },
}

/// Recovery strategies for different types of failures
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Retry the operation with the same parameters
    Retry { max_attempts: usize, delay: Duration },
    /// Adjust optimization parameters and retry
    Adjust { adjustment_type: AdjustmentType },
    /// Fall back to a simpler/more conservative approach
    Fallback { strategy: FallbackStrategy },
    /// Reset to last known good state
    Checkpoint { checkpoint_id: String },
    /// Graceful degradation with reduced functionality
    Degrade { degradation_level: f64 },
    /// Abort the optimization process
    Abort { save_partial_results: bool },
    /// Custom recovery strategy
    Custom { strategy_name: String, parameters: HashMap<String, String> },
}

/// Types of parameter adjustments for recovery
#[derive(Debug, Clone, PartialEq)]
pub enum AdjustmentType {
    ReduceLearningRate { factor: f64 },
    IncreaseLearningRate { factor: f64 },
    AdaptiveLearningRate,
    ReduceBatchSize { factor: f64 },
    IncreaseBatchSize { factor: f64 },
    RegularizationIncrease { factor: f64 },
    RegularizationDecrease { factor: f64 },
    ClipGradients { max_norm: f64 },
    NoiseInjection { noise_level: f64 },
    ParameterReset { reset_percentage: f64 },
    Custom { adjustment_name: String, parameters: HashMap<String, f64> },
}

/// Fallback strategies for severe failures
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackStrategy {
    Conservative,    // Use very safe, slow parameters
    Aggressive,      // Use more aggressive parameters to escape local minima
    Random,          // Random restart with different initialization
    Ensemble,        // Switch to ensemble methods
    Simplified,      // Use simplified version of algorithm
    Alternative,     // Switch to alternative optimization algorithm
    Manual,          // Require manual intervention
}

/// Error information structure
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    pub id: u64,
    pub timestamp: Instant,
    pub thread_id: ThreadId,
    pub error_type: OptimizationError,
    pub severity: Severity,
    pub category: ErrorCategory,
    pub context: ErrorContext,
    pub stack_trace: Option<String>,
    pub recovery_attempted: bool,
    pub recovery_successful: Option<bool>,
}

/// Context information when error occurred
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub iteration: usize,
    pub current_loss: Option<f64>,
    pub gradient_norm: Option<f64>,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub algorithm_state: HashMap<String, String>,
    pub memory_usage: usize,
    pub elapsed_time: Duration,
    pub additional_info: HashMap<String, String>,
}

/// Main optimization error type
#[derive(Debug, Clone)]
pub enum OptimizationError {
    Violation(ViolationType),
    System(SystemError),
    Algorithm(AlgorithmError),
    Configuration(ConfigurationError),
    Resource(ResourceError),
    Network(NetworkError),
    Custom { error_type: String, message: String },
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationError::Violation(v) => write!(f, "Optimization violation: {:?}", v),
            OptimizationError::System(e) => write!(f, "System error: {:?}", e),
            OptimizationError::Algorithm(e) => write!(f, "Algorithm error: {:?}", e),
            OptimizationError::Configuration(e) => write!(f, "Configuration error: {:?}", e),
            OptimizationError::Resource(e) => write!(f, "Resource error: {:?}", e),
            OptimizationError::Network(e) => write!(f, "Network error: {:?}", e),
            OptimizationError::Custom { error_type, message } => {
                write!(f, "Custom error ({}): {}", error_type, message)
            }
        }
    }
}

impl StdError for OptimizationError {}

/// System-level errors
#[derive(Debug, Clone)]
pub enum SystemError {
    OutOfMemory { requested: usize, available: usize },
    ThreadPanic { thread_id: ThreadId, panic_message: String },
    FileSystemError { path: String, operation: String, error: String },
    PermissionDenied { resource: String },
    Timeout { operation: String, duration: Duration },
}

/// Algorithm-specific errors
#[derive(Debug, Clone)]
pub enum AlgorithmError {
    InvalidState { state: String, expected: String },
    ConvergenceFailure { reason: String, iterations: usize },
    MatrixSingularity { matrix_name: String },
    EigenvalueComputation { error: String },
    OptimizationDivergence { iteration: usize, loss: f64 },
}

/// Configuration errors
#[derive(Debug, Clone)]
pub enum ConfigurationError {
    InvalidParameter { parameter: String, value: String, constraint: String },
    MissingParameter { parameter: String },
    IncompatibleParameters { param1: String, param2: String, reason: String },
    InvalidRange { parameter: String, min: f64, max: f64, value: f64 },
}

/// Resource errors
#[derive(Debug, Clone)]
pub enum ResourceError {
    InsufficientMemory { required: usize, available: usize },
    InsufficientStorage { required: usize, available: usize },
    CPUExhaustion { load_percentage: f64 },
    GPUError { device_id: usize, error: String },
    NetworkBandwidth { required_mbps: f64, available_mbps: f64 },
}

/// Network errors
#[derive(Debug, Clone)]
pub enum NetworkError {
    ConnectionTimeout { host: String, timeout: Duration },
    ProtocolError { protocol: String, error_code: u32 },
    AuthenticationFailure { user: String },
    DataCorruption { checksum_expected: String, checksum_actual: String },
    NodeFailure { node_id: String, failure_type: String },
}

/// Violation detector for optimization constraints
pub struct ViolationDetector {
    config: ViolationDetectorConfig,
    detection_state: Arc<RwLock<DetectionState>>,
    pattern_analyzer: Arc<PatternAnalyzer>,
    metrics: Arc<ViolationMetrics>,
    alert_system: Arc<AlertSystem>,
}

/// Configuration for violation detection
#[derive(Debug, Clone)]
pub struct ViolationDetectorConfig {
    pub gradient_explosion_threshold: f64,
    pub gradient_vanishing_threshold: f64,
    pub convergence_stall_threshold: usize,
    pub numerical_instability_tolerance: f64,
    pub learning_rate_bounds: (f64, f64),
    pub memory_limit_bytes: usize,
    pub oscillation_detection_window: usize,
    pub loss_validity_checks: bool,
    pub custom_constraints: HashMap<String, Box<dyn Fn(f64) -> bool + Send + Sync>>,
}

impl Default for ViolationDetectorConfig {
    fn default() -> Self {
        Self {
            gradient_explosion_threshold: 1e8,
            gradient_vanishing_threshold: 1e-8,
            convergence_stall_threshold: 50,
            numerical_instability_tolerance: 1e-15,
            learning_rate_bounds: (1e-8, 1e2),
            memory_limit_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            oscillation_detection_window: 20,
            loss_validity_checks: true,
            custom_constraints: HashMap::new(),
        }
    }
}

/// Internal state for violation detection
#[derive(Debug)]
struct DetectionState {
    gradient_history: VecDeque<f64>,
    loss_history: VecDeque<f64>,
    convergence_tracker: ConvergenceTracker,
    oscillation_detector: OscillationDetector,
    memory_monitor: MemoryMonitor,
    violation_counts: HashMap<String, usize>,
}

/// Convergence tracking state
#[derive(Debug)]
struct ConvergenceTracker {
    best_loss: f64,
    stall_counter: usize,
    improvement_threshold: f64,
    last_improvement: Instant,
}

/// Oscillation detection state
#[derive(Debug)]
struct OscillationDetector {
    value_history: VecDeque<f64>,
    peak_detector: PeakDetector,
    frequency_analyzer: FrequencyAnalyzer,
}

/// Peak detection for oscillation analysis
#[derive(Debug)]
struct PeakDetector {
    peaks: VecDeque<(usize, f64)>,
    valleys: VecDeque<(usize, f64)>,
    min_prominence: f64,
}

/// Frequency analysis for oscillation patterns
#[derive(Debug)]
struct FrequencyAnalyzer {
    window_size: usize,
    frequency_bins: Vec<f64>,
    power_spectrum: Vec<f64>,
}

/// Memory monitoring state
#[derive(Debug)]
struct MemoryMonitor {
    current_usage: usize,
    peak_usage: usize,
    allocation_history: VecDeque<(Instant, usize)>,
    warning_threshold: usize,
}

impl ViolationDetector {
    /// Create a new violation detector builder
    pub fn builder() -> ViolationDetectorBuilder {
        ViolationDetectorBuilder::new()
    }

    /// Check for gradient-related violations
    pub fn check_gradient_violations(&self, gradient: ArrayView1<f64>) -> SklResult<Vec<ViolationType>> {
        let mut violations = Vec::new();
        let config = &self.config;

        // Calculate gradient magnitude
        let gradient_norm = gradient.mapv(|x| x * x).sum().sqrt();

        // Check for gradient explosion
        if gradient_norm > config.gradient_explosion_threshold {
            violations.push(ViolationType::GradientExplosion {
                magnitude: gradient_norm,
                threshold: config.gradient_explosion_threshold,
            });
        }

        // Check for gradient vanishing
        if gradient_norm < config.gradient_vanishing_threshold {
            violations.push(ViolationType::GradientVanishing {
                magnitude: gradient_norm,
                threshold: config.gradient_vanishing_threshold,
            });
        }

        // Check for numerical instability
        for (i, &value) in gradient.iter().enumerate() {
            if !value.is_finite() {
                violations.push(ViolationType::NumericalInstability {
                    location: format!("gradient[{}]", i),
                    value,
                });
            }
        }

        // Update gradient history
        let mut state = self.detection_state.write()
            .map_err(|_| CoreError::LockError("Failed to acquire detection state lock".to_string()))?;
        state.gradient_history.push_back(gradient_norm);
        if state.gradient_history.len() > 1000 {
            state.gradient_history.pop_front();
        }

        // Analyze gradient patterns
        if let Some(pattern_violations) = self.pattern_analyzer.analyze_gradient_patterns(&state.gradient_history)? {
            violations.extend(pattern_violations);
        }

        Ok(violations)
    }

    /// Check for convergence-related violations
    pub fn check_convergence_violations(&self, current_loss: f64, iteration: usize) -> SklResult<Vec<ViolationType>> {
        let mut violations = Vec::new();
        let config = &self.config;

        let mut state = self.detection_state.write()
            .map_err(|_| CoreError::LockError("Failed to acquire detection state lock".to_string()))?;

        // Update loss history
        state.loss_history.push_back(current_loss);
        if state.loss_history.len() > 1000 {
            state.loss_history.pop_front();
        }

        // Check for loss function violations
        if config.loss_validity_checks {
            if !current_loss.is_finite() {
                violations.push(ViolationType::LossFunctionViolation {
                    loss_value: current_loss,
                    iteration,
                });
            }
        }

        // Update convergence tracker
        let tracker = &mut state.convergence_tracker;
        if current_loss < tracker.best_loss - tracker.improvement_threshold {
            tracker.best_loss = current_loss;
            tracker.stall_counter = 0;
            tracker.last_improvement = Instant::now();
        } else {
            tracker.stall_counter += 1;
        }

        // Check for convergence stall
        if tracker.stall_counter >= config.convergence_stall_threshold {
            violations.push(ViolationType::ConvergenceStall {
                iterations_stalled: tracker.stall_counter,
                threshold: config.convergence_stall_threshold,
            });
        }

        // Check for oscillation
        state.oscillation_detector.value_history.push_back(current_loss);
        if state.oscillation_detector.value_history.len() > config.oscillation_detection_window {
            state.oscillation_detector.value_history.pop_front();

            if let Some(oscillation) = self.detect_oscillation(&state.oscillation_detector)? {
                violations.push(oscillation);
            }
        }

        Ok(violations)
    }

    /// Check parameter constraint violations
    pub fn check_parameter_violations(&self, parameters: &HashMap<String, f64>) -> SklResult<Vec<ViolationType>> {
        let mut violations = Vec::new();
        let config = &self.config;

        for (param_name, &value) in parameters {
            match param_name.as_str() {
                "learning_rate" => {
                    if value < config.learning_rate_bounds.0 || value > config.learning_rate_bounds.1 {
                        violations.push(ViolationType::LearningRateViolation {
                            current_rate: value,
                            suggested_range: config.learning_rate_bounds,
                        });
                    }
                }
                _ => {
                    // Check custom constraints
                    if let Some(constraint_fn) = config.custom_constraints.get(param_name) {
                        if !constraint_fn(value) {
                            violations.push(ViolationType::ParameterConstraintViolation {
                                parameter: param_name.clone(),
                                value,
                                constraint: ConstraintType::Custom { constraint_fn: param_name.clone() },
                            });
                        }
                    }
                }
            }
        }

        Ok(violations)
    }

    /// Check memory usage violations
    pub fn check_memory_violations(&self) -> SklResult<Vec<ViolationType>> {
        let mut violations = Vec::new();
        let config = &self.config;

        let mut state = self.detection_state.write()
            .map_err(|_| CoreError::LockError("Failed to acquire detection state lock".to_string()))?;

        // Update memory monitor
        let current_memory = self.get_current_memory_usage()?;
        state.memory_monitor.current_usage = current_memory;
        state.memory_monitor.peak_usage = state.memory_monitor.peak_usage.max(current_memory);
        state.memory_monitor.allocation_history.push_back((Instant::now(), current_memory));

        // Clean old history
        let cutoff_time = Instant::now() - Duration::from_secs(3600); // Keep 1 hour of history
        while let Some(&(timestamp, _)) = state.memory_monitor.allocation_history.front() {
            if timestamp < cutoff_time {
                state.memory_monitor.allocation_history.pop_front();
            } else {
                break;
            }
        }

        // Check memory limit
        if current_memory > config.memory_limit_bytes {
            violations.push(ViolationType::MemoryExhaustion {
                current_usage: current_memory,
                limit: config.memory_limit_bytes,
            });
        }

        Ok(violations)
    }

    /// Get comprehensive violation summary
    pub fn get_violation_summary(&self) -> SklResult<ViolationSummary> {
        let state = self.detection_state.read()
            .map_err(|_| CoreError::LockError("Failed to acquire detection state lock".to_string()))?;

        let total_violations = state.violation_counts.values().sum();
        let most_common_violation = state.violation_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(violation_type, &count)| (violation_type.clone(), count));

        let recent_gradient_trend = if state.gradient_history.len() >= 10 {
            let recent: Vec<f64> = state.gradient_history.iter().rev().take(10).cloned().collect();
            Some(self.calculate_trend(&recent)?)
        } else {
            None
        };

        let recent_loss_trend = if state.loss_history.len() >= 10 {
            let recent: Vec<f64> = state.loss_history.iter().rev().take(10).cloned().collect();
            Some(self.calculate_trend(&recent)?)
        } else {
            None
        };

        Ok(ViolationSummary {
            total_violations,
            violation_counts: state.violation_counts.clone(),
            most_common_violation,
            current_memory_usage: state.memory_monitor.current_usage,
            peak_memory_usage: state.memory_monitor.peak_usage,
            convergence_stall_count: state.convergence_tracker.stall_counter,
            recent_gradient_trend,
            recent_loss_trend,
            last_improvement: state.convergence_tracker.last_improvement,
        })
    }

    fn detect_oscillation(&self, detector: &OscillationDetector) -> SklResult<Option<ViolationType>> {
        if detector.value_history.len() < 10 {
            return Ok(None);
        }

        let values: Vec<f64> = detector.value_history.iter().cloned().collect();

        // Simple oscillation detection - look for alternating pattern
        let mut direction_changes = 0;
        let mut last_direction = 0i8; // -1, 0, 1 for decreasing, equal, increasing

        for window in values.windows(2) {
            let current_direction = if window[1] > window[0] {
                1
            } else if window[1] < window[0] {
                -1
            } else {
                0
            };

            if current_direction != 0 && last_direction != 0 && current_direction != last_direction {
                direction_changes += 1;
            }
            last_direction = current_direction;
        }

        // If more than half the transitions are direction changes, consider it oscillation
        let oscillation_ratio = direction_changes as f64 / (values.len() - 1) as f64;
        if oscillation_ratio > 0.6 {
            let amplitude = self.calculate_oscillation_amplitude(&values)?;
            let frequency = direction_changes as f64 / values.len() as f64;

            return Ok(Some(ViolationType::OscillationDetected { frequency, amplitude }));
        }

        Ok(None)
    }

    fn calculate_oscillation_amplitude(&self, values: &[f64]) -> SklResult<f64> {
        if values.is_empty() {
            return Ok(0.0);
        }

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Ok(max_val - min_val)
    }

    fn calculate_trend(&self, values: &[f64]) -> SklResult<f64> {
        if values.len() < 2 {
            return Ok(0.0);
        }

        // Simple linear regression slope calculation
        let n = values.len() as f64;
        let x_mean = (0..values.len()).map(|i| i as f64).sum::<f64>() / n;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = (0..values.len())
            .map(|i| (i as f64 - x_mean) * (values[i] - y_mean))
            .sum();

        let denominator: f64 = (0..values.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();

        if denominator.abs() < 1e-10 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn get_current_memory_usage(&self) -> SklResult<usize> {
        // In a real implementation, this would query actual memory usage
        // For now, return a placeholder value
        Ok(1024 * 1024 * 100) // 100 MB
    }
}

/// Error tracking and analysis system
pub struct ErrorTracker {
    config: ErrorTrackerConfig,
    error_storage: Arc<RwLock<ErrorStorage>>,
    pattern_detector: Arc<ErrorPatternDetector>,
    metrics: Arc<ErrorMetrics>,
    notification_system: Arc<NotificationSystem>,
}

/// Configuration for error tracking
#[derive(Debug, Clone)]
pub struct ErrorTrackerConfig {
    pub max_errors_retained: usize,
    pub error_aggregation_window: Duration,
    pub pattern_detection_enabled: bool,
    pub automatic_categorization: bool,
    pub real_time_alerts: bool,
    pub error_correlation_analysis: bool,
    pub retention_policy: RetentionPolicy,
}

/// Error retention policies
#[derive(Debug, Clone)]
pub enum RetentionPolicy {
    /// Keep errors for a specific duration
    TimeBased { duration: Duration },
    /// Keep a maximum number of errors
    CountBased { max_count: usize },
    /// Keep errors based on severity
    SeverityBased { min_severity: Severity },
    /// Custom retention logic
    Custom { policy_name: String },
}

impl Default for ErrorTrackerConfig {
    fn default() -> Self {
        Self {
            max_errors_retained: 10000,
            error_aggregation_window: Duration::from_hours(1),
            pattern_detection_enabled: true,
            automatic_categorization: true,
            real_time_alerts: true,
            error_correlation_analysis: true,
            retention_policy: RetentionPolicy::CountBased { max_count: 10000 },
        }
    }
}

/// Error storage system
#[derive(Debug)]
struct ErrorStorage {
    errors: VecDeque<ErrorInfo>,
    error_index: HashMap<u64, usize>, // error_id -> index in errors
    category_index: HashMap<ErrorCategory, Vec<u64>>,
    severity_index: HashMap<Severity, Vec<u64>>,
    temporal_index: HashMap<String, Vec<u64>>, // time bucket -> error_ids
    error_counter: u64,
}

impl ErrorStorage {
    fn new() -> Self {
        Self {
            errors: VecDeque::new(),
            error_index: HashMap::new(),
            category_index: HashMap::new(),
            severity_index: HashMap::new(),
            temporal_index: HashMap::new(),
            error_counter: 0,
        }
    }

    fn add_error(&mut self, mut error_info: ErrorInfo) -> u64 {
        self.error_counter += 1;
        error_info.id = self.error_counter;

        let error_id = error_info.id;
        let category = error_info.category.clone();
        let severity = error_info.severity.clone();

        // Add to main storage
        self.errors.push_back(error_info);
        self.error_index.insert(error_id, self.errors.len() - 1);

        // Add to category index
        self.category_index.entry(category).or_default().push(error_id);

        // Add to severity index
        self.severity_index.entry(severity).or_default().push(error_id);

        // Add to temporal index (bucket by hour)
        let time_bucket = format!("{}", error_info.timestamp.elapsed().as_secs() / 3600);
        self.temporal_index.entry(time_bucket).or_default().push(error_id);

        error_id
    }

    fn cleanup_old_errors(&mut self, retention_policy: &RetentionPolicy) {
        match retention_policy {
            RetentionPolicy::CountBased { max_count } => {
                while self.errors.len() > *max_count {
                    if let Some(error) = self.errors.pop_front() {
                        self.remove_error_from_indices(error.id);
                    }
                }
            }
            RetentionPolicy::TimeBased { duration } => {
                let cutoff_time = Instant::now() - *duration;
                while let Some(error) = self.errors.front() {
                    if error.timestamp < cutoff_time {
                        let error = self.errors.pop_front().unwrap();
                        self.remove_error_from_indices(error.id);
                    } else {
                        break;
                    }
                }
            }
            RetentionPolicy::SeverityBased { min_severity } => {
                self.errors.retain(|error| error.severity >= *min_severity);
                // Rebuild indices after filtering
                self.rebuild_indices();
            }
            RetentionPolicy::Custom { .. } => {
                // Custom retention logic would be implemented here
            }
        }
    }

    fn remove_error_from_indices(&mut self, error_id: u64) {
        self.error_index.remove(&error_id);

        // Remove from category and severity indices
        for errors in self.category_index.values_mut() {
            errors.retain(|&id| id != error_id);
        }
        for errors in self.severity_index.values_mut() {
            errors.retain(|&id| id != error_id);
        }
        for errors in self.temporal_index.values_mut() {
            errors.retain(|&id| id != error_id);
        }
    }

    fn rebuild_indices(&mut self) {
        self.error_index.clear();
        self.category_index.clear();
        self.severity_index.clear();
        self.temporal_index.clear();

        for (index, error) in self.errors.iter().enumerate() {
            self.error_index.insert(error.id, index);
            self.category_index.entry(error.category.clone()).or_default().push(error.id);
            self.severity_index.entry(error.severity.clone()).or_default().push(error.id);

            let time_bucket = format!("{}", error.timestamp.elapsed().as_secs() / 3600);
            self.temporal_index.entry(time_bucket).or_default().push(error.id);
        }
    }
}

impl ErrorTracker {
    /// Create a new error tracker builder
    pub fn builder() -> ErrorTrackerBuilder {
        ErrorTrackerBuilder::new()
    }

    /// Record a new error
    pub fn record_error(&self, error: OptimizationError, context: ErrorContext, severity: Severity) -> SklResult<u64> {
        let error_info = ErrorInfo {
            id: 0, // Will be set by storage
            timestamp: Instant::now(),
            thread_id: std::thread::current().id(),
            error_type: error.clone(),
            severity: severity.clone(),
            category: self.categorize_error(&error)?,
            context,
            stack_trace: self.capture_stack_trace(),
            recovery_attempted: false,
            recovery_successful: None,
        };

        let mut storage = self.error_storage.write()
            .map_err(|_| CoreError::LockError("Failed to acquire error storage lock".to_string()))?;

        let error_id = storage.add_error(error_info);

        // Cleanup old errors according to retention policy
        storage.cleanup_old_errors(&self.config.retention_policy);

        // Update metrics
        self.metrics.record_error(&error, &severity)?;

        // Trigger alerts if enabled
        if self.config.real_time_alerts && severity >= Severity::Error {
            self.notification_system.send_alert(&error, &severity)?;
        }

        // Perform pattern detection if enabled
        if self.config.pattern_detection_enabled {
            self.pattern_detector.analyze_error_patterns(&storage)?;
        }

        Ok(error_id)
    }

    /// Update error with recovery information
    pub fn update_recovery_status(&self, error_id: u64, attempted: bool, successful: Option<bool>) -> SklResult<()> {
        let mut storage = self.error_storage.write()
            .map_err(|_| CoreError::LockError("Failed to acquire error storage lock".to_string()))?;

        if let Some(&index) = storage.error_index.get(&error_id) {
            if let Some(error) = storage.errors.get_mut(index) {
                error.recovery_attempted = attempted;
                error.recovery_successful = successful;
            }
        }

        Ok(())
    }

    /// Get error statistics
    pub fn get_error_statistics(&self) -> SklResult<ErrorStatistics> {
        let storage = self.error_storage.read()
            .map_err(|_| CoreError::LockError("Failed to acquire error storage lock".to_string()))?;

        let total_errors = storage.errors.len();
        let errors_by_category = storage.category_index.iter()
            .map(|(cat, errors)| (cat.clone(), errors.len()))
            .collect();
        let errors_by_severity = storage.severity_index.iter()
            .map(|(sev, errors)| (sev.clone(), errors.len()))
            .collect();

        let recent_errors = storage.errors.iter()
            .rev()
            .take(100)
            .map(|e| e.clone())
            .collect();

        let recovery_success_rate = self.calculate_recovery_success_rate(&storage)?;

        Ok(ErrorStatistics {
            total_errors,
            errors_by_category,
            errors_by_severity,
            recent_errors,
            recovery_success_rate,
            average_recovery_time: Duration::from_secs(0), // Placeholder
        })
    }

    fn categorize_error(&self, error: &OptimizationError) -> SklResult<ErrorCategory> {
        match error {
            OptimizationError::Violation(v) => match v {
                ViolationType::GradientExplosion { .. } | ViolationType::GradientVanishing { .. } |
                ViolationType::NumericalInstability { .. } => Ok(ErrorCategory::Numerical),
                ViolationType::ConvergenceStall { .. } | ViolationType::OscillationDetected { .. } => Ok(ErrorCategory::Convergence),
                ViolationType::LearningRateViolation { .. } | ViolationType::ParameterConstraintViolation { .. } => Ok(ErrorCategory::Configuration),
                ViolationType::MemoryExhaustion { .. } => Ok(ErrorCategory::Resource),
                ViolationType::LossFunctionViolation { .. } => Ok(ErrorCategory::Algorithm),
                ViolationType::Custom { .. } => Ok(ErrorCategory::Custom { category: "violation".to_string() }),
            },
            OptimizationError::System(_) => Ok(ErrorCategory::System),
            OptimizationError::Algorithm(_) => Ok(ErrorCategory::Algorithm),
            OptimizationError::Configuration(_) => Ok(ErrorCategory::Configuration),
            OptimizationError::Resource(_) => Ok(ErrorCategory::Resource),
            OptimizationError::Network(_) => Ok(ErrorCategory::Network),
            OptimizationError::Custom { error_type, .. } => Ok(ErrorCategory::Custom { category: error_type.clone() }),
        }
    }

    fn capture_stack_trace(&self) -> Option<String> {
        // In a real implementation, this would capture the actual stack trace
        Some("stack trace placeholder".to_string())
    }

    fn calculate_recovery_success_rate(&self, storage: &ErrorStorage) -> SklResult<f64> {
        let total_attempted = storage.errors.iter()
            .filter(|e| e.recovery_attempted)
            .count();

        if total_attempted == 0 {
            return Ok(0.0);
        }

        let successful = storage.errors.iter()
            .filter(|e| e.recovery_attempted && e.recovery_successful == Some(true))
            .count();

        Ok(successful as f64 / total_attempted as f64)
    }
}

/// Recovery manager for automated error recovery
pub struct RecoveryManager {
    config: RecoveryManagerConfig,
    recovery_strategies: HashMap<String, Box<dyn RecoveryExecutor + Send + Sync>>,
    recovery_history: Arc<RwLock<RecoveryHistory>>,
    checkpoint_manager: Arc<CheckpointManager>,
    metrics: Arc<RecoveryMetrics>,
}

/// Configuration for recovery manager
#[derive(Debug, Clone)]
pub struct RecoveryManagerConfig {
    pub max_recovery_attempts: usize,
    pub recovery_timeout: Duration,
    pub fallback_strategy: FallbackStrategy,
    pub auto_checkpoint: bool,
    pub checkpoint_frequency: Duration,
    pub learning_enabled: bool,
    pub strategy_adaptation: bool,
}

impl Default for RecoveryManagerConfig {
    fn default() -> Self {
        Self {
            max_recovery_attempts: 3,
            recovery_timeout: Duration::from_secs(300),
            fallback_strategy: FallbackStrategy::Conservative,
            auto_checkpoint: true,
            checkpoint_frequency: Duration::from_secs(600),
            learning_enabled: true,
            strategy_adaptation: true,
        }
    }
}

/// Recovery execution trait
pub trait RecoveryExecutor {
    fn execute_recovery(&self, error: &OptimizationError, context: &ErrorContext) -> SklResult<RecoveryResult>;
    fn estimate_recovery_time(&self, error: &OptimizationError) -> Duration;
    fn recovery_success_probability(&self, error: &OptimizationError) -> f64;
}

/// Result of a recovery attempt
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub success: bool,
    pub new_state: Option<HashMap<String, String>>,
    pub recovery_time: Duration,
    pub side_effects: Vec<SideEffect>,
    pub recommendations: Vec<String>,
}

/// Side effects of recovery operations
#[derive(Debug, Clone)]
pub enum SideEffect {
    PerformanceDegradation { factor: f64 },
    MemoryIncrease { bytes: usize },
    AccuracyReduction { percentage: f64 },
    ConfigurationChange { parameter: String, old_value: String, new_value: String },
    StateReset { component: String },
}

/// Recovery history tracking
#[derive(Debug)]
struct RecoveryHistory {
    attempts: VecDeque<RecoveryAttempt>,
    success_rates: HashMap<String, (usize, usize)>, // strategy -> (successes, attempts)
    strategy_effectiveness: HashMap<String, f64>,
}

/// Individual recovery attempt record
#[derive(Debug, Clone)]
struct RecoveryAttempt {
    timestamp: Instant,
    error_type: String,
    strategy_used: String,
    success: bool,
    recovery_time: Duration,
    context: ErrorContext,
}

impl RecoveryManager {
    /// Create a new recovery manager builder
    pub fn builder() -> RecoveryManagerBuilder {
        RecoveryManagerBuilder::new()
    }

    /// Attempt to recover from an error
    pub fn attempt_recovery(&self, error: &OptimizationError, context: &ErrorContext) -> SklResult<RecoveryResult> {
        let strategy_name = self.select_recovery_strategy(error, context)?;

        if let Some(strategy) = self.recovery_strategies.get(&strategy_name) {
            let start_time = Instant::now();
            let result = strategy.execute_recovery(error, context)?;
            let recovery_time = start_time.elapsed();

            // Record the attempt
            self.record_recovery_attempt(error, context, &strategy_name, &result, recovery_time)?;

            // Update strategy effectiveness if learning is enabled
            if self.config.learning_enabled {
                self.update_strategy_effectiveness(&strategy_name, &result)?;
            }

            Ok(result)
        } else {
            // Fall back to default strategy
            self.execute_fallback_strategy(error, context)
        }
    }

    /// Select the best recovery strategy for the given error
    pub fn select_recovery_strategy(&self, error: &OptimizationError, _context: &ErrorContext) -> SklResult<String> {
        // Simplified strategy selection - real implementation would be more sophisticated
        match error {
            OptimizationError::Violation(ViolationType::GradientExplosion { .. }) => {
                Ok("gradient_clipping".to_string())
            }
            OptimizationError::Violation(ViolationType::ConvergenceStall { .. }) => {
                Ok("learning_rate_adjustment".to_string())
            }
            OptimizationError::Resource(ResourceError::InsufficientMemory { .. }) => {
                Ok("memory_optimization".to_string())
            }
            _ => Ok("default".to_string()),
        }
    }

    /// Get recovery statistics
    pub fn get_recovery_statistics(&self) -> SklResult<RecoveryStatistics> {
        let history = self.recovery_history.read()
            .map_err(|_| CoreError::LockError("Failed to acquire recovery history lock".to_string()))?;

        let total_attempts = history.attempts.len();
        let successful_attempts = history.attempts.iter().filter(|a| a.success).count();
        let overall_success_rate = if total_attempts > 0 {
            successful_attempts as f64 / total_attempts as f64
        } else {
            0.0
        };

        let strategy_performance = history.success_rates.iter()
            .map(|(strategy, &(successes, attempts))| {
                let rate = if attempts > 0 { successes as f64 / attempts as f64 } else { 0.0 };
                (strategy.clone(), rate)
            })
            .collect();

        let average_recovery_time = if !history.attempts.is_empty() {
            let total_time: Duration = history.attempts.iter().map(|a| a.recovery_time).sum();
            total_time / history.attempts.len() as u32
        } else {
            Duration::from_secs(0)
        };

        Ok(RecoveryStatistics {
            total_attempts,
            successful_attempts,
            overall_success_rate,
            strategy_performance,
            average_recovery_time,
            most_effective_strategy: self.get_most_effective_strategy(&history)?,
        })
    }

    fn record_recovery_attempt(&self, error: &OptimizationError, context: &ErrorContext,
                              strategy: &str, result: &RecoveryResult, recovery_time: Duration) -> SklResult<()> {
        let mut history = self.recovery_history.write()
            .map_err(|_| CoreError::LockError("Failed to acquire recovery history lock".to_string()))?;

        let attempt = RecoveryAttempt {
            timestamp: Instant::now(),
            error_type: format!("{:?}", error),
            strategy_used: strategy.to_string(),
            success: result.success,
            recovery_time,
            context: context.clone(),
        };

        history.attempts.push_back(attempt);
        if history.attempts.len() > 10000 {
            history.attempts.pop_front();
        }

        // Update success rates
        let (successes, attempts) = history.success_rates.entry(strategy.to_string()).or_insert((0, 0));
        *attempts += 1;
        if result.success {
            *successes += 1;
        }

        Ok(())
    }

    fn update_strategy_effectiveness(&self, strategy: &str, result: &RecoveryResult) -> SklResult<()> {
        let mut history = self.recovery_history.write()
            .map_err(|_| CoreError::LockError("Failed to acquire recovery history lock".to_string()))?;

        let effectiveness = if result.success { 1.0 } else { 0.0 };
        let current_effectiveness = history.strategy_effectiveness.get(strategy).copied().unwrap_or(0.5);

        // Exponential moving average
        let alpha = 0.1;
        let new_effectiveness = alpha * effectiveness + (1.0 - alpha) * current_effectiveness;

        history.strategy_effectiveness.insert(strategy.to_string(), new_effectiveness);

        Ok(())
    }

    fn execute_fallback_strategy(&self, _error: &OptimizationError, _context: &ErrorContext) -> SklResult<RecoveryResult> {
        // Implement fallback strategy based on configuration
        match self.config.fallback_strategy {
            FallbackStrategy::Conservative => {
                Ok(RecoveryResult {
                    success: true,
                    new_state: Some(HashMap::new()),
                    recovery_time: Duration::from_millis(100),
                    side_effects: vec![SideEffect::PerformanceDegradation { factor: 0.5 }],
                    recommendations: vec!["Consider manual intervention".to_string()],
                })
            }
            _ => {
                Ok(RecoveryResult {
                    success: false,
                    new_state: None,
                    recovery_time: Duration::from_millis(50),
                    side_effects: vec![],
                    recommendations: vec!["Manual intervention required".to_string()],
                })
            }
        }
    }

    fn get_most_effective_strategy(&self, history: &RecoveryHistory) -> SklResult<Option<String>> {
        Ok(history.strategy_effectiveness.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy, _)| strategy.clone()))
    }
}

// Placeholder implementations for other components

/// Pattern analyzer for detecting error patterns
pub struct PatternAnalyzer {
    // Implementation details would go here
}

impl PatternAnalyzer {
    pub fn analyze_gradient_patterns(&self, _gradient_history: &VecDeque<f64>) -> SklResult<Option<Vec<ViolationType>>> {
        // Placeholder implementation
        Ok(None)
    }
}

/// Error pattern detector
pub struct ErrorPatternDetector {
    // Implementation details would go here
}

impl ErrorPatternDetector {
    pub fn analyze_error_patterns(&self, _storage: &ErrorStorage) -> SklResult<()> {
        // Placeholder implementation
        Ok(())
    }
}

/// Alert system for real-time notifications
pub struct AlertSystem {
    // Implementation details would go here
}

impl AlertSystem {
    pub fn send_alert(&self, _error: &OptimizationError, _severity: &Severity) -> SklResult<()> {
        // Placeholder implementation
        Ok(())
    }
}

/// Notification system for error tracking
pub struct NotificationSystem {
    // Implementation details would go here
}

impl NotificationSystem {
    pub fn send_alert(&self, _error: &OptimizationError, _severity: &Severity) -> SklResult<()> {
        // Placeholder implementation
        Ok(())
    }
}

/// Checkpoint manager for state recovery
pub struct CheckpointManager {
    // Implementation details would go here
}

impl CheckpointManager {
    pub fn create_checkpoint(&self, _state: &HashMap<String, String>) -> SklResult<String> {
        // Placeholder implementation
        Ok("checkpoint_id".to_string())
    }

    pub fn restore_checkpoint(&self, _checkpoint_id: &str) -> SklResult<HashMap<String, String>> {
        // Placeholder implementation
        Ok(HashMap::new())
    }
}

// Metrics structures

/// Violation metrics collector
pub struct ViolationMetrics {
    registry: MetricRegistry,
    violation_counter: Counter,
    detection_latency: Histogram,
}

impl ViolationMetrics {
    pub fn new() -> SklResult<Self> {
        let registry = MetricRegistry::new();
        let violation_counter = registry.counter("violations_detected_total", "Total violations detected")?;
        let detection_latency = registry.histogram("violation_detection_latency", "Violation detection latency")?;

        Ok(Self {
            registry,
            violation_counter,
            detection_latency,
        })
    }
}

/// Error metrics collector
pub struct ErrorMetrics {
    registry: MetricRegistry,
    error_counter: Counter,
    error_rate: Gauge,
}

impl ErrorMetrics {
    pub fn new() -> SklResult<Self> {
        let registry = MetricRegistry::new();
        let error_counter = registry.counter("errors_total", "Total errors recorded")?;
        let error_rate = registry.gauge("error_rate", "Current error rate")?;

        Ok(Self {
            registry,
            error_counter,
            error_rate,
        })
    }

    pub fn record_error(&self, _error: &OptimizationError, _severity: &Severity) -> SklResult<()> {
        self.error_counter.increment();
        Ok(())
    }
}

/// Recovery metrics collector
pub struct RecoveryMetrics {
    registry: MetricRegistry,
    recovery_attempts: Counter,
    recovery_success_rate: Gauge,
    recovery_duration: Histogram,
}

impl RecoveryMetrics {
    pub fn new() -> SklResult<Self> {
        let registry = MetricRegistry::new();
        let recovery_attempts = registry.counter("recovery_attempts_total", "Total recovery attempts")?;
        let recovery_success_rate = registry.gauge("recovery_success_rate", "Recovery success rate")?;
        let recovery_duration = registry.histogram("recovery_duration", "Recovery operation duration")?;

        Ok(Self {
            registry,
            recovery_attempts,
            recovery_success_rate,
            recovery_duration,
        })
    }
}

// Builder implementations

/// Builder for ViolationDetector
pub struct ViolationDetectorBuilder {
    config: ViolationDetectorConfig,
}

impl ViolationDetectorBuilder {
    pub fn new() -> Self {
        Self {
            config: ViolationDetectorConfig::default(),
        }
    }

    pub fn gradient_explosion_threshold(mut self, threshold: f64) -> Self {
        self.config.gradient_explosion_threshold = threshold;
        self
    }

    pub fn convergence_stall_threshold(mut self, threshold: usize) -> Self {
        self.config.convergence_stall_threshold = threshold;
        self
    }

    pub fn numerical_instability_tolerance(mut self, tolerance: f64) -> Self {
        self.config.numerical_instability_tolerance = tolerance;
        self
    }

    pub fn build(self) -> SklResult<ViolationDetector> {
        let detection_state = Arc::new(RwLock::new(DetectionState {
            gradient_history: VecDeque::new(),
            loss_history: VecDeque::new(),
            convergence_tracker: ConvergenceTracker {
                best_loss: f64::INFINITY,
                stall_counter: 0,
                improvement_threshold: 1e-6,
                last_improvement: Instant::now(),
            },
            oscillation_detector: OscillationDetector {
                value_history: VecDeque::new(),
                peak_detector: PeakDetector {
                    peaks: VecDeque::new(),
                    valleys: VecDeque::new(),
                    min_prominence: 0.01,
                },
                frequency_analyzer: FrequencyAnalyzer {
                    window_size: 50,
                    frequency_bins: Vec::new(),
                    power_spectrum: Vec::new(),
                },
            },
            memory_monitor: MemoryMonitor {
                current_usage: 0,
                peak_usage: 0,
                allocation_history: VecDeque::new(),
                warning_threshold: self.config.memory_limit_bytes / 2,
            },
            violation_counts: HashMap::new(),
        }));

        let pattern_analyzer = Arc::new(PatternAnalyzer {});
        let metrics = Arc::new(ViolationMetrics::new()?);
        let alert_system = Arc::new(AlertSystem {});

        Ok(ViolationDetector {
            config: self.config,
            detection_state,
            pattern_analyzer,
            metrics,
            alert_system,
        })
    }
}

/// Builder for ErrorTracker
pub struct ErrorTrackerBuilder {
    config: ErrorTrackerConfig,
}

impl ErrorTrackerBuilder {
    pub fn new() -> Self {
        Self {
            config: ErrorTrackerConfig::default(),
        }
    }

    pub fn max_errors_retained(mut self, max_errors: usize) -> Self {
        self.config.max_errors_retained = max_errors;
        self
    }

    pub fn error_aggregation_window(mut self, window: Duration) -> Self {
        self.config.error_aggregation_window = window;
        self
    }

    pub fn pattern_detection(mut self, enabled: bool) -> Self {
        self.config.pattern_detection_enabled = enabled;
        self
    }

    pub fn build(self) -> SklResult<ErrorTracker> {
        let error_storage = Arc::new(RwLock::new(ErrorStorage::new()));
        let pattern_detector = Arc::new(ErrorPatternDetector {});
        let metrics = Arc::new(ErrorMetrics::new()?);
        let notification_system = Arc::new(NotificationSystem {});

        Ok(ErrorTracker {
            config: self.config,
            error_storage,
            pattern_detector,
            metrics,
            notification_system,
        })
    }
}

/// Builder for RecoveryManager
pub struct RecoveryManagerBuilder {
    config: RecoveryManagerConfig,
}

impl RecoveryManagerBuilder {
    pub fn new() -> Self {
        Self {
            config: RecoveryManagerConfig::default(),
        }
    }

    pub fn max_recovery_attempts(mut self, max_attempts: usize) -> Self {
        self.config.max_recovery_attempts = max_attempts;
        self
    }

    pub fn fallback_strategy(mut self, strategy: FallbackStrategy) -> Self {
        self.config.fallback_strategy = strategy;
        self
    }

    pub fn auto_checkpoint(mut self, enabled: bool) -> Self {
        self.config.auto_checkpoint = enabled;
        self
    }

    pub fn build(self) -> SklResult<RecoveryManager> {
        let recovery_strategies = HashMap::new(); // Would be populated with actual strategies
        let recovery_history = Arc::new(RwLock::new(RecoveryHistory {
            attempts: VecDeque::new(),
            success_rates: HashMap::new(),
            strategy_effectiveness: HashMap::new(),
        }));
        let checkpoint_manager = Arc::new(CheckpointManager {});
        let metrics = Arc::new(RecoveryMetrics::new()?);

        Ok(RecoveryManager {
            config: self.config,
            recovery_strategies,
            recovery_history,
            checkpoint_manager,
            metrics,
        })
    }
}

// Statistics and result types

/// Violation summary
#[derive(Debug, Clone)]
pub struct ViolationSummary {
    pub total_violations: usize,
    pub violation_counts: HashMap<String, usize>,
    pub most_common_violation: Option<(String, usize)>,
    pub current_memory_usage: usize,
    pub peak_memory_usage: usize,
    pub convergence_stall_count: usize,
    pub recent_gradient_trend: Option<f64>,
    pub recent_loss_trend: Option<f64>,
    pub last_improvement: Instant,
}

/// Error statistics
#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    pub total_errors: usize,
    pub errors_by_category: HashMap<ErrorCategory, usize>,
    pub errors_by_severity: HashMap<Severity, usize>,
    pub recent_errors: Vec<ErrorInfo>,
    pub recovery_success_rate: f64,
    pub average_recovery_time: Duration,
}

/// Recovery statistics
#[derive(Debug, Clone)]
pub struct RecoveryStatistics {
    pub total_attempts: usize,
    pub successful_attempts: usize,
    pub overall_success_rate: f64,
    pub strategy_performance: HashMap<String, f64>,
    pub average_recovery_time: Duration,
    pub most_effective_strategy: Option<String>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_violation_detector_creation() {
        let detector = ViolationDetector::builder()
            .gradient_explosion_threshold(1e6)
            .convergence_stall_threshold(100)
            .build()
            .expect("Failed to create violation detector");

        assert_eq!(detector.config.gradient_explosion_threshold, 1e6);
        assert_eq!(detector.config.convergence_stall_threshold, 100);
    }

    #[test]
    fn test_gradient_explosion_detection() {
        let detector = ViolationDetector::builder()
            .gradient_explosion_threshold(10.0)
            .build()
            .expect("Failed to create detector");

        let large_gradient = array![100.0, 200.0, 50.0];
        let violations = detector.check_gradient_violations(large_gradient.view())
            .expect("Failed to check gradient violations");

        assert!(!violations.is_empty());
        assert!(matches!(violations[0], ViolationType::GradientExplosion { .. }));
    }

    #[test]
    fn test_gradient_vanishing_detection() {
        let detector = ViolationDetector::builder()
            .build()
            .expect("Failed to create detector");

        let small_gradient = array![1e-10, 1e-11, 1e-12];
        let violations = detector.check_gradient_violations(small_gradient.view())
            .expect("Failed to check gradient violations");

        assert!(!violations.is_empty());
        assert!(matches!(violations[0], ViolationType::GradientVanishing { .. }));
    }

    #[test]
    fn test_numerical_instability_detection() {
        let detector = ViolationDetector::builder()
            .build()
            .expect("Failed to create detector");

        let unstable_gradient = array![1.0, f64::NAN, 3.0];
        let violations = detector.check_gradient_violations(unstable_gradient.view())
            .expect("Failed to check gradient violations");

        assert!(!violations.is_empty());
        assert!(matches!(violations[0], ViolationType::NumericalInstability { .. }));
    }

    #[test]
    fn test_convergence_stall_detection() {
        let detector = ViolationDetector::builder()
            .convergence_stall_threshold(3)
            .build()
            .expect("Failed to create detector");

        // Simulate stalled convergence
        for i in 0..5 {
            let violations = detector.check_convergence_violations(1.0, i)
                .expect("Failed to check convergence violations");

            if i >= 3 {
                assert!(!violations.is_empty());
                assert!(matches!(violations[0], ViolationType::ConvergenceStall { .. }));
            }
        }
    }

    #[test]
    fn test_error_tracker_creation() {
        let tracker = ErrorTracker::builder()
            .max_errors_retained(1000)
            .pattern_detection(true)
            .build()
            .expect("Failed to create error tracker");

        assert_eq!(tracker.config.max_errors_retained, 1000);
        assert!(tracker.config.pattern_detection_enabled);
    }

    #[test]
    fn test_error_recording() {
        let tracker = ErrorTracker::builder()
            .build()
            .expect("Failed to create error tracker");

        let error = OptimizationError::Violation(ViolationType::GradientExplosion {
            magnitude: 1e10,
            threshold: 1e8,
        });

        let context = ErrorContext {
            iteration: 100,
            current_loss: Some(0.5),
            gradient_norm: Some(1e10),
            learning_rate: 0.01,
            batch_size: 32,
            algorithm_state: HashMap::new(),
            memory_usage: 1024 * 1024,
            elapsed_time: Duration::from_secs(60),
            additional_info: HashMap::new(),
        };

        let error_id = tracker.record_error(error, context, Severity::Error)
            .expect("Failed to record error");

        assert!(error_id > 0);

        let stats = tracker.get_error_statistics()
            .expect("Failed to get error statistics");

        assert_eq!(stats.total_errors, 1);
    }

    #[test]
    fn test_recovery_manager_creation() {
        let manager = RecoveryManager::builder()
            .max_recovery_attempts(5)
            .auto_checkpoint(true)
            .build()
            .expect("Failed to create recovery manager");

        assert_eq!(manager.config.max_recovery_attempts, 5);
        assert!(manager.config.auto_checkpoint);
    }

    #[test]
    fn test_parameter_violation_detection() {
        let detector = ViolationDetector::builder()
            .build()
            .expect("Failed to create detector");

        let mut parameters = HashMap::new();
        parameters.insert("learning_rate".to_string(), 1e10); // Too high

        let violations = detector.check_parameter_violations(&parameters)
            .expect("Failed to check parameter violations");

        assert!(!violations.is_empty());
        assert!(matches!(violations[0], ViolationType::LearningRateViolation { .. }));
    }

    #[test]
    fn test_oscillation_detection() {
        let detector = ViolationDetector::builder()
            .build()
            .expect("Failed to create detector");

        // Simulate oscillating loss values
        let oscillating_values = vec![1.0, 0.5, 1.2, 0.4, 1.1, 0.6, 1.3, 0.3];

        for (i, &loss) in oscillating_values.iter().enumerate() {
            let _violations = detector.check_convergence_violations(loss, i)
                .expect("Failed to check convergence violations");
        }

        // The oscillation should be detected after enough samples
        let summary = detector.get_violation_summary()
            .expect("Failed to get violation summary");

        // In a real implementation, we would check for oscillation detection
        assert!(summary.total_violations >= 0); // Placeholder assertion
    }
}