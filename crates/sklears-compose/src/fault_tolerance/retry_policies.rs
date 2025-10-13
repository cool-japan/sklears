//! Retry Policies Module
//!
//! Implements sophisticated retry strategies and backoff algorithms for fault tolerance:
//! - Configurable retry strategies (exponential, linear, fixed, jittered)
//! - Advanced backoff algorithms with customizable parameters
//! - Retry condition evaluation and smart retry decisions
//! - Comprehensive metrics and monitoring
//! - Integration with circuit breakers and fault detection

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use scirs2_core::random::{thread_rng, Rng};
use tokio::time::sleep;
use uuid::Uuid;

/// Retry strategy defining how retries are executed
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    /// Fixed delay between retries
    Fixed { delay: Duration },
    /// Linear backoff with increasing delays
    Linear { initial_delay: Duration, increment: Duration },
    /// Exponential backoff with configurable base and multiplier
    Exponential { initial_delay: Duration, multiplier: f64, max_delay: Duration },
    /// Exponential backoff with jitter to avoid thundering herd
    ExponentialJitter {
        initial_delay: Duration,
        multiplier: f64,
        max_delay: Duration,
        jitter_factor: f64,
    },
    /// Custom backoff algorithm
    Custom { calculate_delay: fn(u32) -> Duration },
}

/// Retry condition determining when to retry an operation
#[derive(Debug, Clone)]
pub enum RetryCondition {
    /// Retry on any error
    Any,
    /// Retry on specific error messages
    ErrorMessage { patterns: Vec<String> },
    /// Retry on specific error types
    ErrorType { error_types: Vec<String> },
    /// Retry based on response time
    ResponseTime { threshold: Duration },
    /// Retry based on HTTP status codes (for HTTP operations)
    HttpStatus { retry_codes: Vec<u16> },
    /// Custom retry condition
    Custom { should_retry: fn(&str) -> bool },
}

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicyConfig {
    /// Policy identifier
    pub policy_id: String,
    /// Retry strategy to use
    pub strategy: RetryStrategy,
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Maximum total time for all retry attempts
    pub max_total_time: Duration,
    /// Conditions for retrying
    pub retry_conditions: Vec<RetryCondition>,
    /// Whether to enable metrics collection
    pub collect_metrics: bool,
    /// Timeout for each individual attempt
    pub attempt_timeout: Option<Duration>,
}

/// Individual retry attempt information
#[derive(Debug, Clone)]
pub struct RetryAttempt {
    pub attempt_number: u32,
    pub start_time: Instant,
    pub duration: Duration,
    pub success: bool,
    pub error_message: Option<String>,
    pub delay_before: Duration,
}

/// Retry execution result
#[derive(Debug, Clone)]
pub struct RetryResult<T> {
    /// Final operation result
    pub result: Result<T, RetryError>,
    /// All retry attempts made
    pub attempts: Vec<RetryAttempt>,
    /// Total execution time
    pub total_duration: Duration,
    /// Final attempt number (successful or last failed)
    pub final_attempt: u32,
    /// Retry metrics
    pub metrics: RetryMetrics,
}

/// Retry policy specific errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum RetryError {
    #[error("Maximum retry attempts ({max_attempts}) exceeded")]
    MaxAttemptsExceeded { max_attempts: u32 },
    #[error("Maximum total time ({max_time:?}) exceeded")]
    MaxTimeExceeded { max_time: Duration },
    #[error("Operation failed after retries: {message}")]
    OperationFailed { message: String },
    #[error("Attempt timeout ({timeout:?}) exceeded")]
    AttemptTimeout { timeout: Duration },
    #[error("Retry condition not met: {reason}")]
    RetryConditionNotMet { reason: String },
    #[error("Policy configuration error: {message}")]
    ConfigurationError { message: String },
}

/// Retry metrics for monitoring and analysis
#[derive(Debug, Clone)]
pub struct RetryMetrics {
    /// Policy identifier
    pub policy_id: String,
    /// Total number of operations
    pub total_operations: u64,
    /// Total number of successful operations
    pub successful_operations: u64,
    /// Total number of failed operations (after all retries)
    pub failed_operations: u64,
    /// Total retry attempts across all operations
    pub total_attempts: u64,
    /// Average number of attempts per operation
    pub average_attempts: f64,
    /// Success rate (operations that eventually succeeded)
    pub success_rate: f64,
    /// Average total time per operation
    pub average_total_time: Duration,
    /// Average time per individual attempt
    pub average_attempt_time: Duration,
    /// Most common error types
    pub error_distribution: HashMap<String, u32>,
}

/// Execution context for retry operations
#[derive(Debug, Clone)]
pub struct RetryExecutionContext {
    /// Operation identifier
    pub operation_id: String,
    /// Operation start time
    pub start_time: Instant,
    /// Operation metadata
    pub metadata: HashMap<String, String>,
    /// Custom retry conditions for this operation
    pub custom_conditions: Vec<RetryCondition>,
}

/// Retry policy implementation
#[derive(Debug)]
pub struct RetryPolicy {
    /// Policy configuration
    config: RetryPolicyConfig,
    /// Retry metrics
    metrics: Arc<RwLock<RetryMetrics>>,
    /// Operation history for analysis
    operation_history: Arc<RwLock<Vec<RetryOperationRecord>>>,
}

/// Record of a complete retry operation
#[derive(Debug, Clone)]
struct RetryOperationRecord {
    /// Operation identifier
    operation_id: String,
    /// Start time
    start_time: Instant,
    /// Total duration
    total_duration: Duration,
    /// All attempts made
    attempts: Vec<RetryAttempt>,
    /// Final success status
    success: bool,
    /// Final error message if failed
    final_error: Option<String>,
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self::ExponentialJitter {
            initial_delay: Duration::from_millis(100),
            multiplier: 2.0,
            max_delay: Duration::from_secs(30),
            jitter_factor: 0.1,
        }
    }
}

impl Default for RetryCondition {
    fn default() -> Self {
        Self::Any
    }
}

impl RetryPolicy {
    /// Create new retry policy with configuration
    pub fn new(config: RetryPolicyConfig) -> Self {
        let metrics = RetryMetrics {
            policy_id: config.policy_id.clone(),
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_attempts: 0,
            average_attempts: 0.0,
            success_rate: 0.0,
            average_total_time: Duration::ZERO,
            average_attempt_time: Duration::ZERO,
            error_distribution: HashMap::new(),
        };

        Self {
            config,
            metrics: Arc::new(RwLock::new(metrics)),
            operation_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create retry policy with default configuration
    pub fn with_defaults(policy_id: String) -> Self {
        let config = RetryPolicyConfig {
            policy_id,
            strategy: RetryStrategy::default(),
            max_attempts: 3,
            max_total_time: Duration::from_secs(60),
            retry_conditions: vec![RetryCondition::default()],
            collect_metrics: true,
            attempt_timeout: Some(Duration::from_secs(30)),
        };
        Self::new(config)
    }

    /// Execute operation with retry policy
    pub async fn execute<T, F, Fut>(&self, operation: F) -> RetryResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
        T: Clone,
    {
        let context = RetryExecutionContext {
            operation_id: Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            metadata: HashMap::new(),
            custom_conditions: vec![],
        };

        self.execute_with_context(operation, context).await
    }

    /// Execute operation with custom context
    pub async fn execute_with_context<T, F, Fut>(
        &self,
        operation: F,
        context: RetryExecutionContext,
    ) -> RetryResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
        T: Clone,
    {
        let operation_start = Instant::now();
        let mut attempts = Vec::new();
        let mut attempt_number = 1;

        loop {
            // Check if we've exceeded maximum attempts
            if attempt_number > self.config.max_attempts {
                let error = RetryError::MaxAttemptsExceeded {
                    max_attempts: self.config.max_attempts,
                };
                return self.create_failed_result(error, attempts, operation_start, attempt_number - 1);
            }

            // Check if we've exceeded maximum total time
            if operation_start.elapsed() > self.config.max_total_time {
                let error = RetryError::MaxTimeExceeded {
                    max_time: self.config.max_total_time,
                };
                return self.create_failed_result(error, attempts, operation_start, attempt_number - 1);
            }

            // Calculate delay for this attempt
            let delay = if attempt_number == 1 {
                Duration::ZERO
            } else {
                self.calculate_delay(attempt_number - 1)
            };

            // Apply delay
            if delay > Duration::ZERO {
                sleep(delay).await;
            }

            // Execute attempt
            let attempt_start = Instant::now();
            let attempt_result = if let Some(timeout) = self.config.attempt_timeout {
                match tokio::time::timeout(timeout, operation()).await {
                    Ok(result) => result,
                    Err(_) => Err(format!("Attempt timeout after {:?}", timeout)),
                }
            } else {
                operation().await
            };

            let attempt_duration = attempt_start.elapsed();
            let success = attempt_result.is_ok();

            // Create attempt record
            let attempt = RetryAttempt {
                attempt_number,
                start_time: attempt_start,
                duration: attempt_duration,
                success,
                error_message: if success { None } else { Some(attempt_result.as_ref().unwrap_err().clone()) },
                delay_before: delay,
            };

            attempts.push(attempt);

            // If successful, return result
            if success {
                let result = attempt_result.unwrap();
                return self.create_successful_result(result, attempts, operation_start, attempt_number, &context);
            }

            // Check if we should retry
            let error_message = attempt_result.unwrap_err();
            if !self.should_retry(&error_message, attempt_number, &context).await {
                let error = RetryError::RetryConditionNotMet {
                    reason: format!("Error '{}' does not meet retry conditions", error_message),
                };
                return self.create_failed_result(error, attempts, operation_start, attempt_number);
            }

            attempt_number += 1;
        }
    }

    /// Calculate delay for retry attempt
    fn calculate_delay(&self, attempt_number: u32) -> Duration {
        match &self.config.strategy {
            RetryStrategy::Fixed { delay } => *delay,
            RetryStrategy::Linear { initial_delay, increment } => {
                *initial_delay + *increment * attempt_number
            },
            RetryStrategy::Exponential { initial_delay, multiplier, max_delay } => {
                let delay = Duration::from_nanos(
                    (initial_delay.as_nanos() as f64 * multiplier.powi(attempt_number as i32)) as u64
                );
                std::cmp::min(delay, *max_delay)
            },
            RetryStrategy::ExponentialJitter { initial_delay, multiplier, max_delay, jitter_factor } => {
                let base_delay = Duration::from_nanos(
                    (initial_delay.as_nanos() as f64 * multiplier.powi(attempt_number as i32)) as u64
                );
                let base_delay = std::cmp::min(base_delay, *max_delay);

                // Add jitter
                let jitter = (base_delay.as_nanos() as f64 * jitter_factor * (thread_rng().gen::<f64>() - 0.5)) as i64;
                let final_delay_nanos = (base_delay.as_nanos() as i64 + jitter).max(0) as u64;
                Duration::from_nanos(final_delay_nanos)
            },
            RetryStrategy::Custom { calculate_delay } => calculate_delay(attempt_number),
        }
    }

    /// Check if operation should be retried based on conditions
    async fn should_retry(&self, error_message: &str, attempt_number: u32, context: &RetryExecutionContext) -> bool {
        // Combine policy conditions with context conditions
        let mut all_conditions = self.config.retry_conditions.clone();
        all_conditions.extend(context.custom_conditions.clone());

        if all_conditions.is_empty() {
            return true; // Default to retry if no conditions specified
        }

        for condition in &all_conditions {
            if self.evaluate_retry_condition(condition, error_message, attempt_number).await {
                return true;
            }
        }

        false
    }

    /// Evaluate individual retry condition
    async fn evaluate_retry_condition(&self, condition: &RetryCondition, error_message: &str, _attempt_number: u32) -> bool {
        match condition {
            RetryCondition::Any => true,
            RetryCondition::ErrorMessage { patterns } => {
                patterns.iter().any(|pattern| error_message.contains(pattern))
            },
            RetryCondition::ErrorType { error_types } => {
                error_types.iter().any(|error_type| error_message.contains(error_type))
            },
            RetryCondition::ResponseTime { threshold: _ } => {
                // Would need response time information in error message
                // For now, assume we should retry timeout-related errors
                error_message.to_lowercase().contains("timeout")
            },
            RetryCondition::HttpStatus { retry_codes } => {
                // Parse HTTP status from error message
                for code in retry_codes {
                    if error_message.contains(&code.to_string()) {
                        return true;
                    }
                }
                false
            },
            RetryCondition::Custom { should_retry } => should_retry(error_message),
        }
    }

    /// Create successful retry result
    fn create_successful_result<T>(
        &self,
        result: T,
        attempts: Vec<RetryAttempt>,
        operation_start: Instant,
        final_attempt: u32,
        context: &RetryExecutionContext,
    ) -> RetryResult<T> {
        let total_duration = operation_start.elapsed();

        if self.config.collect_metrics {
            self.record_operation(true, &attempts, total_duration, None, context);
        }

        let metrics = self.get_current_metrics();

        RetryResult {
            result: Ok(result),
            attempts,
            total_duration,
            final_attempt,
            metrics,
        }
    }

    /// Create failed retry result
    fn create_failed_result<T>(
        &self,
        error: RetryError,
        attempts: Vec<RetryAttempt>,
        operation_start: Instant,
        final_attempt: u32,
    ) -> RetryResult<T> {
        let total_duration = operation_start.elapsed();

        if self.config.collect_metrics {
            let context = RetryExecutionContext {
                operation_id: Uuid::new_v4().to_string(),
                start_time: operation_start,
                metadata: HashMap::new(),
                custom_conditions: vec![],
            };
            self.record_operation(false, &attempts, total_duration, Some(error.to_string()), &context);
        }

        let metrics = self.get_current_metrics();

        RetryResult {
            result: Err(error),
            attempts,
            total_duration,
            final_attempt,
            metrics,
        }
    }

    /// Record operation for metrics
    fn record_operation(
        &self,
        success: bool,
        attempts: &[RetryAttempt],
        total_duration: Duration,
        final_error: Option<String>,
        context: &RetryExecutionContext,
    ) {
        // Record in operation history
        let record = RetryOperationRecord {
            operation_id: context.operation_id.clone(),
            start_time: context.start_time,
            total_duration,
            attempts: attempts.to_vec(),
            success,
            final_error,
        };

        self.operation_history.write().unwrap().push(record);

        // Update metrics
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_operations += 1;

        if success {
            metrics.successful_operations += 1;
        } else {
            metrics.failed_operations += 1;

            // Update error distribution
            if let Some(error) = &final_error {
                *metrics.error_distribution.entry(error.clone()).or_insert(0) += 1;
            }
        }

        metrics.total_attempts += attempts.len() as u64;

        // Recalculate derived metrics
        self.recalculate_metrics(&mut metrics);
    }

    /// Recalculate derived metrics
    fn recalculate_metrics(&self, metrics: &mut RetryMetrics) {
        if metrics.total_operations > 0 {
            metrics.success_rate = metrics.successful_operations as f64 / metrics.total_operations as f64;
            metrics.average_attempts = metrics.total_attempts as f64 / metrics.total_operations as f64;

            // Calculate average times from operation history
            let history = self.operation_history.read().unwrap();
            if !history.is_empty() {
                let total_time: Duration = history.iter().map(|r| r.total_duration).sum();
                metrics.average_total_time = total_time / history.len() as u32;

                let all_attempts: Vec<&RetryAttempt> = history.iter()
                    .flat_map(|r| &r.attempts)
                    .collect();
                if !all_attempts.is_empty() {
                    let total_attempt_time: Duration = all_attempts.iter().map(|a| a.duration).sum();
                    metrics.average_attempt_time = total_attempt_time / all_attempts.len() as u32;
                }
            }
        }
    }

    /// Get current metrics snapshot
    pub fn get_current_metrics(&self) -> RetryMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Get policy configuration
    pub fn get_config(&self) -> &RetryPolicyConfig {
        &self.config
    }

    /// Reset policy metrics and history
    pub fn reset(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_operations = 0;
        metrics.successful_operations = 0;
        metrics.failed_operations = 0;
        metrics.total_attempts = 0;
        metrics.average_attempts = 0.0;
        metrics.success_rate = 0.0;
        metrics.average_total_time = Duration::ZERO;
        metrics.average_attempt_time = Duration::ZERO;
        metrics.error_distribution.clear();

        self.operation_history.write().unwrap().clear();
    }
}

/// Retry policies manager handling multiple retry policies
#[derive(Debug)]
pub struct RetryPoliciesManager {
    /// Manager identifier
    manager_id: String,
    /// Managed retry policies
    policies: Arc<RwLock<HashMap<String, Arc<RetryPolicy>>>>,
    /// Default policy configuration
    default_config: RetryPolicyConfig,
}

impl RetryPoliciesManager {
    /// Create new retry policies manager
    pub fn new(manager_id: String, default_config: RetryPolicyConfig) -> Self {
        Self {
            manager_id,
            policies: Arc::new(RwLock::new(HashMap::new())),
            default_config,
        }
    }

    /// Create manager with default configuration
    pub fn with_defaults(manager_id: String) -> Self {
        let default_config = RetryPolicyConfig {
            policy_id: "default".to_string(),
            strategy: RetryStrategy::default(),
            max_attempts: 3,
            max_total_time: Duration::from_secs(60),
            retry_conditions: vec![RetryCondition::default()],
            collect_metrics: true,
            attempt_timeout: Some(Duration::from_secs(30)),
        };
        Self::new(manager_id, default_config)
    }

    /// Get or create retry policy
    pub async fn get_policy(&self, policy_id: &str) -> Arc<RetryPolicy> {
        {
            let policies = self.policies.read().unwrap();
            if let Some(policy) = policies.get(policy_id) {
                return policy.clone();
            }
        }

        // Create new policy
        let mut config = self.default_config.clone();
        config.policy_id = policy_id.to_string();

        let policy = Arc::new(RetryPolicy::new(config));

        let mut policies = self.policies.write().unwrap();
        policies.insert(policy_id.to_string(), policy.clone());

        policy
    }

    /// Create policy with custom configuration
    pub async fn create_policy(&self, config: RetryPolicyConfig) -> Arc<RetryPolicy> {
        let policy = Arc::new(RetryPolicy::new(config.clone()));

        let mut policies = self.policies.write().unwrap();
        policies.insert(config.policy_id.clone(), policy.clone());

        policy
    }

    /// Remove retry policy
    pub async fn remove_policy(&self, policy_id: &str) -> bool {
        let mut policies = self.policies.write().unwrap();
        policies.remove(policy_id).is_some()
    }

    /// Get all policy metrics
    pub async fn get_all_metrics(&self) -> HashMap<String, RetryMetrics> {
        let mut all_metrics = HashMap::new();
        let policies = self.policies.read().unwrap();

        for (id, policy) in policies.iter() {
            let metrics = policy.get_current_metrics();
            all_metrics.insert(id.clone(), metrics);
        }

        all_metrics
    }

    /// Reset all policies
    pub async fn reset_all(&self) {
        let policies = self.policies.read().unwrap();
        for policy in policies.values() {
            policy.reset();
        }
    }

    /// Get manager health status
    pub async fn get_health_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("manager_id".to_string(), self.manager_id.clone());

        let policies = self.policies.read().unwrap();
        status.insert("total_policies".to_string(), policies.len().to_string());

        let mut total_operations = 0u64;
        let mut total_successful = 0u64;
        let mut total_failed = 0u64;

        for policy in policies.values() {
            let metrics = policy.get_current_metrics();
            total_operations += metrics.total_operations;
            total_successful += metrics.successful_operations;
            total_failed += metrics.failed_operations;
        }

        status.insert("total_operations".to_string(), total_operations.to_string());
        status.insert("total_successful".to_string(), total_successful.to_string());
        status.insert("total_failed".to_string(), total_failed.to_string());

        if total_operations > 0 {
            let overall_success_rate = total_successful as f64 / total_operations as f64;
            status.insert("overall_success_rate".to_string(), format!("{:.2}", overall_success_rate));
        }

        status
    }
}

// Temporary rand implementation for jitter calculation
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};

    static RNG_STATE: AtomicU64 = AtomicU64::new(1);

    pub fn random<T>() -> T
    where
        T: From<f64>,
    {
        // Simple linear congruential generator
        let prev = RNG_STATE.load(Ordering::Relaxed);
        let next = prev.wrapping_mul(1103515245).wrapping_add(12345);
        RNG_STATE.store(next, Ordering::Relaxed);

        let normalized = (next as f64) / (u64::MAX as f64);
        T::from(normalized)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_retry_policy_basic_functionality() {
        let config = RetryPolicyConfig {
            policy_id: "test".to_string(),
            strategy: RetryStrategy::Fixed { delay: Duration::from_millis(10) },
            max_attempts: 3,
            max_total_time: Duration::from_secs(10),
            retry_conditions: vec![RetryCondition::Any],
            collect_metrics: true,
            attempt_timeout: None,
        };

        let policy = RetryPolicy::new(config);

        // Test successful operation
        let result = policy.execute(|| async { Ok::<_, String>("success") }).await;
        assert!(result.result.is_ok());
        assert_eq!(result.final_attempt, 1);
        assert_eq!(result.attempts.len(), 1);
    }

    #[tokio::test]
    async fn test_retry_with_failures() {
        let config = RetryPolicyConfig {
            policy_id: "test".to_string(),
            strategy: RetryStrategy::Fixed { delay: Duration::from_millis(1) },
            max_attempts: 3,
            max_total_time: Duration::from_secs(10),
            retry_conditions: vec![RetryCondition::Any],
            collect_metrics: true,
            attempt_timeout: None,
        };

        let policy = RetryPolicy::new(config);
        let counter = Arc::new(AtomicU32::new(0));

        // Fail first two attempts, succeed on third
        let counter_clone = counter.clone();
        let result = policy.execute(move || {
            let counter = counter_clone.clone();
            async move {
                let attempt = counter.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt < 3 {
                    Err(format!("Attempt {} failed", attempt))
                } else {
                    Ok("success")
                }
            }
        }).await;

        assert!(result.result.is_ok());
        assert_eq!(result.final_attempt, 3);
        assert_eq!(result.attempts.len(), 3);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_max_attempts_exceeded() {
        let config = RetryPolicyConfig {
            policy_id: "test".to_string(),
            strategy: RetryStrategy::Fixed { delay: Duration::from_millis(1) },
            max_attempts: 2,
            max_total_time: Duration::from_secs(10),
            retry_conditions: vec![RetryCondition::Any],
            collect_metrics: true,
            attempt_timeout: None,
        };

        let policy = RetryPolicy::new(config);

        // Always fail
        let result = policy.execute(|| async { Err::<(), String>("always fail".to_string()) }).await;

        assert!(result.result.is_err());
        match result.result.unwrap_err() {
            RetryError::MaxAttemptsExceeded { max_attempts } => assert_eq!(max_attempts, 2),
            _ => panic!("Expected MaxAttemptsExceeded error"),
        }
        assert_eq!(result.attempts.len(), 2);
    }

    #[tokio::test]
    async fn test_exponential_backoff() {
        let config = RetryPolicyConfig {
            policy_id: "test".to_string(),
            strategy: RetryStrategy::Exponential {
                initial_delay: Duration::from_millis(10),
                multiplier: 2.0,
                max_delay: Duration::from_millis(100),
            },
            max_attempts: 4,
            max_total_time: Duration::from_secs(10),
            retry_conditions: vec![RetryCondition::Any],
            collect_metrics: true,
            attempt_timeout: None,
        };

        let policy = RetryPolicy::new(config);
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        let result = policy.execute(move || {
            let counter = counter_clone.clone();
            async move {
                let attempt = counter.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt < 4 {
                    Err(format!("Attempt {} failed", attempt))
                } else {
                    Ok("success")
                }
            }
        }).await;

        assert!(result.result.is_ok());
        assert_eq!(result.attempts.len(), 4);

        // Check that delays are increasing (exponential backoff)
        let delays: Vec<Duration> = result.attempts.iter().map(|a| a.delay_before).collect();
        assert_eq!(delays[0], Duration::ZERO); // First attempt has no delay
        assert!(delays[1] > Duration::ZERO);
        assert!(delays[2] > delays[1]);
        assert!(delays[3] > delays[2]);
    }

    #[tokio::test]
    async fn test_retry_condition_evaluation() {
        let config = RetryPolicyConfig {
            policy_id: "test".to_string(),
            strategy: RetryStrategy::Fixed { delay: Duration::from_millis(1) },
            max_attempts: 3,
            max_total_time: Duration::from_secs(10),
            retry_conditions: vec![RetryCondition::ErrorMessage {
                patterns: vec!["retryable".to_string()],
            }],
            collect_metrics: true,
            attempt_timeout: None,
        };

        let policy = RetryPolicy::new(config);

        // Should not retry on non-retryable error
        let result = policy.execute(|| async { Err::<(), String>("non-retryable error".to_string()) }).await;
        assert!(result.result.is_err());
        assert_eq!(result.attempts.len(), 1); // Only one attempt

        // Should retry on retryable error
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        let result = policy.execute(move || {
            let counter = counter_clone.clone();
            async move {
                let attempt = counter.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt < 2 {
                    Err("retryable error".to_string())
                } else {
                    Ok("success")
                }
            }
        }).await;

        assert!(result.result.is_ok());
        assert_eq!(result.attempts.len(), 2);
    }

    #[tokio::test]
    async fn test_retry_metrics() {
        let policy = RetryPolicy::with_defaults("test_metrics".to_string());

        // Execute several operations
        let _ = policy.execute(|| async { Ok::<_, String>("success") }).await;
        let _ = policy.execute(|| async { Err::<(), String>("failure".to_string()) }).await;
        let _ = policy.execute(|| async { Ok::<_, String>("success") }).await;

        let metrics = policy.get_current_metrics();
        assert_eq!(metrics.total_operations, 3);
        assert_eq!(metrics.successful_operations, 2);
        assert_eq!(metrics.failed_operations, 1);
        assert!((metrics.success_rate - 0.67).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_retry_policies_manager() {
        let manager = RetryPoliciesManager::with_defaults("test_manager".to_string());

        let policy1 = manager.get_policy("policy1").await;
        let policy2 = manager.get_policy("policy2").await;

        // Execute operations on different policies
        let _ = policy1.execute(|| async { Ok::<_, String>("success") }).await;
        let _ = policy2.execute(|| async { Err::<(), String>("failure".to_string()) }).await;

        let all_metrics = manager.get_all_metrics().await;
        assert_eq!(all_metrics.len(), 2);
        assert!(all_metrics.contains_key("policy1"));
        assert!(all_metrics.contains_key("policy2"));

        let health = manager.get_health_status().await;
        assert_eq!(health.get("total_policies").unwrap(), "2");
        assert_eq!(health.get("total_operations").unwrap(), "2");
    }

    #[tokio::test]
    async fn test_concurrent_retries() {
        let policy = Arc::new(RetryPolicy::with_defaults("concurrent_test".to_string()));
        let mut handles = vec![];

        // Spawn multiple concurrent operations
        for i in 0..5 {
            let policy_clone = policy.clone();
            let handle = tokio::spawn(async move {
                let result = policy_clone.execute(|| async {
                    // Some operations succeed, some fail
                    if i % 2 == 0 {
                        Ok::<_, String>(format!("success_{}", i))
                    } else {
                        Err(format!("failure_{}", i))
                    }
                }).await;
                result.result.is_ok()
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let results: Vec<bool> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Should have mix of successes and failures
        let successes = results.iter().filter(|&&x| x).count();
        let failures = results.iter().filter(|&&x| !x).count();

        assert_eq!(successes + failures, 5);
        assert!(successes > 0);
        assert!(failures > 0);

        let metrics = policy.get_current_metrics();
        assert_eq!(metrics.total_operations, 5);
    }
}