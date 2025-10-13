//! Circuit Breaker Module
//!
//! Implements sophisticated circuit breaker patterns for fault tolerance, providing:
//! - State-based circuit breaking (Closed, Open, Half-Open)
//! - Adaptive failure detection algorithms
//! - Threshold management and configuration
//! - Performance monitoring and metrics
//! - Integration with fault tolerance coordination systems

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::time::sleep;
use uuid::Uuid;

/// Circuit breaker states defining the operational mode
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed - requests flow normally
    Closed,
    /// Circuit is open - requests are blocked
    Open,
    /// Circuit is half-open - testing if service has recovered
    HalfOpen,
}

/// Circuit breaker failure detection strategy
#[derive(Debug, Clone)]
pub enum FailureDetectionStrategy {
    /// Failure count based detection
    FailureCount { threshold: u32 },
    /// Failure rate based detection
    FailureRate { threshold: f64, window_size: Duration },
    /// Response time based detection
    ResponseTime { threshold: Duration },
    /// Hybrid detection combining multiple strategies
    Hybrid {
        failure_count_threshold: u32,
        failure_rate_threshold: f64,
        response_time_threshold: Duration,
        window_size: Duration,
    },
}

/// Circuit breaker recovery strategy
#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    /// Timeout before transitioning from Open to Half-Open
    pub timeout: Duration,
    /// Number of successful requests required to close circuit
    pub success_threshold: u32,
    /// Maximum number of requests allowed in Half-Open state
    pub half_open_max_requests: u32,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Unique circuit breaker identifier
    pub circuit_id: String,
    /// Failure detection strategy
    pub detection_strategy: FailureDetectionStrategy,
    /// Recovery strategy configuration
    pub recovery_strategy: RecoveryStrategy,
    /// Whether to enable automatic recovery
    pub auto_recovery: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
}

/// Circuit breaker execution result
#[derive(Debug, Clone)]
pub struct CircuitBreakerResult<T> {
    /// Operation result
    pub result: Result<T, CircuitBreakerError>,
    /// Execution duration
    pub duration: Duration,
    /// Circuit state after execution
    pub circuit_state: CircuitBreakerState,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

/// Circuit breaker specific errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open - requests blocked")]
    CircuitOpen,
    #[error("Half-open circuit limit exceeded")]
    HalfOpenLimitExceeded,
    #[error("Operation failed: {message}")]
    OperationFailed { message: String },
    #[error("Timeout exceeded: {timeout:?}")]
    TimeoutExceeded { timeout: Duration },
    #[error("Circuit breaker configuration error: {message}")]
    ConfigurationError { message: String },
}

/// Circuit breaker metrics and monitoring data
#[derive(Debug, Clone)]
pub struct CircuitBreakerMetrics {
    /// Circuit identifier
    pub circuit_id: String,
    /// Current circuit state
    pub current_state: CircuitBreakerState,
    /// Total number of requests
    pub total_requests: u64,
    /// Number of successful requests
    pub successful_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Number of rejected requests (circuit open)
    pub rejected_requests: u64,
    /// Current failure rate
    pub failure_rate: f64,
    /// Average response time
    pub average_response_time: Duration,
    /// Time since last state change
    pub time_in_current_state: Duration,
    /// Last state change timestamp
    pub last_state_change: Instant,
    /// Success streak in half-open state
    pub half_open_success_streak: u32,
}

/// Request execution context for circuit breaker operations
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Request identifier
    pub request_id: String,
    /// Request timestamp
    pub timestamp: Instant,
    /// Request timeout
    pub timeout: Option<Duration>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Circuit breaker implementation providing comprehensive fault tolerance
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Circuit breaker identifier
    circuit_id: String,
    /// Current circuit state
    state: Arc<RwLock<CircuitBreakerState>>,
    /// Circuit breaker configuration
    config: CircuitBreakerConfig,
    /// Execution metrics
    metrics: Arc<RwLock<CircuitBreakerMetrics>>,
    /// Request history for failure detection
    request_history: Arc<RwLock<Vec<RequestRecord>>>,
    /// Last state change timestamp
    last_state_change: Arc<RwLock<Instant>>,
    /// Current half-open request count
    half_open_requests: Arc<RwLock<u32>>,
}

/// Individual request record for metrics and analysis
#[derive(Debug, Clone)]
struct RequestRecord {
    /// Request timestamp
    timestamp: Instant,
    /// Request success status
    success: bool,
    /// Request duration
    duration: Duration,
    /// Request context
    context: ExecutionContext,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            success_threshold: 3,
            half_open_max_requests: 5,
        }
    }
}

impl Default for FailureDetectionStrategy {
    fn default() -> Self {
        Self::Hybrid {
            failure_count_threshold: 5,
            failure_rate_threshold: 0.5,
            response_time_threshold: Duration::from_secs(5),
            window_size: Duration::from_secs(60),
        }
    }
}

impl CircuitBreaker {
    /// Create new circuit breaker with configuration
    pub fn new(config: CircuitBreakerConfig) -> Self {
        let circuit_id = config.circuit_id.clone();
        let now = Instant::now();

        let metrics = CircuitBreakerMetrics {
            circuit_id: circuit_id.clone(),
            current_state: CircuitBreakerState::Closed,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            rejected_requests: 0,
            failure_rate: 0.0,
            average_response_time: Duration::ZERO,
            time_in_current_state: Duration::ZERO,
            last_state_change: now,
            half_open_success_streak: 0,
        };

        Self {
            circuit_id,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            config,
            metrics: Arc::new(RwLock::new(metrics)),
            request_history: Arc::new(RwLock::new(Vec::new())),
            last_state_change: Arc::new(RwLock::new(now)),
            half_open_requests: Arc::new(RwLock::new(0)),
        }
    }

    /// Create circuit breaker with default configuration
    pub fn with_defaults(circuit_id: String) -> Self {
        let config = CircuitBreakerConfig {
            circuit_id,
            detection_strategy: FailureDetectionStrategy::default(),
            recovery_strategy: RecoveryStrategy::default(),
            auto_recovery: true,
            metrics_interval: Duration::from_secs(10),
        };
        Self::new(config)
    }

    /// Execute operation with circuit breaker protection
    pub async fn execute<T, F, Fut>(&self, operation: F) -> CircuitBreakerResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
    {
        let context = ExecutionContext {
            request_id: Uuid::new_v4().to_string(),
            timestamp: Instant::now(),
            timeout: None,
            metadata: HashMap::new(),
        };

        self.execute_with_context(operation, context).await
    }

    /// Execute operation with custom context
    pub async fn execute_with_context<T, F, Fut>(
        &self,
        operation: F,
        context: ExecutionContext,
    ) -> CircuitBreakerResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
    {
        let start_time = Instant::now();

        // Check if request can proceed
        match self.can_proceed().await {
            Ok(()) => {},
            Err(error) => {
                self.record_rejection().await;
                return CircuitBreakerResult {
                    result: Err(error),
                    duration: start_time.elapsed(),
                    circuit_state: self.get_current_state().await,
                    metadata: context.metadata,
                };
            }
        }

        // Execute operation with timeout
        let result = if let Some(timeout) = context.timeout {
            match tokio::time::timeout(timeout, operation()).await {
                Ok(result) => result,
                Err(_) => Err(format!("Operation timed out after {:?}", timeout)),
            }
        } else {
            operation().await
        };

        let duration = start_time.elapsed();
        let success = result.is_ok();

        // Record request and update circuit state
        self.record_request(success, duration, context.clone()).await;
        self.update_circuit_state().await;

        let circuit_state = self.get_current_state().await;

        CircuitBreakerResult {
            result: result.map_err(|msg| CircuitBreakerError::OperationFailed { message: msg }),
            duration,
            circuit_state,
            metadata: context.metadata,
        }
    }

    /// Check if request can proceed based on current circuit state
    async fn can_proceed(&self) -> Result<(), CircuitBreakerError> {
        let state = self.get_current_state().await;

        match state {
            CircuitBreakerState::Closed => Ok(()),
            CircuitBreakerState::Open => {
                // Check if enough time has passed to transition to half-open
                if self.should_attempt_reset().await {
                    self.transition_to_half_open().await;
                    Ok(())
                } else {
                    Err(CircuitBreakerError::CircuitOpen)
                }
            },
            CircuitBreakerState::HalfOpen => {
                let current_requests = *self.half_open_requests.read().unwrap();
                if current_requests >= self.config.recovery_strategy.half_open_max_requests {
                    Err(CircuitBreakerError::HalfOpenLimitExceeded)
                } else {
                    *self.half_open_requests.write().unwrap() += 1;
                    Ok(())
                }
            }
        }
    }

    /// Record request execution and outcome
    async fn record_request(&self, success: bool, duration: Duration, context: ExecutionContext) {
        let record = RequestRecord {
            timestamp: context.timestamp,
            success,
            duration,
            context,
        };

        // Add to request history
        {
            let mut history = self.request_history.write().unwrap();
            history.push(record);

            // Clean old records based on detection strategy window
            if let FailureDetectionStrategy::FailureRate { window_size, .. } = &self.config.detection_strategy {
                let cutoff = Instant::now() - *window_size;
                history.retain(|r| r.timestamp > cutoff);
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_requests += 1;

            if success {
                metrics.successful_requests += 1;
                if self.get_current_state().await == CircuitBreakerState::HalfOpen {
                    metrics.half_open_success_streak += 1;
                }
            } else {
                metrics.failed_requests += 1;
                metrics.half_open_success_streak = 0;
            }

            // Update failure rate and average response time
            self.calculate_metrics().await;
        }
    }

    /// Record rejected request (circuit open)
    async fn record_rejection(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.rejected_requests += 1;
        metrics.total_requests += 1;
    }

    /// Calculate current metrics from request history
    async fn calculate_metrics(&self) {
        let history = self.request_history.read().unwrap();
        let mut metrics = self.metrics.write().unwrap();

        if !history.is_empty() {
            let total_requests = history.len() as f64;
            let failed_requests = history.iter().filter(|r| !r.success).count() as f64;

            metrics.failure_rate = failed_requests / total_requests;

            let total_duration: Duration = history.iter().map(|r| r.duration).sum();
            metrics.average_response_time = total_duration / history.len() as u32;
        }

        let now = Instant::now();
        let last_change = *self.last_state_change.read().unwrap();
        metrics.time_in_current_state = now - last_change;
        metrics.last_state_change = last_change;
    }

    /// Update circuit state based on current metrics and detection strategy
    async fn update_circuit_state(&self) {
        let current_state = self.get_current_state().await;
        let should_open = self.should_open_circuit().await;
        let should_close = self.should_close_circuit().await;

        match current_state {
            CircuitBreakerState::Closed => {
                if should_open {
                    self.transition_to_open().await;
                }
            },
            CircuitBreakerState::HalfOpen => {
                if should_close {
                    self.transition_to_closed().await;
                } else if should_open {
                    self.transition_to_open().await;
                }
            },
            CircuitBreakerState::Open => {
                // Handled in can_proceed method
            }
        }
    }

    /// Check if circuit should be opened based on failure detection strategy
    async fn should_open_circuit(&self) -> bool {
        match &self.config.detection_strategy {
            FailureDetectionStrategy::FailureCount { threshold } => {
                let metrics = self.metrics.read().unwrap();
                metrics.failed_requests >= *threshold as u64
            },
            FailureDetectionStrategy::FailureRate { threshold, .. } => {
                let metrics = self.metrics.read().unwrap();
                metrics.failure_rate >= *threshold
            },
            FailureDetectionStrategy::ResponseTime { threshold } => {
                let metrics = self.metrics.read().unwrap();
                metrics.average_response_time >= *threshold
            },
            FailureDetectionStrategy::Hybrid {
                failure_count_threshold,
                failure_rate_threshold,
                response_time_threshold,
                ..
            } => {
                let metrics = self.metrics.read().unwrap();
                metrics.failed_requests >= *failure_count_threshold as u64
                    || metrics.failure_rate >= *failure_rate_threshold
                    || metrics.average_response_time >= *response_time_threshold
            }
        }
    }

    /// Check if circuit should be closed (from half-open state)
    async fn should_close_circuit(&self) -> bool {
        if self.get_current_state().await != CircuitBreakerState::HalfOpen {
            return false;
        }

        let metrics = self.metrics.read().unwrap();
        metrics.half_open_success_streak >= self.config.recovery_strategy.success_threshold
    }

    /// Check if circuit should attempt reset (transition to half-open)
    async fn should_attempt_reset(&self) -> bool {
        if !self.config.auto_recovery {
            return false;
        }

        let last_change = *self.last_state_change.read().unwrap();
        let elapsed = Instant::now() - last_change;
        elapsed >= self.config.recovery_strategy.timeout
    }

    /// Transition circuit to Open state
    async fn transition_to_open(&self) {
        *self.state.write().unwrap() = CircuitBreakerState::Open;
        *self.last_state_change.write().unwrap() = Instant::now();
        *self.half_open_requests.write().unwrap() = 0;

        let mut metrics = self.metrics.write().unwrap();
        metrics.current_state = CircuitBreakerState::Open;
        metrics.half_open_success_streak = 0;
    }

    /// Transition circuit to Closed state
    async fn transition_to_closed(&self) {
        *self.state.write().unwrap() = CircuitBreakerState::Closed;
        *self.last_state_change.write().unwrap() = Instant::now();
        *self.half_open_requests.write().unwrap() = 0;

        let mut metrics = self.metrics.write().unwrap();
        metrics.current_state = CircuitBreakerState::Closed;
        metrics.half_open_success_streak = 0;

        // Reset failure metrics when closing
        metrics.failed_requests = 0;
        metrics.failure_rate = 0.0;
    }

    /// Transition circuit to Half-Open state
    async fn transition_to_half_open(&self) {
        *self.state.write().unwrap() = CircuitBreakerState::HalfOpen;
        *self.last_state_change.write().unwrap() = Instant::now();
        *self.half_open_requests.write().unwrap() = 0;

        let mut metrics = self.metrics.write().unwrap();
        metrics.current_state = CircuitBreakerState::HalfOpen;
        metrics.half_open_success_streak = 0;
    }

    /// Get current circuit state
    pub async fn get_current_state(&self) -> CircuitBreakerState {
        self.state.read().unwrap().clone()
    }

    /// Get current circuit breaker metrics
    pub async fn get_metrics(&self) -> CircuitBreakerMetrics {
        self.calculate_metrics().await;
        self.metrics.read().unwrap().clone()
    }

    /// Get circuit breaker configuration
    pub fn get_config(&self) -> &CircuitBreakerConfig {
        &self.config
    }

    /// Reset circuit breaker to closed state
    pub async fn reset(&self) {
        self.transition_to_closed().await;

        // Clear request history
        self.request_history.write().unwrap().clear();

        // Reset metrics
        let now = Instant::now();
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_requests = 0;
        metrics.successful_requests = 0;
        metrics.failed_requests = 0;
        metrics.rejected_requests = 0;
        metrics.failure_rate = 0.0;
        metrics.average_response_time = Duration::ZERO;
        metrics.time_in_current_state = Duration::ZERO;
        metrics.last_state_change = now;
        metrics.half_open_success_streak = 0;
    }

    /// Force circuit to open state (for testing/maintenance)
    pub async fn force_open(&self) {
        self.transition_to_open().await;
    }

    /// Force circuit to closed state (override automatic behavior)
    pub async fn force_closed(&self) {
        self.transition_to_closed().await;
    }
}

/// Circuit breaker system managing multiple circuit breakers
#[derive(Debug)]
pub struct CircuitBreakerSystem {
    /// System identifier
    system_id: String,
    /// Managed circuit breakers
    circuit_breakers: Arc<RwLock<HashMap<String, Arc<CircuitBreaker>>>>,
    /// System-wide configuration
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerSystem {
    /// Create new circuit breaker system
    pub fn new(system_id: String, default_config: CircuitBreakerConfig) -> Self {
        Self {
            system_id,
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            default_config,
        }
    }

    /// Create system with default configuration
    pub fn with_defaults(system_id: String) -> Self {
        let default_config = CircuitBreakerConfig {
            circuit_id: "default".to_string(),
            detection_strategy: FailureDetectionStrategy::default(),
            recovery_strategy: RecoveryStrategy::default(),
            auto_recovery: true,
            metrics_interval: Duration::from_secs(10),
        };
        Self::new(system_id, default_config)
    }

    /// Get or create circuit breaker
    pub async fn get_circuit_breaker(&self, circuit_id: &str) -> Arc<CircuitBreaker> {
        {
            let breakers = self.circuit_breakers.read().unwrap();
            if let Some(breaker) = breakers.get(circuit_id) {
                return breaker.clone();
            }
        }

        // Create new circuit breaker
        let mut config = self.default_config.clone();
        config.circuit_id = circuit_id.to_string();

        let breaker = Arc::new(CircuitBreaker::new(config));

        let mut breakers = self.circuit_breakers.write().unwrap();
        breakers.insert(circuit_id.to_string(), breaker.clone());

        breaker
    }

    /// Remove circuit breaker
    pub async fn remove_circuit_breaker(&self, circuit_id: &str) -> bool {
        let mut breakers = self.circuit_breakers.write().unwrap();
        breakers.remove(circuit_id).is_some()
    }

    /// Get all circuit breaker metrics
    pub async fn get_all_metrics(&self) -> HashMap<String, CircuitBreakerMetrics> {
        let mut all_metrics = HashMap::new();
        let breakers = self.circuit_breakers.read().unwrap();

        for (id, breaker) in breakers.iter() {
            let metrics = breaker.get_metrics().await;
            all_metrics.insert(id.clone(), metrics);
        }

        all_metrics
    }

    /// Reset all circuit breakers
    pub async fn reset_all(&self) {
        let breakers = self.circuit_breakers.read().unwrap();
        for breaker in breakers.values() {
            breaker.reset().await;
        }
    }

    /// Get system health status
    pub async fn get_health_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("system_id".to_string(), self.system_id.clone());

        let breakers = self.circuit_breakers.read().unwrap();
        status.insert("total_circuits".to_string(), breakers.len().to_string());

        let mut open_count = 0;
        let mut half_open_count = 0;
        let mut closed_count = 0;

        for breaker in breakers.values() {
            match breaker.get_current_state().await {
                CircuitBreakerState::Open => open_count += 1,
                CircuitBreakerState::HalfOpen => half_open_count += 1,
                CircuitBreakerState::Closed => closed_count += 1,
            }
        }

        status.insert("open_circuits".to_string(), open_count.to_string());
        status.insert("half_open_circuits".to_string(), half_open_count.to_string());
        status.insert("closed_circuits".to_string(), closed_count.to_string());

        status
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_circuit_breaker_basic_functionality() {
        let config = CircuitBreakerConfig {
            circuit_id: "test".to_string(),
            detection_strategy: FailureDetectionStrategy::FailureCount { threshold: 3 },
            recovery_strategy: RecoveryStrategy::default(),
            auto_recovery: true,
            metrics_interval: Duration::from_millis(100),
        };

        let circuit = CircuitBreaker::new(config);

        // Test successful operation
        let result = circuit.execute(|| async { Ok::<_, String>("success") }).await;
        assert!(result.result.is_ok());
        assert_eq!(result.circuit_state, CircuitBreakerState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_failure_detection() {
        let config = CircuitBreakerConfig {
            circuit_id: "test".to_string(),
            detection_strategy: FailureDetectionStrategy::FailureCount { threshold: 2 },
            recovery_strategy: RecoveryStrategy::default(),
            auto_recovery: true,
            metrics_interval: Duration::from_millis(100),
        };

        let circuit = CircuitBreaker::new(config);

        // Trigger failures
        for _ in 0..3 {
            let _ = circuit.execute(|| async { Err::<(), String>("failure".to_string()) }).await;
        }

        // Circuit should be open
        assert_eq!(circuit.get_current_state().await, CircuitBreakerState::Open);

        // Next request should be rejected
        let result = circuit.execute(|| async { Ok::<_, String>("should be blocked") }).await;
        assert!(matches!(result.result, Err(CircuitBreakerError::CircuitOpen)));
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let config = CircuitBreakerConfig {
            circuit_id: "test".to_string(),
            detection_strategy: FailureDetectionStrategy::FailureCount { threshold: 1 },
            recovery_strategy: RecoveryStrategy {
                timeout: Duration::from_millis(100),
                success_threshold: 2,
                half_open_max_requests: 5,
            },
            auto_recovery: true,
            metrics_interval: Duration::from_millis(50),
        };

        let circuit = CircuitBreaker::new(config);

        // Trigger failure to open circuit
        let _ = circuit.execute(|| async { Err::<(), String>("failure".to_string()) }).await;
        assert_eq!(circuit.get_current_state().await, CircuitBreakerState::Open);

        // Wait for recovery timeout
        sleep(Duration::from_millis(150)).await;

        // Should transition to half-open and allow requests
        let result = circuit.execute(|| async { Ok::<_, String>("success") }).await;
        assert!(result.result.is_ok());
        assert_eq!(result.circuit_state, CircuitBreakerState::HalfOpen);

        // Another success should close the circuit
        let result = circuit.execute(|| async { Ok::<_, String>("success") }).await;
        assert!(result.result.is_ok());
        assert_eq!(result.circuit_state, CircuitBreakerState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_metrics() {
        let circuit = CircuitBreaker::with_defaults("test".to_string());

        // Execute some operations
        let _ = circuit.execute(|| async { Ok::<_, String>("success") }).await;
        let _ = circuit.execute(|| async { Err::<(), String>("failure".to_string()) }).await;
        let _ = circuit.execute(|| async { Ok::<_, String>("success") }).await;

        let metrics = circuit.get_metrics().await;
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
        assert!((metrics.failure_rate - 0.33).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_circuit_breaker_system() {
        let system = CircuitBreakerSystem::with_defaults("test_system".to_string());

        let circuit1 = system.get_circuit_breaker("circuit1").await;
        let circuit2 = system.get_circuit_breaker("circuit2").await;

        // Execute operations on different circuits
        let _ = circuit1.execute(|| async { Ok::<_, String>("success") }).await;
        let _ = circuit2.execute(|| async { Err::<(), String>("failure".to_string()) }).await;

        let all_metrics = system.get_all_metrics().await;
        assert_eq!(all_metrics.len(), 2);
        assert!(all_metrics.contains_key("circuit1"));
        assert!(all_metrics.contains_key("circuit2"));

        let health = system.get_health_status().await;
        assert_eq!(health.get("total_circuits").unwrap(), "2");
    }

    #[tokio::test]
    async fn test_concurrent_circuit_breaker() {
        let circuit = Arc::new(CircuitBreaker::with_defaults("concurrent_test".to_string()));
        let counter = Arc::new(AtomicU32::new(0));

        let mut handles = vec![];

        // Spawn multiple concurrent operations
        for _ in 0..10 {
            let circuit_clone = circuit.clone();
            let counter_clone = counter.clone();

            let handle = tokio::spawn(async move {
                let result = circuit_clone.execute(|| async {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, String>("success")
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

        // All operations should succeed
        assert_eq!(results.iter().filter(|&&x| x).count(), 10);
        assert_eq!(counter.load(Ordering::SeqCst), 10);

        let metrics = circuit.get_metrics().await;
        assert_eq!(metrics.successful_requests, 10);
    }
}