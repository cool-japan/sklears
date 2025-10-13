//! Circuit Breaker Statistics Tracking
//!
//! This module provides comprehensive statistics tracking for circuit breakers,
//! including request counters, response time tracking, error tracking, state
//! transition monitoring, health metrics, and global statistics aggregation.

use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, RwLock,
};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

use crate::fault_core::{CircuitBreakerState, CircuitBreakerStats, FaultSeverity};

use super::error_types::CircuitBreakerError;

/// Circuit breaker statistics tracker
#[derive(Debug)]
pub struct CircuitBreakerStatsTracker {
    /// Request counters
    request_counters: Arc<Mutex<RequestCounters>>,
    /// Response time tracker
    response_times: Arc<Mutex<ResponseTimeTracker>>,
    /// Error tracker
    error_tracker: Arc<Mutex<ErrorTracker>>,
    /// State transition tracker
    state_transitions: Arc<Mutex<StateTransitionTracker>>,
    /// Health metrics
    health_metrics: Arc<Mutex<HealthMetrics>>,
}

/// Request counters
#[derive(Debug, Default)]
pub struct RequestCounters {
    /// Total requests
    pub total: AtomicU64,
    /// Successful requests
    pub successful: AtomicU64,
    /// Failed requests
    pub failed: AtomicU64,
    /// Rejected requests (circuit open)
    pub rejected: AtomicU64,
    /// Timeout requests
    pub timeout: AtomicU64,
    /// Requests in progress
    pub in_progress: AtomicU64,
}

/// Response time tracker
#[derive(Debug)]
pub struct ResponseTimeTracker {
    /// Response time history (sliding window)
    pub history: VecDeque<Duration>,
    /// Window size
    pub window_size: usize,
    /// Sum of response times in window
    pub sum: Duration,
    /// Minimum response time
    pub min: Option<Duration>,
    /// Maximum response time
    pub max: Option<Duration>,
    /// Percentiles cache
    pub percentiles: HashMap<u8, Duration>,
}

/// Error tracker
#[derive(Debug, Default)]
pub struct ErrorTracker {
    /// Error counts by type
    pub error_counts: HashMap<String, u64>,
    /// Error patterns
    pub error_patterns: Vec<ErrorPattern>,
    /// Recent errors (sliding window)
    pub recent_errors: VecDeque<ErrorEvent>,
    /// Error rate calculation
    pub error_rate: f64,
}

/// Error pattern
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern name
    pub name: String,
    /// Pattern regex
    pub pattern: String,
    /// Match count
    pub count: u64,
    /// First occurrence
    pub first_seen: SystemTime,
    /// Last occurrence
    pub last_seen: SystemTime,
    /// Pattern severity
    pub severity: FaultSeverity,
}

/// Error event
#[derive(Debug, Clone)]
pub struct ErrorEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Error severity
    pub severity: FaultSeverity,
    /// Request context
    pub context: RequestContext,
}

/// Request context
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Request identifier
    pub request_id: String,
    /// User identifier
    pub user_id: Option<String>,
    /// Session identifier
    pub session_id: Option<String>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// State transition tracker
#[derive(Debug)]
pub struct StateTransitionTracker {
    /// Transition history
    pub transitions: VecDeque<StateTransition>,
    /// Current state start time
    pub current_state_start: SystemTime,
    /// State durations
    pub state_durations: HashMap<CircuitBreakerState, Duration>,
    /// Transition counts
    pub transition_counts: HashMap<(CircuitBreakerState, CircuitBreakerState), u64>,
}

/// State transition
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Transition timestamp
    pub timestamp: SystemTime,
    /// Previous state
    pub from_state: CircuitBreakerState,
    /// New state
    pub to_state: CircuitBreakerState,
    /// Transition reason
    pub reason: TransitionReason,
    /// Transition metadata
    pub metadata: HashMap<String, String>,
}

/// Transition reason enumeration
#[derive(Debug, Clone)]
pub enum TransitionReason {
    /// Failure threshold exceeded
    FailureThresholdExceeded,
    /// Recovery timeout elapsed
    RecoveryTimeoutElapsed,
    /// Successful recovery test
    SuccessfulRecoveryTest,
    /// Failed recovery test
    FailedRecoveryTest,
    /// Manual override
    ManualOverride,
    /// Health check failure
    HealthCheckFailure,
    /// Policy violation
    PolicyViolation,
    /// Custom reason
    Custom(String),
}

/// Health metrics
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,
    /// Availability (0.0 to 1.0)
    pub availability: f64,
    /// Reliability (0.0 to 1.0)
    pub reliability: f64,
    /// Performance score (0.0 to 1.0)
    pub performance_score: f64,
    /// Last health check
    pub last_health_check: SystemTime,
    /// Health trend
    pub health_trend: HealthTrend,
}

/// Health trend enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum HealthTrend {
    /// Improving
    Improving,
    /// Stable
    Stable,
    /// Degrading
    Degrading,
    /// Critical
    Critical,
    /// Unknown
    Unknown,
}

/// Circuit breaker statistics aggregator for global statistics
#[derive(Debug)]
pub struct CircuitBreakerStatsAggregator {
    /// Aggregated statistics
    stats: Arc<RwLock<AggregatedStats>>,
    /// Aggregation configuration
    config: AggregationConfig,
}

/// Aggregated statistics structure
#[derive(Debug, Default)]
pub struct AggregatedStats {
    /// Total circuit breakers
    pub total_breakers: u64,
    /// Active circuit breakers
    pub active_breakers: u64,
    /// Open circuit breakers
    pub open_breakers: u64,
    /// Half-open circuit breakers
    pub half_open_breakers: u64,
    /// Global request statistics
    pub global_requests: RequestCounters,
    /// Global health score
    pub global_health: f64,
}

/// Aggregation configuration
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Aggregation interval
    pub interval: Duration,
    /// Enable real-time aggregation
    pub real_time: bool,
    /// Aggregation algorithms
    pub algorithms: Vec<String>,
}

/// Request result enumeration for statistics tracking
#[derive(Debug, Clone, PartialEq)]
pub enum RequestResult {
    /// Success
    Success,
    /// Failure
    Failure,
    /// Timeout
    Timeout,
    /// Rejected
    Rejected,
}

impl Default for CircuitBreakerStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreakerStatsTracker {
    /// Create a new statistics tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            request_counters: Arc::new(Mutex::new(RequestCounters::default())),
            response_times: Arc::new(Mutex::new(ResponseTimeTracker::default())),
            error_tracker: Arc::new(Mutex::new(ErrorTracker::default())),
            state_transitions: Arc::new(Mutex::new(StateTransitionTracker::default())),
            health_metrics: Arc::new(Mutex::new(HealthMetrics::default())),
        }
    }

    /// Record a successful request
    pub fn record_success(&self, duration: Duration) {
        let counters = self.request_counters.lock().unwrap();
        counters.successful.fetch_add(1, Ordering::Relaxed);
        counters.total.fetch_add(1, Ordering::Relaxed);

        let mut response_times = self.response_times.lock().unwrap();
        response_times.history.push_back(duration);
        if response_times.history.len() > response_times.window_size {
            response_times.history.pop_front();
        }

        // Update min/max response times
        response_times.min = Some(response_times.min.map_or(duration, |min| min.min(duration)));
        response_times.max = Some(response_times.max.map_or(duration, |max| max.max(duration)));

        // Update sum for average calculation
        response_times.sum += duration;
        if response_times.history.len() > response_times.window_size {
            let first_duration = response_times.history[0];
            response_times.sum -= first_duration;
        }
    }

    /// Record a failed request
    pub fn record_failure(&self, error: &CircuitBreakerError) {
        let counters = self.request_counters.lock().unwrap();
        counters.failed.fetch_add(1, Ordering::Relaxed);
        counters.total.fetch_add(1, Ordering::Relaxed);

        let mut error_tracker = self.error_tracker.lock().unwrap();
        let error_type = format!("{error:?}");
        *error_tracker
            .error_counts
            .entry(error_type.clone())
            .or_insert(0) += 1;

        error_tracker.recent_errors.push_back(ErrorEvent {
            timestamp: SystemTime::now(),
            error_type,
            message: error.to_string(),
            severity: FaultSeverity::Medium,
            context: RequestContext {
                request_id: Uuid::new_v4().to_string(),
                user_id: None,
                session_id: None,
                metadata: HashMap::new(),
            },
        });

        // Maintain sliding window for recent errors
        if error_tracker.recent_errors.len() > 1000 {
            error_tracker.recent_errors.pop_front();
        }
    }

    /// Record a rejected request (circuit open)
    pub fn record_rejection(&self) {
        let counters = self.request_counters.lock().unwrap();
        counters.rejected.fetch_add(1, Ordering::Relaxed);
        counters.total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a timeout request
    pub fn record_timeout(&self) {
        let counters = self.request_counters.lock().unwrap();
        counters.timeout.fetch_add(1, Ordering::Relaxed);
        counters.total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record state transition
    pub fn record_state_transition(
        &self,
        from_state: CircuitBreakerState,
        to_state: CircuitBreakerState,
        reason: TransitionReason,
    ) {
        let mut transitions = self.state_transitions.lock().unwrap();

        let transition = StateTransition {
            timestamp: SystemTime::now(),
            from_state,
            to_state,
            reason,
            metadata: HashMap::new(),
        };

        transitions.transitions.push_back(transition);

        // Update transition counts
        let key = (from_state, to_state);
        *transitions.transition_counts.entry(key).or_insert(0) += 1;

        // Maintain sliding window for transitions
        if transitions.transitions.len() > 100 {
            transitions.transitions.pop_front();
        }

        // Update current state start time
        transitions.current_state_start = SystemTime::now();
    }

    /// Update health metrics
    pub fn update_health_metrics(&self, health_score: f64, availability: f64, reliability: f64) {
        let mut health = self.health_metrics.lock().unwrap();

        let previous_score = health.health_score;
        health.health_score = health_score;
        health.availability = availability;
        health.reliability = reliability;
        health.last_health_check = SystemTime::now();

        // Determine trend
        health.health_trend = if health_score > previous_score + 0.1 {
            HealthTrend::Improving
        } else if health_score < previous_score - 0.1 {
            HealthTrend::Degrading
        } else if health_score < 0.3 {
            HealthTrend::Critical
        } else {
            HealthTrend::Stable
        };
    }

    /// Reset all statistics
    pub fn reset(&self) {
        let counters = self.request_counters.lock().unwrap();
        counters.total.store(0, Ordering::Relaxed);
        counters.successful.store(0, Ordering::Relaxed);
        counters.failed.store(0, Ordering::Relaxed);
        counters.rejected.store(0, Ordering::Relaxed);
        counters.timeout.store(0, Ordering::Relaxed);
        counters.in_progress.store(0, Ordering::Relaxed);

        let mut response_times = self.response_times.lock().unwrap();
        response_times.history.clear();
        response_times.sum = Duration::default();
        response_times.min = None;
        response_times.max = None;
        response_times.percentiles.clear();

        let mut error_tracker = self.error_tracker.lock().unwrap();
        error_tracker.error_counts.clear();
        error_tracker.recent_errors.clear();
        error_tracker.error_rate = 0.0;

        let mut transitions = self.state_transitions.lock().unwrap();
        transitions.transitions.clear();
        transitions.transition_counts.clear();

        let mut health = self.health_metrics.lock().unwrap();
        *health = HealthMetrics::default();
    }

    /// Get current statistics
    #[must_use]
    pub fn get_stats(&self) -> CircuitBreakerStats {
        let counters = self.request_counters.lock().unwrap();
        /// CircuitBreakerStats
        CircuitBreakerStats {
            total_requests: counters.total.load(Ordering::Relaxed),
            successful_requests: counters.successful.load(Ordering::Relaxed),
            failed_requests: counters.failed.load(Ordering::Relaxed),
            consecutive_failures: 0, // Would need to track this separately
            state_changes: 0,        // Would need to track this separately
            last_failure_time: None,
            last_success_time: None,
        }
    }

    /// Get request counters
    #[must_use]
    pub fn get_request_counters(&self) -> RequestCounters {
        let counters = self.request_counters.lock().unwrap();
        /// RequestCounters
        RequestCounters {
            total: AtomicU64::new(counters.total.load(Ordering::Relaxed)),
            successful: AtomicU64::new(counters.successful.load(Ordering::Relaxed)),
            failed: AtomicU64::new(counters.failed.load(Ordering::Relaxed)),
            rejected: AtomicU64::new(counters.rejected.load(Ordering::Relaxed)),
            timeout: AtomicU64::new(counters.timeout.load(Ordering::Relaxed)),
            in_progress: AtomicU64::new(counters.in_progress.load(Ordering::Relaxed)),
        }
    }

    /// Get response time statistics
    #[must_use]
    pub fn get_response_time_stats(&self) -> ResponseTimeStats {
        let response_times = self.response_times.lock().unwrap();

        let avg = if response_times.history.is_empty() {
            Duration::default()
        } else {
            response_times.sum / response_times.history.len() as u32
        };

        /// ResponseTimeStats
        ResponseTimeStats {
            average: avg,
            min: response_times.min,
            max: response_times.max,
            percentiles: response_times.percentiles.clone(),
            sample_count: response_times.history.len(),
        }
    }

    /// Get error statistics
    #[must_use]
    pub fn get_error_stats(&self) -> ErrorStats {
        let error_tracker = self.error_tracker.lock().unwrap();

        /// ErrorStats
        ErrorStats {
            error_counts: error_tracker.error_counts.clone(),
            error_rate: error_tracker.error_rate,
            recent_error_count: error_tracker.recent_errors.len(),
            patterns: error_tracker.error_patterns.clone(),
        }
    }

    /// Get health metrics
    #[must_use]
    pub fn get_health_metrics(&self) -> HealthMetrics {
        let health = self.health_metrics.lock().unwrap();
        (*health).clone()
    }
}

impl CircuitBreakerStatsAggregator {
    /// Create a new statistics aggregator
    #[must_use]
    pub fn new(config: AggregationConfig) -> Self {
        Self {
            stats: Arc::new(RwLock::new(AggregatedStats::default())),
            config,
        }
    }

    /// Update aggregated statistics
    pub fn update_stats(&self, breaker_stats: &[CircuitBreakerStats]) {
        let mut stats = self.stats.write().unwrap();

        stats.total_breakers = breaker_stats.len() as u64;
        stats.active_breakers = breaker_stats.len() as u64; // Simplified

        // Aggregate request counts
        let mut total_requests = 0;
        let mut successful_requests = 0;
        let mut failed_requests = 0;

        for stat in breaker_stats {
            total_requests += stat.total_requests;
            successful_requests += stat.successful_requests;
            failed_requests += stat.failed_requests;
        }

        stats
            .global_requests
            .total
            .store(total_requests, Ordering::Relaxed);
        stats
            .global_requests
            .successful
            .store(successful_requests, Ordering::Relaxed);
        stats
            .global_requests
            .failed
            .store(failed_requests, Ordering::Relaxed);

        // Calculate global health score
        stats.global_health = if total_requests > 0 {
            successful_requests as f64 / total_requests as f64
        } else {
            1.0
        };
    }

    /// Get aggregated statistics
    #[must_use]
    pub fn get_stats(&self) -> AggregatedStats {
        let stats = self.stats.read().unwrap();
        /// AggregatedStats
        AggregatedStats {
            total_breakers: stats.total_breakers,
            active_breakers: stats.active_breakers,
            open_breakers: stats.open_breakers,
            half_open_breakers: stats.half_open_breakers,
            global_requests: RequestCounters {
                total: AtomicU64::new(stats.global_requests.total.load(Ordering::Relaxed)),
                successful: AtomicU64::new(
                    stats.global_requests.successful.load(Ordering::Relaxed),
                ),
                failed: AtomicU64::new(stats.global_requests.failed.load(Ordering::Relaxed)),
                rejected: AtomicU64::new(stats.global_requests.rejected.load(Ordering::Relaxed)),
                timeout: AtomicU64::new(stats.global_requests.timeout.load(Ordering::Relaxed)),
                in_progress: AtomicU64::new(
                    stats.global_requests.in_progress.load(Ordering::Relaxed),
                ),
            },
            global_health: stats.global_health,
        }
    }

    /// Reset aggregated statistics
    pub fn reset(&self) {
        let mut stats = self.stats.write().unwrap();
        *stats = AggregatedStats::default();
    }
}

/// Response time statistics summary
#[derive(Debug, Clone)]
pub struct ResponseTimeStats {
    /// Average response time
    pub average: Duration,
    /// Minimum response time
    pub min: Option<Duration>,
    /// Maximum response time
    pub max: Option<Duration>,
    /// Percentile values
    pub percentiles: HashMap<u8, Duration>,
    /// Number of samples
    pub sample_count: usize,
}

/// Error statistics summary
#[derive(Debug, Clone)]
pub struct ErrorStats {
    /// Error counts by type
    pub error_counts: HashMap<String, u64>,
    /// Current error rate
    pub error_rate: f64,
    /// Count of recent errors
    pub recent_error_count: usize,
    /// Detected error patterns
    pub patterns: Vec<ErrorPattern>,
}

impl Default for ResponseTimeTracker {
    fn default() -> Self {
        Self {
            history: VecDeque::new(),
            window_size: 1000, // Default window size
            sum: Duration::default(),
            min: None,
            max: None,
            percentiles: HashMap::new(),
        }
    }
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            real_time: true,
            algorithms: vec!["simple".to_string()],
        }
    }
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            health_score: 1.0,
            availability: 1.0,
            reliability: 1.0,
            performance_score: 1.0,
            last_health_check: SystemTime::now(),
            health_trend: HealthTrend::Unknown,
        }
    }
}

impl Default for StateTransitionTracker {
    fn default() -> Self {
        Self {
            transitions: VecDeque::new(),
            current_state_start: SystemTime::now(),
            state_durations: HashMap::new(),
            transition_counts: HashMap::new(),
        }
    }
}
