//! Circuit Breaker Module
//!
//! This module provides comprehensive circuit breaker implementations following the
//! circuit breaker pattern for fault tolerance. It includes state management,
//! failure detection, recovery strategies, statistics tracking, event management,
//! and advanced analytics.

use sklears_core::error::Result as SklResult;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

use crate::fault_core::{CircuitBreakerConfig, RetryConfig};

// Module declarations
pub mod analytics_engine;
pub mod circuit_breaker_core;
pub mod error_types;
pub mod event_system;
pub mod failure_detection;
pub mod recovery_management;
pub mod statistics_tracking;

// Re-export core types
pub use analytics_engine::{
    AnalyticsInsight, AnalyticsProcessor, AnalyticsRecommendation, AnalyticsResult,
    CircuitBreakerAnalytics,
};
pub use circuit_breaker_core::{AdvancedCircuitBreaker, CircuitBreaker};
pub use error_types::CircuitBreakerError;
pub use event_system::{
    CircuitBreakerEvent, CircuitBreakerEventRecorder, CircuitBreakerEventType,
    ConsoleEventPublisher, EventPublisher, FileEventPublisher,
};
pub use failure_detection::{
    CircuitBreakerFailureDetector, PatternDetector, SlidingWindow, StatisticalAnalyzer,
    ThresholdManager,
};
pub use recovery_management::{
    CircuitBreakerRecoveryManager, RecoveryContext, RecoveryResult, RecoveryStrategy,
    ValidationResult,
};
pub use statistics_tracking::{
    CircuitBreakerStatsAggregator, CircuitBreakerStatsTracker, ErrorTracker, HealthMetrics,
    RequestCounters, ResponseTimeTracker,
};

/// Global configuration for circuit breakers
#[derive(Debug, Clone, Default)]
pub struct CircuitBreakerGlobalConfig {
    /// Default configuration
    pub default_config: CircuitBreakerConfig,
    /// Global thresholds
    pub global_thresholds: HashMap<String, f64>,
    /// Feature flags
    pub features: HashMap<String, bool>,
    /// Integration settings
    pub integrations: HashMap<String, String>,
}

/// Publishing configuration for event distribution
#[derive(Debug, Clone)]
pub struct PublishingConfig {
    /// Enable publishing
    pub enabled: bool,
    /// Async publishing
    pub async_publishing: bool,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Buffer size
    pub buffer_size: usize,
}

/// Health monitoring configuration
#[derive(Debug, Clone)]
pub struct HealthMonitoringConfig {
    /// Enable health monitoring
    pub enabled: bool,
    /// Check interval
    pub check_interval: Duration,
    /// Health thresholds
    pub thresholds: HashMap<String, f64>,
}

/// Health checker trait for circuit breaker health monitoring
pub trait HealthChecker: Send + Sync {
    /// Check health
    fn check_health(&self, circuit_id: &str) -> HealthCheckResult;

    /// Get checker name
    fn name(&self) -> &str;
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Health status
    pub healthy: bool,
    /// Health score (0.0 to 1.0)
    pub score: f64,
    /// Check details
    pub details: String,
    /// Check timestamp
    pub timestamp: SystemTime,
}

/// Circuit breaker health monitor
pub struct CircuitBreakerHealthMonitor {
    /// Health checkers
    checkers: HashMap<String, Box<dyn HealthChecker + Send + Sync>>,
    /// Monitoring configuration
    config: HealthMonitoringConfig,
}

impl std::fmt::Debug for CircuitBreakerHealthMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreakerHealthMonitor")
            .field("checkers", &format!("<{} checkers>", self.checkers.len()))
            .field("config", &self.config)
            .finish()
    }
}

/// Circuit breaker policy manager
#[derive(Debug)]
pub struct CircuitBreakerPolicyManager {
    /// Policies
    policies: Arc<RwLock<HashMap<String, CircuitBreakerPolicy>>>,
    /// Policy engine
    engine: Arc<PolicyEngine>,
}

/// Circuit breaker policy definition
#[derive(Debug)]
pub struct CircuitBreakerPolicy {
    /// Policy identifier
    pub id: String,
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    /// Policy conditions
    pub conditions: Vec<PolicyCondition>,
}

/// Policy rule definition
#[derive(Debug, Clone)]
pub struct PolicyRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: String,
    /// Rule priority
    pub priority: i32,
}

/// Policy condition definition
#[derive(Debug, Clone)]
pub struct PolicyCondition {
    /// Condition field
    pub field: String,
    /// Condition operator
    pub operator: String,
    /// Condition value
    pub value: String,
}

/// Policy engine for rule evaluation and action execution
#[derive(Debug)]
pub struct PolicyEngine {
    /// Rule evaluator
    evaluator: Arc<RuleEvaluator>,
    /// Action executor
    executor: Arc<ActionExecutor>,
}

/// Rule evaluator for policy rules
#[derive(Debug)]
pub struct RuleEvaluator;

/// Action executor for policy actions
#[derive(Debug)]
pub struct ActionExecutor;

/// Circuit breaker event publisher for event distribution
pub struct CircuitBreakerEventPublisher {
    /// Event publishers
    publishers: Vec<Box<dyn EventPublisher + Send + Sync>>,
    /// Publishing configuration
    config: PublishingConfig,
}

impl std::fmt::Debug for CircuitBreakerEventPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreakerEventPublisher")
            .field(
                "publishers",
                &format!("<{} publishers>", self.publishers.len()),
            )
            .field("config", &self.config)
            .finish()
    }
}

/// Circuit breaker builder for easy construction
#[derive(Debug)]
pub struct CircuitBreakerBuilder {
    /// Circuit identifier
    id: Option<String>,
    /// Circuit name
    name: Option<String>,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Enable analytics
    analytics_enabled: bool,
    /// Enable event recording
    events_enabled: bool,
}

impl CircuitBreakerBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: None,
            name: None,
            config: CircuitBreakerConfig::default(),
            analytics_enabled: true,
            events_enabled: true,
        }
    }

    /// Set circuit breaker ID
    #[must_use]
    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Set circuit breaker name
    #[must_use]
    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Set configuration
    #[must_use]
    pub fn config(mut self, config: CircuitBreakerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set failure threshold
    #[must_use]
    pub fn failure_threshold(mut self, threshold: u32) -> Self {
        self.config.failure_threshold = threshold as usize;
        self
    }

    /// Set success threshold
    #[must_use]
    pub fn success_threshold(mut self, threshold: u32) -> Self {
        self.config.success_threshold = threshold as usize;
        self
    }

    /// Set timeout
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Enable or disable analytics
    #[must_use]
    pub fn analytics(mut self, enabled: bool) -> Self {
        self.analytics_enabled = enabled;
        self
    }

    /// Enable or disable event recording
    #[must_use]
    pub fn events(mut self, enabled: bool) -> Self {
        self.events_enabled = enabled;
        self
    }

    /// Build the circuit breaker
    pub fn build(self) -> SklResult<AdvancedCircuitBreaker> {
        let id = self.id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = self.name.unwrap_or_else(|| format!("circuit-{id}"));

        let circuit_breaker = AdvancedCircuitBreaker::new(name, self.config)?;

        Ok(circuit_breaker)
    }
}

impl Default for CircuitBreakerHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreakerHealthMonitor {
    /// Create a new health monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            checkers: HashMap::new(),
            config: HealthMonitoringConfig {
                enabled: true,
                check_interval: Duration::from_secs(30),
                thresholds: HashMap::new(),
            },
        }
    }

    /// Add a health checker
    pub fn add_checker(&mut self, name: String, checker: Box<dyn HealthChecker + Send + Sync>) {
        self.checkers.insert(name, checker);
    }

    /// Check health of a circuit
    #[must_use]
    pub fn check_health(&self, circuit_id: &str) -> Vec<HealthCheckResult> {
        let mut results = Vec::new();

        for (name, checker) in &self.checkers {
            let result = checker.check_health(circuit_id);
            results.push(result);
        }

        results
    }

    /// Get overall health score
    #[must_use]
    pub fn get_overall_health(&self, circuit_id: &str) -> f64 {
        let results = self.check_health(circuit_id);
        if results.is_empty() {
            return 1.0;
        }

        let total_score: f64 = results.iter().map(|r| r.score).sum();
        total_score / results.len() as f64
    }
}

impl Default for CircuitBreakerEventPublisher {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreakerEventPublisher {
    /// Create a new event publisher
    #[must_use]
    pub fn new() -> Self {
        Self {
            publishers: Vec::new(),
            config: PublishingConfig {
                enabled: true,
                async_publishing: true,
                retry_config: RetryConfig::default(),
                buffer_size: 1000,
            },
        }
    }

    /// Add a publisher
    pub fn add_publisher(&mut self, publisher: Box<dyn EventPublisher + Send + Sync>) {
        self.publishers.push(publisher);
    }

    /// Publish an event
    pub fn publish(&self, event: &CircuitBreakerEvent) -> SklResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        for publisher in &self.publishers {
            if let Err(e) = publisher.publish(event) {
                eprintln!("Failed to publish event with {}: {:?}", publisher.name(), e);
            }
        }

        Ok(())
    }
}

impl Default for CircuitBreakerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PublishingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            async_publishing: true,
            retry_config: RetryConfig::default(),
            buffer_size: 1000,
        }
    }
}

impl Default for HealthMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval: Duration::from_secs(30),
            thresholds: HashMap::new(),
        }
    }
}

/// Utility functions for circuit breaker management
pub mod utils {
    use super::{
        AdvancedCircuitBreaker, AnalyticsProcessor, CircuitBreakerBuilder, Duration,
        EventPublisher, HealthChecker, RecoveryStrategy, SklResult,
    };

    /// Create a simple circuit breaker with default settings
    pub fn create_simple_circuit_breaker(
        id: String,
        name: String,
    ) -> SklResult<AdvancedCircuitBreaker> {
        CircuitBreakerBuilder::new().id(id).name(name).build()
    }

    /// Create a circuit breaker with custom failure threshold
    pub fn create_circuit_breaker_with_threshold(
        id: String,
        name: String,
        failure_threshold: u32,
    ) -> SklResult<AdvancedCircuitBreaker> {
        CircuitBreakerBuilder::new()
            .id(id)
            .name(name)
            .failure_threshold(failure_threshold)
            .build()
    }

    /// Create a high-availability circuit breaker
    pub fn create_ha_circuit_breaker(
        id: String,
        name: String,
    ) -> SklResult<AdvancedCircuitBreaker> {
        CircuitBreakerBuilder::new()
            .id(id)
            .name(name)
            .failure_threshold(10)
            .success_threshold(5)
            .timeout(Duration::from_secs(30))
            .analytics(true)
            .events(true)
            .build()
    }
}

/// Test utilities for circuit breaker testing
#[allow(non_snake_case)]
#[cfg(test)]
pub mod test_utils {
    use super::*;

    /// Create a test circuit breaker
    pub fn create_test_circuit_breaker() -> AdvancedCircuitBreaker {
        CircuitBreakerBuilder::new()
            .id("test-circuit".to_string())
            .name("Test Circuit".to_string())
            .failure_threshold(3)
            .success_threshold(2)
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }

    /// Create multiple test circuit breakers
    pub fn create_test_circuit_breakers(count: usize) -> Vec<AdvancedCircuitBreaker> {
        (0..count)
            .map(|i| {
                CircuitBreakerBuilder::new()
                    .id(format!("test-circuit-{}", i))
                    .name(format!("Test Circuit {}", i))
                    .build()
                    .unwrap()
            })
            .collect()
    }

    /// Simulate circuit breaker failures
    pub fn simulate_failures(circuit: &AdvancedCircuitBreaker, count: u32) {
        for _ in 0..count {
            circuit.record_failure(CircuitBreakerError::ExecutionError(
                "Test failure".to_string(),
            ));
        }
    }

    /// Simulate circuit breaker successes
    pub fn simulate_successes(circuit: &AdvancedCircuitBreaker, count: u32) {
        for _ in 0..count {
            circuit.record_success(Duration::from_millis(100));
        }
    }
}

// Note: Core types are already re-exported at the top of the file
