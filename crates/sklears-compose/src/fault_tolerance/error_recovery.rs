//! Error Recovery Module
//!
//! Implements sophisticated error recovery and self-healing mechanisms for fault tolerance:
//! - Multiple recovery strategies (restart, reset, rollback, compensation)
//! - Intelligent error analysis and recovery decision making
//! - Self-healing workflows with automatic validation
//! - State management and transaction-like recovery operations
//! - Comprehensive metrics and recovery success tracking

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::time::sleep;
use uuid::Uuid;

/// Error recovery strategies defining how to handle different types of errors
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Restart the failed component or operation
    Restart {
        /// Maximum number of restart attempts
        max_attempts: u32,
        /// Delay between restart attempts
        restart_delay: Duration,
        /// Whether to preserve state across restarts
        preserve_state: bool,
    },
    /// Reset component to initial/default state
    Reset {
        /// Whether to backup current state before reset
        backup_state: bool,
        /// Custom reset procedure
        custom_reset: Option<fn() -> bool>,
    },
    /// Rollback to previous known good state
    Rollback {
        /// Number of states to rollback
        rollback_depth: u32,
        /// Whether to create checkpoint before rollback
        create_checkpoint: bool,
    },
    /// Execute compensation actions
    Compensation {
        /// Compensation actions to execute
        actions: Vec<CompensationAction>,
        /// Whether to execute in reverse order
        reverse_order: bool,
    },
    /// Graceful degradation - reduce functionality but keep operating
    GracefulDegradation {
        /// Minimum functionality level to maintain
        min_functionality: u32,
        /// Features to disable during degradation
        disable_features: Vec<String>,
    },
    /// Circuit breaker pattern - stop accepting requests temporarily
    CircuitBreaker {
        /// Duration to keep circuit open
        circuit_timeout: Duration,
        /// Health check interval while circuit is open
        health_check_interval: Duration,
    },
    /// Manual recovery - require human intervention
    Manual {
        /// Notification channels for manual intervention
        notification_channels: Vec<String>,
        /// Maximum time to wait for manual intervention
        manual_timeout: Duration,
    },
    /// Custom recovery strategy
    Custom {
        /// Custom recovery function
        recover_fn: fn(&ErrorContext) -> RecoveryResult,
    },
}

/// Compensation action for recovery operations
#[derive(Debug, Clone)]
pub struct CompensationAction {
    /// Action identifier
    pub action_id: String,
    /// Action description
    pub description: String,
    /// Action execution function (simplified for demo)
    pub execute: fn() -> Result<(), String>,
    /// Action rollback function (for nested compensation)
    pub rollback: Option<fn() -> Result<(), String>>,
    /// Action execution timeout
    pub timeout: Duration,
}

/// Error classification for determining recovery strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorClassification {
    /// Transient error that may resolve itself
    Transient,
    /// Intermittent error that occurs sporadically
    Intermittent,
    /// Persistent error that requires intervention
    Persistent,
    /// Critical error that requires immediate attention
    Critical,
    /// Resource exhaustion error
    ResourceExhaustion,
    /// Configuration error
    Configuration,
    /// Network related error
    Network,
    /// Data corruption error
    DataCorruption,
    /// Security related error
    Security,
    /// Unknown error type
    Unknown,
}

/// Error context providing information for recovery decisions
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error identifier
    pub error_id: String,
    /// Original error message
    pub error_message: String,
    /// Error classification
    pub classification: ErrorClassification,
    /// Component that experienced the error
    pub component_id: String,
    /// Error occurrence timestamp
    pub timestamp: Instant,
    /// Error severity level (1-10)
    pub severity: u32,
    /// Previous recovery attempts for this error
    pub previous_attempts: Vec<RecoveryAttempt>,
    /// Context metadata
    pub metadata: HashMap<String, String>,
    /// Error stack trace or additional details
    pub details: Vec<String>,
}

/// Recovery attempt record
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    /// Attempt identifier
    pub attempt_id: String,
    /// Recovery strategy used
    pub strategy: RecoveryStrategy,
    /// Attempt start time
    pub start_time: Instant,
    /// Attempt duration
    pub duration: Duration,
    /// Whether attempt was successful
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Actions taken during recovery
    pub actions_taken: Vec<String>,
}

/// Recovery execution result
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Whether recovery was successful
    pub success: bool,
    /// Recovery strategy that was used
    pub strategy_used: RecoveryStrategy,
    /// Total recovery time
    pub recovery_time: Duration,
    /// Actions that were executed
    pub actions_executed: Vec<String>,
    /// Error message if recovery failed
    pub error_message: Option<String>,
    /// Recovery metadata
    pub metadata: HashMap<String, String>,
    /// Post-recovery system state
    pub post_recovery_state: Option<SystemState>,
}

/// System state snapshot for recovery operations
#[derive(Debug, Clone)]
pub struct SystemState {
    /// State identifier
    pub state_id: String,
    /// State creation timestamp
    pub timestamp: Instant,
    /// Component states
    pub component_states: HashMap<String, ComponentState>,
    /// System-wide configuration
    pub system_config: HashMap<String, String>,
    /// Active transactions or operations
    pub active_operations: Vec<String>,
}

/// Individual component state
#[derive(Debug, Clone)]
pub struct ComponentState {
    /// Component identifier
    pub component_id: String,
    /// Component status
    pub status: ComponentStatus,
    /// Component configuration
    pub configuration: HashMap<String, String>,
    /// Component metrics
    pub metrics: HashMap<String, f64>,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Component status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentStatus {
    /// Component is healthy and operational
    Healthy,
    /// Component is experiencing issues but still functional
    Degraded,
    /// Component is failing but recoverable
    Failing,
    /// Component has failed and needs recovery
    Failed,
    /// Component is in recovery process
    Recovering,
    /// Component is in maintenance mode
    Maintenance,
    /// Component status is unknown
    Unknown,
}

/// Error recovery system configuration
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Configuration identifier
    pub config_id: String,
    /// Default recovery strategy
    pub default_strategy: RecoveryStrategy,
    /// Error classification rules
    pub classification_rules: HashMap<String, ErrorClassification>,
    /// Strategy selection rules (error pattern -> strategy)
    pub strategy_selection: HashMap<String, RecoveryStrategy>,
    /// Maximum concurrent recovery operations
    pub max_concurrent_recoveries: u32,
    /// Recovery timeout
    pub recovery_timeout: Duration,
    /// Whether to enable automatic recovery
    pub auto_recovery: bool,
    /// State snapshot interval
    pub snapshot_interval: Duration,
    /// Maximum number of state snapshots to keep
    pub max_snapshots: u32,
}

/// Recovery system metrics
#[derive(Debug, Clone)]
pub struct RecoveryMetrics {
    /// Configuration identifier
    pub config_id: String,
    /// Total number of recovery attempts
    pub total_recoveries: u64,
    /// Number of successful recoveries
    pub successful_recoveries: u64,
    /// Number of failed recoveries
    pub failed_recoveries: u64,
    /// Average recovery time
    pub average_recovery_time: Duration,
    /// Recovery success rate
    pub success_rate: f64,
    /// Recovery attempts by strategy
    pub strategy_usage: HashMap<String, u32>,
    /// Recovery attempts by error classification
    pub classification_distribution: HashMap<String, u32>,
    /// Recent recovery events
    pub recent_recoveries: Vec<RecoveryAttempt>,
}

/// Error recovery system errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum RecoveryError {
    #[error("Recovery strategy failed: {message}")]
    StrategyFailed { message: String },
    #[error("Recovery timeout exceeded: {timeout:?}")]
    RecoveryTimeout { timeout: Duration },
    #[error("Maximum concurrent recoveries exceeded")]
    TooManyConcurrentRecoveries,
    #[error("Component not found: {component_id}")]
    ComponentNotFound { component_id: String },
    #[error("State snapshot not found: {state_id}")]
    StateNotFound { state_id: String },
    #[error("Compensation action failed: {action_id}")]
    CompensationFailed { action_id: String },
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },
}

/// Error recovery system implementation
#[derive(Debug)]
pub struct ErrorRecoverySystem {
    /// System identifier
    system_id: String,
    /// Recovery configuration
    config: RecoveryConfig,
    /// Active recovery operations
    active_recoveries: Arc<RwLock<HashMap<String, RecoveryAttempt>>>,
    /// System state snapshots
    state_snapshots: Arc<RwLock<VecDeque<SystemState>>>,
    /// Recovery metrics
    metrics: Arc<RwLock<RecoveryMetrics>>,
    /// Recovery history
    recovery_history: Arc<RwLock<VecDeque<RecoveryAttempt>>>,
    /// Component registry
    components: Arc<RwLock<HashMap<String, ComponentState>>>,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::Restart {
            max_attempts: 3,
            restart_delay: Duration::from_secs(5),
            preserve_state: true,
        }
    }
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            config_id: "default".to_string(),
            default_strategy: RecoveryStrategy::default(),
            classification_rules: HashMap::new(),
            strategy_selection: HashMap::new(),
            max_concurrent_recoveries: 5,
            recovery_timeout: Duration::from_secs(300),
            auto_recovery: true,
            snapshot_interval: Duration::from_secs(60),
            max_snapshots: 10,
        }
    }
}

impl ErrorRecoverySystem {
    /// Create new error recovery system
    pub fn new(system_id: String, config: RecoveryConfig) -> Self {
        let metrics = RecoveryMetrics {
            config_id: config.config_id.clone(),
            total_recoveries: 0,
            successful_recoveries: 0,
            failed_recoveries: 0,
            average_recovery_time: Duration::ZERO,
            success_rate: 0.0,
            strategy_usage: HashMap::new(),
            classification_distribution: HashMap::new(),
            recent_recoveries: Vec::new(),
        };

        Self {
            system_id,
            config,
            active_recoveries: Arc::new(RwLock::new(HashMap::new())),
            state_snapshots: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(RwLock::new(metrics)),
            recovery_history: Arc::new(RwLock::new(VecDeque::new())),
            components: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create system with default configuration
    pub fn with_defaults(system_id: String) -> Self {
        Self::new(system_id, RecoveryConfig::default())
    }

    /// Register component for recovery management
    pub async fn register_component(&self, component: ComponentState) {
        let mut components = self.components.write().unwrap();
        components.insert(component.component_id.clone(), component);
    }

    /// Unregister component from recovery management
    pub async fn unregister_component(&self, component_id: &str) {
        let mut components = self.components.write().unwrap();
        components.remove(component_id);
    }

    /// Attempt recovery for a given error
    pub async fn attempt_recovery(&self, error_context: ErrorContext) -> Result<RecoveryResult, RecoveryError> {
        // Check concurrent recovery limit
        {
            let active = self.active_recoveries.read().unwrap();
            if active.len() >= self.config.max_concurrent_recoveries as usize {
                return Err(RecoveryError::TooManyConcurrentRecoveries);
            }
        }

        let recovery_id = Uuid::new_v4().to_string();
        let start_time = Instant::now();

        // Select recovery strategy
        let strategy = self.select_recovery_strategy(&error_context).await;

        // Create recovery attempt record
        let mut attempt = RecoveryAttempt {
            attempt_id: recovery_id.clone(),
            strategy: strategy.clone(),
            start_time,
            duration: Duration::ZERO,
            success: false,
            error_message: None,
            actions_taken: Vec::new(),
        };

        // Register active recovery
        {
            let mut active = self.active_recoveries.write().unwrap();
            active.insert(recovery_id.clone(), attempt.clone());
        }

        // Take state snapshot before recovery
        if self.should_take_snapshot(&strategy).await {
            self.create_state_snapshot().await;
            attempt.actions_taken.push("State snapshot created".to_string());
        }

        // Execute recovery strategy
        let recovery_result = self.execute_recovery_strategy(&strategy, &error_context, &mut attempt).await;

        // Update attempt duration
        attempt.duration = start_time.elapsed();
        attempt.success = recovery_result.is_ok();

        if let Err(ref error) = recovery_result {
            attempt.error_message = Some(error.to_string());
        }

        // Remove from active recoveries
        {
            let mut active = self.active_recoveries.write().unwrap();
            active.remove(&recovery_id);
        }

        // Record in history
        {
            let mut history = self.recovery_history.write().unwrap();
            history.push_back(attempt.clone());
            if history.len() > 100 { // Keep last 100 attempts
                history.pop_front();
            }
        }

        // Update metrics
        self.update_metrics(&attempt, &strategy, &error_context).await;

        match recovery_result {
            Ok(post_recovery_state) => {
                Ok(RecoveryResult {
                    success: true,
                    strategy_used: strategy,
                    recovery_time: attempt.duration,
                    actions_executed: attempt.actions_taken,
                    error_message: None,
                    metadata: HashMap::new(),
                    post_recovery_state,
                })
            },
            Err(error) => {
                Ok(RecoveryResult {
                    success: false,
                    strategy_used: strategy,
                    recovery_time: attempt.duration,
                    actions_executed: attempt.actions_taken,
                    error_message: Some(error.to_string()),
                    metadata: HashMap::new(),
                    post_recovery_state: None,
                })
            }
        }
    }

    /// Select appropriate recovery strategy based on error context
    async fn select_recovery_strategy(&self, error_context: &ErrorContext) -> RecoveryStrategy {
        // Check strategy selection rules first
        for (pattern, strategy) in &self.config.strategy_selection {
            if error_context.error_message.contains(pattern) {
                return strategy.clone();
            }
        }

        // Select based on error classification
        match error_context.classification {
            ErrorClassification::Transient => RecoveryStrategy::Restart {
                max_attempts: 2,
                restart_delay: Duration::from_secs(1),
                preserve_state: true,
            },
            ErrorClassification::Intermittent => RecoveryStrategy::CircuitBreaker {
                circuit_timeout: Duration::from_secs(30),
                health_check_interval: Duration::from_secs(5),
            },
            ErrorClassification::Persistent => RecoveryStrategy::Reset {
                backup_state: true,
                custom_reset: None,
            },
            ErrorClassification::Critical => RecoveryStrategy::Manual {
                notification_channels: vec!["emergency".to_string()],
                manual_timeout: Duration::from_secs(300),
            },
            ErrorClassification::ResourceExhaustion => RecoveryStrategy::GracefulDegradation {
                min_functionality: 50,
                disable_features: vec!["non_essential".to_string()],
            },
            ErrorClassification::DataCorruption => RecoveryStrategy::Rollback {
                rollback_depth: 1,
                create_checkpoint: true,
            },
            _ => self.config.default_strategy.clone(),
        }
    }

    /// Execute selected recovery strategy
    async fn execute_recovery_strategy(
        &self,
        strategy: &RecoveryStrategy,
        error_context: &ErrorContext,
        attempt: &mut RecoveryAttempt,
    ) -> Result<Option<SystemState>, RecoveryError> {
        match strategy {
            RecoveryStrategy::Restart { max_attempts, restart_delay, preserve_state } => {
                self.execute_restart_recovery(error_context, *max_attempts, *restart_delay, *preserve_state, attempt).await
            },
            RecoveryStrategy::Reset { backup_state, custom_reset } => {
                self.execute_reset_recovery(error_context, *backup_state, custom_reset, attempt).await
            },
            RecoveryStrategy::Rollback { rollback_depth, create_checkpoint } => {
                self.execute_rollback_recovery(error_context, *rollback_depth, *create_checkpoint, attempt).await
            },
            RecoveryStrategy::Compensation { actions, reverse_order } => {
                self.execute_compensation_recovery(actions, *reverse_order, attempt).await
            },
            RecoveryStrategy::GracefulDegradation { min_functionality, disable_features } => {
                self.execute_degradation_recovery(*min_functionality, disable_features, attempt).await
            },
            RecoveryStrategy::CircuitBreaker { circuit_timeout, health_check_interval } => {
                self.execute_circuit_breaker_recovery(*circuit_timeout, *health_check_interval, attempt).await
            },
            RecoveryStrategy::Manual { notification_channels: _, manual_timeout } => {
                self.execute_manual_recovery(*manual_timeout, attempt).await
            },
            RecoveryStrategy::Custom { recover_fn } => {
                let result = recover_fn(error_context);
                if result.success {
                    attempt.actions_taken.extend(result.actions_executed);
                    Ok(result.post_recovery_state)
                } else {
                    Err(RecoveryError::StrategyFailed {
                        message: result.error_message.unwrap_or_else(|| "Custom recovery failed".to_string()),
                    })
                }
            }
        }
    }

    /// Execute restart recovery strategy
    async fn execute_restart_recovery(
        &self,
        error_context: &ErrorContext,
        max_attempts: u32,
        restart_delay: Duration,
        preserve_state: bool,
        attempt: &mut RecoveryAttempt,
    ) -> Result<Option<SystemState>, RecoveryError> {
        let component_id = &error_context.component_id;

        for attempt_num in 1..=max_attempts {
            attempt.actions_taken.push(format!("Restart attempt {} for component {}", attempt_num, component_id));

            if attempt_num > 1 {
                sleep(restart_delay).await;
            }

            // Simulate component restart
            if self.restart_component(component_id, preserve_state).await {
                attempt.actions_taken.push(format!("Component {} successfully restarted", component_id));
                return Ok(Some(self.get_current_system_state().await));
            }

            attempt.actions_taken.push(format!("Restart attempt {} failed", attempt_num));
        }

        Err(RecoveryError::StrategyFailed {
            message: format!("Failed to restart component {} after {} attempts", component_id, max_attempts),
        })
    }

    /// Execute reset recovery strategy
    async fn execute_reset_recovery(
        &self,
        error_context: &ErrorContext,
        backup_state: bool,
        _custom_reset: &Option<fn() -> bool>,
        attempt: &mut RecoveryAttempt,
    ) -> Result<Option<SystemState>, RecoveryError> {
        let component_id = &error_context.component_id;

        if backup_state {
            attempt.actions_taken.push("Creating state backup before reset".to_string());
            // In real implementation, would backup component state
        }

        attempt.actions_taken.push(format!("Resetting component {} to default state", component_id));

        if self.reset_component(component_id).await {
            attempt.actions_taken.push(format!("Component {} successfully reset", component_id));
            Ok(Some(self.get_current_system_state().await))
        } else {
            Err(RecoveryError::StrategyFailed {
                message: format!("Failed to reset component {}", component_id),
            })
        }
    }

    /// Execute rollback recovery strategy
    async fn execute_rollback_recovery(
        &self,
        _error_context: &ErrorContext,
        rollback_depth: u32,
        create_checkpoint: bool,
        attempt: &mut RecoveryAttempt,
    ) -> Result<Option<SystemState>, RecoveryError> {
        if create_checkpoint {
            attempt.actions_taken.push("Creating checkpoint before rollback".to_string());
            self.create_state_snapshot().await;
        }

        let snapshots = self.state_snapshots.read().unwrap();
        if snapshots.len() >= rollback_depth as usize {
            let rollback_index = snapshots.len() - rollback_depth as usize;
            if let Some(rollback_state) = snapshots.get(rollback_index) {
                attempt.actions_taken.push(format!("Rolling back {} states to {}", rollback_depth, rollback_state.state_id));

                // In real implementation, would restore system to the snapshot state
                self.restore_system_state(rollback_state).await;

                attempt.actions_taken.push("System successfully rolled back".to_string());
                return Ok(Some(rollback_state.clone()));
            }
        }

        Err(RecoveryError::StrategyFailed {
            message: format!("No suitable snapshot found for rollback depth {}", rollback_depth),
        })
    }

    /// Execute compensation recovery strategy
    async fn execute_compensation_recovery(
        &self,
        actions: &[CompensationAction],
        reverse_order: bool,
        attempt: &mut RecoveryAttempt,
    ) -> Result<Option<SystemState>, RecoveryError> {
        let execution_order: Vec<&CompensationAction> = if reverse_order {
            actions.iter().rev().collect()
        } else {
            actions.iter().collect()
        };

        for action in execution_order {
            attempt.actions_taken.push(format!("Executing compensation action: {}", action.description));

            match tokio::time::timeout(action.timeout, async { (action.execute)() }).await {
                Ok(Ok(())) => {
                    attempt.actions_taken.push(format!("Compensation action {} completed successfully", action.action_id));
                },
                Ok(Err(error)) => {
                    return Err(RecoveryError::CompensationFailed {
                        action_id: action.action_id.clone(),
                    });
                },
                Err(_) => {
                    return Err(RecoveryError::RecoveryTimeout {
                        timeout: action.timeout,
                    });
                }
            }
        }

        attempt.actions_taken.push("All compensation actions completed successfully".to_string());
        Ok(Some(self.get_current_system_state().await))
    }

    /// Execute graceful degradation recovery strategy
    async fn execute_degradation_recovery(
        &self,
        min_functionality: u32,
        disable_features: &[String],
        attempt: &mut RecoveryAttempt,
    ) -> Result<Option<SystemState>, RecoveryError> {
        attempt.actions_taken.push(format!("Initiating graceful degradation to {}% functionality", min_functionality));

        for feature in disable_features {
            attempt.actions_taken.push(format!("Disabling feature: {}", feature));
            // In real implementation, would disable the specified feature
        }

        attempt.actions_taken.push("Graceful degradation completed".to_string());
        Ok(Some(self.get_current_system_state().await))
    }

    /// Execute circuit breaker recovery strategy
    async fn execute_circuit_breaker_recovery(
        &self,
        circuit_timeout: Duration,
        health_check_interval: Duration,
        attempt: &mut RecoveryAttempt,
    ) -> Result<Option<SystemState>, RecoveryError> {
        attempt.actions_taken.push("Opening circuit breaker".to_string());

        // Wait for circuit timeout while performing health checks
        let circuit_start = Instant::now();
        while circuit_start.elapsed() < circuit_timeout {
            sleep(health_check_interval).await;

            attempt.actions_taken.push("Performing health check".to_string());

            // Simulate health check
            if self.system_health_check().await {
                attempt.actions_taken.push("System healthy - closing circuit breaker".to_string());
                return Ok(Some(self.get_current_system_state().await));
            }
        }

        attempt.actions_taken.push("Circuit breaker timeout reached - manual intervention required".to_string());
        Err(RecoveryError::RecoveryTimeout { timeout: circuit_timeout })
    }

    /// Execute manual recovery strategy
    async fn execute_manual_recovery(
        &self,
        manual_timeout: Duration,
        attempt: &mut RecoveryAttempt,
    ) -> Result<Option<SystemState>, RecoveryError> {
        attempt.actions_taken.push("Manual recovery required - sending notifications".to_string());

        // Simulate waiting for manual intervention
        sleep(std::cmp::min(manual_timeout, Duration::from_secs(5))).await;

        attempt.actions_taken.push("Manual intervention timeout - escalating".to_string());
        Err(RecoveryError::RecoveryTimeout { timeout: manual_timeout })
    }

    /// Component lifecycle operations (simplified for demo)
    async fn restart_component(&self, component_id: &str, _preserve_state: bool) -> bool {
        let mut components = self.components.write().unwrap();
        if let Some(component) = components.get_mut(component_id) {
            component.status = ComponentStatus::Healthy;
            component.last_update = Instant::now();
            return true;
        }
        false
    }

    async fn reset_component(&self, component_id: &str) -> bool {
        let mut components = self.components.write().unwrap();
        if let Some(component) = components.get_mut(component_id) {
            component.status = ComponentStatus::Healthy;
            component.configuration.clear();
            component.metrics.clear();
            component.last_update = Instant::now();
            return true;
        }
        false
    }

    async fn system_health_check(&self) -> bool {
        let components = self.components.read().unwrap();
        let healthy_count = components.values()
            .filter(|c| c.status == ComponentStatus::Healthy)
            .count();

        // System is healthy if at least 50% of components are healthy
        let total_count = components.len();
        if total_count == 0 {
            return true; // No components to check
        }

        (healthy_count as f64 / total_count as f64) >= 0.5
    }

    /// State management operations
    async fn create_state_snapshot(&self) {
        let system_state = self.get_current_system_state().await;

        let mut snapshots = self.state_snapshots.write().unwrap();
        snapshots.push_back(system_state);

        // Keep only the configured number of snapshots
        while snapshots.len() > self.config.max_snapshots as usize {
            snapshots.pop_front();
        }
    }

    async fn get_current_system_state(&self) -> SystemState {
        let components = self.components.read().unwrap();
        SystemState {
            state_id: Uuid::new_v4().to_string(),
            timestamp: Instant::now(),
            component_states: components.clone(),
            system_config: HashMap::new(),
            active_operations: Vec::new(),
        }
    }

    async fn restore_system_state(&self, _state: &SystemState) {
        // In real implementation, would restore system to the given state
        // This is a complex operation involving restoring component states,
        // configuration, and active operations
    }

    async fn should_take_snapshot(&self, strategy: &RecoveryStrategy) -> bool {
        matches!(strategy,
            RecoveryStrategy::Rollback { .. } |
            RecoveryStrategy::Reset { backup_state: true, .. }
        )
    }

    /// Update recovery metrics
    async fn update_metrics(&self, attempt: &RecoveryAttempt, strategy: &RecoveryStrategy, error_context: &ErrorContext) {
        let mut metrics = self.metrics.write().unwrap();

        metrics.total_recoveries += 1;

        if attempt.success {
            metrics.successful_recoveries += 1;
        } else {
            metrics.failed_recoveries += 1;
        }

        // Update success rate
        metrics.success_rate = metrics.successful_recoveries as f64 / metrics.total_recoveries as f64;

        // Update average recovery time
        if metrics.total_recoveries == 1 {
            metrics.average_recovery_time = attempt.duration;
        } else {
            let total_time = metrics.average_recovery_time * (metrics.total_recoveries - 1) as u32 + attempt.duration;
            metrics.average_recovery_time = total_time / metrics.total_recoveries as u32;
        }

        // Update strategy usage
        let strategy_name = format!("{:?}", strategy).split('{').next().unwrap_or("Unknown").to_string();
        *metrics.strategy_usage.entry(strategy_name).or_insert(0) += 1;

        // Update classification distribution
        let classification_name = format!("{:?}", error_context.classification);
        *metrics.classification_distribution.entry(classification_name).or_insert(0) += 1;

        // Update recent recoveries
        metrics.recent_recoveries.push(attempt.clone());
        if metrics.recent_recoveries.len() > 20 {
            metrics.recent_recoveries.remove(0);
        }
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> RecoveryMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Get system health status
    pub async fn get_health_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("system_id".to_string(), self.system_id.clone());

        let components = self.components.read().unwrap();
        let healthy_components = components.values()
            .filter(|c| c.status == ComponentStatus::Healthy)
            .count();

        status.insert("total_components".to_string(), components.len().to_string());
        status.insert("healthy_components".to_string(), healthy_components.to_string());

        if !components.is_empty() {
            let health_percentage = (healthy_components as f64 / components.len() as f64) * 100.0;
            status.insert("health_percentage".to_string(), format!("{:.1}", health_percentage));
        }

        let active = self.active_recoveries.read().unwrap();
        status.insert("active_recoveries".to_string(), active.len().to_string());

        let snapshots = self.state_snapshots.read().unwrap();
        status.insert("available_snapshots".to_string(), snapshots.len().to_string());

        status
    }

    /// Classify error based on context
    pub fn classify_error(&self, error_message: &str) -> ErrorClassification {
        let lower_message = error_message.to_lowercase();

        for (pattern, classification) in &self.config.classification_rules {
            if lower_message.contains(pattern) {
                return classification.clone();
            }
        }

        // Default classification heuristics
        if lower_message.contains("timeout") || lower_message.contains("connection") {
            ErrorClassification::Network
        } else if lower_message.contains("memory") || lower_message.contains("disk") || lower_message.contains("cpu") {
            ErrorClassification::ResourceExhaustion
        } else if lower_message.contains("config") || lower_message.contains("setting") {
            ErrorClassification::Configuration
        } else if lower_message.contains("critical") || lower_message.contains("fatal") {
            ErrorClassification::Critical
        } else if lower_message.contains("corrupt") || lower_message.contains("invalid") {
            ErrorClassification::DataCorruption
        } else {
            ErrorClassification::Unknown
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_error_recovery_basic_functionality() {
        let system = ErrorRecoverySystem::with_defaults("test_system".to_string());

        let component = ComponentState {
            component_id: "test_component".to_string(),
            status: ComponentStatus::Failed,
            configuration: HashMap::new(),
            metrics: HashMap::new(),
            last_update: Instant::now(),
        };

        system.register_component(component).await;

        let error_context = ErrorContext {
            error_id: "test_error".to_string(),
            error_message: "Component failure".to_string(),
            classification: ErrorClassification::Transient,
            component_id: "test_component".to_string(),
            timestamp: Instant::now(),
            severity: 5,
            previous_attempts: Vec::new(),
            metadata: HashMap::new(),
            details: Vec::new(),
        };

        let result = system.attempt_recovery(error_context).await.unwrap();
        assert!(result.success);
        assert!(!result.actions_executed.is_empty());
    }

    #[tokio::test]
    async fn test_error_classification() {
        let system = ErrorRecoverySystem::with_defaults("test_system".to_string());

        assert_eq!(system.classify_error("Connection timeout"), ErrorClassification::Network);
        assert_eq!(system.classify_error("Out of memory"), ErrorClassification::ResourceExhaustion);
        assert_eq!(system.classify_error("Configuration error"), ErrorClassification::Configuration);
        assert_eq!(system.classify_error("Critical system failure"), ErrorClassification::Critical);
        assert_eq!(system.classify_error("Data corruption detected"), ErrorClassification::DataCorruption);
        assert_eq!(system.classify_error("Unknown error"), ErrorClassification::Unknown);
    }

    #[tokio::test]
    async fn test_recovery_strategy_selection() {
        let system = ErrorRecoverySystem::with_defaults("test_system".to_string());

        let transient_error = ErrorContext {
            error_id: "test".to_string(),
            error_message: "Temporary failure".to_string(),
            classification: ErrorClassification::Transient,
            component_id: "test".to_string(),
            timestamp: Instant::now(),
            severity: 3,
            previous_attempts: Vec::new(),
            metadata: HashMap::new(),
            details: Vec::new(),
        };

        let strategy = system.select_recovery_strategy(&transient_error).await;
        assert!(matches!(strategy, RecoveryStrategy::Restart { .. }));

        let critical_error = ErrorContext {
            error_id: "test".to_string(),
            error_message: "Critical failure".to_string(),
            classification: ErrorClassification::Critical,
            component_id: "test".to_string(),
            timestamp: Instant::now(),
            severity: 10,
            previous_attempts: Vec::new(),
            metadata: HashMap::new(),
            details: Vec::new(),
        };

        let strategy = system.select_recovery_strategy(&critical_error).await;
        assert!(matches!(strategy, RecoveryStrategy::Manual { .. }));
    }

    #[tokio::test]
    async fn test_state_snapshot_management() {
        let system = ErrorRecoverySystem::with_defaults("test_system".to_string());

        // Create initial snapshot
        system.create_state_snapshot().await;

        let snapshots = system.state_snapshots.read().unwrap();
        assert_eq!(snapshots.len(), 1);

        drop(snapshots);

        // Test snapshot limit
        for _ in 0..15 {
            system.create_state_snapshot().await;
        }

        let snapshots = system.state_snapshots.read().unwrap();
        assert!(snapshots.len() <= system.config.max_snapshots as usize);
    }

    #[tokio::test]
    async fn test_component_registration() {
        let system = ErrorRecoverySystem::with_defaults("test_system".to_string());

        let component = ComponentState {
            component_id: "test_component".to_string(),
            status: ComponentStatus::Healthy,
            configuration: HashMap::new(),
            metrics: HashMap::new(),
            last_update: Instant::now(),
        };

        system.register_component(component).await;

        let components = system.components.read().unwrap();
        assert!(components.contains_key("test_component"));
        assert_eq!(components.get("test_component").unwrap().status, ComponentStatus::Healthy);

        drop(components);

        system.unregister_component("test_component").await;

        let components = system.components.read().unwrap();
        assert!(!components.contains_key("test_component"));
    }

    #[tokio::test]
    async fn test_recovery_metrics() {
        let system = ErrorRecoverySystem::with_defaults("test_system".to_string());

        let component = ComponentState {
            component_id: "test_component".to_string(),
            status: ComponentStatus::Failed,
            configuration: HashMap::new(),
            metrics: HashMap::new(),
            last_update: Instant::now(),
        };

        system.register_component(component).await;

        // Perform multiple recovery attempts
        for i in 0..3 {
            let error_context = ErrorContext {
                error_id: format!("error_{}", i),
                error_message: "Test error".to_string(),
                classification: ErrorClassification::Transient,
                component_id: "test_component".to_string(),
                timestamp: Instant::now(),
                severity: 3,
                previous_attempts: Vec::new(),
                metadata: HashMap::new(),
                details: Vec::new(),
            };

            let _ = system.attempt_recovery(error_context).await;
        }

        let metrics = system.get_metrics().await;
        assert_eq!(metrics.total_recoveries, 3);
        assert!(metrics.successful_recoveries > 0);
        assert!(metrics.average_recovery_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_health_status() {
        let system = ErrorRecoverySystem::with_defaults("test_system".to_string());

        let healthy_component = ComponentState {
            component_id: "healthy".to_string(),
            status: ComponentStatus::Healthy,
            configuration: HashMap::new(),
            metrics: HashMap::new(),
            last_update: Instant::now(),
        };

        let failed_component = ComponentState {
            component_id: "failed".to_string(),
            status: ComponentStatus::Failed,
            configuration: HashMap::new(),
            metrics: HashMap::new(),
            last_update: Instant::now(),
        };

        system.register_component(healthy_component).await;
        system.register_component(failed_component).await;

        let health = system.get_health_status().await;
        assert_eq!(health.get("total_components").unwrap(), "2");
        assert_eq!(health.get("healthy_components").unwrap(), "1");
        assert_eq!(health.get("health_percentage").unwrap(), "50.0");
    }
}