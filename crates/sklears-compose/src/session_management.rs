//! Session Management and Lifecycle Control
//!
//! This module provides comprehensive session management capabilities for monitoring
//! execution workflows. It handles session lifecycle, metadata tracking, state
//! management, and resource allocation for monitoring operations.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;

use crate::monitoring_core::{MonitoringSessionStatus, MonitoringCapabilities};
use crate::metrics_collection::PerformanceMetric;
use crate::event_tracking::TaskExecutionEvent;
use crate::alerting_system::AlertState;
use crate::health_monitoring::SystemHealth;
use crate::configuration_management::MonitoringConfig;
use crate::resource_management::{ResourceAllocation, ResourceUtilization};

/// Monitoring session handle for tracking active monitoring
///
/// Provides comprehensive control and metadata for an active monitoring session
/// including lifecycle management, configuration, resource tracking, and
/// operational statistics.
#[derive(Debug, Clone)]
pub struct MonitoringSession {
    /// Unique session identifier
    pub session_id: String,

    /// Session start time
    pub start_time: SystemTime,

    /// Session end time (if completed)
    pub end_time: Option<SystemTime>,

    /// Monitoring configuration for this session
    pub config: MonitoringConfig,

    /// Current session status
    pub status: MonitoringSessionStatus,

    /// Number of metrics collected
    pub metrics_collected: u64,

    /// Number of events recorded
    pub events_recorded: u64,

    /// Number of active alerts
    pub active_alerts: usize,

    /// Session metadata and context
    pub metadata: SessionMetadata,

    /// Resource allocation for monitoring
    pub resource_allocation: ResourceAllocation,

    /// Session statistics
    pub statistics: SessionStatistics,

    /// Session configuration history
    pub config_history: Vec<ConfigurationChange>,

    /// Session tags for categorization
    pub tags: Vec<String>,

    /// Custom session properties
    pub properties: HashMap<String, String>,
}

impl MonitoringSession {
    /// Create a new monitoring session
    pub fn new(session_id: String, config: MonitoringConfig) -> Self {
        let start_time = SystemTime::now();

        Self {
            session_id: session_id.clone(),
            start_time,
            end_time: None,
            config: config.clone(),
            status: MonitoringSessionStatus::Starting,
            metrics_collected: 0,
            events_recorded: 0,
            active_alerts: 0,
            metadata: SessionMetadata::new(session_id),
            resource_allocation: ResourceAllocation::default(),
            statistics: SessionStatistics::new(start_time),
            config_history: vec![ConfigurationChange::initial(config)],
            tags: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Get session duration
    pub fn duration(&self) -> Duration {
        let end_time = self.end_time.unwrap_or_else(SystemTime::now);
        end_time.duration_since(self.start_time).unwrap_or(Duration::from_secs(0))
    }

    /// Check if session is active
    pub fn is_active(&self) -> bool {
        matches!(self.status, MonitoringSessionStatus::Active)
    }

    /// Check if session is stopped
    pub fn is_stopped(&self) -> bool {
        matches!(self.status, MonitoringSessionStatus::Stopped)
    }

    /// Check if session has failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, MonitoringSessionStatus::Failed { .. })
    }

    /// Update session status
    pub fn update_status(&mut self, new_status: MonitoringSessionStatus) {
        self.status = new_status;
        self.statistics.last_status_change = SystemTime::now();

        if self.is_stopped() {
            self.end_time = Some(SystemTime::now());
        }
    }

    /// Add tag to session
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Set custom property
    pub fn set_property(&mut self, key: String, value: String) {
        self.properties.insert(key, value);
    }

    /// Get custom property
    pub fn get_property(&self, key: &str) -> Option<&String> {
        self.properties.get(key)
    }

    /// Update configuration with change tracking
    pub fn update_config(&mut self, new_config: MonitoringConfig) {
        let change = ConfigurationChange {
            timestamp: SystemTime::now(),
            old_config: self.config.clone(),
            new_config: new_config.clone(),
            change_reason: "Manual update".to_string(),
            changed_by: "system".to_string(),
        };

        self.config = new_config;
        self.config_history.push(change);
    }

    /// Increment metrics counter
    pub fn increment_metrics(&mut self, count: u64) {
        self.metrics_collected += count;
        self.statistics.total_operations += count;
        self.statistics.last_activity = SystemTime::now();
    }

    /// Increment events counter
    pub fn increment_events(&mut self, count: u64) {
        self.events_recorded += count;
        self.statistics.total_operations += count;
        self.statistics.last_activity = SystemTime::now();
    }

    /// Update alert count
    pub fn update_alert_count(&mut self, count: usize) {
        self.active_alerts = count;
    }

    /// Calculate operational efficiency
    pub fn operational_efficiency(&self) -> f64 {
        if self.statistics.total_operations == 0 {
            return 1.0;
        }

        let duration_secs = self.duration().as_secs() as f64;
        if duration_secs == 0.0 {
            return 1.0;
        }

        let ops_per_second = self.statistics.total_operations as f64 / duration_secs;

        // Simple efficiency calculation - can be enhanced
        (ops_per_second / 1000.0).min(1.0)
    }
}

/// Session metadata for additional context and tracking
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    /// Session identifier
    pub session_id: String,

    /// User or system that initiated the session
    pub initiator: String,

    /// Purpose or description of the session
    pub purpose: Option<String>,

    /// Environment information
    pub environment: HashMap<String, String>,

    /// Execution context
    pub execution_context: ExecutionContext,

    /// Session priority level
    pub priority: SessionPriority,

    /// Expected session duration
    pub expected_duration: Option<Duration>,

    /// Session dependencies
    pub dependencies: Vec<String>,

    /// Related sessions
    pub related_sessions: Vec<String>,

    /// Creation timestamp
    pub created_at: SystemTime,

    /// Last modified timestamp
    pub modified_at: SystemTime,

    /// Session version
    pub version: u32,
}

impl SessionMetadata {
    /// Create new session metadata
    pub fn new(session_id: String) -> Self {
        let now = SystemTime::now();

        Self {
            session_id,
            initiator: "system".to_string(),
            purpose: None,
            environment: HashMap::new(),
            execution_context: ExecutionContext::default(),
            priority: SessionPriority::Normal,
            expected_duration: None,
            dependencies: Vec::new(),
            related_sessions: Vec::new(),
            created_at: now,
            modified_at: now,
            version: 1,
        }
    }

    /// Update metadata with version increment
    pub fn update(&mut self) {
        self.modified_at = SystemTime::now();
        self.version += 1;
    }

    /// Set environment variable
    pub fn set_env(&mut self, key: String, value: String) {
        self.environment.insert(key, value);
        self.update();
    }

    /// Add dependency
    pub fn add_dependency(&mut self, dependency: String) {
        if !self.dependencies.contains(&dependency) {
            self.dependencies.push(dependency);
            self.update();
        }
    }

    /// Link related session
    pub fn link_session(&mut self, session_id: String) {
        if !self.related_sessions.contains(&session_id) {
            self.related_sessions.push(session_id);
            self.update();
        }
    }
}

/// Execution context information
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Execution environment (development, staging, production)
    pub environment_type: EnvironmentType,

    /// Application or service name
    pub application: String,

    /// Application version
    pub version: String,

    /// Deployment identifier
    pub deployment_id: Option<String>,

    /// Host or node information
    pub host_info: HostInfo,

    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            environment_type: EnvironmentType::Development,
            application: "unknown".to_string(),
            version: "unknown".to_string(),
            deployment_id: None,
            host_info: HostInfo::default(),
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

/// Environment type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum EnvironmentType {
    Development,
    Testing,
    Staging,
    Production,
    Experimental,
}

/// Host information
#[derive(Debug, Clone)]
pub struct HostInfo {
    /// Hostname
    pub hostname: String,

    /// Operating system
    pub os: String,

    /// Architecture
    pub architecture: String,

    /// CPU count
    pub cpu_count: usize,

    /// Total memory (bytes)
    pub total_memory: u64,

    /// Available memory (bytes)
    pub available_memory: u64,
}

impl Default for HostInfo {
    fn default() -> Self {
        Self {
            hostname: "localhost".to_string(),
            os: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            cpu_count: num_cpus::get(),
            total_memory: 0,
            available_memory: 0,
        }
    }
}

/// Resource constraints
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum CPU percentage (0.0 to 1.0)
    pub max_cpu_percent: f64,

    /// Maximum memory usage (bytes)
    pub max_memory_bytes: u64,

    /// Maximum disk usage (bytes)
    pub max_disk_bytes: u64,

    /// Maximum network bandwidth (bytes per second)
    pub max_network_bps: u64,

    /// Monitoring overhead limit (0.0 to 1.0)
    pub max_monitoring_overhead: f64,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_percent: 0.1,  // 10% CPU limit
            max_memory_bytes: 1024 * 1024 * 1024,  // 1GB memory limit
            max_disk_bytes: 10 * 1024 * 1024 * 1024,  // 10GB disk limit
            max_network_bps: 100 * 1024 * 1024,  // 100 Mbps network limit
            max_monitoring_overhead: 0.05,  // 5% overhead limit
        }
    }
}

/// Session priority levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SessionPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Session statistics and operational metrics
#[derive(Debug, Clone)]
pub struct SessionStatistics {
    /// Session creation time
    pub created_at: SystemTime,

    /// Last activity timestamp
    pub last_activity: SystemTime,

    /// Last status change timestamp
    pub last_status_change: SystemTime,

    /// Total operations performed
    pub total_operations: u64,

    /// Error count
    pub error_count: u64,

    /// Warning count
    pub warning_count: u64,

    /// Average operation latency
    pub avg_latency: Duration,

    /// Peak memory usage (bytes)
    pub peak_memory_usage: u64,

    /// Peak CPU usage (percentage)
    pub peak_cpu_usage: f64,

    /// Data volume processed (bytes)
    pub data_volume: u64,

    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,

    /// Efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

impl SessionStatistics {
    /// Create new session statistics
    pub fn new(created_at: SystemTime) -> Self {
        Self {
            created_at,
            last_activity: created_at,
            last_status_change: created_at,
            total_operations: 0,
            error_count: 0,
            warning_count: 0,
            avg_latency: Duration::from_millis(0),
            peak_memory_usage: 0,
            peak_cpu_usage: 0.0,
            data_volume: 0,
            cache_hit_rate: 0.0,
            efficiency_score: 1.0,
        }
    }

    /// Update operation statistics
    pub fn update_operation(&mut self, latency: Duration, data_size: u64) {
        self.total_operations += 1;
        self.last_activity = SystemTime::now();
        self.data_volume += data_size;

        // Update average latency
        let current_avg_ms = self.avg_latency.as_millis() as f64;
        let new_latency_ms = latency.as_millis() as f64;
        let operations = self.total_operations as f64;

        let new_avg_ms = ((current_avg_ms * (operations - 1.0)) + new_latency_ms) / operations;
        self.avg_latency = Duration::from_millis(new_avg_ms as u64);
    }

    /// Record error
    pub fn record_error(&mut self) {
        self.error_count += 1;
        self.last_activity = SystemTime::now();
    }

    /// Record warning
    pub fn record_warning(&mut self) {
        self.warning_count += 1;
        self.last_activity = SystemTime::now();
    }

    /// Update resource usage peaks
    pub fn update_resource_usage(&mut self, memory_usage: u64, cpu_usage: f64) {
        if memory_usage > self.peak_memory_usage {
            self.peak_memory_usage = memory_usage;
        }

        if cpu_usage > self.peak_cpu_usage {
            self.peak_cpu_usage = cpu_usage;
        }
    }

    /// Calculate operations per second
    pub fn operations_per_second(&self) -> f64 {
        let duration = self.last_activity.duration_since(self.created_at)
            .unwrap_or(Duration::from_secs(1));
        let seconds = duration.as_secs_f64();

        if seconds > 0.0 {
            self.total_operations as f64 / seconds
        } else {
            0.0
        }
    }

    /// Calculate error rate
    pub fn error_rate(&self) -> f64 {
        if self.total_operations > 0 {
            self.error_count as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }
}

/// Configuration change tracking
#[derive(Debug, Clone)]
pub struct ConfigurationChange {
    /// Change timestamp
    pub timestamp: SystemTime,

    /// Previous configuration
    pub old_config: MonitoringConfig,

    /// New configuration
    pub new_config: MonitoringConfig,

    /// Reason for change
    pub change_reason: String,

    /// Who made the change
    pub changed_by: String,
}

impl ConfigurationChange {
    /// Create initial configuration change
    pub fn initial(config: MonitoringConfig) -> Self {
        Self {
            timestamp: SystemTime::now(),
            old_config: MonitoringConfig::default(),
            new_config: config,
            change_reason: "Initial configuration".to_string(),
            changed_by: "system".to_string(),
        }
    }
}

/// Internal session state for monitoring operations
#[derive(Debug)]
pub struct MonitoringSessionState {
    /// Public session information
    pub session: MonitoringSession,

    /// Collected metrics for this session
    pub collected_metrics: Vec<PerformanceMetric>,

    /// Recorded events for this session
    pub recorded_events: Vec<TaskExecutionEvent>,

    /// Alert state tracking
    pub alert_state: HashMap<String, AlertState>,

    /// Current health state
    pub health_state: SystemHealth,

    /// Resource utilization history
    pub resource_history: Vec<ResourceUtilization>,

    /// Session locks for thread safety
    pub locks: SessionLocks,

    /// Performance counters
    pub counters: PerformanceCounters,
}

impl MonitoringSessionState {
    /// Create new session state
    pub fn new(session: MonitoringSession, health_state: SystemHealth) -> Self {
        Self {
            session,
            collected_metrics: Vec::new(),
            recorded_events: Vec::new(),
            alert_state: HashMap::new(),
            health_state,
            resource_history: Vec::new(),
            locks: SessionLocks::new(),
            counters: PerformanceCounters::new(),
        }
    }

    /// Add metric to session
    pub fn add_metric(&mut self, metric: PerformanceMetric) {
        self.collected_metrics.push(metric);
        self.session.increment_metrics(1);
        self.counters.metrics_added += 1;
    }

    /// Add event to session
    pub fn add_event(&mut self, event: TaskExecutionEvent) {
        self.recorded_events.push(event);
        self.session.increment_events(1);
        self.counters.events_added += 1;
    }

    /// Update resource utilization
    pub fn update_resource_utilization(&mut self, utilization: ResourceUtilization) {
        self.resource_history.push(utilization.clone());

        // Update session statistics
        if let Some(memory) = utilization.memory_usage_bytes {
            if let Some(cpu) = utilization.cpu_usage_percent {
                self.session.statistics.update_resource_usage(memory, cpu);
            }
        }
    }

    /// Get session duration
    pub fn duration(&self) -> Duration {
        self.session.duration()
    }

    /// Check if session is within resource limits
    pub fn within_resource_limits(&self) -> bool {
        let constraints = &self.session.metadata.execution_context.resource_constraints;

        // Check memory limit
        if self.session.statistics.peak_memory_usage > constraints.max_memory_bytes {
            return false;
        }

        // Check CPU limit
        if self.session.statistics.peak_cpu_usage > constraints.max_cpu_percent {
            return false;
        }

        true
    }
}

/// Session locks for thread-safe operations
#[derive(Debug)]
pub struct SessionLocks {
    /// Metrics lock
    pub metrics_lock: Arc<Mutex<()>>,

    /// Events lock
    pub events_lock: Arc<Mutex<()>>,

    /// Configuration lock
    pub config_lock: Arc<RwLock<()>>,
}

impl SessionLocks {
    fn new() -> Self {
        Self {
            metrics_lock: Arc::new(Mutex::new(())),
            events_lock: Arc::new(Mutex::new(())),
            config_lock: Arc::new(RwLock::new(())),
        }
    }
}

/// Performance counters for session operations
#[derive(Debug, Clone)]
pub struct PerformanceCounters {
    /// Number of metrics added
    pub metrics_added: u64,

    /// Number of events added
    pub events_added: u64,

    /// Number of alerts triggered
    pub alerts_triggered: u64,

    /// Number of configuration changes
    pub config_changes: u64,

    /// Number of health checks performed
    pub health_checks: u64,

    /// Number of resource updates
    pub resource_updates: u64,
}

impl PerformanceCounters {
    fn new() -> Self {
        Self {
            metrics_added: 0,
            events_added: 0,
            alerts_triggered: 0,
            config_changes: 0,
            health_checks: 0,
            resource_updates: 0,
        }
    }
}

/// Session manager for handling multiple monitoring sessions
#[derive(Debug)]
pub struct SessionManager {
    /// Active sessions
    sessions: HashMap<String, MonitoringSessionState>,

    /// Session creation lock
    creation_lock: Arc<Mutex<()>>,

    /// Global session limits
    limits: SessionLimits,

    /// Session cleanup configuration
    cleanup_config: SessionCleanupConfig,
}

impl SessionManager {
    /// Create new session manager
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            creation_lock: Arc::new(Mutex::new(())),
            limits: SessionLimits::default(),
            cleanup_config: SessionCleanupConfig::default(),
        }
    }

    /// Create new monitoring session
    pub fn create_session(&mut self, session_id: String, config: MonitoringConfig) -> SklResult<MonitoringSession> {
        let _lock = self.creation_lock.lock().unwrap();

        // Check session limits
        if self.sessions.len() >= self.limits.max_concurrent_sessions {
            return Err(SklearsError::ResourceExhausted(
                "Maximum number of concurrent sessions reached".to_string()
            ));
        }

        // Check if session already exists
        if self.sessions.contains_key(&session_id) {
            return Err(SklearsError::InvalidInput(
                format!("Session '{}' already exists", session_id)
            ));
        }

        // Create new session
        let session = MonitoringSession::new(session_id.clone(), config);
        let health_state = SystemHealth {
            status: crate::monitoring_core::HealthStatus::Healthy,
            components: HashMap::new(),
            score: 1.0,
            issues: Vec::new(),
        };

        let session_state = MonitoringSessionState::new(session.clone(), health_state);
        self.sessions.insert(session_id.clone(), session_state);

        Ok(session)
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: &str) -> Option<&MonitoringSessionState> {
        self.sessions.get(session_id)
    }

    /// Get mutable session by ID
    pub fn get_session_mut(&mut self, session_id: &str) -> Option<&mut MonitoringSessionState> {
        self.sessions.get_mut(session_id)
    }

    /// Remove session
    pub fn remove_session(&mut self, session_id: &str) -> Option<MonitoringSessionState> {
        self.sessions.remove(session_id)
    }

    /// List all session IDs
    pub fn list_sessions(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }

    /// Get session count
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Cleanup inactive sessions
    pub fn cleanup_sessions(&mut self) -> usize {
        let mut removed_count = 0;
        let now = SystemTime::now();

        self.sessions.retain(|_session_id, state| {
            let should_remove = match state.session.status {
                MonitoringSessionStatus::Stopped => {
                    // Remove stopped sessions after retention period
                    if let Some(end_time) = state.session.end_time {
                        let age = now.duration_since(end_time).unwrap_or(Duration::from_secs(0));
                        if age > self.cleanup_config.stopped_session_retention {
                            removed_count += 1;
                            return false;
                        }
                    }
                    true
                }
                MonitoringSessionStatus::Failed { .. } => {
                    // Remove failed sessions after retention period
                    let age = now.duration_since(state.session.start_time).unwrap_or(Duration::from_secs(0));
                    if age > self.cleanup_config.failed_session_retention {
                        removed_count += 1;
                        return false;
                    }
                    true
                }
                _ => true,
            };

            should_remove
        });

        removed_count
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Session limits configuration
#[derive(Debug, Clone)]
pub struct SessionLimits {
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,

    /// Maximum session duration
    pub max_session_duration: Duration,

    /// Maximum metrics per session
    pub max_metrics_per_session: usize,

    /// Maximum events per session
    pub max_events_per_session: usize,

    /// Maximum memory per session (bytes)
    pub max_memory_per_session: u64,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_concurrent_sessions: 100,
            max_session_duration: Duration::from_secs(24 * 3600), // 24 hours
            max_metrics_per_session: 1_000_000,
            max_events_per_session: 100_000,
            max_memory_per_session: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Session cleanup configuration
#[derive(Debug, Clone)]
pub struct SessionCleanupConfig {
    /// Retention period for stopped sessions
    pub stopped_session_retention: Duration,

    /// Retention period for failed sessions
    pub failed_session_retention: Duration,

    /// Cleanup interval
    pub cleanup_interval: Duration,

    /// Enable automatic cleanup
    pub enable_auto_cleanup: bool,
}

impl Default for SessionCleanupConfig {
    fn default() -> Self {
        Self {
            stopped_session_retention: Duration::from_secs(3600), // 1 hour
            failed_session_retention: Duration::from_secs(7200), // 2 hours
            cleanup_interval: Duration::from_secs(600), // 10 minutes
            enable_auto_cleanup: true,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_session_creation() {
        let config = MonitoringConfig::default();
        let session = MonitoringSession::new("test_session".to_string(), config);

        assert_eq!(session.session_id, "test_session");
        assert!(session.is_active() == false); // Starts in Starting state
        assert_eq!(session.metrics_collected, 0);
        assert_eq!(session.events_recorded, 0);
    }

    #[test]
    fn test_session_duration() {
        let config = MonitoringConfig::default();
        let session = MonitoringSession::new("test_session".to_string(), config);

        let duration = session.duration();
        assert!(duration.as_millis() < 100); // Should be very small
    }

    #[test]
    fn test_session_metadata() {
        let mut metadata = SessionMetadata::new("test_session".to_string());

        metadata.set_env("key".to_string(), "value".to_string());
        assert_eq!(metadata.environment.get("key"), Some(&"value".to_string()));

        metadata.add_dependency("dep1".to_string());
        assert!(metadata.dependencies.contains(&"dep1".to_string()));
    }

    #[test]
    fn test_session_statistics() {
        let start_time = SystemTime::now();
        let mut stats = SessionStatistics::new(start_time);

        stats.update_operation(Duration::from_millis(100), 1024);
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.data_volume, 1024);

        stats.record_error();
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.error_rate(), 1.0);
    }

    #[test]
    fn test_session_manager() {
        let mut manager = SessionManager::new();
        let config = MonitoringConfig::default();

        // Create session
        let session = manager.create_session("test_session".to_string(), config.clone()).unwrap();
        assert_eq!(session.session_id, "test_session");
        assert_eq!(manager.session_count(), 1);

        // Try to create duplicate session
        let result = manager.create_session("test_session".to_string(), config);
        assert!(result.is_err());

        // List sessions
        let sessions = manager.list_sessions();
        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains(&"test_session".to_string()));
    }

    #[test]
    fn test_session_state() {
        let config = MonitoringConfig::default();
        let session = MonitoringSession::new("test_session".to_string(), config);
        let health_state = SystemHealth {
            status: crate::monitoring_core::HealthStatus::Healthy,
            components: HashMap::new(),
            score: 1.0,
            issues: Vec::new(),
        };

        let mut session_state = MonitoringSessionState::new(session, health_state);

        // Test resource limits
        assert!(session_state.within_resource_limits());

        // Update resource usage beyond limits
        session_state.session.statistics.peak_memory_usage = u64::MAX;
        assert!(!session_state.within_resource_limits());
    }

    #[test]
    fn test_configuration_change_tracking() {
        let config1 = MonitoringConfig::default();
        let config2 = MonitoringConfig::default();

        let mut session = MonitoringSession::new("test_session".to_string(), config1);
        assert_eq!(session.config_history.len(), 1);

        session.update_config(config2);
        assert_eq!(session.config_history.len(), 2);
    }
}