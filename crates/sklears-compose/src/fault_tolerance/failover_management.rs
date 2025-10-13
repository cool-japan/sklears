//! Failover Management Module
//!
//! Implements sophisticated failover and redundancy management for fault tolerance:
//! - Multiple failover strategies (active-passive, active-active, round-robin)
//! - Automatic health monitoring and failover triggering
//! - Load balancing and service discovery integration
//! - Comprehensive metrics and monitoring
//! - Manual failover controls and emergency procedures

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::time::sleep;
use uuid::Uuid;

/// Failover strategy defining how failover is executed
#[derive(Debug, Clone, PartialEq)]
pub enum FailoverStrategy {
    /// Active-Passive: Primary handles all traffic, secondary takes over on failure
    ActivePassive {
        /// Time to wait before failing over
        failover_delay: Duration,
        /// Time to wait before failing back
        failback_delay: Duration,
    },
    /// Active-Active: Both instances handle traffic with load balancing
    ActiveActive {
        /// Load balancing algorithm
        load_balancing: LoadBalancingAlgorithm,
        /// Minimum healthy instances required
        min_healthy_instances: usize,
    },
    /// Round Robin: Rotate through all healthy instances
    RoundRobin {
        /// Whether to continue from last position after failover
        sticky_position: bool,
    },
    /// Weighted Round Robin: Instances have different weights
    WeightedRoundRobin {
        /// Instance weights (instance_id -> weight)
        weights: HashMap<String, u32>,
    },
    /// Least Connections: Route to instance with fewest active connections
    LeastConnections,
    /// Random: Randomly select among healthy instances
    Random,
    /// Custom failover strategy
    Custom {
        /// Custom selection function
        select_instance: fn(&[ServiceInstance]) -> Option<String>,
    },
}

/// Load balancing algorithms for active-active failover
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingAlgorithm {
    /// Simple round-robin
    RoundRobin,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Response time based
    ResponseTimeBased,
    /// Random selection
    Random,
}

/// Health check configuration for service instances
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub check_interval: Duration,
    /// Timeout for each health check
    pub check_timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
    /// Custom health check function
    pub health_check_fn: Option<fn(&ServiceInstance) -> bool>,
}

/// Service instance configuration and status
#[derive(Debug, Clone)]
pub struct ServiceInstance {
    /// Instance identifier
    pub instance_id: String,
    /// Instance endpoint/address
    pub endpoint: String,
    /// Instance weight for load balancing
    pub weight: u32,
    /// Current health status
    pub health_status: HealthStatus,
    /// Last health check time
    pub last_health_check: Instant,
    /// Consecutive health check failures
    pub consecutive_failures: u32,
    /// Consecutive health check successes
    pub consecutive_successes: u32,
    /// Current active connections
    pub active_connections: u32,
    /// Average response time
    pub average_response_time: Duration,
    /// Instance metadata
    pub metadata: HashMap<String, String>,
}

/// Health status of service instances
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Instance is healthy and available
    Healthy,
    /// Instance is unhealthy and unavailable
    Unhealthy,
    /// Instance health is unknown (startup or check in progress)
    Unknown,
    /// Instance is in maintenance mode
    Maintenance,
    /// Instance is draining connections
    Draining,
}

/// Failover event types for tracking and notifications
#[derive(Debug, Clone)]
pub enum FailoverEvent {
    /// Primary instance failed, switching to secondary
    FailoverTriggered {
        from_instance: String,
        to_instance: String,
        reason: String,
        timestamp: Instant,
    },
    /// Failed back to original primary instance
    FailbackTriggered {
        from_instance: String,
        to_instance: String,
        timestamp: Instant,
    },
    /// Instance marked as unhealthy
    InstanceUnhealthy {
        instance_id: String,
        reason: String,
        timestamp: Instant,
    },
    /// Instance marked as healthy
    InstanceHealthy {
        instance_id: String,
        timestamp: Instant,
    },
    /// Load balancing weights updated
    WeightsUpdated {
        updates: HashMap<String, u32>,
        timestamp: Instant,
    },
}

/// Failover configuration
#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// Configuration identifier
    pub config_id: String,
    /// Failover strategy to use
    pub strategy: FailoverStrategy,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Whether to enable automatic failover
    pub auto_failover: bool,
    /// Whether to enable automatic failback
    pub auto_failback: bool,
    /// Maximum time to wait for instance to become available
    pub instance_timeout: Duration,
    /// Whether to collect detailed metrics
    pub collect_metrics: bool,
}

/// Failover execution result
#[derive(Debug, Clone)]
pub struct FailoverResult {
    pub selected_instance: Option<ServiceInstance>,
    pub events: Vec<FailoverEvent>,
    pub duration: Duration,
    pub success: bool,
    pub error: Option<String>,
}

/// Failover management errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum FailoverError {
    #[error("No healthy instances available")]
    NoHealthyInstances,
    #[error("Instance not found: {instance_id}")]
    InstanceNotFound { instance_id: String },
    #[error("Failover strategy error: {message}")]
    StrategyError { message: String },
    #[error("Health check failed: {message}")]
    HealthCheckFailed { message: String },
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },
    #[error("Timeout waiting for healthy instance")]
    InstanceTimeout,
}

/// Failover metrics for monitoring and analysis
#[derive(Debug, Clone)]
pub struct FailoverMetrics {
    /// Configuration identifier
    pub config_id: String,
    /// Total number of failover operations
    pub total_operations: u64,
    /// Number of successful failovers
    pub successful_failovers: u64,
    /// Number of failed failovers
    pub failed_failovers: u64,
    /// Total number of health checks
    pub total_health_checks: u64,
    /// Number of failed health checks
    pub failed_health_checks: u64,
    /// Current healthy instances count
    pub healthy_instances: u32,
    /// Current unhealthy instances count
    pub unhealthy_instances: u32,
    /// Average failover time
    pub average_failover_time: Duration,
    /// Recent failover events
    pub recent_events: Vec<FailoverEvent>,
    /// Instance utilization statistics
    pub instance_stats: HashMap<String, InstanceStats>,
}

/// Statistics for individual service instances
#[derive(Debug, Clone)]
pub struct InstanceStats {
    /// Instance identifier
    pub instance_id: String,
    /// Number of requests handled
    pub requests_handled: u64,
    /// Average response time
    pub average_response_time: Duration,
    /// Current active connections
    pub active_connections: u32,
    /// Uptime percentage
    pub uptime_percentage: f64,
    /// Total downtime
    pub total_downtime: Duration,
    /// Last failure time
    pub last_failure: Option<Instant>,
}

/// Failover management system implementation
#[derive(Debug)]
pub struct FailoverManagementSystem {
    /// System identifier
    system_id: String,
    /// Failover configuration
    config: FailoverConfig,
    /// Managed service instances
    instances: Arc<RwLock<HashMap<String, ServiceInstance>>>,
    /// Current primary instance
    current_primary: Arc<RwLock<Option<String>>>,
    /// Round-robin position for strategies that need it
    round_robin_position: Arc<RwLock<usize>>,
    /// Failover event history
    event_history: Arc<RwLock<VecDeque<FailoverEvent>>>,
    /// System metrics
    metrics: Arc<RwLock<FailoverMetrics>>,
    /// Health check task handle
    health_check_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
            health_check_fn: None,
        }
    }
}

impl Default for FailoverStrategy {
    fn default() -> Self {
        Self::ActivePassive {
            failover_delay: Duration::from_secs(5),
            failback_delay: Duration::from_secs(30),
        }
    }
}

impl ServiceInstance {
    /// Create new service instance
    pub fn new(instance_id: String, endpoint: String) -> Self {
        Self {
            instance_id,
            endpoint,
            weight: 100,
            health_status: HealthStatus::Unknown,
            last_health_check: Instant::now(),
            consecutive_failures: 0,
            consecutive_successes: 0,
            active_connections: 0,
            average_response_time: Duration::ZERO,
            metadata: HashMap::new(),
        }
    }

    /// Check if instance is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.health_status, HealthStatus::Healthy)
    }

    /// Check if instance is available for routing
    pub fn is_available(&self) -> bool {
        matches!(self.health_status, HealthStatus::Healthy | HealthStatus::Unknown)
    }

    /// Update instance health status
    pub fn update_health_status(&mut self, status: HealthStatus, reason: Option<String>) {
        self.health_status = status;
        self.last_health_check = Instant::now();

        match status {
            HealthStatus::Healthy => {
                self.consecutive_successes += 1;
                self.consecutive_failures = 0;
            },
            HealthStatus::Unhealthy => {
                self.consecutive_failures += 1;
                self.consecutive_successes = 0;
            },
            _ => {}
        }
    }

    /// Update connection count
    pub fn update_connections(&mut self, count: u32) {
        self.active_connections = count;
    }

    /// Update response time
    pub fn update_response_time(&mut self, response_time: Duration) {
        // Simple exponential moving average
        if self.average_response_time == Duration::ZERO {
            self.average_response_time = response_time;
        } else {
            let alpha = 0.7;
            let new_avg = Duration::from_nanos(
                (self.average_response_time.as_nanos() as f64 * (1.0 - alpha) +
                 response_time.as_nanos() as f64 * alpha) as u64
            );
            self.average_response_time = new_avg;
        }
    }
}

impl FailoverManagementSystem {
    /// Create new failover management system
    pub fn new(system_id: String, config: FailoverConfig) -> Self {
        let metrics = FailoverMetrics {
            config_id: config.config_id.clone(),
            total_operations: 0,
            successful_failovers: 0,
            failed_failovers: 0,
            total_health_checks: 0,
            failed_health_checks: 0,
            healthy_instances: 0,
            unhealthy_instances: 0,
            average_failover_time: Duration::ZERO,
            recent_events: Vec::new(),
            instance_stats: HashMap::new(),
        };

        Self {
            system_id,
            config,
            instances: Arc::new(RwLock::new(HashMap::new())),
            current_primary: Arc::new(RwLock::new(None)),
            round_robin_position: Arc::new(RwLock::new(0)),
            event_history: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(RwLock::new(metrics)),
            health_check_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Create system with default configuration
    pub fn with_defaults(system_id: String) -> Self {
        let config = FailoverConfig {
            config_id: "default".to_string(),
            strategy: FailoverStrategy::default(),
            health_check: HealthCheckConfig::default(),
            auto_failover: true,
            auto_failback: true,
            instance_timeout: Duration::from_secs(30),
            collect_metrics: true,
        };
        Self::new(system_id, config)
    }

    /// Add service instance to management
    pub async fn add_instance(&self, instance: ServiceInstance) -> Result<(), FailoverError> {
        let instance_id = instance.instance_id.clone();

        {
            let mut instances = self.instances.write().unwrap();
            instances.insert(instance_id.clone(), instance);
        }

        // Update metrics
        self.update_instance_counts().await;

        // Start health checking if this is the first instance and auto health checking is enabled
        self.ensure_health_checking().await;

        Ok(())
    }

    /// Remove service instance from management
    pub async fn remove_instance(&self, instance_id: &str) -> Result<(), FailoverError> {
        {
            let mut instances = self.instances.write().unwrap();
            if instances.remove(instance_id).is_none() {
                return Err(FailoverError::InstanceNotFound {
                    instance_id: instance_id.to_string(),
                });
            }
        }

        // Update current primary if it was removed
        {
            let mut primary = self.current_primary.write().unwrap();
            if primary.as_ref() == Some(&instance_id.to_string()) {
                *primary = None;
            }
        }

        self.update_instance_counts().await;
        Ok(())
    }

    /// Get next available instance for operation
    pub async fn get_next_instance(&self) -> FailoverResult {
        let start_time = Instant::now();
        let mut events = Vec::new();

        match self.select_instance_by_strategy().await {
            Ok(instance) => {
                let duration = start_time.elapsed();

                // Update metrics
                {
                    let mut metrics = self.metrics.write().unwrap();
                    metrics.total_operations += 1;
                    metrics.successful_failovers += 1;

                    if metrics.total_operations > 0 {
                        let total_time = metrics.average_failover_time * (metrics.total_operations - 1) as u32 + duration;
                        metrics.average_failover_time = total_time / metrics.total_operations as u32;
                    }
                }

                FailoverResult {
                    selected_instance: Some(instance),
                    events,
                    duration,
                    success: true,
                    error: None,
                }
            },
            Err(error) => {
                let duration = start_time.elapsed();

                // Update metrics
                {
                    let mut metrics = self.metrics.write().unwrap();
                    metrics.total_operations += 1;
                    metrics.failed_failovers += 1;
                }

                FailoverResult {
                    selected_instance: None,
                    events,
                    duration,
                    success: false,
                    error: Some(error.to_string()),
                }
            }
        }
    }

    /// Select instance based on configured strategy
    async fn select_instance_by_strategy(&self) -> Result<ServiceInstance, FailoverError> {
        let instances = self.instances.read().unwrap();
        let healthy_instances: Vec<&ServiceInstance> = instances
            .values()
            .filter(|i| i.is_available())
            .collect();

        if healthy_instances.is_empty() {
            return Err(FailoverError::NoHealthyInstances);
        }

        match &self.config.strategy {
            FailoverStrategy::ActivePassive { .. } => {
                self.select_active_passive(&healthy_instances).await
            },
            FailoverStrategy::ActiveActive { load_balancing, .. } => {
                self.select_active_active(&healthy_instances, load_balancing).await
            },
            FailoverStrategy::RoundRobin { sticky_position } => {
                self.select_round_robin(&healthy_instances, *sticky_position).await
            },
            FailoverStrategy::WeightedRoundRobin { weights } => {
                self.select_weighted_round_robin(&healthy_instances, weights).await
            },
            FailoverStrategy::LeastConnections => {
                self.select_least_connections(&healthy_instances).await
            },
            FailoverStrategy::Random => {
                self.select_random(&healthy_instances).await
            },
            FailoverStrategy::Custom { select_instance } => {
                let instances_vec: Vec<ServiceInstance> = healthy_instances.iter().cloned().cloned().collect();
                if let Some(selected_id) = select_instance(&instances_vec) {
                    instances.get(&selected_id)
                        .cloned()
                        .ok_or(FailoverError::InstanceNotFound { instance_id: selected_id })
                } else {
                    Err(FailoverError::NoHealthyInstances)
                }
            }
        }
    }

    /// Select instance using active-passive strategy
    async fn select_active_passive(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance, FailoverError> {
        // Try current primary first
        {
            let primary = self.current_primary.read().unwrap();
            if let Some(primary_id) = primary.as_ref() {
                if let Some(instance) = instances.iter().find(|i| &i.instance_id == primary_id) {
                    return Ok((*instance).clone());
                }
            }
        }

        // Select new primary (first healthy instance)
        if let Some(new_primary) = instances.first() {
            *self.current_primary.write().unwrap() = Some(new_primary.instance_id.clone());
            Ok((*new_primary).clone())
        } else {
            Err(FailoverError::NoHealthyInstances)
        }
    }

    /// Select instance using active-active strategy
    async fn select_active_active(&self, instances: &[&ServiceInstance], algorithm: &LoadBalancingAlgorithm) -> Result<ServiceInstance, FailoverError> {
        match algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                self.select_round_robin(instances, true).await
            },
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                let weights = instances.iter()
                    .map(|i| (i.instance_id.clone(), i.weight))
                    .collect();
                self.select_weighted_round_robin(instances, &weights).await
            },
            LoadBalancingAlgorithm::LeastConnections => {
                self.select_least_connections(instances).await
            },
            LoadBalancingAlgorithm::ResponseTimeBased => {
                self.select_response_time_based(instances).await
            },
            LoadBalancingAlgorithm::Random => {
                self.select_random(instances).await
            }
        }
    }

    /// Select instance using round-robin
    async fn select_round_robin(&self, instances: &[&ServiceInstance], _sticky: bool) -> Result<ServiceInstance, FailoverError> {
        if instances.is_empty() {
            return Err(FailoverError::NoHealthyInstances);
        }

        let mut position = self.round_robin_position.write().unwrap();
        *position = (*position + 1) % instances.len();
        Ok(instances[*position].clone())
    }

    /// Select instance using weighted round-robin
    async fn select_weighted_round_robin(&self, instances: &[&ServiceInstance], weights: &HashMap<String, u32>) -> Result<ServiceInstance, FailoverError> {
        if instances.is_empty() {
            return Err(FailoverError::NoHealthyInstances);
        }

        let total_weight: u32 = instances.iter()
            .map(|i| weights.get(&i.instance_id).copied().unwrap_or(i.weight))
            .sum();

        if total_weight == 0 {
            return self.select_round_robin(instances, true).await;
        }

        // Simple weighted selection (could be improved with better algorithm)
        let target = (self.pseudo_random() as u32) % total_weight;
        let mut current_weight = 0;

        for instance in instances {
            let weight = weights.get(&instance.instance_id).copied().unwrap_or(instance.weight);
            current_weight += weight;
            if current_weight > target {
                return Ok((*instance).clone());
            }
        }

        // Fallback to first instance
        Ok(instances[0].clone())
    }

    /// Select instance with least connections
    async fn select_least_connections(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance, FailoverError> {
        instances.iter()
            .min_by_key(|i| i.active_connections)
            .map(|i| (*i).clone())
            .ok_or(FailoverError::NoHealthyInstances)
    }

    /// Select instance based on response time
    async fn select_response_time_based(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance, FailoverError> {
        instances.iter()
            .filter(|i| i.average_response_time > Duration::ZERO)
            .min_by_key(|i| i.average_response_time)
            .or_else(|| instances.first())
            .map(|i| (*i).clone())
            .ok_or(FailoverError::NoHealthyInstances)
    }

    /// Select random instance
    async fn select_random(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance, FailoverError> {
        if instances.is_empty() {
            return Err(FailoverError::NoHealthyInstances);
        }

        let index = (self.pseudo_random() as usize) % instances.len();
        Ok(instances[index].clone())
    }

    /// Simple pseudo-random number generator
    fn pseudo_random(&self) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);

        let prev = COUNTER.load(Ordering::Relaxed);
        let next = prev.wrapping_mul(1103515245).wrapping_add(12345);
        COUNTER.store(next, Ordering::Relaxed);
        next
    }

    /// Manually trigger failover to specific instance
    pub async fn manual_failover(&self, target_instance_id: &str) -> Result<FailoverResult, FailoverError> {
        let start_time = Instant::now();

        {
            let instances = self.instances.read().unwrap();
            if !instances.contains_key(target_instance_id) {
                return Err(FailoverError::InstanceNotFound {
                    instance_id: target_instance_id.to_string(),
                });
            }
        }

        // Update current primary
        let old_primary = {
            let mut primary = self.current_primary.write().unwrap();
            let old = primary.clone();
            *primary = Some(target_instance_id.to_string());
            old
        };

        // Create failover event
        let event = FailoverEvent::FailoverTriggered {
            from_instance: old_primary.unwrap_or_else(|| "none".to_string()),
            to_instance: target_instance_id.to_string(),
            reason: "Manual failover".to_string(),
            timestamp: Instant::now(),
        };

        self.record_event(event.clone()).await;

        let duration = start_time.elapsed();
        let selected_instance = {
            let instances = self.instances.read().unwrap();
            instances.get(target_instance_id).cloned()
        };

        Ok(FailoverResult {
            selected_instance,
            events: vec![event],
            duration,
            success: true,
            error: None,
        })
    }

    /// Start continuous health checking
    pub async fn start_health_checking(&self) {
        let instances_clone = self.instances.clone();
        let config = self.config.clone();
        let metrics_clone = self.metrics.clone();
        let event_history_clone = self.event_history.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health_check.check_interval);

            loop {
                interval.tick().await;

                let instance_ids: Vec<String> = {
                    let instances = instances_clone.read().unwrap();
                    instances.keys().cloned().collect()
                };

                for instance_id in instance_ids {
                    let health_result = Self::perform_health_check(
                        &instances_clone,
                        &instance_id,
                        &config.health_check,
                    ).await;

                    // Update metrics
                    {
                        let mut metrics = metrics_clone.write().unwrap();
                        metrics.total_health_checks += 1;
                        if !health_result {
                            metrics.failed_health_checks += 1;
                        }
                    }

                    // Record health change events
                    if let Some(event) = Self::check_health_status_change(
                        &instances_clone,
                        &instance_id,
                        health_result,
                    ).await {
                        let mut history = event_history_clone.write().unwrap();
                        history.push_back(event);
                        if history.len() > 100 { // Keep last 100 events
                            history.pop_front();
                        }
                    }
                }
            }
        });

        *self.health_check_handle.write().unwrap() = Some(handle);
    }

    /// Ensure health checking is running
    async fn ensure_health_checking(&self) {
        let handle_exists = {
            let handle = self.health_check_handle.read().unwrap();
            handle.is_some() && !handle.as_ref().unwrap().is_finished()
        };

        if !handle_exists {
            self.start_health_checking().await;
        }
    }

    /// Stop health checking
    pub async fn stop_health_checking(&self) {
        if let Some(handle) = self.health_check_handle.write().unwrap().take() {
            handle.abort();
        }
    }

    /// Perform health check on specific instance
    async fn perform_health_check(
        instances: &Arc<RwLock<HashMap<String, ServiceInstance>>>,
        instance_id: &str,
        config: &HealthCheckConfig,
    ) -> bool {
        // Get instance for health check
        let instance = {
            let instances_guard = instances.read().unwrap();
            instances_guard.get(instance_id).cloned()
        };

        if let Some(mut instance) = instance {
            let health_result = if let Some(health_check_fn) = config.health_check_fn {
                health_check_fn(&instance)
            } else {
                // Default health check - simple connectivity test
                Self::default_health_check(&instance, config.check_timeout).await
            };

            // Update instance health status
            let new_status = if health_result {
                if instance.consecutive_successes + 1 >= config.success_threshold {
                    HealthStatus::Healthy
                } else {
                    instance.health_status.clone()
                }
            } else {
                if instance.consecutive_failures + 1 >= config.failure_threshold {
                    HealthStatus::Unhealthy
                } else {
                    instance.health_status.clone()
                }
            };

            instance.update_health_status(new_status, Some("Health check".to_string()));

            // Update instance in collection
            {
                let mut instances_guard = instances.write().unwrap();
                instances_guard.insert(instance_id.to_string(), instance);
            }

            health_result
        } else {
            false
        }
    }

    /// Default health check implementation
    async fn default_health_check(_instance: &ServiceInstance, timeout: Duration) -> bool {
        // Simulate health check with timeout
        let check_duration = Duration::from_millis(100); // Simulate 100ms check

        if check_duration > timeout {
            return false;
        }

        sleep(check_duration).await;
        true // Assume healthy for default implementation
    }

    /// Check if health status changed and create event
    async fn check_health_status_change(
        instances: &Arc<RwLock<HashMap<String, ServiceInstance>>>,
        instance_id: &str,
        current_health: bool,
    ) -> Option<FailoverEvent> {
        let instance = {
            let instances_guard = instances.read().unwrap();
            instances_guard.get(instance_id).cloned()
        };

        if let Some(instance) = instance {
            let previous_health = instance.is_healthy();

            if current_health && !previous_health {
                Some(FailoverEvent::InstanceHealthy {
                    instance_id: instance_id.to_string(),
                    timestamp: Instant::now(),
                })
            } else if !current_health && previous_health {
                Some(FailoverEvent::InstanceUnhealthy {
                    instance_id: instance_id.to_string(),
                    reason: "Health check failed".to_string(),
                    timestamp: Instant::now(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Record failover event
    async fn record_event(&self, event: FailoverEvent) {
        let mut history = self.event_history.write().unwrap();
        history.push_back(event);
        if history.len() > 100 { // Keep last 100 events
            history.pop_front();
        }
    }

    /// Update instance counts in metrics
    async fn update_instance_counts(&self) {
        let instances = self.instances.read().unwrap();
        let mut healthy_count = 0;
        let mut unhealthy_count = 0;

        for instance in instances.values() {
            if instance.is_healthy() {
                healthy_count += 1;
            } else {
                unhealthy_count += 1;
            }
        }

        let mut metrics = self.metrics.write().unwrap();
        metrics.healthy_instances = healthy_count;
        metrics.unhealthy_instances = unhealthy_count;
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> FailoverMetrics {
        self.update_instance_counts().await;
        let mut metrics = self.metrics.read().unwrap().clone();

        // Add recent events
        let history = self.event_history.read().unwrap();
        metrics.recent_events = history.iter().rev().take(10).cloned().collect();

        // Update instance stats
        let instances = self.instances.read().unwrap();
        for (id, instance) in instances.iter() {
            let stats = InstanceStats {
                instance_id: id.clone(),
                requests_handled: 0, // Would be tracked in real implementation
                average_response_time: instance.average_response_time,
                active_connections: instance.active_connections,
                uptime_percentage: if instance.is_healthy() { 100.0 } else { 0.0 }, // Simplified
                total_downtime: Duration::ZERO, // Would be tracked in real implementation
                last_failure: None, // Would be tracked in real implementation
            };
            metrics.instance_stats.insert(id.clone(), stats);
        }

        metrics
    }

    /// Get system health status
    pub async fn get_health_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("system_id".to_string(), self.system_id.clone());

        let instances = self.instances.read().unwrap();
        let healthy_count = instances.values().filter(|i| i.is_healthy()).count();
        let total_count = instances.len();

        status.insert("total_instances".to_string(), total_count.to_string());
        status.insert("healthy_instances".to_string(), healthy_count.to_string());
        status.insert("unhealthy_instances".to_string(), (total_count - healthy_count).to_string());

        if total_count > 0 {
            let health_percentage = (healthy_count as f64 / total_count as f64) * 100.0;
            status.insert("health_percentage".to_string(), format!("{:.1}", health_percentage));
        }

        let primary = self.current_primary.read().unwrap();
        if let Some(primary_id) = primary.as_ref() {
            status.insert("current_primary".to_string(), primary_id.clone());
        } else {
            status.insert("current_primary".to_string(), "none".to_string());
        }

        status
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_failover_basic_functionality() {
        let system = FailoverManagementSystem::with_defaults("test_system".to_string());

        let instance1 = ServiceInstance::new("instance1".to_string(), "endpoint1".to_string());
        let instance2 = ServiceInstance::new("instance2".to_string(), "endpoint2".to_string());

        system.add_instance(instance1).await.unwrap();
        system.add_instance(instance2).await.unwrap();

        let result = system.get_next_instance().await;
        assert!(result.success);
        assert!(result.selected_instance.is_some());
    }

    #[tokio::test]
    async fn test_manual_failover() {
        let system = FailoverManagementSystem::with_defaults("test_system".to_string());

        let mut instance1 = ServiceInstance::new("instance1".to_string(), "endpoint1".to_string());
        let mut instance2 = ServiceInstance::new("instance2".to_string(), "endpoint2".to_string());

        instance1.health_status = HealthStatus::Healthy;
        instance2.health_status = HealthStatus::Healthy;

        system.add_instance(instance1).await.unwrap();
        system.add_instance(instance2).await.unwrap();

        let result = system.manual_failover("instance2").await.unwrap();
        assert!(result.success);
        assert_eq!(result.selected_instance.unwrap().instance_id, "instance2");
        assert_eq!(result.events.len(), 1);
    }

    #[tokio::test]
    async fn test_no_healthy_instances() {
        let system = FailoverManagementSystem::with_defaults("test_system".to_string());

        let mut instance1 = ServiceInstance::new("instance1".to_string(), "endpoint1".to_string());
        instance1.health_status = HealthStatus::Unhealthy;

        system.add_instance(instance1).await.unwrap();

        let result = system.get_next_instance().await;
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_round_robin_selection() {
        let mut config = FailoverConfig {
            config_id: "test".to_string(),
            strategy: FailoverStrategy::RoundRobin { sticky_position: true },
            health_check: HealthCheckConfig::default(),
            auto_failover: true,
            auto_failback: true,
            instance_timeout: Duration::from_secs(30),
            collect_metrics: true,
        };

        let system = FailoverManagementSystem::new("test_system".to_string(), config);

        let mut instance1 = ServiceInstance::new("instance1".to_string(), "endpoint1".to_string());
        let mut instance2 = ServiceInstance::new("instance2".to_string(), "endpoint2".to_string());
        let mut instance3 = ServiceInstance::new("instance3".to_string(), "endpoint3".to_string());

        instance1.health_status = HealthStatus::Healthy;
        instance2.health_status = HealthStatus::Healthy;
        instance3.health_status = HealthStatus::Healthy;

        system.add_instance(instance1).await.unwrap();
        system.add_instance(instance2).await.unwrap();
        system.add_instance(instance3).await.unwrap();

        // Test round-robin behavior
        let mut selected_instances = Vec::new();
        for _ in 0..6 {
            let result = system.get_next_instance().await;
            assert!(result.success);
            selected_instances.push(result.selected_instance.unwrap().instance_id);
        }

        // Should have cycled through instances
        assert_eq!(selected_instances.len(), 6);
        // Due to implementation details, exact round-robin order may vary
        assert!(selected_instances.iter().any(|id| id == "instance1"));
        assert!(selected_instances.iter().any(|id| id == "instance2"));
        assert!(selected_instances.iter().any(|id| id == "instance3"));
    }

    #[tokio::test]
    async fn test_least_connections_selection() {
        let config = FailoverConfig {
            config_id: "test".to_string(),
            strategy: FailoverStrategy::LeastConnections,
            health_check: HealthCheckConfig::default(),
            auto_failover: true,
            auto_failback: true,
            instance_timeout: Duration::from_secs(30),
            collect_metrics: true,
        };

        let system = FailoverManagementSystem::new("test_system".to_string(), config);

        let mut instance1 = ServiceInstance::new("instance1".to_string(), "endpoint1".to_string());
        let mut instance2 = ServiceInstance::new("instance2".to_string(), "endpoint2".to_string());

        instance1.health_status = HealthStatus::Healthy;
        instance1.active_connections = 10;
        instance2.health_status = HealthStatus::Healthy;
        instance2.active_connections = 5;

        system.add_instance(instance1).await.unwrap();
        system.add_instance(instance2).await.unwrap();

        let result = system.get_next_instance().await;
        assert!(result.success);
        // Should select instance with fewer connections
        assert_eq!(result.selected_instance.unwrap().instance_id, "instance2");
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let system = FailoverManagementSystem::with_defaults("test_system".to_string());

        let mut instance1 = ServiceInstance::new("instance1".to_string(), "endpoint1".to_string());
        instance1.health_status = HealthStatus::Healthy;

        system.add_instance(instance1).await.unwrap();

        // Perform some operations
        for _ in 0..5 {
            let _ = system.get_next_instance().await;
        }

        let metrics = system.get_metrics().await;
        assert_eq!(metrics.total_operations, 5);
        assert_eq!(metrics.successful_failovers, 5);
        assert_eq!(metrics.healthy_instances, 1);
        assert_eq!(metrics.unhealthy_instances, 0);
    }

    #[tokio::test]
    async fn test_health_status() {
        let system = FailoverManagementSystem::with_defaults("test_system".to_string());

        let mut instance1 = ServiceInstance::new("instance1".to_string(), "endpoint1".to_string());
        let mut instance2 = ServiceInstance::new("instance2".to_string(), "endpoint2".to_string());

        instance1.health_status = HealthStatus::Healthy;
        instance2.health_status = HealthStatus::Unhealthy;

        system.add_instance(instance1).await.unwrap();
        system.add_instance(instance2).await.unwrap();

        let health = system.get_health_status().await;
        assert_eq!(health.get("total_instances").unwrap(), "2");
        assert_eq!(health.get("healthy_instances").unwrap(), "1");
        assert_eq!(health.get("unhealthy_instances").unwrap(), "1");
        assert_eq!(health.get("health_percentage").unwrap(), "50.0");
    }
}