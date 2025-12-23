//! Load Balancing Module
//!
//! This module provides comprehensive load balancing capabilities including
//! multiple distribution algorithms, performance-based routing, health integration,
//! and adaptive load balancing for notification channel optimization.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};

/// Performance load balancer
#[derive(Debug, Clone)]
pub struct PerformanceLoadBalancer {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Performance weights
    pub weights: HashMap<String, f64>,
    /// Load distribution
    pub distribution: LoadDistribution,
    /// Balancer configuration
    pub config: LoadBalancerConfig,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    PerformanceBased,
    AdaptivePerformance,
    Custom(String),
}

/// Load distribution tracking
#[derive(Debug, Clone)]
pub struct LoadDistribution {
    /// Current loads
    pub current_loads: HashMap<String, f64>,
    /// Load history
    pub load_history: HashMap<String, VecDeque<f64>>,
    /// Load predictions
    pub predictions: HashMap<String, f64>,
    /// Distribution efficiency
    pub efficiency: f64,
}

/// Load balancer configuration
#[derive(Debug, Clone)]
pub struct LoadBalancerConfig {
    /// Enable load balancing
    pub enabled: bool,
    /// Rebalancing threshold
    pub rebalancing_threshold: f64,
    /// Rebalancing interval
    pub rebalancing_interval: Duration,
    /// Health check integration
    pub health_check_integration: bool,
    /// Performance monitoring integration
    pub performance_monitoring: bool,
}

/// Load balancing request
#[derive(Debug, Clone)]
pub struct LoadBalancingRequest {
    /// Request identifier
    pub request_id: String,
    /// Available targets
    pub targets: Vec<LoadBalancingTarget>,
    /// Request priority
    pub priority: RequestPriority,
    /// Request characteristics
    pub characteristics: RequestCharacteristics,
    /// Constraints
    pub constraints: LoadBalancingConstraints,
}

/// Load balancing target
#[derive(Debug, Clone)]
pub struct LoadBalancingTarget {
    /// Target identifier
    pub target_id: String,
    /// Target name
    pub name: String,
    /// Target capacity
    pub capacity: TargetCapacity,
    /// Current performance metrics
    pub performance: TargetPerformance,
    /// Health status
    pub health: TargetHealth,
    /// Target weight
    pub weight: f64,
}

/// Target capacity information
#[derive(Debug, Clone)]
pub struct TargetCapacity {
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Current active requests
    pub current_requests: usize,
    /// CPU capacity
    pub cpu_capacity: f64,
    /// Memory capacity
    pub memory_capacity: f64,
    /// Network capacity
    pub network_capacity: f64,
}

/// Target performance metrics
#[derive(Debug, Clone)]
pub struct TargetPerformance {
    /// Average response time
    pub avg_response_time: Duration,
    /// Request success rate
    pub success_rate: f64,
    /// Throughput (requests/second)
    pub throughput: f64,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Performance score
    pub performance_score: f64,
}

/// Target health status
#[derive(Debug, Clone)]
pub struct TargetHealth {
    /// Is target healthy
    pub healthy: bool,
    /// Health score (0.0 to 1.0)
    pub health_score: f64,
    /// Last health check
    pub last_check: SystemTime,
    /// Health details
    pub details: String,
}

/// Request priority levels
#[derive(Debug, Clone)]
pub enum RequestPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Request characteristics
#[derive(Debug, Clone)]
pub struct RequestCharacteristics {
    /// Expected processing time
    pub expected_processing_time: Option<Duration>,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Request size
    pub request_size: usize,
    /// Request type
    pub request_type: String,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// CPU requirement
    pub cpu: f64,
    /// Memory requirement
    pub memory: f64,
    /// Network bandwidth requirement
    pub network: f64,
    /// Storage requirement
    pub storage: f64,
}

/// Load balancing constraints
#[derive(Debug, Clone)]
pub struct LoadBalancingConstraints {
    /// Excluded targets
    pub excluded_targets: Vec<String>,
    /// Preferred targets
    pub preferred_targets: Vec<String>,
    /// Maximum response time
    pub max_response_time: Option<Duration>,
    /// Minimum health score
    pub min_health_score: Option<f64>,
}

/// Load balancing result
#[derive(Debug, Clone)]
pub struct LoadBalancingResult {
    /// Request identifier
    pub request_id: String,
    /// Selected target
    pub selected_target: String,
    /// Selection reasoning
    pub reasoning: SelectionReasoning,
    /// Predicted performance
    pub predicted_performance: PredictedPerformance,
    /// Load balancing metrics
    pub metrics: LoadBalancingMetrics,
}

/// Selection reasoning
#[derive(Debug, Clone)]
pub struct SelectionReasoning {
    /// Selection strategy used
    pub strategy: LoadBalancingStrategy,
    /// Selection factors
    pub factors: HashMap<String, f64>,
    /// Decision confidence
    pub confidence: f64,
    /// Alternative targets considered
    pub alternatives: Vec<String>,
}

/// Predicted performance for selected target
#[derive(Debug, Clone)]
pub struct PredictedPerformance {
    /// Predicted response time
    pub response_time: Duration,
    /// Predicted success probability
    pub success_probability: f64,
    /// Predicted resource utilization
    pub resource_utilization: f64,
    /// Performance confidence
    pub confidence: f64,
}

/// Load balancing metrics
#[derive(Debug, Clone)]
pub struct LoadBalancingMetrics {
    /// Selection time
    pub selection_time: Duration,
    /// Targets evaluated
    pub targets_evaluated: usize,
    /// Load distribution variance
    pub load_variance: f64,
    /// Overall efficiency
    pub efficiency: f64,
}

/// Round-robin state for round-robin strategies
#[derive(Debug, Clone)]
pub struct RoundRobinState {
    /// Current index
    pub current_index: usize,
    /// Target list
    pub targets: Vec<String>,
    /// Last update time
    pub last_updated: SystemTime,
}

impl PerformanceLoadBalancer {
    /// Create a new performance load balancer
    pub fn new() -> Self {
        Self {
            strategy: LoadBalancingStrategy::PerformanceBased,
            weights: HashMap::new(),
            distribution: LoadDistribution::default(),
            config: LoadBalancerConfig::default(),
        }
    }

    /// Create with custom strategy
    pub fn with_strategy(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            weights: HashMap::new(),
            distribution: LoadDistribution::default(),
            config: LoadBalancerConfig::default(),
        }
    }

    /// Process load balancing request
    pub fn process_request(&mut self, request: LoadBalancingRequest) -> Result<LoadBalancingResult, String> {
        if !self.config.enabled {
            return Err("Load balancing is disabled".to_string());
        }

        let start_time = SystemTime::now();

        // Filter targets based on constraints
        let eligible_targets = self.filter_targets(&request.targets, &request.constraints)?;

        if eligible_targets.is_empty() {
            return Err("No eligible targets available".to_string());
        }

        // Select target based on strategy
        let selected_target = self.select_target(&eligible_targets, &request)?;

        // Update load distribution
        self.update_load_distribution(&selected_target.target_id, 1.0);

        // Calculate metrics
        let selection_time = start_time.elapsed().unwrap_or(Duration::from_millis(0));
        let load_variance = self.calculate_load_variance();

        Ok(LoadBalancingResult {
            request_id: request.request_id,
            selected_target: selected_target.target_id.clone(),
            reasoning: SelectionReasoning {
                strategy: self.strategy.clone(),
                factors: self.calculate_selection_factors(&selected_target),
                confidence: self.calculate_selection_confidence(&selected_target, &eligible_targets),
                alternatives: eligible_targets.iter().take(3).map(|t| t.target_id.clone()).collect(),
            },
            predicted_performance: self.predict_performance(&selected_target, &request.characteristics),
            metrics: LoadBalancingMetrics {
                selection_time,
                targets_evaluated: eligible_targets.len(),
                load_variance,
                efficiency: self.distribution.efficiency,
            },
        })
    }

    /// Select target based on current strategy
    pub fn select_target(&self, targets: &[LoadBalancingTarget], request: &LoadBalancingRequest) -> Result<LoadBalancingTarget, String> {
        if targets.is_empty() {
            return Err("No targets available".to_string());
        }

        match &self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(targets),
            LoadBalancingStrategy::WeightedRoundRobin => self.select_weighted_round_robin(targets),
            LoadBalancingStrategy::LeastConnections => self.select_least_connections(targets),
            LoadBalancingStrategy::PerformanceBased => self.select_performance_based(targets, request),
            LoadBalancingStrategy::AdaptivePerformance => self.select_adaptive_performance(targets, request),
            LoadBalancingStrategy::Custom(name) => self.select_custom(targets, request, name),
        }
    }

    /// Update target weights
    pub fn update_weights(&mut self, weights: HashMap<String, f64>) {
        self.weights = weights;
    }

    /// Set target weight
    pub fn set_target_weight(&mut self, target_id: String, weight: f64) {
        self.weights.insert(target_id, weight);
    }

    /// Get target weight
    pub fn get_target_weight(&self, target_id: &str) -> f64 {
        self.weights.get(target_id).copied().unwrap_or(1.0)
    }

    /// Update load distribution
    pub fn update_load_distribution(&mut self, target_id: &str, load_increment: f64) {
        // Update current load
        let current_load = self.distribution.current_loads.get(target_id).copied().unwrap_or(0.0);
        self.distribution.current_loads.insert(target_id.to_string(), current_load + load_increment);

        // Update load history
        let history = self.distribution.load_history.entry(target_id.to_string()).or_insert_with(VecDeque::new);
        history.push_back(current_load + load_increment);

        // Limit history size
        if history.len() > 100 {
            history.pop_front();
        }

        // Update efficiency
        self.update_distribution_efficiency();
    }

    /// Rebalance loads if necessary
    pub fn rebalance_if_needed(&mut self) -> Result<Vec<RebalancingAction>, String> {
        if !self.should_rebalance() {
            return Ok(Vec::new());
        }

        self.perform_rebalancing()
    }

    /// Get load distribution statistics
    pub fn get_distribution_stats(&self) -> LoadDistributionStats {
        let loads: Vec<f64> = self.distribution.current_loads.values().copied().collect();

        let total_load: f64 = loads.iter().sum();
        let avg_load = if loads.is_empty() { 0.0 } else { total_load / loads.len() as f64 };
        let max_load = loads.iter().copied().fold(0.0, f64::max);
        let min_load = loads.iter().copied().fold(f64::MAX, f64::min);

        let variance = if loads.len() > 1 {
            loads.iter().map(|&x| (x - avg_load).powi(2)).sum::<f64>() / loads.len() as f64
        } else {
            0.0
        };

        LoadDistributionStats {
            total_targets: self.distribution.current_loads.len(),
            total_load,
            average_load: avg_load,
            max_load,
            min_load,
            load_variance: variance,
            efficiency: self.distribution.efficiency,
        }
    }

    fn filter_targets(&self, targets: &[LoadBalancingTarget], constraints: &LoadBalancingConstraints) -> Result<Vec<LoadBalancingTarget>, String> {
        let mut eligible_targets = Vec::new();

        for target in targets {
            // Check if target is excluded
            if constraints.excluded_targets.contains(&target.target_id) {
                continue;
            }

            // Check health requirements
            if let Some(min_health) = constraints.min_health_score {
                if target.health.health_score < min_health {
                    continue;
                }
            }

            // Check if target is healthy
            if !target.health.healthy {
                continue;
            }

            // Check capacity
            if target.capacity.current_requests >= target.capacity.max_concurrent_requests {
                continue;
            }

            eligible_targets.push(target.clone());
        }

        Ok(eligible_targets)
    }

    fn select_round_robin(&self, targets: &[LoadBalancingTarget]) -> Result<LoadBalancingTarget, String> {
        // Simple round-robin - in practice would maintain state
        let index = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs() as usize % targets.len();
        Ok(targets[index].clone())
    }

    fn select_weighted_round_robin(&self, targets: &[LoadBalancingTarget]) -> Result<LoadBalancingTarget, String> {
        let mut best_target = None;
        let mut best_score = f64::NEG_INFINITY;

        for target in targets {
            let weight = self.get_target_weight(&target.target_id);
            let current_load = self.distribution.current_loads.get(&target.target_id).copied().unwrap_or(0.0);
            let score = weight / (current_load + 1.0); // Avoid division by zero

            if score > best_score {
                best_score = score;
                best_target = Some(target.clone());
            }
        }

        best_target.ok_or_else(|| "No suitable target found".to_string())
    }

    fn select_least_connections(&self, targets: &[LoadBalancingTarget]) -> Result<LoadBalancingTarget, String> {
        targets.iter()
            .min_by_key(|target| target.capacity.current_requests)
            .cloned()
            .ok_or_else(|| "No suitable target found".to_string())
    }

    fn select_performance_based(&self, targets: &[LoadBalancingTarget], _request: &LoadBalancingRequest) -> Result<LoadBalancingTarget, String> {
        targets.iter()
            .max_by(|a, b| a.performance.performance_score.partial_cmp(&b.performance.performance_score).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .ok_or_else(|| "No suitable target found".to_string())
    }

    fn select_adaptive_performance(&self, targets: &[LoadBalancingTarget], request: &LoadBalancingRequest) -> Result<LoadBalancingTarget, String> {
        let mut best_target = None;
        let mut best_score = f64::NEG_INFINITY;

        for target in targets {
            let performance_score = target.performance.performance_score;
            let health_score = target.health.health_score;
            let capacity_score = 1.0 - (target.capacity.current_requests as f64 / target.capacity.max_concurrent_requests as f64);
            let weight = self.get_target_weight(&target.target_id);

            // Priority adjustment
            let priority_multiplier = match request.priority {
                RequestPriority::Critical => 1.5,
                RequestPriority::High => 1.2,
                RequestPriority::Normal => 1.0,
                RequestPriority::Low => 0.8,
            };

            let combined_score = (performance_score * 0.4 + health_score * 0.3 + capacity_score * 0.3) * weight * priority_multiplier;

            if combined_score > best_score {
                best_score = combined_score;
                best_target = Some(target.clone());
            }
        }

        best_target.ok_or_else(|| "No suitable target found".to_string())
    }

    fn select_custom(&self, targets: &[LoadBalancingTarget], _request: &LoadBalancingRequest, _strategy_name: &str) -> Result<LoadBalancingTarget, String> {
        // Fallback to performance-based selection
        self.select_performance_based(targets, _request)
    }

    fn calculate_selection_factors(&self, target: &LoadBalancingTarget) -> HashMap<String, f64> {
        let mut factors = HashMap::new();
        factors.insert("performance_score".to_string(), target.performance.performance_score);
        factors.insert("health_score".to_string(), target.health.health_score);
        factors.insert("capacity_utilization".to_string(),
            target.capacity.current_requests as f64 / target.capacity.max_concurrent_requests as f64);
        factors.insert("weight".to_string(), self.get_target_weight(&target.target_id));
        factors
    }

    fn calculate_selection_confidence(&self, _selected: &LoadBalancingTarget, candidates: &[LoadBalancingTarget]) -> f64 {
        if candidates.len() <= 1 {
            return 1.0;
        }

        // Simple confidence calculation based on score differences
        let scores: Vec<f64> = candidates.iter().map(|t| t.performance.performance_score).collect();
        let max_score = scores.iter().copied().fold(0.0, f64::max);
        let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;

        if max_score > 0.0 {
            (max_score - avg_score) / max_score
        } else {
            0.5
        }
    }

    fn predict_performance(&self, target: &LoadBalancingTarget, _characteristics: &RequestCharacteristics) -> PredictedPerformance {
        // Calculate prediction confidence based on:
        // 1. Historical stability (variance in load history)
        // 2. Health score (unhealthy targets have lower confidence)
        // 3. Amount of historical data available
        let confidence = self.calculate_prediction_confidence(target);

        PredictedPerformance {
            response_time: target.performance.avg_response_time,
            success_probability: target.performance.success_rate,
            resource_utilization: target.performance.resource_utilization,
            confidence,
        }
    }

    fn calculate_prediction_confidence(&self, target: &LoadBalancingTarget) -> f64 {
        let mut confidence = 1.0;

        // Factor 1: Historical stability (lower variance = higher confidence)
        if let Some(history) = self.distribution.load_history.get(&target.target_id) {
            if history.len() > 1 {
                let history_vec: Vec<f64> = history.iter().copied().collect();
                let avg = history_vec.iter().sum::<f64>() / history_vec.len() as f64;
                let variance = history_vec.iter()
                    .map(|&x| (x - avg).powi(2))
                    .sum::<f64>() / history_vec.len() as f64;

                // Normalize variance contribution (higher variance = lower confidence)
                let stability_factor = if variance > 0.0 {
                    1.0 / (1.0 + variance)
                } else {
                    1.0
                };
                confidence *= stability_factor;

                // Factor 2: Data availability (more history = higher confidence)
                let data_factor = (history.len() as f64 / 10.0).min(1.0);
                confidence *= 0.7 + 0.3 * data_factor;
            } else {
                // Very little history available
                confidence *= 0.6;
            }
        } else {
            // No history available
            confidence *= 0.5;
        }

        // Factor 3: Health score (unhealthy targets have lower prediction confidence)
        confidence *= target.health.health_score;

        // Ensure confidence is in valid range [0.0, 1.0]
        confidence.clamp(0.0, 1.0)
    }

    fn calculate_load_variance(&self) -> f64 {
        let loads: Vec<f64> = self.distribution.current_loads.values().copied().collect();
        if loads.len() <= 1 {
            return 0.0;
        }

        let avg_load = loads.iter().sum::<f64>() / loads.len() as f64;
        loads.iter().map(|&x| (x - avg_load).powi(2)).sum::<f64>() / loads.len() as f64
    }

    fn update_distribution_efficiency(&mut self) {
        let variance = self.calculate_load_variance();
        // Lower variance means higher efficiency
        self.distribution.efficiency = if variance > 0.0 { 1.0 / (1.0 + variance) } else { 1.0 };
    }

    fn should_rebalance(&self) -> bool {
        let variance = self.calculate_load_variance();
        variance > self.config.rebalancing_threshold
    }

    fn perform_rebalancing(&mut self) -> Result<Vec<RebalancingAction>, String> {
        let mut actions = Vec::new();

        // Calculate average load
        let loads: Vec<(String, f64)> = self.distribution.current_loads
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        if loads.is_empty() {
            return Ok(actions);
        }

        let avg_load = loads.iter().map(|(_, load)| load).sum::<f64>() / loads.len() as f64;

        // Identify overloaded and underloaded targets
        let mut overloaded: Vec<(String, f64)> = loads.iter()
            .filter(|(_, load)| *load > avg_load * (1.0 + self.config.rebalancing_threshold))
            .map(|(id, load)| (id.clone(), *load))
            .collect();

        let mut underloaded: Vec<(String, f64)> = loads.iter()
            .filter(|(_, load)| *load < avg_load * (1.0 - self.config.rebalancing_threshold))
            .map(|(id, load)| (id.clone(), *load))
            .collect();

        // Sort by load difference from average (descending for overloaded, ascending for underloaded)
        overloaded.sort_by(|a, b| (b.1 - avg_load).partial_cmp(&(a.1 - avg_load)).unwrap());
        underloaded.sort_by(|a, b| (avg_load - a.1).partial_cmp(&(avg_load - b.1)).unwrap());

        // Create rebalancing actions by moving load from overloaded to underloaded targets
        for (source_id, source_load) in &overloaded {
            if underloaded.is_empty() {
                break;
            }

            let excess_load = source_load - avg_load;
            let (dest_id, dest_load) = underloaded.first().unwrap();
            let deficit_load = avg_load - dest_load;

            // Move the minimum of excess and deficit
            let load_to_move = excess_load.min(deficit_load);

            if load_to_move > 0.0 {
                // Determine priority based on magnitude of imbalance
                let priority = if load_to_move > avg_load * 0.5 {
                    ActionPriority::Urgent
                } else if load_to_move > avg_load * 0.3 {
                    ActionPriority::High
                } else if load_to_move > avg_load * 0.1 {
                    ActionPriority::Normal
                } else {
                    ActionPriority::Low
                };

                actions.push(RebalancingAction {
                    action_type: RebalancingActionType::MoveLoad,
                    source_target: source_id.clone(),
                    destination_target: dest_id.clone(),
                    load_amount: load_to_move,
                    priority,
                });

                // Update loads for next iteration
                if deficit_load <= excess_load {
                    // Underloaded target is now balanced, remove it
                    underloaded.remove(0);
                } else {
                    // Still underloaded, update its load
                    underloaded[0].1 += load_to_move;
                }
            }
        }

        Ok(actions)
    }
}

/// Load distribution statistics
#[derive(Debug, Clone)]
pub struct LoadDistributionStats {
    /// Total number of targets
    pub total_targets: usize,
    /// Total load across all targets
    pub total_load: f64,
    /// Average load per target
    pub average_load: f64,
    /// Maximum load on any target
    pub max_load: f64,
    /// Minimum load on any target
    pub min_load: f64,
    /// Load variance
    pub load_variance: f64,
    /// Distribution efficiency
    pub efficiency: f64,
}

/// Rebalancing action
#[derive(Debug, Clone)]
pub struct RebalancingAction {
    /// Action type
    pub action_type: RebalancingActionType,
    /// Source target
    pub source_target: String,
    /// Destination target
    pub destination_target: String,
    /// Load amount to move
    pub load_amount: f64,
    /// Action priority
    pub priority: ActionPriority,
}

/// Rebalancing action types
#[derive(Debug, Clone)]
pub enum RebalancingActionType {
    MoveLoad,
    AdjustWeight,
    AddTarget,
    RemoveTarget,
}

/// Action priority levels
#[derive(Debug, Clone)]
pub enum ActionPriority {
    Low,
    Normal,
    High,
    Urgent,
}

// Default implementations
impl Default for LoadDistribution {
    fn default() -> Self {
        Self {
            current_loads: HashMap::new(),
            load_history: HashMap::new(),
            predictions: HashMap::new(),
            efficiency: 1.0,
        }
    }
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rebalancing_threshold: 0.3, // 30% variance threshold
            rebalancing_interval: Duration::from_secs(60),
            health_check_integration: true,
            performance_monitoring: true,
        }
    }
}

impl Default for TargetCapacity {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 100,
            current_requests: 0,
            cpu_capacity: 1.0,
            memory_capacity: 1.0,
            network_capacity: 1.0,
        }
    }
}

impl Default for TargetPerformance {
    fn default() -> Self {
        Self {
            avg_response_time: Duration::from_millis(100),
            success_rate: 0.99,
            throughput: 100.0,
            resource_utilization: 0.5,
            performance_score: 0.8,
        }
    }
}

impl Default for TargetHealth {
    fn default() -> Self {
        Self {
            healthy: true,
            health_score: 1.0,
            last_check: SystemTime::now(),
            details: "Healthy".to_string(),
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu: 0.1,
            memory: 0.1,
            network: 0.1,
            storage: 0.1,
        }
    }
}