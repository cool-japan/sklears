//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use super::node_management::{NodeInfo, ResourceAllocation, AllocationPriority};

use super::types_3::{AffinityPreference, AllocatedResources, EnergyAwareOptimizer, LoadBalancingPerformanceMonitor, LoadBalancingSimdAccelerator, LoadPrediction, PerformanceMetrics, RebalancingOperation, ScalingAction, ScalingDecision, ScalingMetrics, UtilizationMetrics, WorkerStatus, WorkloadPortion};


#[derive(Debug)]
pub struct LoadTrends {
    pub trending_up: Vec<String>,
    pub trending_down: Vec<String>,
    pub stable_workers: Vec<String>,
}
/// Comprehensive optimization load balancer with SIMD acceleration
#[derive(Debug)]
pub struct OptimizationLoadBalancer {
    load_balancing_strategy: LoadBalancingStrategy,
    worker_loads: HashMap<String, WorkerLoad>,
    resource_allocations: HashMap<String, ResourceAllocation>,
    allocation_history: VecDeque<AllocationSnapshot>,
    load_threshold: f64,
    imbalance_tolerance: f64,
    simd_accelerator: Arc<Mutex<LoadBalancingSimdAccelerator>>,
    prediction_engine: Arc<Mutex<LoadPredictionEngine>>,
    adaptive_controller: Arc<Mutex<AdaptiveLoadController>>,
    resource_optimizer: Arc<Mutex<ResourceOptimizer>>,
    performance_monitor: Arc<Mutex<LoadBalancingPerformanceMonitor>>,
    auto_scaling_manager: Arc<Mutex<AutoScalingManager>>,
    energy_optimizer: Arc<Mutex<EnergyAwareOptimizer>>,
}
impl OptimizationLoadBalancer {
    pub fn new() -> Self {
        Self {
            load_balancing_strategy: LoadBalancingStrategy::WeightedRoundRobin,
            worker_loads: HashMap::new(),
            resource_allocations: HashMap::new(),
            allocation_history: VecDeque::new(),
            load_threshold: 0.8,
            imbalance_tolerance: 0.15,
            simd_accelerator: Arc::new(Mutex::new(LoadBalancingSimdAccelerator::new())),
            prediction_engine: Arc::new(Mutex::new(LoadPredictionEngine::new())),
            adaptive_controller: Arc::new(Mutex::new(AdaptiveLoadController::new())),
            resource_optimizer: Arc::new(Mutex::new(ResourceOptimizer::new())),
            performance_monitor: Arc::new(
                Mutex::new(LoadBalancingPerformanceMonitor::new()),
            ),
            auto_scaling_manager: Arc::new(Mutex::new(AutoScalingManager::new())),
            energy_optimizer: Arc::new(Mutex::new(EnergyAwareOptimizer::new())),
        }
    }
    /// Initialize load balancer with nodes
    pub fn initialize(&mut self, nodes: &[NodeInfo]) -> SklResult<()> {
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match simd_accelerator.initialize_worker_loads(nodes) {
            Ok(loads) => {
                self.worker_loads = loads;
            }
            Err(_) => {
                for node in nodes {
                    let worker_load = WorkerLoad::new(&node.node_id);
                    self.worker_loads.insert(node.node_id.clone(), worker_load);
                }
            }
        }
        {
            let mut prediction_engine = self
                .prediction_engine
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            prediction_engine.initialize_models(nodes)?;
        }
        {
            let mut adaptive_controller = self
                .adaptive_controller
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            adaptive_controller.initialize_control_parameters(nodes)?;
        }
        Ok(())
    }
    /// Distribute workload using SIMD-accelerated algorithms
    pub fn distribute_workload(
        &mut self,
        workload: &WorkloadDistributionRequest,
    ) -> SklResult<WorkloadDistributionPlan> {
        let strategy = {
            let adaptive_controller = self
                .adaptive_controller
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            adaptive_controller.select_optimal_strategy(&self.worker_loads, workload)?
        };
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let distribution_plan = match strategy {
            LoadBalancingStrategy::WeightedRoundRobin => {
                simd_accelerator
                    .weighted_round_robin_distribution(&self.worker_loads, workload)?
            }
            LoadBalancingStrategy::LeastConnectionsAdvanced => {
                simd_accelerator
                    .least_connections_distribution(&self.worker_loads, workload)?
            }
            LoadBalancingStrategy::CapacityAware => {
                simd_accelerator
                    .capacity_aware_distribution(&self.worker_loads, workload)?
            }
            LoadBalancingStrategy::PredictiveLoad => {
                let prediction_engine = self
                    .prediction_engine
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let predictions = prediction_engine
                    .predict_future_loads(&self.worker_loads, Duration::from_secs(300))?;
                simd_accelerator
                    .predictive_distribution(&self.worker_loads, workload, &predictions)?
            }
            LoadBalancingStrategy::EnergyAware => {
                let energy_optimizer = self
                    .energy_optimizer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let energy_costs = energy_optimizer
                    .calculate_energy_costs(&self.worker_loads)?;
                simd_accelerator
                    .energy_aware_distribution(
                        &self.worker_loads,
                        workload,
                        &energy_costs,
                    )?
            }
        };
        self.update_worker_loads_from_distribution(&distribution_plan)?;
        self.record_allocation_snapshot(&distribution_plan)?;
        Ok(distribution_plan)
    }
    /// Balance load across workers using SIMD acceleration
    pub fn load_balance(&mut self, worker_loads: &[WorkerLoad]) -> SklResult<()> {
        for load in worker_loads {
            self.worker_loads.insert(load.worker_id.clone(), load.clone());
        }
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let imbalance_metrics = simd_accelerator
            .calculate_load_imbalance(&self.worker_loads)?;
        if imbalance_metrics.max_imbalance > self.imbalance_tolerance {
            let rebalancing_plan = simd_accelerator
                .compute_rebalancing_plan(
                    &self.worker_loads,
                    &imbalance_metrics,
                    self.imbalance_tolerance,
                )?;
            self.execute_rebalancing_plan(rebalancing_plan)?;
        }
        {
            let mut monitor = self
                .performance_monitor
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            monitor.update_performance_metrics(&self.worker_loads, &imbalance_metrics)?;
        }
        Ok(())
    }
    /// Redistribute workload from failed node using SIMD optimization
    pub fn redistribute_from_failed_node(&mut self, failed_node: &str) -> SklResult<()> {
        let failed_workload = self
            .worker_loads
            .get(failed_node)
            .ok_or("Failed node not found in worker loads")?
            .clone();
        self.worker_loads.remove(failed_node);
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let redistribution_plan = simd_accelerator
            .compute_failure_redistribution(&self.worker_loads, &failed_workload)?;
        for redistribution in redistribution_plan.redistributions {
            if let Some(worker_load) = self
                .worker_loads
                .get_mut(&redistribution.target_worker)
            {
                worker_load.add_workload(&redistribution.workload_portion)?;
            }
        }
        {
            let mut auto_scaling = self
                .auto_scaling_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            auto_scaling
                .evaluate_scaling_needs(
                    &self.worker_loads,
                    ScalingTrigger::NodeFailure,
                )?;
        }
        Ok(())
    }
    /// Optimize resource allocation using SIMD algorithms
    pub fn optimize_resource_allocation(
        &mut self,
    ) -> SklResult<ResourceOptimizationResult> {
        let resource_optimizer = self
            .resource_optimizer
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let optimization_result = simd_accelerator
            .optimize_resource_distribution(
                &self.worker_loads,
                &self.resource_allocations,
                &resource_optimizer,
            )?;
        for (worker_id, optimized_allocation) in optimization_result
            .optimized_allocations
        {
            self.resource_allocations.insert(worker_id, optimized_allocation);
        }
        Ok(optimization_result)
    }
    /// Predict future load distribution using SIMD-accelerated ML models
    pub fn predict_future_load(
        &self,
        time_horizon: Duration,
    ) -> SklResult<LoadPrediction> {
        let prediction_engine = self
            .prediction_engine
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let load_trends = simd_accelerator
            .extract_load_trends(&self.worker_loads, &self.allocation_history)?;
        let future_prediction = prediction_engine
            .predict_load_distribution(load_trends, time_horizon)?;
        Ok(future_prediction)
    }
    /// Auto-scale cluster based on load patterns
    pub fn auto_scale_cluster(&mut self) -> SklResult<AutoScalingResult> {
        let mut auto_scaling = self
            .auto_scaling_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let scaling_metrics = simd_accelerator
            .analyze_scaling_requirements(&self.worker_loads)?;
        let scaling_decision = auto_scaling.make_scaling_decision(scaling_metrics)?;
        match scaling_decision.action {
            ScalingAction::ScaleUp => {
                let new_workers = auto_scaling
                    .provision_new_workers(scaling_decision.target_count)?;
                for worker in new_workers {
                    self.worker_loads.insert(worker.id, WorkerLoad::new(&worker.id));
                }
            }
            ScalingAction::ScaleDown => {
                let workers_to_remove = auto_scaling
                    .select_workers_for_removal(
                        &self.worker_loads,
                        scaling_decision.target_count,
                    )?;
                for worker_id in workers_to_remove {
                    self.redistribute_from_failed_node(&worker_id)?;
                }
            }
            ScalingAction::NoAction => {}
        }
        Ok(AutoScalingResult {
            action_taken: scaling_decision.action,
            workers_changed: scaling_decision.target_count,
            estimated_cost_impact: scaling_decision.cost_impact,
            performance_improvement: scaling_decision.performance_improvement,
        })
    }
    /// Get comprehensive load balancing metrics
    pub fn get_load_balancing_metrics(&self) -> SklResult<LoadBalancingMetrics> {
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let monitor = self.performance_monitor.lock().unwrap_or_else(|e| e.into_inner());
        let load_distribution = simd_accelerator
            .calculate_load_distribution_metrics(&self.worker_loads)?;
        let performance_metrics = monitor.get_current_performance_metrics();
        let resource_utilization = simd_accelerator
            .calculate_resource_utilization(
                &self.worker_loads,
                &self.resource_allocations,
            )?;
        Ok(LoadBalancingMetrics {
            load_distribution,
            performance_metrics,
            resource_utilization,
            total_workers: self.worker_loads.len(),
            active_allocations: self.resource_allocations.len(),
            load_balancing_quality: self.calculate_load_balancing_quality()?,
        })
    }
    /// Calculate overall load balancing quality
    fn calculate_load_balancing_quality(&self) -> SklResult<f64> {
        let simd_accelerator = self
            .simd_accelerator
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let quality_factors = simd_accelerator
            .calculate_quality_factors(&self.worker_loads)?;
        Ok(quality_factors.overall_quality)
    }
    /// Execute rebalancing plan
    fn execute_rebalancing_plan(&mut self, plan: RebalancingPlan) -> SklResult<()> {
        for operation in plan.operations {
            match operation.operation_type {
                RebalancingOperationType::MoveWorkload => {
                    self.move_workload_between_workers(
                        &operation.source_worker,
                        &operation.target_worker,
                        operation.workload_amount,
                    )?;
                }
                RebalancingOperationType::SplitWorkload => {
                    self.split_workload(
                        &operation.source_worker,
                        &operation.split_targets,
                        operation.workload_amount,
                    )?;
                }
                RebalancingOperationType::MergeWorkload => {
                    self.merge_workloads(
                        &operation.merge_sources,
                        &operation.target_worker,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// Move workload between workers
    fn move_workload_between_workers(
        &mut self,
        source: &str,
        target: &str,
        amount: f64,
    ) -> SklResult<()> {
        if let (Some(source_load), Some(target_load)) = (
            self.worker_loads.get_mut(source),
            self.worker_loads.get_mut(target),
        ) {
            source_load.reduce_workload(amount)?;
            target_load.increase_workload(amount)?;
        }
        Ok(())
    }
    /// Split workload among multiple workers
    fn split_workload(
        &mut self,
        source: &str,
        targets: &[String],
        total_amount: f64,
    ) -> SklResult<()> {
        let amount_per_target = total_amount / targets.len() as f64;
        if let Some(source_load) = self.worker_loads.get_mut(source) {
            source_load.reduce_workload(total_amount)?;
        }
        for target in targets {
            if let Some(target_load) = self.worker_loads.get_mut(target) {
                target_load.increase_workload(amount_per_target)?;
            }
        }
        Ok(())
    }
    /// Merge workloads from multiple sources
    fn merge_workloads(&mut self, sources: &[String], target: &str) -> SklResult<()> {
        let mut total_amount = 0.0;
        for source in sources {
            if let Some(source_load) = self.worker_loads.get_mut(source) {
                total_amount += source_load.current_load;
                source_load.reset_workload()?;
            }
        }
        if let Some(target_load) = self.worker_loads.get_mut(target) {
            target_load.increase_workload(total_amount)?;
        }
        Ok(())
    }
    /// Update worker loads from distribution plan
    fn update_worker_loads_from_distribution(
        &mut self,
        plan: &WorkloadDistributionPlan,
    ) -> SklResult<()> {
        for assignment in &plan.assignments {
            if let Some(worker_load) = self.worker_loads.get_mut(&assignment.worker_id) {
                worker_load.add_task_assignment(&assignment.task_assignment)?;
            }
        }
        Ok(())
    }
    /// Record allocation snapshot for analysis
    fn record_allocation_snapshot(
        &mut self,
        plan: &WorkloadDistributionPlan,
    ) -> SklResult<()> {
        let snapshot = AllocationSnapshot {
            timestamp: SystemTime::now(),
            worker_loads: self.worker_loads.clone(),
            distribution_plan: plan.clone(),
            load_imbalance: self.calculate_current_imbalance()?,
        };
        self.allocation_history.push_back(snapshot);
        if self.allocation_history.len() > 1000 {
            self.allocation_history.pop_front();
        }
        Ok(())
    }
    /// Calculate current load imbalance
    fn calculate_current_imbalance(&self) -> SklResult<f64> {
        if self.worker_loads.is_empty() {
            return Ok(0.0);
        }
        let loads: Vec<f64> = self
            .worker_loads
            .values()
            .map(|w| w.current_load)
            .collect();
        if loads.len() >= 8 {
            let simd_accelerator = self
                .simd_accelerator
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            simd_accelerator.calculate_load_variance(&loads)
        } else {
            let mean = loads.iter().sum::<f64>() / loads.len() as f64;
            let variance = loads.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / loads.len() as f64;
            Ok(variance.sqrt() / mean)
        }
    }
}
/// No-op resource optimizer placeholder (reserved for future SIMD-based allocation tuning).
#[derive(Debug)]
pub struct ResourceOptimizer;
impl ResourceOptimizer {
    pub fn new() -> Self {
        Self
    }
}
#[derive(Debug)]
pub struct LoadBalancingMetrics {
    pub load_distribution: LoadDistributionMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub resource_utilization: ResourceUtilizationMetrics,
    pub total_workers: usize,
    pub active_allocations: usize,
    pub load_balancing_quality: f64,
}
#[derive(Debug)]
pub struct ResourceOptimizationResult {
    pub optimized_allocations: HashMap<String, ResourceAllocation>,
    pub efficiency_improvement: f64,
    pub cost_reduction: f64,
}
#[derive(Debug)]
pub struct WorkerInfo {
    pub id: String,
    pub capacity: f64,
}
#[derive(Debug)]
pub struct LoadDistributionMetrics {
    pub distribution_evenness: f64,
    pub hotspot_count: usize,
    pub coldspot_count: usize,
}
/// Auto-scaling manager that makes data-driven scale-up / scale-down decisions.
///
/// Decision thresholds:
///   average_utilization > 0.80 → scale up by ceil(trending-up-fraction × current count)
///   average_utilization < 0.25 → scale down by floor(underutilized fraction × current count)
///   otherwise → no action
#[derive(Debug)]
pub struct AutoScalingManager {
    current_worker_count: usize,
    /// Average utilization threshold above which we scale up
    scale_up_threshold: f64,
    /// Average utilization threshold below which we scale down
    scale_down_threshold: f64,
}
impl AutoScalingManager {
    pub fn new() -> Self {
        Self {
            current_worker_count: 0,
            scale_up_threshold: 0.80,
            scale_down_threshold: 0.25,
        }
    }
    pub fn evaluate_scaling_needs(
        &mut self,
        workers: &HashMap<String, WorkerLoad>,
        _trigger: ScalingTrigger,
    ) -> SklResult<()> {
        self.current_worker_count = workers.len();
        Ok(())
    }
    /// Returns a scaling decision based on `ScalingMetrics`.
    pub fn make_scaling_decision(
        &self,
        metrics: ScalingMetrics,
    ) -> SklResult<ScalingDecision> {
        let avg = metrics.average_utilization.clamp(0.0, 1.0);
        if avg > self.scale_up_threshold {
            let additional = if self.current_worker_count > 0 {
                let target_count = ((self.current_worker_count as f64 * avg) / 0.70)
                    .ceil() as usize;
                target_count.saturating_sub(self.current_worker_count).max(1)
            } else {
                1
            };
            return Ok(ScalingDecision {
                action: ScalingAction::ScaleUp,
                target_count: self.current_worker_count + additional,
                cost_impact: additional as f64 * 0.15,
                performance_improvement: (avg - 0.70).max(0.0),
            });
        }
        if avg < self.scale_down_threshold && self.current_worker_count > 1 {
            let remove = ((self.current_worker_count as f64
                * (1.0 - avg / self.scale_down_threshold))
                .floor() as usize)
                .min(self.current_worker_count - 1);
            let target_count = self.current_worker_count.saturating_sub(remove);
            return Ok(ScalingDecision {
                action: ScalingAction::ScaleDown,
                target_count,
                cost_impact: -(remove as f64 * 0.15),
                performance_improvement: 0.0,
            });
        }
        Ok(ScalingDecision {
            action: ScalingAction::NoAction,
            target_count: self.current_worker_count,
            cost_impact: 0.0,
            performance_improvement: 0.0,
        })
    }
    /// Creates `count` synthetic `WorkerInfo` entries (real provisioning would call cloud APIs).
    pub fn provision_new_workers(&self, count: usize) -> SklResult<Vec<WorkerInfo>> {
        let workers = (0..count)
            .map(|i| WorkerInfo {
                id: format!("auto-worker-{}", i),
                capacity: 100.0,
            })
            .collect();
        Ok(workers)
    }
    /// Selects workers for removal — picks the `count` workers with lowest utilization.
    pub fn select_workers_for_removal(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        count: usize,
    ) -> SklResult<Vec<String>> {
        let mut sorted: Vec<(&String, f64)> = workers
            .iter()
            .map(|(id, w)| (id, w.load_percentage()))
            .collect();
        sorted
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let removed: Vec<String> = sorted
            .into_iter()
            .take(count)
            .map(|(id, _)| id.clone())
            .collect();
        Ok(removed)
    }
}
#[derive(Debug)]
pub struct ResourceUtilizationMetrics {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub network_utilization: f64,
    pub overall_efficiency: f64,
}
#[derive(Debug, PartialEq)]
pub enum WorkerHealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
}
#[derive(Debug)]
pub struct RebalancingPlan {
    pub plan_id: String,
    pub operations: Vec<RebalancingOperation>,
    pub estimated_improvement: f64,
    pub execution_time: Duration,
}
#[derive(Debug)]
pub enum RebalancingOperationType {
    MoveWorkload,
    SplitWorkload,
    MergeWorkload,
}
#[derive(Debug, Clone, PartialEq)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}
#[derive(Debug, Clone)]
pub struct LoadMeasurement {
    pub timestamp: SystemTime,
    pub load_value: f64,
    pub utilization_metrics: UtilizationMetrics,
}
#[derive(Debug)]
pub struct FailureRedistributionPlan {
    pub failed_worker: String,
    pub redistributions: Vec<WorkloadRedistribution>,
    pub total_redistributed_load: f64,
    pub success_probability: f64,
}
/// Individual worker assignment
#[derive(Debug, Clone)]
pub struct WorkerAssignment {
    pub worker_id: String,
    pub task_assignment: TaskAssignment,
    pub allocated_resources: AllocatedResources,
    pub estimated_execution_time: Duration,
    pub priority_boost: f64,
}
/// Load prediction engine using EWMA extrapolation per worker.
///
/// For each worker, an Exponentially Weighted Moving Average of recent load
/// samples is maintained and projected forward using a linear trend derived
/// from the last N measurements.  The horizon scales the projection.
///
/// OLS trend: given samples (x_0 … x_{n-1}), the slope is computed in O(n)
/// without any external dependency.
#[derive(Debug)]
pub struct LoadPredictionEngine {
    /// EWMA smoothing factor α
    alpha: f64,
    /// Cached EWMA per worker (updated on `predict_future_loads`)
    ewma_loads: HashMap<String, f64>,
    /// Number of samples used for linear trend estimation
    trend_window: usize,
}
impl LoadPredictionEngine {
    pub fn new() -> Self {
        Self {
            alpha: 0.3,
            ewma_loads: HashMap::new(),
            trend_window: 8,
        }
    }
    pub fn initialize_models(&mut self, nodes: &[NodeInfo]) -> SklResult<()> {
        for node in nodes {
            self.ewma_loads.entry(node.node_id.clone()).or_insert(0.0);
        }
        Ok(())
    }
    /// Predict future load for each worker given their recent `load_history`.
    ///
    /// Algorithm:
    ///  1. Compute EWMA of historical load samples.
    ///  2. Estimate linear slope via simple OLS on the last `trend_window` samples.
    ///  3. Project: predicted = ewma + slope × horizon_steps
    ///  4. Clamp to [0, max_capacity].
    pub fn predict_future_loads(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        horizon: Duration,
    ) -> SklResult<HashMap<String, f64>> {
        let horizon_steps = (horizon.as_secs_f64() / 60.0).max(1.0);
        let mut predictions = HashMap::with_capacity(workers.len());
        for (worker_id, worker) in workers {
            let history: Vec<f64> = worker
                .load_history
                .iter()
                .rev()
                .take(self.trend_window.max(2))
                .map(|m| m.load_value)
                .collect();
            if history.is_empty() {
                predictions.insert(worker_id.clone(), worker.current_load);
                continue;
            }
            let ewma = history
                .iter()
                .fold(
                    *history.last().unwrap_or(&worker.current_load),
                    |acc, &x| self.alpha * x + (1.0 - self.alpha) * acc,
                );
            let slope = if history.len() >= 2 {
                self.compute_slope(&history)
            } else {
                0.0
            };
            let predicted = (ewma + slope * horizon_steps)
                .clamp(0.0, worker.max_capacity);
            predictions.insert(worker_id.clone(), predicted);
        }
        Ok(predictions)
    }
    /// Predict aggregate load distribution given trends and horizon.
    pub fn predict_load_distribution(
        &self,
        trends: LoadTrends,
        horizon: Duration,
    ) -> SklResult<LoadPrediction> {
        let horizon_steps = (horizon.as_secs_f64() / 60.0).max(1.0);
        let mut predicted_loads = HashMap::new();
        for worker_id in &trends.trending_up {
            let cached = self.ewma_loads.get(worker_id).copied().unwrap_or(0.5);
            predicted_loads
                .insert(
                    worker_id.clone(),
                    (cached * (1.0 + 0.15 * horizon_steps)).clamp(0.0, 1.0),
                );
        }
        for worker_id in &trends.trending_down {
            let cached = self.ewma_loads.get(worker_id).copied().unwrap_or(0.5);
            predicted_loads
                .insert(
                    worker_id.clone(),
                    (cached * (1.0 - 0.10 * horizon_steps)).max(0.0),
                );
        }
        for worker_id in &trends.stable_workers {
            let cached = self.ewma_loads.get(worker_id).copied().unwrap_or(0.5);
            predicted_loads.insert(worker_id.clone(), cached);
        }
        let total_workers = trends.trending_up.len() + trends.trending_down.len()
            + trends.stable_workers.len();
        let trending_fraction = if total_workers > 0 {
            (trends.trending_up.len() + trends.trending_down.len()) as f64
                / total_workers as f64
        } else {
            0.0
        };
        let confidence = (1.0 - 0.4 * trending_fraction).clamp(0.5, 0.95);
        Ok(LoadPrediction {
            predicted_loads,
            confidence,
            horizon,
        })
    }
    /// Simple OLS slope: β₁ = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
    /// where x is sample index (0, 1, …, n-1) and y is the load value.
    fn compute_slope(&self, samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let sum_x: f64 = (0..samples.len()).map(|i| i as f64).sum();
        let sum_y: f64 = samples.iter().sum();
        let sum_xy: f64 = samples.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..samples.len()).map(|i| (i as f64).powi(2)).sum();
        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }
}
/// Comprehensive worker load tracking with SIMD optimization
#[derive(Debug, Clone)]
pub struct WorkerLoad {
    pub worker_id: String,
    pub current_load: f64,
    pub max_capacity: f64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub network_utilization: f64,
    pub active_tasks: Vec<TaskAssignment>,
    pub load_history: VecDeque<LoadMeasurement>,
    pub performance_score: f64,
    pub availability_score: f64,
    pub energy_consumption: f64,
    pub last_update: SystemTime,
}
impl WorkerLoad {
    pub fn new(worker_id: &str) -> Self {
        Self {
            worker_id: worker_id.to_string(),
            current_load: 0.0,
            max_capacity: 100.0,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            network_utilization: 0.0,
            active_tasks: Vec::new(),
            load_history: VecDeque::new(),
            performance_score: 1.0,
            availability_score: 1.0,
            energy_consumption: 0.0,
            last_update: SystemTime::now(),
        }
    }
    /// Calculate load percentage
    pub fn load_percentage(&self) -> f64 {
        if self.max_capacity > 0.0 {
            (self.current_load / self.max_capacity).min(1.0)
        } else {
            0.0
        }
    }
    /// Check if worker can accept additional load
    pub fn can_accept_load(&self, additional_load: f64) -> bool {
        (self.current_load + additional_load) <= self.max_capacity
            && self.availability_score > 0.8 && self.cpu_utilization < 0.9
            && self.memory_utilization < 0.9
    }
    /// Add workload to worker
    pub fn add_workload(&mut self, workload: &WorkloadPortion) -> SklResult<()> {
        if !self.can_accept_load(workload.computational_load) {
            return Err("Worker cannot accept additional workload".into());
        }
        self.current_load += workload.computational_load;
        self.active_tasks.extend(workload.tasks.clone());
        self.update_utilization_metrics()?;
        Ok(())
    }
    /// Add task assignment
    pub fn add_task_assignment(&mut self, assignment: &TaskAssignment) -> SklResult<()> {
        self.current_load += assignment.computational_cost;
        self.active_tasks.push(assignment.clone());
        self.update_utilization_metrics()?;
        Ok(())
    }
    /// Increase workload by amount
    pub fn increase_workload(&mut self, amount: f64) -> SklResult<()> {
        if !self.can_accept_load(amount) {
            return Err("Cannot increase workload beyond capacity".into());
        }
        self.current_load += amount;
        self.update_utilization_metrics()?;
        Ok(())
    }
    /// Reduce workload by amount
    pub fn reduce_workload(&mut self, amount: f64) -> SklResult<()> {
        self.current_load = (self.current_load - amount).max(0.0);
        self.update_utilization_metrics()?;
        Ok(())
    }
    /// Reset workload to zero
    pub fn reset_workload(&mut self) -> SklResult<()> {
        self.current_load = 0.0;
        self.active_tasks.clear();
        self.update_utilization_metrics()?;
        Ok(())
    }
    /// Update utilization metrics using SIMD if available
    fn update_utilization_metrics(&mut self) -> SklResult<()> {
        self.cpu_utilization = (self.current_load / self.max_capacity).min(1.0);
        self.memory_utilization = self.cpu_utilization * 0.8;
        self.network_utilization = self.cpu_utilization * 0.6;
        if self.load_history.len() >= 8 {
            let recent_loads: Vec<f64> = self
                .load_history
                .iter()
                .rev()
                .take(8)
                .map(|m| m.load_value)
                .collect();
            match simd_dot_product(
                &Array1::from(recent_loads.clone()),
                &Array1::ones(recent_loads.len()),
            ) {
                Ok(sum) => {
                    let avg_load = sum / recent_loads.len() as f64;
                    self.performance_score = 1.0 - avg_load;
                }
                Err(_) => {
                    let avg_load = recent_loads.iter().sum::<f64>()
                        / recent_loads.len() as f64;
                    self.performance_score = 1.0 - avg_load;
                }
            }
        } else {
            self.performance_score = 1.0 - self.load_percentage();
        }
        self.load_history
            .push_back(LoadMeasurement {
                timestamp: SystemTime::now(),
                load_value: self.current_load,
                utilization_metrics: UtilizationMetrics {
                    cpu: self.cpu_utilization,
                    memory: self.memory_utilization,
                    network: self.network_utilization,
                },
            });
        if self.load_history.len() > 100 {
            self.load_history.pop_front();
        }
        self.last_update = SystemTime::now();
        Ok(())
    }
    /// Get comprehensive worker status
    pub fn get_worker_status(&self) -> WorkerStatus {
        WorkerStatus {
            worker_id: self.worker_id.clone(),
            current_load_percentage: self.load_percentage(),
            is_available: self.availability_score > 0.8,
            is_overloaded: self.load_percentage() > 0.9,
            performance_score: self.performance_score,
            active_task_count: self.active_tasks.len(),
            estimated_remaining_capacity: self.max_capacity - self.current_load,
            health_status: if self.availability_score > 0.9 {
                WorkerHealthStatus::Excellent
            } else if self.availability_score > 0.7 {
                WorkerHealthStatus::Good
            } else if self.availability_score > 0.5 {
                WorkerHealthStatus::Fair
            } else {
                WorkerHealthStatus::Poor
            },
        }
    }
}
/// Workload distribution request
#[derive(Debug, Clone)]
pub struct WorkloadDistributionRequest {
    pub request_id: String,
    pub workload_type: WorkloadType,
    pub total_computational_load: f64,
    pub memory_requirements: f64,
    pub network_requirements: f64,
    pub priority: TaskPriority,
    pub deadline: Option<SystemTime>,
    pub parallelizable: bool,
    pub resource_constraints: Vec<ResourceConstraint>,
    pub affinity_preferences: Vec<AffinityPreference>,
}
/// Load balancing strategy enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    WeightedRoundRobin,
    LeastConnectionsAdvanced,
    CapacityAware,
    PredictiveLoad,
    EnergyAware,
}
/// Task assignment details
#[derive(Debug, Clone)]
pub struct TaskAssignment {
    pub task_id: String,
    pub task_type: String,
    pub computational_cost: f64,
    pub memory_requirement: f64,
    pub network_requirement: f64,
    pub estimated_duration: Duration,
    pub dependencies: Vec<String>,
}
#[derive(Debug, Clone)]
pub enum WorkloadType {
    Optimization,
    DataProcessing,
    MachineLearning,
    Simulation,
    Analytics,
}
#[derive(Debug)]
pub struct AutoScalingResult {
    pub action_taken: ScalingAction,
    pub workers_changed: usize,
    pub estimated_cost_impact: f64,
    pub performance_improvement: f64,
}
#[derive(Debug)]
pub enum ScalingTrigger {
    NodeFailure,
    HighLoad,
    LowLoad,
    Scheduled,
}
#[derive(Debug)]
pub enum ScalingRecommendation {
    ScaleUp,
    ScaleDown,
    NoAction,
}
#[derive(Debug)]
pub struct WorkloadRedistribution {
    pub target_worker: String,
    pub workload_portion: WorkloadPortion,
}
#[derive(Debug)]
pub struct ResourceConstraint {
    pub constraint_type: String,
    pub min_value: f64,
    pub max_value: f64,
}
/// Workload distribution plan
#[derive(Debug, Clone)]
pub struct WorkloadDistributionPlan {
    pub plan_id: String,
    pub request_id: String,
    pub assignments: Vec<WorkerAssignment>,
    pub total_workers_used: usize,
    pub estimated_completion_time: Duration,
    pub load_balance_quality: f64,
    pub resource_efficiency: f64,
    pub energy_efficiency: f64,
}
#[derive(Debug)]
pub struct AllocationSnapshot {
    pub timestamp: SystemTime,
    pub worker_loads: HashMap<String, WorkerLoad>,
    pub distribution_plan: WorkloadDistributionPlan,
    pub load_imbalance: f64,
}
/// Adaptive load controller that picks the best `LoadBalancingStrategy` for the
/// current system state.
///
/// Decision logic (UCB-inspired heuristic):
///  - If any worker is overloaded (>90% capacity) → `CapacityAware`
///  - If all workers are below 30% → `EnergyAware`
///  - If workload is `parallelizable` → `PredictiveLoad`
///  - Otherwise → `WeightedRoundRobin`
#[derive(Debug)]
pub struct AdaptiveLoadController {
    node_count: usize,
}
impl AdaptiveLoadController {
    pub fn new() -> Self {
        Self { node_count: 0 }
    }
    pub fn initialize_control_parameters(
        &mut self,
        nodes: &[NodeInfo],
    ) -> SklResult<()> {
        self.node_count = nodes.len();
        Ok(())
    }
    /// Selects the optimal load-balancing strategy based on current worker loads
    /// and workload characteristics.
    pub fn select_optimal_strategy(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        request: &WorkloadDistributionRequest,
    ) -> SklResult<LoadBalancingStrategy> {
        if workers.is_empty() {
            return Ok(LoadBalancingStrategy::WeightedRoundRobin);
        }
        let loads: Vec<f64> = workers.values().map(|w| w.load_percentage()).collect();
        let max_load = loads.iter().cloned().fold(0.0_f64, f64::max);
        let min_load = loads.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_load = loads.iter().sum::<f64>() / loads.len() as f64;
        if max_load > 0.90 {
            return Ok(LoadBalancingStrategy::CapacityAware);
        }
        if max_load < 0.30 {
            return Ok(LoadBalancingStrategy::EnergyAware);
        }
        if request.parallelizable && workers.len() >= 2 {
            return Ok(LoadBalancingStrategy::PredictiveLoad);
        }
        if mean_load > 0.0 {
            let variance = loads.iter().map(|&x| (x - mean_load).powi(2)).sum::<f64>()
                / loads.len() as f64;
            let cv = variance.sqrt() / mean_load;
            if cv > 0.30 {
                return Ok(LoadBalancingStrategy::LeastConnectionsAdvanced);
            }
        }
        let _ = min_load;
        Ok(LoadBalancingStrategy::WeightedRoundRobin)
    }
}
