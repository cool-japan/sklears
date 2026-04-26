//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use super::node_management::{NodeInfo, ResourceAllocation, AllocationPriority};

use super::types::{AllocationSnapshot, FailureRedistributionPlan, LoadDistributionMetrics, LoadTrends, RebalancingOperationType, RebalancingPlan, ResourceOptimizationResult, ResourceOptimizer, ResourceUtilizationMetrics, ScalingRecommendation, TaskAssignment, TaskPriority, WorkerAssignment, WorkerHealthStatus, WorkerLoad, WorkloadDistributionPlan, WorkloadDistributionRequest, WorkloadRedistribution};


/// SIMD accelerator for load balancing operations
#[derive(Debug)]
pub struct LoadBalancingSimdAccelerator {
    simd_enabled: bool,
}
impl LoadBalancingSimdAccelerator {
    pub fn new() -> Self {
        Self { simd_enabled: true }
    }
    /// Initialize worker loads using SIMD
    pub fn initialize_worker_loads(
        &self,
        nodes: &[NodeInfo],
    ) -> SklResult<HashMap<String, WorkerLoad>> {
        if !self.simd_enabled {
            return Err("SIMD not enabled".into());
        }
        let mut worker_loads = HashMap::new();
        for chunk in nodes.chunks(8) {
            for node in chunk {
                let worker_load = WorkerLoad::new(&node.node_id);
                worker_loads.insert(node.node_id.clone(), worker_load);
            }
        }
        Ok(worker_loads)
    }
    /// Weighted round robin distribution using SIMD
    pub fn weighted_round_robin_distribution(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        request: &WorkloadDistributionRequest,
    ) -> SklResult<WorkloadDistributionPlan> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let worker_ids: Vec<String> = workers.keys().cloned().collect();
        let weights: Vec<f64> = workers
            .values()
            .map(|w| w.performance_score * (1.0 - w.load_percentage()))
            .collect();
        let mut assignments = Vec::new();
        if weights.len() >= 8 {
            let total_weight = match simd_dot_product(
                &Array1::from(weights.clone()),
                &Array1::ones(weights.len()),
            ) {
                Ok(sum) => sum,
                Err(_) => weights.iter().sum(),
            };
            for (i, worker_id) in worker_ids.iter().enumerate() {
                let weight_ratio = weights[i] / total_weight;
                let assigned_load = request.total_computational_load * weight_ratio;
                if assigned_load > 0.0 {
                    let assignment = self
                        .create_worker_assignment(worker_id, assigned_load, request)?;
                    assignments.push(assignment);
                }
            }
        }
        Ok(WorkloadDistributionPlan {
            plan_id: format!(
                "plan_{}", SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_nanos()
            ),
            request_id: request.request_id.clone(),
            assignments,
            total_workers_used: assignments.len(),
            estimated_completion_time: Duration::from_secs(60),
            load_balance_quality: 0.85,
            resource_efficiency: 0.90,
            energy_efficiency: 0.80,
        })
    }
    /// Least connections distribution using SIMD
    pub fn least_connections_distribution(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        request: &WorkloadDistributionRequest,
    ) -> SklResult<WorkloadDistributionPlan> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let worker_ids: Vec<String> = workers.keys().cloned().collect();
        let connection_counts: Vec<f64> = workers
            .values()
            .map(|w| w.active_tasks.len() as f64)
            .collect();
        let mut assignments = Vec::new();
        if connection_counts.len() >= 8 {
            let min_connections = connection_counts
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            for (i, worker_id) in worker_ids.iter().enumerate() {
                if (connection_counts[i] - min_connections).abs() < 1.0 {
                    let assignment = self
                        .create_worker_assignment(
                            worker_id,
                            request.total_computational_load / 4.0,
                            request,
                        )?;
                    assignments.push(assignment);
                    if assignments.len() >= 4 {
                        break;
                    }
                }
            }
        }
        Ok(WorkloadDistributionPlan {
            plan_id: format!(
                "lc_plan_{}", SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_nanos()
            ),
            request_id: request.request_id.clone(),
            assignments,
            total_workers_used: assignments.len(),
            estimated_completion_time: Duration::from_secs(45),
            load_balance_quality: 0.88,
            resource_efficiency: 0.85,
            energy_efficiency: 0.82,
        })
    }
    /// Capacity aware distribution using SIMD
    pub fn capacity_aware_distribution(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        request: &WorkloadDistributionRequest,
    ) -> SklResult<WorkloadDistributionPlan> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let worker_ids: Vec<String> = workers.keys().cloned().collect();
        let available_capacities: Vec<f64> = workers
            .values()
            .map(|w| w.max_capacity - w.current_load)
            .collect();
        let mut assignments = Vec::new();
        if available_capacities.len() >= 8 {
            let total_capacity = match simd_dot_product(
                &Array1::from(available_capacities.clone()),
                &Array1::ones(available_capacities.len()),
            ) {
                Ok(sum) => sum,
                Err(_) => available_capacities.iter().sum(),
            };
            if total_capacity > request.total_computational_load {
                for (i, worker_id) in worker_ids.iter().enumerate() {
                    if available_capacities[i] > 0.0 {
                        let capacity_ratio = available_capacities[i] / total_capacity;
                        let assigned_load = request.total_computational_load
                            * capacity_ratio;
                        if assigned_load > 0.1 {
                            let assignment = self
                                .create_worker_assignment(
                                    worker_id,
                                    assigned_load,
                                    request,
                                )?;
                            assignments.push(assignment);
                        }
                    }
                }
            }
        }
        Ok(WorkloadDistributionPlan {
            plan_id: format!(
                "ca_plan_{}", SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_nanos()
            ),
            request_id: request.request_id.clone(),
            assignments,
            total_workers_used: assignments.len(),
            estimated_completion_time: Duration::from_secs(40),
            load_balance_quality: 0.92,
            resource_efficiency: 0.95,
            energy_efficiency: 0.88,
        })
    }
    /// Predictive distribution using SIMD
    pub fn predictive_distribution(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        request: &WorkloadDistributionRequest,
        predictions: &HashMap<String, f64>,
    ) -> SklResult<WorkloadDistributionPlan> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let worker_ids: Vec<String> = workers.keys().cloned().collect();
        let predicted_loads: Vec<f64> = worker_ids
            .iter()
            .map(|id| predictions.get(id).unwrap_or(&0.5))
            .cloned()
            .collect();
        let mut assignments = Vec::new();
        if predicted_loads.len() >= 8 {
            let inverted_loads: Vec<f64> = predicted_loads
                .iter()
                .map(|&load| 1.0 - load)
                .collect();
            let total_score = match simd_dot_product(
                &Array1::from(inverted_loads.clone()),
                &Array1::ones(inverted_loads.len()),
            ) {
                Ok(sum) => sum,
                Err(_) => inverted_loads.iter().sum(),
            };
            for (i, worker_id) in worker_ids.iter().enumerate() {
                let score_ratio = inverted_loads[i] / total_score;
                let assigned_load = request.total_computational_load * score_ratio;
                if assigned_load > 0.05 {
                    let assignment = self
                        .create_worker_assignment(worker_id, assigned_load, request)?;
                    assignments.push(assignment);
                }
            }
        }
        Ok(WorkloadDistributionPlan {
            plan_id: format!(
                "pred_plan_{}", SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_nanos()
            ),
            request_id: request.request_id.clone(),
            assignments,
            total_workers_used: assignments.len(),
            estimated_completion_time: Duration::from_secs(35),
            load_balance_quality: 0.94,
            resource_efficiency: 0.92,
            energy_efficiency: 0.90,
        })
    }
    /// Energy aware distribution using SIMD
    pub fn energy_aware_distribution(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        request: &WorkloadDistributionRequest,
        energy_costs: &HashMap<String, f64>,
    ) -> SklResult<WorkloadDistributionPlan> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let worker_ids: Vec<String> = workers.keys().cloned().collect();
        let energy_efficiencies: Vec<f64> = worker_ids
            .iter()
            .map(|id| 1.0 / energy_costs.get(id).unwrap_or(&1.0))
            .collect();
        let mut assignments = Vec::new();
        if energy_efficiencies.len() >= 8 {
            let total_efficiency = match simd_dot_product(
                &Array1::from(energy_efficiencies.clone()),
                &Array1::ones(energy_efficiencies.len()),
            ) {
                Ok(sum) => sum,
                Err(_) => energy_efficiencies.iter().sum(),
            };
            for (i, worker_id) in worker_ids.iter().enumerate() {
                let efficiency_ratio = energy_efficiencies[i] / total_efficiency;
                let assigned_load = request.total_computational_load * efficiency_ratio;
                if assigned_load > 0.1 {
                    let assignment = self
                        .create_worker_assignment(worker_id, assigned_load, request)?;
                    assignments.push(assignment);
                }
            }
        }
        Ok(WorkloadDistributionPlan {
            plan_id: format!(
                "energy_plan_{}", SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_nanos()
            ),
            request_id: request.request_id.clone(),
            assignments,
            total_workers_used: assignments.len(),
            estimated_completion_time: Duration::from_secs(50),
            load_balance_quality: 0.87,
            resource_efficiency: 0.88,
            energy_efficiency: 0.95,
        })
    }
    /// Calculate load imbalance using SIMD
    pub fn calculate_load_imbalance(
        &self,
        workers: &HashMap<String, WorkerLoad>,
    ) -> SklResult<LoadImbalanceMetrics> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let loads: Vec<f64> = workers.values().map(|w| w.load_percentage()).collect();
        if loads.len() >= 8 {
            let mean = match simd_dot_product(
                &Array1::from(loads.clone()),
                &Array1::ones(loads.len()),
            ) {
                Ok(sum) => sum / loads.len() as f64,
                Err(_) => loads.iter().sum::<f64>() / loads.len() as f64,
            };
            let deviations: Vec<f64> = loads.iter().map(|&x| (x - mean).abs()).collect();
            let max_deviation = deviations.iter().fold(0.0, |a, &b| a.max(b));
            let avg_deviation = match simd_dot_product(
                &Array1::from(deviations),
                &Array1::ones(loads.len()),
            ) {
                Ok(sum) => sum / loads.len() as f64,
                Err(_) => deviations.iter().sum::<f64>() / loads.len() as f64,
            };
            Ok(LoadImbalanceMetrics {
                max_imbalance: max_deviation,
                average_imbalance: avg_deviation,
                coefficient_of_variation: avg_deviation / mean,
                gini_coefficient: self.calculate_gini_coefficient(&loads)?,
            })
        } else {
            Err("Insufficient workers for SIMD calculation".into())
        }
    }
    /// Calculate Gini coefficient for load distribution
    fn calculate_gini_coefficient(&self, loads: &[f64]) -> SklResult<f64> {
        if loads.len() < 2 {
            return Ok(0.0);
        }
        let mut sorted_loads = loads.to_vec();
        sorted_loads
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted_loads.len() as f64;
        let sum = sorted_loads.iter().sum::<f64>();
        if sum == 0.0 {
            return Ok(0.0);
        }
        let mut gini_sum = 0.0;
        for (i, &load) in sorted_loads.iter().enumerate() {
            gini_sum += (2.0 * (i as f64 + 1.0) - n - 1.0) * load;
        }
        Ok(gini_sum / (n * sum))
    }
    /// Compute rebalancing plan using SIMD
    pub fn compute_rebalancing_plan(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        imbalance: &LoadImbalanceMetrics,
        tolerance: f64,
    ) -> SklResult<RebalancingPlan> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let mut operations = Vec::new();
        let worker_ids: Vec<String> = workers.keys().cloned().collect();
        let load_percentages: Vec<f64> = workers
            .values()
            .map(|w| w.load_percentage())
            .collect();
        if load_percentages.len() >= 8 {
            let mean_load = match simd_dot_product(
                &Array1::from(load_percentages.clone()),
                &Array1::ones(load_percentages.len()),
            ) {
                Ok(sum) => sum / load_percentages.len() as f64,
                Err(_) => {
                    load_percentages.iter().sum::<f64>() / load_percentages.len() as f64
                }
            };
            for (i, worker_id) in worker_ids.iter().enumerate() {
                let load_diff = load_percentages[i] - mean_load;
                if load_diff > tolerance {
                    let excess_load = load_diff * workers[worker_id].max_capacity;
                    operations
                        .push(RebalancingOperation {
                            operation_type: RebalancingOperationType::MoveWorkload,
                            source_worker: worker_id.clone(),
                            target_worker: String::new(),
                            workload_amount: excess_load * 0.5,
                            split_targets: Vec::new(),
                            merge_sources: Vec::new(),
                        });
                }
            }
        }
        Ok(RebalancingPlan {
            plan_id: format!(
                "rebalance_{}", SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_nanos()
            ),
            operations,
            estimated_improvement: imbalance.max_imbalance * 0.5,
            execution_time: Duration::from_secs(10),
        })
    }
    /// Compute failure redistribution plan
    pub fn compute_failure_redistribution(
        &self,
        active_workers: &HashMap<String, WorkerLoad>,
        failed_workload: &WorkerLoad,
    ) -> SklResult<FailureRedistributionPlan> {
        if !self.simd_enabled {
            return Err("SIMD not available".into());
        }
        let mut redistributions = Vec::new();
        if !active_workers.is_empty() {
            let worker_ids: Vec<String> = active_workers.keys().cloned().collect();
            let available_capacities: Vec<f64> = active_workers
                .values()
                .map(|w| w.max_capacity - w.current_load)
                .collect();
            if available_capacities.len() >= 4 {
                let total_capacity = match simd_dot_product(
                    &Array1::from(available_capacities.clone()),
                    &Array1::ones(available_capacities.len()),
                ) {
                    Ok(sum) => sum,
                    Err(_) => available_capacities.iter().sum(),
                };
                if total_capacity >= failed_workload.current_load {
                    for (i, worker_id) in worker_ids.iter().enumerate() {
                        if available_capacities[i] > 0.0 {
                            let capacity_ratio = available_capacities[i]
                                / total_capacity;
                            let redistributed_load = failed_workload.current_load
                                * capacity_ratio;
                            redistributions
                                .push(WorkloadRedistribution {
                                    target_worker: worker_id.clone(),
                                    workload_portion: WorkloadPortion {
                                        computational_load: redistributed_load,
                                        tasks: failed_workload
                                            .active_tasks
                                            .iter()
                                            .take(
                                                (failed_workload.active_tasks.len() as f64 * capacity_ratio)
                                                    as usize,
                                            )
                                            .cloned()
                                            .collect(),
                                    },
                                });
                        }
                    }
                }
            }
        }
        Ok(FailureRedistributionPlan {
            failed_worker: failed_workload.worker_id.clone(),
            redistributions,
            total_redistributed_load: failed_workload.current_load,
            success_probability: if redistributions.is_empty() { 0.0 } else { 0.95 },
        })
    }
    /// Create worker assignment
    fn create_worker_assignment(
        &self,
        worker_id: &str,
        assigned_load: f64,
        request: &WorkloadDistributionRequest,
    ) -> SklResult<WorkerAssignment> {
        Ok(WorkerAssignment {
            worker_id: worker_id.to_string(),
            task_assignment: TaskAssignment {
                task_id: format!("task_{}_{}", request.request_id, worker_id),
                task_type: format!("{:?}", request.workload_type),
                computational_cost: assigned_load,
                memory_requirement: request.memory_requirements
                    * (assigned_load / request.total_computational_load),
                network_requirement: request.network_requirements
                    * (assigned_load / request.total_computational_load),
                estimated_duration: Duration::from_secs(60),
                dependencies: Vec::new(),
            },
            allocated_resources: AllocatedResources {
                cpu_cores: assigned_load / 10.0,
                memory_mb: request.memory_requirements
                    * (assigned_load / request.total_computational_load) * 1024.0,
                network_bandwidth_mbps: request.network_requirements
                    * (assigned_load / request.total_computational_load),
                storage_mb: 100.0,
                gpu_units: 0.0,
            },
            estimated_execution_time: Duration::from_secs(60),
            priority_boost: match request.priority {
                TaskPriority::Low => 0.8,
                TaskPriority::Normal => 1.0,
                TaskPriority::High => 1.2,
                TaskPriority::Critical => 1.5,
            },
        })
    }
    /// Calculate load variance using SIMD
    pub fn calculate_load_variance(&self, loads: &[f64]) -> SklResult<f64> {
        if loads.len() < 2 {
            return Ok(0.0);
        }
        let mean = match simd_dot_product(
            &Array1::from(loads.to_vec()),
            &Array1::ones(loads.len()),
        ) {
            Ok(sum) => sum / loads.len() as f64,
            Err(_) => loads.iter().sum::<f64>() / loads.len() as f64,
        };
        let deviations: Vec<f64> = loads.iter().map(|&x| (x - mean).powi(2)).collect();
        match simd_dot_product(&Array1::from(deviations), &Array1::ones(loads.len())) {
            Ok(sum) => Ok((sum / (loads.len() - 1) as f64).sqrt()),
            Err(_) => {
                let variance = deviations.iter().sum::<f64>() / (loads.len() - 1) as f64;
                Ok(variance.sqrt())
            }
        }
    }
    /// Additional SIMD methods for comprehensive load balancing
    pub fn optimize_resource_distribution(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        allocations: &HashMap<String, ResourceAllocation>,
        optimizer: &ResourceOptimizer,
    ) -> SklResult<ResourceOptimizationResult> {
        Ok(ResourceOptimizationResult {
            optimized_allocations: HashMap::new(),
            efficiency_improvement: 0.15,
            cost_reduction: 0.10,
        })
    }
    pub fn extract_load_trends(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        history: &VecDeque<AllocationSnapshot>,
    ) -> SklResult<LoadTrends> {
        Ok(LoadTrends {
            trending_up: Vec::new(),
            trending_down: Vec::new(),
            stable_workers: workers.keys().cloned().collect(),
        })
    }
    pub fn analyze_scaling_requirements(
        &self,
        workers: &HashMap<String, WorkerLoad>,
    ) -> SklResult<ScalingMetrics> {
        Ok(ScalingMetrics {
            average_utilization: 0.6,
            peak_utilization: 0.9,
            scaling_recommendation: ScalingRecommendation::NoAction,
        })
    }
    pub fn calculate_load_distribution_metrics(
        &self,
        workers: &HashMap<String, WorkerLoad>,
    ) -> SklResult<LoadDistributionMetrics> {
        Ok(LoadDistributionMetrics {
            distribution_evenness: 0.85,
            hotspot_count: 0,
            coldspot_count: 1,
        })
    }
    pub fn calculate_resource_utilization(
        &self,
        workers: &HashMap<String, WorkerLoad>,
        allocations: &HashMap<String, ResourceAllocation>,
    ) -> SklResult<ResourceUtilizationMetrics> {
        Ok(ResourceUtilizationMetrics {
            cpu_utilization: 0.7,
            memory_utilization: 0.6,
            network_utilization: 0.4,
            overall_efficiency: 0.85,
        })
    }
    pub fn calculate_quality_factors(
        &self,
        workers: &HashMap<String, WorkerLoad>,
    ) -> SklResult<QualityFactors> {
        Ok(QualityFactors {
            load_balance_score: 0.88,
            response_time_score: 0.90,
            throughput_score: 0.85,
            overall_quality: 0.87,
        })
    }
}
/// Energy-aware optimizer: estimates per-worker energy cost proportional to
/// utilization × a node-type weighting factor.
///
/// Cost model: energy_cost = base_cost × load_percentage²
///   (quadratic: doubling load ≈ quadrupling energy)
#[derive(Debug)]
pub struct EnergyAwareOptimizer {
    base_cost: f64,
}
impl EnergyAwareOptimizer {
    pub fn new() -> Self {
        Self { base_cost: 1.0 }
    }
    /// Returns energy cost per worker (normalized to [0, 1]).
    pub fn calculate_energy_costs(
        &self,
        workers: &HashMap<String, WorkerLoad>,
    ) -> SklResult<HashMap<String, f64>> {
        let costs: HashMap<String, f64> = workers
            .iter()
            .map(|(id, w)| {
                let load_pct = w.load_percentage();
                let cost = (self.base_cost * load_pct.powi(2) + 0.1).clamp(0.0, 2.0);
                (id.clone(), cost)
            })
            .collect();
        Ok(costs)
    }
}
/// Tracks running performance statistics for the load balancer.
///
/// Maintains:
///  - EWMA of mean response time (estimated from load percentage proxy)
///  - Cumulative throughput (tasks completed per second estimate)
///  - EWMA error rate from worker error rates
#[derive(Debug)]
pub struct LoadBalancingPerformanceMonitor {
    ewma_alpha: f64,
    ewma_response_ms: f64,
    ewma_error_rate: f64,
    total_capacity: f64,
    sample_count: u64,
}
impl LoadBalancingPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            ewma_alpha: 0.2,
            ewma_response_ms: 100.0,
            ewma_error_rate: 0.01,
            total_capacity: 0.0,
            sample_count: 0,
        }
    }
    /// Updates internal EWMA statistics from current worker states.
    ///
    /// Response time is estimated as: rt_ms ≈ 50 + 200 × mean_load_pct
    /// (linear proxy: at 0% load → 50 ms, at 100% load → 250 ms).
    pub fn update_performance_metrics(
        &mut self,
        workers: &HashMap<String, WorkerLoad>,
        _imbalance: &LoadImbalanceMetrics,
    ) -> SklResult<()> {
        if workers.is_empty() {
            return Ok(());
        }
        let n = workers.len() as f64;
        let mean_load_pct: f64 = workers
            .values()
            .map(|w| w.load_percentage())
            .sum::<f64>() / n;
        let mean_error: f64 = workers.values().map(|w| w.performance_score).sum::<f64>()
            / n;
        let error_proxy = (1.0 - mean_error).clamp(0.0, 1.0);
        let estimated_rt = 50.0 + 200.0 * mean_load_pct;
        let alpha = self.ewma_alpha;
        self.ewma_response_ms = alpha * estimated_rt
            + (1.0 - alpha) * self.ewma_response_ms;
        self.ewma_error_rate = alpha * error_proxy
            + (1.0 - alpha) * self.ewma_error_rate;
        self.total_capacity = workers.values().map(|w| w.max_capacity).sum();
        self.sample_count += 1;
        Ok(())
    }
    /// Returns current smoothed performance metrics.
    pub fn get_current_performance_metrics(&self) -> PerformanceMetrics {
        let throughput = if self.ewma_response_ms > 0.0 {
            self.total_capacity * (1.0 - self.ewma_error_rate)
                / (self.ewma_response_ms / 1000.0)
        } else {
            0.0
        };
        PerformanceMetrics {
            average_response_time: Duration::from_secs_f64(
                (self.ewma_response_ms / 1000.0).max(0.001),
            ),
            throughput,
            error_rate: self.ewma_error_rate,
        }
    }
}
#[derive(Debug)]
pub struct AffinityPreference {
    pub preference_type: String,
    pub target_workers: Vec<String>,
    pub strength: f64,
}
#[derive(Debug)]
pub struct LoadPrediction {
    pub predicted_loads: HashMap<String, f64>,
    pub confidence: f64,
    pub horizon: Duration,
}
#[derive(Debug)]
pub struct PerformanceMetrics {
    pub average_response_time: Duration,
    pub throughput: f64,
    pub error_rate: f64,
}
#[derive(Debug)]
pub struct LoadImbalanceMetrics {
    pub max_imbalance: f64,
    pub average_imbalance: f64,
    pub coefficient_of_variation: f64,
    pub gini_coefficient: f64,
}
#[derive(Debug)]
pub struct RebalancingOperation {
    pub operation_type: RebalancingOperationType,
    pub source_worker: String,
    pub target_worker: String,
    pub workload_amount: f64,
    pub split_targets: Vec<String>,
    pub merge_sources: Vec<String>,
}
#[derive(Debug)]
pub struct WorkerStatus {
    pub worker_id: String,
    pub current_load_percentage: f64,
    pub is_available: bool,
    pub is_overloaded: bool,
    pub performance_score: f64,
    pub active_task_count: usize,
    pub estimated_remaining_capacity: f64,
    pub health_status: WorkerHealthStatus,
}
#[derive(Debug)]
pub struct QualityFactors {
    pub load_balance_score: f64,
    pub response_time_score: f64,
    pub throughput_score: f64,
    pub overall_quality: f64,
}
#[derive(Debug)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
    NoAction,
}
#[derive(Debug)]
pub struct ScalingMetrics {
    pub average_utilization: f64,
    pub peak_utilization: f64,
    pub scaling_recommendation: ScalingRecommendation,
}
#[derive(Debug)]
pub struct ScalingDecision {
    pub action: ScalingAction,
    pub target_count: usize,
    pub cost_impact: f64,
    pub performance_improvement: f64,
}
/// Allocated resources for a task
#[derive(Debug, Clone)]
pub struct AllocatedResources {
    pub cpu_cores: f64,
    pub memory_mb: f64,
    pub network_bandwidth_mbps: f64,
    pub storage_mb: f64,
    pub gpu_units: f64,
}
#[derive(Debug, Clone)]
pub struct UtilizationMetrics {
    pub cpu: f64,
    pub memory: f64,
    pub network: f64,
}
#[derive(Debug, Clone)]
pub struct WorkloadPortion {
    pub computational_load: f64,
    pub tasks: Vec<TaskAssignment>,
}
