//! Emergency Resource Management System
//!
//! This module provides comprehensive emergency resource allocation and management
//! capabilities including resource pools, allocation tracking, capacity planning,
//! cost management, and emergency resource provisioning.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

// Import types from sibling modules
use super::detection::{EmergencyEvent, EmergencyType, EmergencySeverity};

/// Emergency resource allocation and management system
///
/// Manages emergency resource pools, allocation tracking, capacity planning,
/// and automated resource provisioning during emergency response operations.
/// Provides intelligent resource optimization and cost management.
#[derive(Debug)]
pub struct EmergencyResourceManager {
    /// Resource pools registry
    resource_pools: Arc<RwLock<HashMap<String, ResourcePool>>>,
    /// Active resource allocations
    allocated_resources: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    /// Resource allocation history
    allocation_history: Arc<RwLock<Vec<HistoricalAllocation>>>,
    /// Resource capacity planner
    capacity_planner: Arc<RwLock<CapacityPlanner>>,
    /// Cost tracker
    cost_tracker: Arc<RwLock<ResourceCostTracker>>,
    /// Resource optimization engine
    optimization_engine: Arc<RwLock<ResourceOptimizationEngine>>,
    /// Allocation policies
    allocation_policies: Arc<RwLock<Vec<AllocationPolicy>>>,
}

impl EmergencyResourceManager {
    pub fn new() -> Self {
        Self {
            resource_pools: Arc::new(RwLock::new(HashMap::new())),
            allocated_resources: Arc::new(RwLock::new(HashMap::new())),
            allocation_history: Arc::new(RwLock::new(Vec::new())),
            capacity_planner: Arc::new(RwLock::new(CapacityPlanner::new())),
            cost_tracker: Arc::new(RwLock::new(ResourceCostTracker::new())),
            optimization_engine: Arc::new(RwLock::new(ResourceOptimizationEngine::new())),
            allocation_policies: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn initialize(&self) -> SklResult<()> {
        self.setup_resource_pools()?;
        self.setup_allocation_policies()?;
        self.initialize_optimization_engine()?;
        Ok(())
    }

    /// Allocate emergency resources based on event requirements
    pub fn allocate_emergency_resources(&self, event: &EmergencyEvent) -> SklResult<Vec<ResourceAllocation>> {
        let resource_requirements = self.analyze_resource_requirements(event)?;
        let mut allocations = Vec::new();

        for requirement in resource_requirements {
            match self.allocate_resource_internal(&requirement, event) {
                Ok(allocation) => allocations.push(allocation),
                Err(e) => {
                    // Log error but continue with other allocations
                    eprintln!("Failed to allocate {}: {}", requirement.resource_type, e);
                }
            }
        }

        // Update capacity planner
        {
            let mut planner = self.capacity_planner.write()
                .map_err(|_| SklearsError::Other("Failed to acquire capacity planner lock".into()))?;
            planner.record_emergency_allocation(event, &allocations)?;
        }

        Ok(allocations)
    }

    /// Allocate a specific resource
    pub fn allocate_resource(
        &self,
        resource_type: String,
        severity: EmergencySeverity,
        duration: Option<Duration>
    ) -> SklResult<ResourceAllocationResult> {
        let requirement = ResourceRequirement {
            resource_type: ResourceType::from_string(&resource_type)?,
            amount_needed: self.calculate_amount_for_severity(severity),
            priority: self.severity_to_priority(severity),
            duration: duration.unwrap_or(Duration::from_hours(4)),
            constraints: self.get_allocation_constraints(&resource_type)?,
        };

        let event = EmergencyEvent {
            event_id: "direct_allocation".to_string(),
            emergency_type: EmergencyType::ResourceExhaustion,
            severity,
            title: "Direct Resource Allocation".to_string(),
            description: format!("Direct allocation of {} resources", resource_type),
            source: "resource_manager".to_string(),
            timestamp: SystemTime::now(),
            affected_systems: vec![],
            estimated_impact: super::detection::EmergencyImpact {
                user_impact: super::detection::UserImpact::Medium,
                business_impact: super::detection::BusinessImpact::Medium,
                system_impact: super::detection::SystemImpact::Medium,
                financial_impact: Some(1000.0),
            },
            estimated_impact_duration: duration,
            detected_by: "resource_manager".to_string(),
            context: HashMap::new(),
            related_events: vec![],
            urgency: super::detection::Urgency::Medium,
            requires_immediate_action: false,
        };

        let allocation = self.allocate_resource_internal(&requirement, &event)?;

        Ok(ResourceAllocationResult {
            success: true,
            allocated_amount: allocation.amount,
            allocation_id: allocation.allocation_id,
            estimated_cost: allocation.cost,
        })
    }

    /// Get resource utilization status
    pub fn get_resource_utilization(&self) -> SklResult<ResourceUtilizationStatus> {
        let pools = self.resource_pools.read()
            .map_err(|_| SklearsError::Other("Failed to acquire pools lock".into()))?;

        let allocations = self.allocated_resources.read()
            .map_err(|_| SklearsError::Other("Failed to acquire allocations lock".into()))?;

        let mut pool_utilization = HashMap::new();
        let mut total_allocated_cost = 0.0;

        for (pool_id, pool) in pools.iter() {
            let allocated_amount: f64 = allocations.values()
                .filter(|a| a.resource_type == pool.resource_type)
                .map(|a| a.amount)
                .sum();

            let utilization = if pool.total_capacity > 0.0 {
                allocated_amount / pool.total_capacity
            } else {
                0.0
            };

            pool_utilization.insert(pool_id.clone(), ResourcePoolUtilization {
                pool_id: pool_id.clone(),
                resource_type: pool.resource_type.clone(),
                total_capacity: pool.total_capacity,
                allocated: allocated_amount,
                available: pool.available_capacity,
                utilization_percentage: utilization * 100.0,
                emergency_reserve: pool.reserved_for_emergency,
                cost_per_unit: pool.cost_per_unit,
            });
        }

        total_allocated_cost = allocations.values()
            .map(|a| a.cost)
            .sum();

        Ok(ResourceUtilizationStatus {
            pool_utilization,
            total_allocated_cost,
            emergency_capacity_available: self.calculate_emergency_capacity(&pools)?,
            resource_constraints: self.identify_resource_constraints(&pools, &allocations)?,
            optimization_opportunities: self.identify_optimization_opportunities(&pools, &allocations)?,
        })
    }

    /// Release specific resource allocation
    pub fn release_allocation(&self, allocation_id: &str) -> SklResult<()> {
        let allocation = {
            let mut allocations = self.allocated_resources.write()
                .map_err(|_| SklearsError::Other("Failed to acquire allocations lock".into()))?;

            allocations.remove(allocation_id)
                .ok_or_else(|| SklearsError::InvalidInput("Allocation not found".into()))?
        };

        // Return resources to pool
        {
            let mut pools = self.resource_pools.write()
                .map_err(|_| SklearsError::Other("Failed to acquire pools lock".into()))?;

            for pool in pools.values_mut() {
                if pool.resource_type == allocation.resource_type {
                    pool.available_capacity += allocation.amount;
                    break;
                }
            }
        }

        // Record in history
        {
            let mut history = self.allocation_history.write()
                .map_err(|_| SklearsError::Other("Failed to acquire history lock".into()))?;

            history.push(HistoricalAllocation {
                allocation,
                released_at: Some(SystemTime::now()),
                actual_duration: None, // Would calculate
                actual_cost: None, // Would calculate final cost
                efficiency_score: 0.85, // Would calculate
                lessons_learned: vec![],
            });

            // Limit history size
            if history.len() > 1000 {
                history.drain(0..100);
            }
        }

        Ok(())
    }

    /// Release all emergency resources
    pub fn release_emergency_resources(&self) -> SklResult<ResourceReleaseResult> {
        let allocation_ids: Vec<String> = {
            let allocations = self.allocated_resources.read()
                .map_err(|_| SklearsError::Other("Failed to acquire allocations lock".into()))?;
            allocations.keys().cloned().collect()
        };

        let mut released_count = 0;
        let mut failed_releases = 0;
        let mut total_cost_released = 0.0;

        for allocation_id in allocation_ids {
            match self.release_allocation(&allocation_id) {
                Ok(_) => {
                    released_count += 1;
                    // Would track cost released
                },
                Err(_) => failed_releases += 1,
            }
        }

        // Update cost tracker
        {
            let mut tracker = self.cost_tracker.write()
                .map_err(|_| SklearsError::Other("Failed to acquire cost tracker lock".into()))?;
            tracker.record_bulk_release(released_count, total_cost_released)?;
        }

        Ok(ResourceReleaseResult {
            total_allocations: released_count + failed_releases,
            successful_releases: released_count,
            failed_releases,
            total_cost_released,
            release_duration: Duration::from_minutes(5), // Estimated
        })
    }

    /// Get resource optimization recommendations
    pub fn get_optimization_recommendations(&self) -> SklResult<Vec<OptimizationRecommendation>> {
        let engine = self.optimization_engine.read()
            .map_err(|_| SklearsError::Other("Failed to acquire optimization engine lock".into()))?;

        let pools = self.resource_pools.read()
            .map_err(|_| SklearsError::Other("Failed to acquire pools lock".into()))?;

        let allocations = self.allocated_resources.read()
            .map_err(|_| SklearsError::Other("Failed to acquire allocations lock".into()))?;

        engine.generate_recommendations(&pools, &allocations)
    }

    /// Get cost analysis
    pub fn get_cost_analysis(&self) -> SklResult<ResourceCostAnalysis> {
        let tracker = self.cost_tracker.read()
            .map_err(|_| SklearsError::Other("Failed to acquire cost tracker lock".into()))?;
        tracker.generate_cost_analysis()
    }

    /// Shutdown resource management
    pub fn shutdown(&self) -> SklResult<()> {
        // Release all resources
        self.release_emergency_resources()?;

        // Clear resource pools
        {
            let mut pools = self.resource_pools.write()
                .map_err(|_| SklearsError::Other("Failed to acquire pools lock".into()))?;
            pools.clear();
        }

        Ok(())
    }

    fn setup_resource_pools(&self) -> SklResult<()> {
        let mut pools = self.resource_pools.write()
            .map_err(|_| SklearsError::Other("Failed to acquire pools lock".into()))?;

        pools.insert("compute".to_string(), ResourcePool {
            pool_id: "compute".to_string(),
            resource_type: ResourceType::Compute,
            total_capacity: 1000.0,
            available_capacity: 800.0,
            reserved_for_emergency: 200.0,
            cost_per_unit: 0.1,
            scaling_policy: ScalingPolicy {
                auto_scaling_enabled: true,
                min_capacity: 500.0,
                max_capacity: 2000.0,
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.3,
                scale_up_increment: 100.0,
                scale_down_increment: 50.0,
                cooldown_period: Duration::from_minutes(5),
            },
            performance_metrics: ResourcePoolMetrics::default(),
        });

        pools.insert("storage".to_string(), ResourcePool {
            pool_id: "storage".to_string(),
            resource_type: ResourceType::Storage,
            total_capacity: 10000.0,
            available_capacity: 7000.0,
            reserved_for_emergency: 1000.0,
            cost_per_unit: 0.01,
            scaling_policy: ScalingPolicy {
                auto_scaling_enabled: true,
                min_capacity: 5000.0,
                max_capacity: 50000.0,
                scale_up_threshold: 0.85,
                scale_down_threshold: 0.4,
                scale_up_increment: 1000.0,
                scale_down_increment: 500.0,
                cooldown_period: Duration::from_minutes(10),
            },
            performance_metrics: ResourcePoolMetrics::default(),
        });

        pools.insert("network".to_string(), ResourcePool {
            pool_id: "network".to_string(),
            resource_type: ResourceType::Network,
            total_capacity: 100.0, // Gbps
            available_capacity: 70.0,
            reserved_for_emergency: 20.0,
            cost_per_unit: 1.0,
            scaling_policy: ScalingPolicy {
                auto_scaling_enabled: false,
                min_capacity: 50.0,
                max_capacity: 500.0,
                scale_up_threshold: 0.9,
                scale_down_threshold: 0.2,
                scale_up_increment: 10.0,
                scale_down_increment: 5.0,
                cooldown_period: Duration::from_minutes(15),
            },
            performance_metrics: ResourcePoolMetrics::default(),
        });

        pools.insert("database".to_string(), ResourcePool {
            pool_id: "database".to_string(),
            resource_type: ResourceType::Database,
            total_capacity: 50.0, // Database instances
            available_capacity: 35.0,
            reserved_for_emergency: 10.0,
            cost_per_unit: 5.0,
            scaling_policy: ScalingPolicy {
                auto_scaling_enabled: true,
                min_capacity: 20.0,
                max_capacity: 100.0,
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.3,
                scale_up_increment: 5.0,
                scale_down_increment: 2.0,
                cooldown_period: Duration::from_minutes(20),
            },
            performance_metrics: ResourcePoolMetrics::default(),
        });

        Ok(())
    }

    fn setup_allocation_policies(&self) -> SklResult<()> {
        let mut policies = self.allocation_policies.write()
            .map_err(|_| SklearsError::Other("Failed to acquire policies lock".into()))?;

        policies.push(AllocationPolicy {
            policy_id: "emergency_priority".to_string(),
            name: "Emergency Priority Policy".to_string(),
            policy_type: PolicyType::Priority,
            conditions: vec![
                PolicyCondition {
                    condition_type: ConditionType::SeverityLevel,
                    operator: ComparisonOperator::GreaterThanOrEqual,
                    value: "Critical".to_string(),
                }
            ],
            actions: vec![
                PolicyAction {
                    action_type: ActionType::ReserveCapacity,
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("percentage".to_string(), "50".to_string());
                        params
                    },
                }
            ],
            priority: 1,
            enabled: true,
        });

        policies.push(AllocationPolicy {
            policy_id: "cost_optimization".to_string(),
            name: "Cost Optimization Policy".to_string(),
            policy_type: PolicyType::CostOptimization,
            conditions: vec![
                PolicyCondition {
                    condition_type: ConditionType::TotalCost,
                    operator: ComparisonOperator::GreaterThan,
                    value: "10000".to_string(),
                }
            ],
            actions: vec![
                PolicyAction {
                    action_type: ActionType::OptimizeAllocation,
                    parameters: HashMap::new(),
                }
            ],
            priority: 2,
            enabled: true,
        });

        Ok(())
    }

    fn initialize_optimization_engine(&self) -> SklResult<()> {
        let mut engine = self.optimization_engine.write()
            .map_err(|_| SklearsError::Other("Failed to acquire optimization engine lock".into()))?;

        engine.configure_optimization_algorithms()?;
        Ok(())
    }

    fn analyze_resource_requirements(&self, event: &EmergencyEvent) -> SklResult<Vec<ResourceRequirement>> {
        let mut requirements = Vec::new();

        // Base requirements based on emergency type
        match event.emergency_type {
            EmergencyType::SystemFailure => {
                requirements.push(ResourceRequirement {
                    resource_type: ResourceType::Compute,
                    amount_needed: self.calculate_amount_for_severity(event.severity) * 2.0,
                    priority: AllocationPriority::High,
                    duration: Duration::from_hours(4),
                    constraints: vec![],
                });
            },
            EmergencyType::ResourceExhaustion => {
                requirements.push(ResourceRequirement {
                    resource_type: ResourceType::Compute,
                    amount_needed: self.calculate_amount_for_severity(event.severity) * 3.0,
                    priority: AllocationPriority::Critical,
                    duration: Duration::from_hours(2),
                    constraints: vec![],
                });
                requirements.push(ResourceRequirement {
                    resource_type: ResourceType::Storage,
                    amount_needed: self.calculate_amount_for_severity(event.severity) * 10.0,
                    priority: AllocationPriority::High,
                    duration: Duration::from_hours(6),
                    constraints: vec![],
                });
            },
            EmergencyType::DatabaseFailure => {
                requirements.push(ResourceRequirement {
                    resource_type: ResourceType::Database,
                    amount_needed: self.calculate_amount_for_severity(event.severity),
                    priority: AllocationPriority::Critical,
                    duration: Duration::from_hours(8),
                    constraints: vec![],
                });
            },
            _ => {
                // Default allocation
                requirements.push(ResourceRequirement {
                    resource_type: ResourceType::Compute,
                    amount_needed: self.calculate_amount_for_severity(event.severity),
                    priority: AllocationPriority::Medium,
                    duration: Duration::from_hours(2),
                    constraints: vec![],
                });
            }
        }

        Ok(requirements)
    }

    fn allocate_resource_internal(&self, requirement: &ResourceRequirement, event: &EmergencyEvent) -> SklResult<ResourceAllocation> {
        let mut pools = self.resource_pools.write()
            .map_err(|_| SklearsError::Other("Failed to acquire pools lock".into()))?;

        // Find suitable pool
        let pool = pools.values_mut()
            .find(|p| p.resource_type == requirement.resource_type)
            .ok_or_else(|| SklearsError::Other("No suitable resource pool found".into()))?;

        // Check availability
        if pool.available_capacity < requirement.amount_needed {
            return Err(SklearsError::Other("Insufficient resource capacity".into()));
        }

        // Create allocation
        let allocation = ResourceAllocation {
            allocation_id: uuid::Uuid::new_v4().to_string(),
            resource_type: requirement.resource_type.clone(),
            amount: requirement.amount_needed,
            allocated_at: SystemTime::now(),
            duration: Some(requirement.duration),
            cost: requirement.amount_needed * pool.cost_per_unit,
            purpose: format!("Emergency response for {}", event.event_id),
            emergency_id: Some(event.event_id.clone()),
            allocation_metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("severity".to_string(), format!("{:?}", event.severity));
                metadata.insert("emergency_type".to_string(), format!("{:?}", event.emergency_type));
                metadata
            },
        };

        // Update pool capacity
        pool.available_capacity -= requirement.amount_needed;

        // Register allocation
        {
            let mut allocations = self.allocated_resources.write()
                .map_err(|_| SklearsError::Other("Failed to acquire allocations lock".into()))?;
            allocations.insert(allocation.allocation_id.clone(), allocation.clone());
        }

        // Update cost tracker
        {
            let mut tracker = self.cost_tracker.write()
                .map_err(|_| SklearsError::Other("Failed to acquire cost tracker lock".into()))?;
            tracker.record_allocation(&allocation)?;
        }

        Ok(allocation)
    }

    fn calculate_amount_for_severity(&self, severity: EmergencySeverity) -> f64 {
        match severity {
            EmergencySeverity::Low => 10.0,
            EmergencySeverity::Medium => 25.0,
            EmergencySeverity::High => 50.0,
            EmergencySeverity::Critical => 100.0,
            EmergencySeverity::Catastrophic => 200.0,
        }
    }

    fn severity_to_priority(&self, severity: EmergencySeverity) -> AllocationPriority {
        match severity {
            EmergencySeverity::Low => AllocationPriority::Low,
            EmergencySeverity::Medium => AllocationPriority::Medium,
            EmergencySeverity::High => AllocationPriority::High,
            EmergencySeverity::Critical => AllocationPriority::Critical,
            EmergencySeverity::Catastrophic => AllocationPriority::Emergency,
        }
    }

    fn get_allocation_constraints(&self, _resource_type: &str) -> SklResult<Vec<AllocationConstraint>> {
        // Would implement constraint logic
        Ok(vec![])
    }

    fn calculate_emergency_capacity(&self, pools: &HashMap<String, ResourcePool>) -> SklResult<f64> {
        let total_emergency_capacity: f64 = pools.values()
            .map(|pool| pool.reserved_for_emergency)
            .sum();
        Ok(total_emergency_capacity)
    }

    fn identify_resource_constraints(&self, pools: &HashMap<String, ResourcePool>, allocations: &HashMap<String, ResourceAllocation>) -> SklResult<Vec<ResourceConstraint>> {
        let mut constraints = Vec::new();

        for pool in pools.values() {
            let allocated_amount: f64 = allocations.values()
                .filter(|a| a.resource_type == pool.resource_type)
                .map(|a| a.amount)
                .sum();

            let utilization = allocated_amount / pool.total_capacity;

            if utilization > 0.8 {
                constraints.push(ResourceConstraint {
                    constraint_type: ConstraintType::HighUtilization,
                    resource_type: pool.resource_type.clone(),
                    severity: if utilization > 0.95 { ConstraintSeverity::Critical } else { ConstraintSeverity::High },
                    description: format!("High utilization: {:.1}%", utilization * 100.0),
                    impact: ConstraintImpact::ReducedCapacity,
                    mitigation_suggestions: vec![
                        "Consider scaling up capacity".to_string(),
                        "Optimize resource usage".to_string(),
                    ],
                });
            }
        }

        Ok(constraints)
    }

    fn identify_optimization_opportunities(&self, pools: &HashMap<String, ResourcePool>, allocations: &HashMap<String, ResourceAllocation>) -> SklResult<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();

        // Look for underutilized resources
        for pool in pools.values() {
            let allocated_amount: f64 = allocations.values()
                .filter(|a| a.resource_type == pool.resource_type)
                .map(|a| a.amount)
                .sum();

            let utilization = allocated_amount / pool.total_capacity;

            if utilization < 0.3 {
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: OpportunityType::Underutilization,
                    resource_type: pool.resource_type.clone(),
                    potential_savings: pool.available_capacity * pool.cost_per_unit * 0.5,
                    description: format!("Underutilized resources: {:.1}% utilization", utilization * 100.0),
                    action_required: OptimizationAction::ScaleDown,
                    confidence_score: 0.85,
                });
            }
        }

        Ok(opportunities)
    }
}

/// Resource pool configuration and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    pub pool_id: String,
    pub resource_type: ResourceType,
    pub total_capacity: f64,
    pub available_capacity: f64,
    pub reserved_for_emergency: f64,
    pub cost_per_unit: f64,
    pub scaling_policy: ScalingPolicy,
    pub performance_metrics: ResourcePoolMetrics,
}

/// Resource types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceType {
    Compute,
    Storage,
    Network,
    Database,
    Cache,
    Queue,
}

impl ResourceType {
    pub fn from_string(s: &str) -> SklResult<Self> {
        match s.to_lowercase().as_str() {
            "compute" => Ok(ResourceType::Compute),
            "storage" => Ok(ResourceType::Storage),
            "network" => Ok(ResourceType::Network),
            "database" => Ok(ResourceType::Database),
            "cache" => Ok(ResourceType::Cache),
            "queue" => Ok(ResourceType::Queue),
            _ => Err(SklearsError::InvalidInput(format!("Unknown resource type: {}", s))),
        }
    }
}

/// Resource allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub allocation_id: String,
    pub resource_type: ResourceType,
    pub amount: f64,
    pub allocated_at: SystemTime,
    pub duration: Option<Duration>,
    pub cost: f64,
    pub purpose: String,
    pub emergency_id: Option<String>,
    pub allocation_metadata: HashMap<String, String>,
}

/// Resource allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocationResult {
    pub success: bool,
    pub allocated_amount: f64,
    pub allocation_id: String,
    pub estimated_cost: f64,
}

/// Resource requirement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub resource_type: ResourceType,
    pub amount_needed: f64,
    pub priority: AllocationPriority,
    pub duration: Duration,
    pub constraints: Vec<AllocationConstraint>,
}

/// Allocation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Allocation constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationConstraint {
    pub constraint_type: ConstraintType,
    pub value: String,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    HighUtilization,
    CostLimit,
    RegionLimit,
    TimeWindow,
    TotalCost,
    SeverityLevel,
}

/// Scaling policy for resource pools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    pub auto_scaling_enabled: bool,
    pub min_capacity: f64,
    pub max_capacity: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub scale_up_increment: f64,
    pub scale_down_increment: f64,
    pub cooldown_period: Duration,
}

/// Resource pool performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePoolMetrics {
    pub average_utilization: f64,
    pub peak_utilization: f64,
    pub allocation_count: u64,
    pub total_cost: f64,
    pub efficiency_score: f64,
}

impl Default for ResourcePoolMetrics {
    fn default() -> Self {
        Self {
            average_utilization: 0.45,
            peak_utilization: 0.85,
            allocation_count: 0,
            total_cost: 0.0,
            efficiency_score: 0.80,
        }
    }
}

/// Resource utilization status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationStatus {
    pub pool_utilization: HashMap<String, ResourcePoolUtilization>,
    pub total_allocated_cost: f64,
    pub emergency_capacity_available: f64,
    pub resource_constraints: Vec<ResourceConstraint>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Pool utilization details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePoolUtilization {
    pub pool_id: String,
    pub resource_type: ResourceType,
    pub total_capacity: f64,
    pub allocated: f64,
    pub available: f64,
    pub utilization_percentage: f64,
    pub emergency_reserve: f64,
    pub cost_per_unit: f64,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraint {
    pub constraint_type: ConstraintType,
    pub resource_type: ResourceType,
    pub severity: ConstraintSeverity,
    pub description: String,
    pub impact: ConstraintImpact,
    pub mitigation_suggestions: Vec<String>,
}

/// Constraint severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Constraint impact types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintImpact {
    ReducedCapacity,
    IncreasedCost,
    PerformanceDegradation,
    ServiceUnavailability,
}

/// Optimization opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OpportunityType,
    pub resource_type: ResourceType,
    pub potential_savings: f64,
    pub description: String,
    pub action_required: OptimizationAction,
    pub confidence_score: f64,
}

/// Optimization opportunity types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpportunityType {
    Underutilization,
    Rightsizing,
    CostOptimization,
    PerformanceImprovement,
}

/// Optimization actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAction {
    ScaleDown,
    ScaleUp,
    Rebalance,
    Migrate,
    Terminate,
}

/// Resource release result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReleaseResult {
    pub total_allocations: usize,
    pub successful_releases: usize,
    pub failed_releases: usize,
    pub total_cost_released: f64,
    pub release_duration: Duration,
}

/// Historical allocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalAllocation {
    pub allocation: ResourceAllocation,
    pub released_at: Option<SystemTime>,
    pub actual_duration: Option<Duration>,
    pub actual_cost: Option<f64>,
    pub efficiency_score: f64,
    pub lessons_learned: Vec<String>,
}

/// Capacity planning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityPlanner {
    pub historical_usage: Vec<UsageRecord>,
    pub forecasting_models: Vec<ForecastingModel>,
    pub capacity_recommendations: Vec<CapacityRecommendation>,
}

impl CapacityPlanner {
    pub fn new() -> Self {
        Self {
            historical_usage: Vec::new(),
            forecasting_models: Vec::new(),
            capacity_recommendations: Vec::new(),
        }
    }

    pub fn record_emergency_allocation(&mut self, _event: &EmergencyEvent, _allocations: &[ResourceAllocation]) -> SklResult<()> {
        // Would implement capacity planning logic
        Ok(())
    }
}

/// Usage record for capacity planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub timestamp: SystemTime,
    pub resource_type: ResourceType,
    pub usage_amount: f64,
    pub context: String,
}

/// Forecasting model for capacity planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingModel {
    pub model_id: String,
    pub resource_type: ResourceType,
    pub model_type: ModelType,
    pub accuracy: f64,
    pub last_trained: SystemTime,
}

/// Model types for forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Linear,
    Exponential,
    Seasonal,
    MachineLearning,
}

/// Capacity recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityRecommendation {
    pub resource_type: ResourceType,
    pub recommended_capacity: f64,
    pub current_capacity: f64,
    pub confidence: f64,
    pub reasoning: String,
    pub implementation_timeline: Duration,
}

/// Cost tracking system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCostTracker {
    pub total_cost: f64,
    pub cost_by_resource_type: HashMap<ResourceType, f64>,
    pub daily_costs: Vec<DailyCostRecord>,
    pub budget_limits: HashMap<ResourceType, f64>,
}

impl ResourceCostTracker {
    pub fn new() -> Self {
        Self {
            total_cost: 0.0,
            cost_by_resource_type: HashMap::new(),
            daily_costs: Vec::new(),
            budget_limits: HashMap::new(),
        }
    }

    pub fn record_allocation(&mut self, allocation: &ResourceAllocation) -> SklResult<()> {
        self.total_cost += allocation.cost;
        *self.cost_by_resource_type.entry(allocation.resource_type.clone()).or_insert(0.0) += allocation.cost;
        Ok(())
    }

    pub fn record_bulk_release(&mut self, _count: usize, _cost: f64) -> SklResult<()> {
        // Would implement bulk release tracking
        Ok(())
    }

    pub fn generate_cost_analysis(&self) -> SklResult<ResourceCostAnalysis> {
        Ok(ResourceCostAnalysis {
            total_cost: self.total_cost,
            cost_by_resource_type: self.cost_by_resource_type.clone(),
            average_daily_cost: self.calculate_average_daily_cost(),
            cost_trends: self.analyze_cost_trends(),
            budget_utilization: self.calculate_budget_utilization(),
        })
    }

    fn calculate_average_daily_cost(&self) -> f64 {
        if self.daily_costs.is_empty() {
            0.0
        } else {
            self.daily_costs.iter().map(|r| r.total_cost).sum::<f64>() / self.daily_costs.len() as f64
        }
    }

    fn analyze_cost_trends(&self) -> Vec<CostTrend> {
        // Would implement trend analysis
        vec![]
    }

    fn calculate_budget_utilization(&self) -> HashMap<ResourceType, f64> {
        let mut utilization = HashMap::new();
        for (resource_type, limit) in &self.budget_limits {
            if let Some(spent) = self.cost_by_resource_type.get(resource_type) {
                utilization.insert(resource_type.clone(), spent / limit);
            }
        }
        utilization
    }
}

/// Daily cost record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyCostRecord {
    pub date: SystemTime,
    pub total_cost: f64,
    pub cost_by_resource_type: HashMap<ResourceType, f64>,
}

/// Cost analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCostAnalysis {
    pub total_cost: f64,
    pub cost_by_resource_type: HashMap<ResourceType, f64>,
    pub average_daily_cost: f64,
    pub cost_trends: Vec<CostTrend>,
    pub budget_utilization: HashMap<ResourceType, f64>,
}

/// Cost trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTrend {
    pub resource_type: ResourceType,
    pub trend_direction: TrendDirection,
    pub change_percentage: f64,
    pub time_period: Duration,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Resource optimization engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationEngine {
    pub optimization_algorithms: Vec<OptimizationAlgorithm>,
    pub optimization_history: Vec<OptimizationResult>,
}

impl ResourceOptimizationEngine {
    pub fn new() -> Self {
        Self {
            optimization_algorithms: Vec::new(),
            optimization_history: Vec::new(),
        }
    }

    pub fn configure_optimization_algorithms(&mut self) -> SklResult<()> {
        // Would configure optimization algorithms
        Ok(())
    }

    pub fn generate_recommendations(&self, _pools: &HashMap<String, ResourcePool>, _allocations: &HashMap<String, ResourceAllocation>) -> SklResult<Vec<OptimizationRecommendation>> {
        // Would generate optimization recommendations
        Ok(vec![])
    }
}

/// Optimization algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAlgorithm {
    pub algorithm_id: String,
    pub algorithm_type: AlgorithmType,
    pub enabled: bool,
    pub weight: f64,
}

/// Algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmType {
    CostMinimization,
    UtilizationMaximization,
    PerformanceOptimization,
    BalancedOptimization,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub optimization_id: String,
    pub timestamp: SystemTime,
    pub cost_savings: f64,
    pub performance_improvement: f64,
    pub actions_taken: Vec<OptimizationAction>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_id: String,
    pub recommendation_type: RecommendationType,
    pub resource_type: ResourceType,
    pub current_state: String,
    pub recommended_state: String,
    pub expected_savings: f64,
    pub implementation_effort: ImplementationEffort,
    pub confidence: f64,
}

/// Recommendation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    Rightsizing,
    Consolidation,
    Migration,
    Termination,
    Scaling,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Allocation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPolicy {
    pub policy_id: String,
    pub name: String,
    pub policy_type: PolicyType,
    pub conditions: Vec<PolicyCondition>,
    pub actions: Vec<PolicyAction>,
    pub priority: u32,
    pub enabled: bool,
}

/// Policy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyType {
    Priority,
    CostOptimization,
    CapacityManagement,
    Security,
}

/// Policy conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCondition {
    pub condition_type: ConditionType,
    pub operator: ComparisonOperator,
    pub value: String,
}

/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Policy actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyAction {
    pub action_type: ActionType,
    pub parameters: HashMap<String, String>,
}

/// Action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    ReserveCapacity,
    OptimizeAllocation,
    ScaleResources,
    SendAlert,
}

impl Default for EmergencyResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager_creation() {
        let manager = EmergencyResourceManager::new();
        assert!(manager.initialize().is_ok());
    }

    #[test]
    fn test_resource_allocation() {
        let manager = EmergencyResourceManager::new();
        manager.initialize().unwrap();

        let result = manager.allocate_resource(
            "compute".to_string(),
            EmergencySeverity::Critical,
            Some(Duration::from_hours(2))
        );
        assert!(result.is_ok());

        let allocation_result = result.unwrap();
        assert!(allocation_result.success);
        assert!(allocation_result.allocated_amount > 0.0);
    }

    #[test]
    fn test_emergency_resource_allocation() {
        let manager = EmergencyResourceManager::new();
        manager.initialize().unwrap();

        let event = EmergencyEvent {
            event_id: "emergency-001".to_string(),
            emergency_type: EmergencyType::SystemFailure,
            severity: EmergencySeverity::Critical,
            title: "System Down".to_string(),
            description: "Primary system is unresponsive".to_string(),
            source: "health_monitor".to_string(),
            timestamp: SystemTime::now(),
            affected_systems: vec!["primary_system".to_string()],
            estimated_impact: super::super::detection::EmergencyImpact {
                user_impact: super::super::detection::UserImpact::High,
                business_impact: super::super::detection::BusinessImpact::High,
                system_impact: super::super::detection::SystemImpact::Critical,
                financial_impact: Some(10000.0),
            },
            estimated_impact_duration: Some(Duration::from_hours(2)),
            detected_by: "detector".to_string(),
            context: std::collections::HashMap::new(),
            related_events: vec![],
            urgency: super::super::detection::Urgency::High,
            requires_immediate_action: true,
        };

        let allocations = manager.allocate_emergency_resources(&event).unwrap();
        assert!(!allocations.is_empty());
    }

    #[test]
    fn test_resource_utilization() {
        let manager = EmergencyResourceManager::new();
        manager.initialize().unwrap();

        let utilization = manager.get_resource_utilization().unwrap();
        assert!(!utilization.pool_utilization.is_empty());
        assert!(utilization.total_allocated_cost >= 0.0);
    }
}