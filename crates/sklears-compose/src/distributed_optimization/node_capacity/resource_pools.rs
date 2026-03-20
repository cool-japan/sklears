use crate::distributed_optimization::core_types::*;
use super::capacity_metrics::NodeCapacity;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Resource pool management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    pub pool_id: String,
    pub pool_name: String,
    pub pool_type: ResourcePoolType,
    pub member_nodes: Vec<NodeId>,
    pub aggregate_capacity: NodeCapacity,
    pub allocation_strategy: AllocationStrategy,
    pub load_balancing: LoadBalancingStrategy,
    pub failover_config: FailoverConfiguration,
}

/// Resource pool types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourcePoolType {
    Compute,
    Storage,
    Network,
    GPU,
    Memory,
    Mixed,
    Custom(String),
}

/// Allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    RoundRobin,
    WeightedRoundRobin,
    LeastUtilized,
    MostUtilized,
    Random,
    Custom(String),
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    LeastResponseTime,
    ResourceBased,
    Geographic,
    Custom(String),
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfiguration {
    pub enable_failover: bool,
    pub failover_threshold: f64,
    pub backup_nodes: Vec<NodeId>,
    pub failover_strategy: FailoverStrategy,
    pub recovery_time_objective: Duration,
    pub recovery_point_objective: Duration,
}

/// Failover strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailoverStrategy {
    ActivePassive,
    ActiveActive,
    LoadRedistribution,
    Graceful,
    Immediate,
    Custom(String),
}

impl Default for FailoverConfiguration {
    fn default() -> Self {
        Self {
            enable_failover: true,
            failover_threshold: 0.8,
            backup_nodes: Vec::new(),
            failover_strategy: FailoverStrategy::ActivePassive,
            recovery_time_objective: Duration::from_secs(300), // 5 minutes
            recovery_point_objective: Duration::from_secs(60), // 1 minute
        }
    }
}