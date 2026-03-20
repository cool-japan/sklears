//! Default implementations for resource management types
//!
//! This module provides default implementations for all the configuration
//! and state types used throughout the resource management system.

use super::resource_types::{
    AlertThresholds, AllocationConstraints, AllocationStrategy, HealthStatus, MemoryProtection,
    MemoryUsage, NetworkUsage, PoolConfig, PoolHealth, PoolStats, ResourceManagerState,
    ResourceManagerStats, ResourceUsage, StoragePermissions, StorageUsage, TrackerConfig,
    UsageStatistics,
};
use std::time::{Duration, SystemTime};

// Default implementations for core resource types
impl Default for ResourceManagerState {
    fn default() -> Self {
        Self {
            initialized: false,
            running: false,
            optimization_enabled: true,
            last_health_check: SystemTime::now(),
            stats: ResourceManagerStats::default(),
        }
    }
}

impl Default for ResourceManagerStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            active_allocations: 0,
            failed_allocations: 0,
            avg_allocation_time: Duration::from_millis(100),
            utilization_efficiency: 0.0,
        }
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 1024 * 1024 * 1024 * 1024, // 1TB
            min_size: 1024 * 1024 * 1024,        // 1GB
            growth_factor: 1.5,
            shrink_threshold: 0.3,
            allocation_strategy: AllocationStrategy::BestFit,
            monitoring_enabled: true,
        }
    }
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            active_allocations: 0,
            peak_utilization: 0.0,
            average_utilization: 0.0,
            allocation_rate: 0.0,
            deallocation_rate: 0.0,
        }
    }
}

impl Default for PoolHealth {
    fn default() -> Self {
        Self {
            status: HealthStatus::Healthy,
            score: 1.0,
            last_check: SystemTime::now(),
            issues: Vec::new(),
        }
    }
}

impl Default for MemoryProtection {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            execute: false,
        }
    }
}

impl Default for StoragePermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            execute: false,
        }
    }
}

impl Default for AllocationConstraints {
    fn default() -> Self {
        Self {
            numa_affinity: None,
            cpu_affinity: None,
            gpu_affinity: None,
            memory_constraints: None,
            security_constraints: None,
        }
    }
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_secs(1),
            history_retention: Duration::from_secs(24 * 60 * 60), // 24 hours
            detailed_metrics: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 85.0,
            gpu_threshold: 90.0,
            network_threshold: 90.0,
            storage_threshold: 95.0,
        }
    }
}

impl Default for UsageStatistics {
    fn default() -> Self {
        Self {
            peak_cpu: 0.0,
            avg_cpu: 0.0,
            peak_memory: 0,
            avg_memory: 0,
            total_allocations: 0,
            failed_allocations: 0,
        }
    }
}

// Default implementations for specialized resource types
impl Default for MemoryUsage {
    fn default() -> Self {
        Self {
            total: 16 * 1024 * 1024 * 1024, // 16GB
            used: 0,
            free: 16 * 1024 * 1024 * 1024,
            cached: 0,
            swap_used: 0,
        }
    }
}

impl Default for NetworkUsage {
    fn default() -> Self {
        Self {
            bytes_received: 0,
            bytes_sent: 0,
            packets_received: 0,
            packets_sent: 0,
            bandwidth_utilization: 0.0,
        }
    }
}

impl Default for StorageUsage {
    fn default() -> Self {
        Self {
            total: 1024 * 1024 * 1024 * 1024, // 1TB
            used: 0,
            free: 1024 * 1024 * 1024 * 1024,
            read_ops: 0.0,
            write_ops: 0.0,
            read_bandwidth: 0.0,
            write_bandwidth: 0.0,
        }
    }
}

// Helper function to create default resource usage
#[must_use]
pub fn create_default_resource_usage() -> ResourceUsage {
    ResourceUsage {
        cpu_percent: 0.0,
        memory_usage: MemoryUsage::default(),
        gpu_usage: Vec::new(),
        network_usage: NetworkUsage::default(),
        storage_usage: StorageUsage::default(),
    }
}
