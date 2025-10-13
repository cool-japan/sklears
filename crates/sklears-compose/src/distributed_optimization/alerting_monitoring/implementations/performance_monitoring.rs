use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Comprehensive performance monitoring system providing advanced system resource monitoring,
/// application performance tracking, and SLA management with configurable thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// CPU monitoring
    pub cpu_monitoring: CPUMonitoringConfig,
    /// Memory monitoring
    pub memory_monitoring: MemoryMonitoringConfig,
    /// Disk monitoring
    pub disk_monitoring: DiskMonitoringConfig,
    /// Network monitoring
    pub network_monitoring: NetworkMonitoringConfig,
    /// Application monitoring
    pub application_monitoring: ApplicationMonitoringConfig,
}

/// CPU monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUMonitoringConfig {
    /// Monitoring enabled
    pub enabled: bool,
    /// Collection interval
    pub interval: Duration,
    /// CPU usage thresholds
    pub thresholds: CPUThresholds,
}

/// CPU thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUThresholds {
    /// Warning threshold
    pub warning: f64,
    /// Critical threshold
    pub critical: f64,
}

/// Memory monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMonitoringConfig {
    /// Monitoring enabled
    pub enabled: bool,
    /// Collection interval
    pub interval: Duration,
    /// Memory usage thresholds
    pub thresholds: MemoryThresholds,
}

/// Memory thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryThresholds {
    /// Warning threshold (percentage)
    pub warning: f64,
    /// Critical threshold (percentage)
    pub critical: f64,
}

/// Disk monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMonitoringConfig {
    /// Monitoring enabled
    pub enabled: bool,
    /// Collection interval
    pub interval: Duration,
    /// Disk usage thresholds
    pub thresholds: DiskThresholds,
}

/// Disk thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskThresholds {
    /// Warning threshold (percentage)
    pub warning: f64,
    /// Critical threshold (percentage)
    pub critical: f64,
}

/// Network monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMonitoringConfig {
    /// Monitoring enabled
    pub enabled: bool,
    /// Collection interval
    pub interval: Duration,
    /// Network thresholds
    pub thresholds: NetworkThresholds,
}

/// Network thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkThresholds {
    /// Bandwidth warning threshold (Mbps)
    pub bandwidth_warning: f64,
    /// Bandwidth critical threshold (Mbps)
    pub bandwidth_critical: f64,
    /// Latency warning threshold (ms)
    pub latency_warning: f64,
    /// Latency critical threshold (ms)
    pub latency_critical: f64,
}

/// Application monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationMonitoringConfig {
    /// Response time monitoring
    pub response_time: ResponseTimeMonitoring,
    /// Error rate monitoring
    pub error_rate: ErrorRateMonitoring,
    /// Throughput monitoring
    pub throughput: ThroughputMonitoring,
}

/// Response time monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeMonitoring {
    /// Enabled status
    pub enabled: bool,
    /// SLA thresholds
    pub sla_thresholds: SLAThresholds,
}

/// SLA thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLAThresholds {
    /// Target response time (ms)
    pub target: f64,
    /// Warning threshold (ms)
    pub warning: f64,
    /// Critical threshold (ms)
    pub critical: f64,
}

/// Error rate monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRateMonitoring {
    /// Enabled status
    pub enabled: bool,
    /// Error rate thresholds
    pub thresholds: ErrorRateThresholds,
}

/// Error rate thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRateThresholds {
    /// Warning threshold (percentage)
    pub warning: f64,
    /// Critical threshold (percentage)
    pub critical: f64,
}

/// Throughput monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMonitoring {
    /// Enabled status
    pub enabled: bool,
    /// Throughput thresholds
    pub thresholds: ThroughputThresholds,
}

/// Throughput thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputThresholds {
    /// Minimum expected throughput (requests/second)
    pub minimum: f64,
    /// Warning threshold (requests/second)
    pub warning: f64,
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            cpu_monitoring: CPUMonitoringConfig {
                enabled: true,
                interval: Duration::from_secs(60),
                thresholds: CPUThresholds {
                    warning: 80.0,
                    critical: 95.0,
                },
            },
            memory_monitoring: MemoryMonitoringConfig {
                enabled: true,
                interval: Duration::from_secs(60),
                thresholds: MemoryThresholds {
                    warning: 80.0,
                    critical: 95.0,
                },
            },
            disk_monitoring: DiskMonitoringConfig {
                enabled: true,
                interval: Duration::from_secs(300),
                thresholds: DiskThresholds {
                    warning: 85.0,
                    critical: 95.0,
                },
            },
            network_monitoring: NetworkMonitoringConfig {
                enabled: true,
                interval: Duration::from_secs(60),
                thresholds: NetworkThresholds {
                    bandwidth_warning: 80.0,
                    bandwidth_critical: 95.0,
                    latency_warning: 100.0,
                    latency_critical: 500.0,
                },
            },
            application_monitoring: ApplicationMonitoringConfig {
                response_time: ResponseTimeMonitoring {
                    enabled: true,
                    sla_thresholds: SLAThresholds {
                        target: 100.0,
                        warning: 200.0,
                        critical: 1000.0,
                    },
                },
                error_rate: ErrorRateMonitoring {
                    enabled: true,
                    thresholds: ErrorRateThresholds {
                        warning: 5.0,
                        critical: 10.0,
                    },
                },
                throughput: ThroughputMonitoring {
                    enabled: true,
                    thresholds: ThroughputThresholds {
                        minimum: 100.0,
                        warning: 50.0,
                    },
                },
            },
        }
    }
}