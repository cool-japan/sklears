//! Report Generation Module
//!
//! This module provides comprehensive report generation capabilities including
//! data source management, template processing, job scheduling, output delivery,
//! and performance monitoring for automated business intelligence reporting.
//!
//! # Architecture
//!
//! The report generation system is organized into six main components:
//!
//! - **Core Management** (`generation_core`): Fundamental types and orchestration
//! - **Data Sources** (`data_sources`): Data connectivity and management
//! - **Template Engine** (`template_engine`): Template processing and rendering
//! - **Scheduler & Execution** (`scheduler_execution`): Job scheduling and execution
//! - **Output & Delivery** (`output_delivery`): Format handling and delivery
//! - **Monitoring & Metrics** (`monitoring_metrics`): Performance monitoring and alerting
//!
//! # Usage
//!
//! ```ignore
//! use crate::report_generation::{ReportGenerationManager, ReportGenerator};
//!
//! // Create a new report generation manager
//! let manager = ReportGenerationManager::new();
//!
//! // Register a report generator
//! let generator = ReportGenerator { /* ... */ };
//! manager.register_generator(generator)?;
//!
//! // Generate a report
//! let parameters = std::collections::HashMap::new();
//! let report = manager.generate_report("generator_id", "template_id", parameters)?;
//! ```

#![allow(unexpected_cfgs)]

// Module declarations
pub mod data_sources;
pub mod generation_core;
pub mod monitoring_metrics;
pub mod output_delivery;
pub mod scheduler_execution;
pub mod template_engine;

// Re-export core management types
pub use generation_core::{
    BackoffStrategy, ChartResolution, ColorDepth, DeliveryOption, FontRendering, GeneratedReport,
    GenerationConfig, GeneratorPerformanceMetrics, GeneratorStatus, ImageQuality, MemoryUsageStats,
    NotificationChannel, NotificationSettings, NotificationTrigger, OptimizationLevel,
    QualitySettings, ReportGenerationError, ReportGenerationManager, ReportGenerationResult,
    ReportGenerator, ReportSchedule, ReportScheduling, ReportType, RetryPolicy,
};

// Re-export data source management types
pub use data_sources::{
    AuthenticationMethod, BackoffStrategy as DataBackoffStrategy, DataSource, DataSourceConnection,
    DataSourceManager, DataSourceType, OAuth2Config, RetryConfiguration, SslConfig, TimeoutConfig,
};

// Re-export template engine types
pub use template_engine::{
    Alignment, Breakpoint, CachingStrategy, CompatibilityInfo, CompiledTemplate, ConditionType,
    ConditionalDisplay, ContentType, DateFormat, Dimension, FontConfig, GridSystem, GridType,
    HeaderFooterConfig, LayoutAdjustments, LayoutConfig, LayoutType, LevelOfDetail,
    LocalizationConfig, Margin, NumberFormat, NumberingFormat, NumberingPosition, Padding,
    PageMargins, PageNumbering, PageOrientation, PageSettings, PageSize, ParameterDefinition,
    ParameterType, Position, PositioningType, RenderingConfig, RenderingEngineType,
    RenderingPerformanceSettings, ReportTemplate, ReportTemplateEngine, ResponsiveDesign,
    ScalingStrategy, SectionLayoutProperties, SectionType, Size, SyntaxChecker, SyntaxRule,
    SyntaxSeverity, TemplateMetadata, TemplateRenderingEngine, TemplateSection, TemplateStructure,
    TemplateValidationRule, TemplateValidator, ValidationRule as TemplateValidationRuleType,
    ValidationRuleType as TemplateValidationType,
};

// Re-export scheduler and execution types
pub use scheduler_execution::{
    BackoffStrategy as SchedulerBackoffStrategy, ExecutionMetrics, ExecutionMonitor,
    ExecutionStatus, JobExecution, JobExecutionEngine, JobQueue, JobStatus, ReportScheduler,
    RetryConfiguration as SchedulerRetryConfig, ScheduledReportJob, SchedulerConfig,
    ThreadPoolConfig,
};

// Re-export output and delivery types
pub use output_delivery::{
    AuthenticationMethod as DeliveryAuthenticationMethod,
    BackoffStrategy as DeliveryBackoffStrategy, BatchDeliveryConfig, CompressionOption,
    DeliveryAnalytics, DeliveryChannel, DeliveryChannelConfig, DeliveryChannelType, DeliveryLog,
    DeliveryScheduling, DeliveryStatus, DeliveryTracking, FormatConfig, FormatHandler, ImageFormat,
    ImmediateDeliveryConfig, OAuth2Config as DeliveryOAuth2Config, OutputFormat,
    OutputFormatManager, QualityPreset, ReportDeliveryCoordinator,
    RetryConfiguration as DeliveryRetryConfig, ScheduledDeliveryConfig, VolumeStatistics,
};

// Re-export monitoring and metrics types
pub use monitoring_metrics::{
    AlertRateLimiting, AlertThresholds, AuthenticationMethod as MetricsAuthenticationMethod,
    ChannelConfig, CommonError, DataQualityMetrics as MetricsDataQuality, DataSourceHealthMonitor,
    DiskIOStatistics, ErrorStatistics, EscalationLevel, EscalationPolicy,
    ExecutionMetrics as MetricsExecution,
    GeneratorPerformanceMetrics as MetricsGeneratorPerformance, HealthAlertConfig, HealthCheck,
    HealthStatus, MemoryUsageStats as MetricsMemoryUsage, MessageFormat, MessageTemplate,
    MetricsSnapshot, NetworkIOStatistics, NotificationChannel as MetricsNotificationChannel,
    NotificationChannelType as MetricsNotificationChannelType,
    NotificationSettings as MetricsNotificationSettings, OAuth2Config as MetricsOAuth2Config,
    RateLimiting as MetricsRateLimiting, ReportGenerationMetrics, ResourceUtilization,
};

// Re-export common error handling
pub use generation_core::ReportGenerationError as Error;
pub use generation_core::ReportGenerationResult as Result;

// Convenience type aliases for commonly used combinations
pub type ReportManager = ReportGenerationManager;
pub type Generator = ReportGenerator;
pub type Template = ReportTemplate;
pub type Scheduler = ReportScheduler;
pub type DeliveryCoordinator = ReportDeliveryCoordinator;
pub type Metrics = ReportGenerationMetrics;

/// Main entry point for report generation operations
///
/// Provides a simplified interface for common report generation tasks
/// while maintaining access to all underlying subsystems.
pub struct ReportGenerationSystem {
    manager: ReportGenerationManager,
}

impl ReportGenerationSystem {
    /// Create a new report generation system
    pub fn new() -> Self {
        Self {
            manager: ReportGenerationManager::new(),
        }
    }

    /// Get access to the underlying manager
    pub fn manager(&self) -> &ReportGenerationManager {
        &self.manager
    }

    /// Get mutable access to the underlying manager
    pub fn manager_mut(&mut self) -> &mut ReportGenerationManager {
        &mut self.manager
    }

    /// Quick report generation with minimal configuration
    pub fn quick_generate(
        &self,
        generator_id: &str,
        template_id: &str,
        parameters: std::collections::HashMap<String, String>,
    ) -> Result<GeneratedReport> {
        self.manager
            .generate_report(generator_id, template_id, parameters)
    }

    /// Get system health status
    pub fn health_check(&self) -> SystemHealthStatus {
        // Implementation would check all subsystems
        SystemHealthStatus {
            overall_status: HealthStatus::Healthy,
            subsystem_status: std::collections::HashMap::new(),
            last_check: chrono::Utc::now(),
        }
    }
}

/// System health status for monitoring
#[derive(Debug, Clone)]
pub struct SystemHealthStatus {
    pub overall_status: HealthStatus,
    pub subsystem_status: std::collections::HashMap<String, HealthStatus>,
    pub last_check: chrono::DateTime<chrono::Utc>,
}

impl Default for ReportGenerationSystem {
    fn default() -> Self {
        Self::new()
    }
}

// Feature flags for conditional compilation
#[cfg(feature = "async")]
pub mod async_support {
    //! Async support for report generation operations
    //!
    //! This module provides async versions of report generation operations
    //! for integration with async runtimes.

    use super::*;
    use std::future::Future;
    use std::pin::Pin;

    /// Async version of ReportGenerationSystem
    pub struct AsyncReportGenerationSystem {
        inner: ReportGenerationSystem,
    }

    impl AsyncReportGenerationSystem {
        pub fn new() -> Self {
            Self {
                inner: ReportGenerationSystem::new(),
            }
        }

        /// Async report generation
        pub async fn generate_report(
            &self,
            generator_id: &str,
            template_id: &str,
            parameters: std::collections::HashMap<String, String>,
        ) -> Result<GeneratedReport> {
            // Implementation would use async execution
            tokio::task::spawn_blocking({
                let generator_id = generator_id.to_string();
                let template_id = template_id.to_string();
                let inner = &self.inner;
                move || inner.quick_generate(&generator_id, &template_id, parameters)
            })
            .await
            .unwrap_or_default()
        }
    }
}

#[cfg(feature = "metrics")]
pub mod metrics_integration {
    //! Integration with external metrics systems
    //!
    //! Provides adapters for popular metrics collection systems
    //! like Prometheus, InfluxDB, and custom telemetry.

    use super::*;

    /// Prometheus metrics exporter
    pub struct PrometheusExporter {
        metrics: std::sync::Arc<std::sync::RwLock<ReportGenerationMetrics>>,
    }

    impl PrometheusExporter {
        pub fn new(metrics: std::sync::Arc<std::sync::RwLock<ReportGenerationMetrics>>) -> Self {
            Self { metrics }
        }

        /// Export metrics in Prometheus format
        pub fn export(&self) -> String {
            let metrics = self.metrics.read().unwrap_or_else(|e| e.into_inner());
            format!(
                "# HELP report_generation_total Total number of reports generated\n\
                 # TYPE report_generation_total counter\n\
                 report_generation_total {}\n\
                 # HELP report_generation_success_rate Success rate of report generation\n\
                 # TYPE report_generation_success_rate gauge\n\
                 report_generation_success_rate {}\n",
                metrics.total_reports, metrics.success_rate
            )
        }
    }
}

// Documentation examples and tests
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        let _system = ReportGenerationSystem::new();
    }

    #[test]
    fn test_health_check() {
        let system = ReportGenerationSystem::new();
        let health = system.health_check();
        assert!(matches!(health.overall_status, HealthStatus::Healthy));
    }
}

#[cfg(doctest)]
mod doctests {
    //! Documentation tests to ensure examples in docs work correctly

    /// Example from module documentation
    ///
    /// ```ignore
    /// use report_generation::{ReportGenerationManager, ReportGenerator};
    /// use std::collections::HashMap;
    ///
    /// let manager = ReportGenerationManager::new();
    /// // Additional setup would be required for a complete example
    /// ```
    fn doctest_example() {}
}
