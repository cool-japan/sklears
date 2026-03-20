//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use crate::monitoring_config::*;
use crate::monitoring_metrics::*;
use crate::monitoring_events::*;
use crate::monitoring_core::*;

use super::types::{ChartConfig, ColorScheme, DashboardLayout, DetailLevel, ExecutiveSummary, KeyPerformanceIndicator, KpiStatus, PageLayout, RecommendationPriority, ReportConfig, ReportFilters, ReportFormat, ReportFormatting, ReportGenerator, ReportGeneratorConfig, ReportSection, ReportType, VisualizationConfig};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_report_generation() {
        let config = ReportGeneratorConfig {
            default_template: "test".to_string(),
            output_directory: "/tmp".to_string(),
            enable_caching: false,
            cache_duration: Duration::from_secs(60),
        };
        let generator = ReportGenerator::new(config);
        let report_config = ReportConfig {
            report_type: ReportType::Summary,
            detail_level: DetailLevel::Standard,
            format: ReportFormat::Json {
                pretty_print: true,
                include_metadata: true,
            },
            sections: vec![
                ReportSection::ExecutiveSummary, ReportSection::PerformanceMetrics
            ],
            visualizations: VisualizationConfig {
                enabled: false,
                chart_types: Vec::new(),
                chart_config: ChartConfig {
                    width: 800,
                    height: 600,
                    color_scheme: "default".to_string(),
                    interactive: false,
                    library: "default".to_string(),
                    custom_css: None,
                },
                layout: DashboardLayout::SingleColumn,
            },
            filters: ReportFilters {
                metric_filters: Vec::new(),
                event_filters: Vec::new(),
                severity_filters: Vec::new(),
                tag_filters: HashMap::new(),
                time_filters: Vec::new(),
            },
            parameters: HashMap::new(),
        };
        let metrics = vec![
            PerformanceMetric::new("test_metric".to_string(), 42.0, "units".to_string())
        ];
        let events = Vec::new();
        let alerts = Vec::new();
        let result = generator
            .generate_report("test_session", &report_config, &metrics, &events, &alerts);
        assert!(result.is_ok());
        let report = result.expect("should succeed");
        assert_eq!(report.metadata.session_id, "test_session");
        assert!(matches!(report.metadata.report_type, ReportType::Summary));
    }
    #[test]
    fn test_kpi_creation() {
        let kpi = KeyPerformanceIndicator {
            name: "Response Time".to_string(),
            current_value: 150.0,
            target_value: Some(100.0),
            previous_value: Some(200.0),
            unit: "ms".to_string(),
            trend: TrendDirection::Improving,
            status: KpiStatus::Good,
            description: "Average response time".to_string(),
        };
        assert_eq!(kpi.name, "Response Time");
        assert_eq!(kpi.current_value, 150.0);
        assert!(matches!(kpi.trend, TrendDirection::Improving));
        assert!(matches!(kpi.status, KpiStatus::Good));
    }
    #[test]
    fn test_report_formatting() {
        let formatting = ReportFormatting {
            theme: "dark".to_string(),
            font_family: "Helvetica".to_string(),
            colors: ColorScheme::default(),
            layout: PageLayout::default(),
            custom_css: Some("body { margin: 0; }".to_string()),
        };
        assert_eq!(formatting.theme, "dark");
        assert_eq!(formatting.font_family, "Helvetica");
        assert!(formatting.custom_css.is_some());
    }
    #[test]
    fn test_recommendation_priority() {
        let high_priority = RecommendationPriority::High;
        let low_priority = RecommendationPriority::Low;
        assert!(matches!(high_priority, RecommendationPriority::High));
        assert!(matches!(low_priority, RecommendationPriority::Low));
    }
}
