//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::collections::HashMap;
use std::time::SystemTime;

use super::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_2 {
    use super::*;
    #[test]
    fn test_regression_detector_creation() {
        let config = RegressionDetectorConfig::default();
        let detector = RegressionDetector::new(config);
        assert_eq!(detector.detection_algorithms.len(), 4);
    }
    #[test]
    fn test_regression_severity_assessment() {
        let detector = RegressionDetector::default();
        let critical_regression = DetectedRegression {
            regression_id: "test".to_string(),
            benchmark_id: "test".to_string(),
            metric_name: "test".to_string(),
            regression_type: RegressionType::PerformanceDegradation,
            severity: RegressionSeverity::Critical,
            current_value: 150.0,
            expected_value: 100.0,
            degradation_percentage: 50.0,
            detection_confidence: 0.9,
            first_detected: SystemTime::now(),
            consecutive_failures: 1,
            detection_method: "Test".to_string(),
            context: RegressionContext {
                environmental_factors: Vec::new(),
                recent_changes: Vec::new(),
                system_metrics: SystemMetrics::default(),
                additional_info: HashMap::new(),
            },
        };
        let assessment = detector
            .assess_regression_severity(&[critical_regression])
            .unwrap_or_default();
        assert_eq!(assessment.critical_regressions, 1);
        assert!(matches!(
            assessment.overall_severity,
            OverallRegressionSeverity::Critical
        ));
    }
    #[test]
    fn test_threshold_management() {
        let mut threshold_mgmt = ThresholdManagement::new();
        threshold_mgmt.update_sensitivity(&DetectionSensitivity::High);
        assert_eq!(threshold_mgmt.threshold_types.len(), 2);
    }
    #[test]
    fn test_alert_system() {
        let alert_system = RegressionAlertSystem::new();
        assert_eq!(alert_system.alert_rules.len(), 2);
        assert_eq!(alert_system.notification_channels.len(), 1);
    }
    #[test]
    fn test_regression_cache() {
        let mut cache = RegressionCache::new();
        assert!(cache.get("test_key").is_none());
        assert_eq!(cache.max_size, 1000);
    }
}
