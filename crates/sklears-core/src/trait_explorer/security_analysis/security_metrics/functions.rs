//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::security_types::*;
use std::collections::HashMap;
use std::time::Duration;

use super::macros::MetricValue;
use super::types_7::SecurityMetricsError;
use super::types_8::SecurityMetricsCollector;
use super::types_9::{MetricCollection, SecurityMetricsResult};

/// Convert a local [`MetricValue`] into a numeric approximation usable in shallow heuristics.
pub(super) fn metric_value_as_f64(value: &MetricValue) -> f64 {
    match value {
        MetricValue::Integer(i) => *i as f64,
        MetricValue::Float(f) => *f,
        MetricValue::Boolean(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        MetricValue::String(_) => 0.0,
    }
}
/// Average data-quality score across a collection of metrics, with a neutral fallback when empty.
pub(super) fn average_quality(metrics: &HashMap<String, MetricCollection>) -> f64 {
    if metrics.is_empty() {
        return 0.75;
    }
    metrics.values().map(|m| m.quality_score).sum::<f64>() / metrics.len() as f64
}
/// Build a shallow [`BusinessImpactAssessment`] from a single 0.0-1.0 severity signal.
pub(super) fn business_impact_from_severity(
    severity: f64,
    personal_data_exposed: bool,
) -> BusinessImpactAssessment {
    let s = severity.clamp(0.0, 1.0);
    BusinessImpactAssessment {
        financial_impact: FinancialImpact {
            direct_costs: s * 10_000.0,
            indirect_costs: s * 5_000.0,
            revenue_loss: s * 20_000.0,
            regulatory_fines: 0.0,
            legal_costs: 0.0,
            recovery_costs: s * 2_000.0,
        },
        operational_impact: OperationalImpact {
            service_disruption_duration: Duration::from_secs((s * 3600.0) as u64),
            affected_business_processes: Vec::new(),
            productivity_loss_percentage: s * 100.0,
            recovery_time_objective: Duration::from_secs(3600),
            recovery_point_objective: Duration::from_secs(900),
        },
        reputational_impact: ReputationalImpact {
            brand_damage_score: s * 10.0,
            customer_trust_impact: s,
            media_attention_level: MediaAttentionLevel::None,
            social_media_sentiment: SentimentLevel::Neutral,
            stakeholder_confidence_impact: s,
        },
        legal_impact: LegalImpact {
            regulatory_violations: Vec::new(),
            potential_lawsuits: 0,
            compliance_breach_severity: ComplianceBreachSeverity::Minor,
            data_protection_violations: Vec::new(),
            contractual_breach_risk: s * 0.5,
        },
        customer_impact: CustomerImpact {
            affected_customer_count: 0,
            customer_data_exposed: personal_data_exposed,
            service_availability_impact: s,
            customer_satisfaction_impact: s,
            churn_risk_percentage: s * 10.0,
        },
        competitive_impact: CompetitiveImpact {
            competitive_advantage_loss: s * 0.3,
            intellectual_property_exposure: false,
            market_share_impact: s * 0.1,
            innovation_capability_impact: s * 0.2,
        },
    }
}
pub fn create_security_metrics_collector() -> SecurityMetricsCollector {
    SecurityMetricsCollector::new()
}
pub fn collect_comprehensive_security_metrics(
    context: &TraitUsageContext,
) -> Result<SecurityMetricsResult, SecurityMetricsError> {
    let mut collector = SecurityMetricsCollector::new();
    collector.collect_security_metrics(context)
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Run the full security-metrics collection entry point against a high-signal context and
    /// assert every top-level section of the result is populated without panicking.
    #[test]
    fn test_collect_comprehensive_security_metrics_smoke() {
        let context = TraitUsageContext {
            trait_name: "Serialize".to_string(),
            traits: vec!["Serialize".to_string(), "Clone".to_string()],
            handles_sensitive_data: true,
            handles_personal_data: true,
            has_unsafe_operations: true,
            has_bounds_checking: false,
            has_audit_logging: false,
            has_data_anonymization: false,
            requires_elevated_privileges: true,
            ..Default::default()
        };
        let result = collect_comprehensive_security_metrics(&context);
        assert!(
            result.is_ok(),
            "metrics collection should succeed: {result:?}"
        );
        let metrics = result.expect("metrics collection should succeed");
        assert!(
            !metrics.metric_collections.is_empty(),
            "expected at least one collected metric"
        );
        assert!((0.0..=10.0).contains(&metrics.overall_security_score));
        assert!((0.0..=1.0).contains(&metrics.analysis_confidence));
        assert!(
            !metrics.health_indicators.is_empty(),
            "expected health indicators derived from collected metrics"
        );
    }
    /// The lower-risk constructor path should also succeed against a default, all-`false` context.
    #[test]
    fn test_create_security_metrics_collector_default_context() {
        let mut collector = create_security_metrics_collector();
        let context = TraitUsageContext::default();
        let result = collector.collect_security_metrics(&context);
        assert!(
            result.is_ok(),
            "collection with a default context should succeed: {result:?}"
        );
        let metrics = result.expect("collection with a default context should succeed");
        assert_eq!(
            metrics.metric_collections.len(),
            12,
            "6 collectors x 2 metric definitions each"
        );
    }
}
