//! Compliance metrics, analytics, and reporting
//!
//! This module provides comprehensive compliance metrics collection, analysis,
//! and reporting capabilities to track compliance program effectiveness,
//! identify trends, and generate insights for continuous improvement.

use std::{
    collections::HashMap,
    time::{Duration, SystemTime},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::compliance_core::RegulatoryFramework;
use crate::compliance_audit::{AuditStatistics, AuditEventType};
use crate::compliance_policy::PolicyStatistics;
use crate::compliance_consent::ConsentStatistics;
use crate::compliance_retention::RetentionStatistics;
use crate::compliance_breach::BreachStatistics;
use crate::compliance_regulatory::{ComplianceSeverity, MaturityLevel};

/// Comprehensive compliance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplianceMetrics {
    /// Total compliance checks performed
    pub compliance_checks: usize,
    /// Successful compliance checks
    pub successful_checks: usize,
    /// Policy violations detected
    pub policy_violations: usize,
    /// Data breaches detected
    pub breaches_detected: usize,
    /// Consent requests processed
    pub consent_requests: usize,
    /// Data retention actions
    pub retention_actions: usize,
    /// Audit findings count
    pub audit_findings: usize,
    /// Overall compliance score (0.0 - 1.0)
    pub compliance_score: f64,
    /// Overall risk score (0.0 - 1.0)
    pub risk_score: f64,
    /// Metrics collection timestamp
    pub last_updated: SystemTime,
    /// Framework-specific metrics
    pub framework_metrics: HashMap<RegulatoryFramework, FrameworkMetrics>,
    /// Trend analysis data
    pub trends: ComplianceTrends,
    /// Custom metrics
    pub custom: HashMap<String, serde_json::Value>,
}

/// Framework-specific compliance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkMetrics {
    /// Framework
    pub framework: RegulatoryFramework,
    /// Implementation completeness (0.0 - 1.0)
    pub implementation_completeness: f64,
    /// Control effectiveness score (0.0 - 1.0)
    pub control_effectiveness: f64,
    /// Maturity level
    pub maturity_level: MaturityLevel,
    /// Violation count for this framework
    pub violations: usize,
    /// Active remediation plans
    pub active_remediations: usize,
    /// Last assessment score
    pub last_assessment_score: f64,
    /// Last assessment date
    pub last_assessed: Option<SystemTime>,
}

/// Compliance trends and analytics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplianceTrends {
    /// Historical compliance scores
    pub compliance_history: Vec<ComplianceSnapshot>,
    /// Risk trend
    pub risk_trend: TrendDirection,
    /// Violation trend
    pub violation_trend: TrendDirection,
    /// Breach trend
    pub breach_trend: TrendDirection,
    /// Consent trend
    pub consent_trend: TrendDirection,
    /// Audit effectiveness trend
    pub audit_trend: TrendDirection,
}

/// Compliance snapshot for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Compliance score at this time
    pub compliance_score: f64,
    /// Risk score at this time
    pub risk_score: f64,
    /// Active violations at this time
    pub violations: usize,
    /// Active breaches at this time
    pub breaches: usize,
    /// Maturity level at this time
    pub maturity_level: MaturityLevel,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Stable trend
    Stable,
    /// Declining trend
    Declining,
    /// Unknown or insufficient data
    Unknown,
}

/// Compliance metrics collector and analyzer
#[derive(Debug)]
pub struct ComplianceMetricsCollector {
    /// Current metrics
    pub metrics: ComplianceMetrics,
    /// Collection configuration
    pub config: MetricsConfig,
    /// Historical snapshots
    pub history: Vec<ComplianceSnapshot>,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable automatic collection
    pub auto_collection: bool,
    /// Collection interval
    pub collection_interval: Duration,
    /// History retention period
    pub history_retention: Duration,
    /// Enable trend analysis
    pub enable_trends: bool,
    /// Minimum data points for trend analysis
    pub min_trend_points: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            auto_collection: true,
            collection_interval: Duration::from_secs(60 * 60), // Hourly
            history_retention: Duration::from_secs(365 * 24 * 60 * 60), // 1 year
            enable_trends: true,
            min_trend_points: 5,
        }
    }
}

/// Compliance dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceDashboard {
    /// Summary metrics
    pub summary: DashboardSummary,
    /// Key performance indicators
    pub kpis: Vec<ComplianceKpi>,
    /// Recent alerts and incidents
    pub alerts: Vec<ComplianceAlert>,
    /// Upcoming deadlines
    pub deadlines: Vec<ComplianceDeadline>,
    /// Framework status overview
    pub frameworks: Vec<FrameworkStatus>,
    /// Generated timestamp
    pub generated_at: SystemTime,
}

/// Dashboard summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    /// Overall health status
    pub health_status: HealthStatus,
    /// Overall compliance score
    pub compliance_score: f64,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Active issues count
    pub active_issues: usize,
    /// Overdue items count
    pub overdue_items: usize,
}

/// Health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Healthy - no issues
    Healthy,
    /// Warning - minor issues
    Warning,
    /// Critical - major issues
    Critical,
    /// Failing - severe issues
    Failing,
}

/// Risk level
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low = 1,
    /// Medium risk
    Medium = 2,
    /// High risk
    High = 3,
    /// Critical risk
    Critical = 4,
}

/// Compliance KPI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceKpi {
    /// KPI name
    pub name: String,
    /// Current value
    pub current_value: f64,
    /// Target value
    pub target_value: f64,
    /// Previous value (for trend)
    pub previous_value: Option<f64>,
    /// Unit of measurement
    pub unit: String,
    /// Trend direction
    pub trend: TrendDirection,
    /// Status relative to target
    pub status: KpiStatus,
}

/// KPI status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KpiStatus {
    /// On target
    OnTarget,
    /// Below target
    BelowTarget,
    /// Above target
    AboveTarget,
    /// Critical deviation
    Critical,
}

/// Compliance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceAlert {
    /// Alert ID
    pub id: Uuid,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: ComplianceSeverity,
    /// Alert message
    pub message: String,
    /// Affected component
    pub component: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Resolution status
    pub status: AlertStatus,
}

/// Alert type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// Policy violation
    PolicyViolation,
    /// Data breach
    DataBreach,
    /// Compliance gap
    ComplianceGap,
    /// Audit finding
    AuditFinding,
    /// Deadline approaching
    DeadlineApproaching,
    /// System anomaly
    SystemAnomaly,
    /// Custom alert
    Custom(String),
}

/// Alert status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Active alert
    Active,
    /// Acknowledged
    Acknowledged,
    /// Resolved
    Resolved,
    /// Dismissed
    Dismissed,
}

/// Compliance deadline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceDeadline {
    /// Deadline ID
    pub id: String,
    /// Description
    pub description: String,
    /// Due date
    pub due_date: SystemTime,
    /// Priority
    pub priority: DeadlinePriority,
    /// Associated framework
    pub framework: Option<RegulatoryFramework>,
    /// Completion status
    pub status: DeadlineStatus,
}

/// Deadline priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DeadlinePriority {
    /// Low priority
    Low = 1,
    /// Medium priority
    Medium = 2,
    /// High priority
    High = 3,
    /// Critical priority
    Critical = 4,
}

/// Deadline status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadlineStatus {
    /// Upcoming
    Upcoming,
    /// Overdue
    Overdue,
    /// Completed
    Completed,
    /// Cancelled
    Cancelled,
}

/// Framework status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkStatus {
    /// Framework
    pub framework: RegulatoryFramework,
    /// Implementation status
    pub implementation_status: String,
    /// Compliance percentage
    pub compliance_percentage: f64,
    /// Active controls
    pub active_controls: usize,
    /// Failed controls
    pub failed_controls: usize,
    /// Last assessment
    pub last_assessment: Option<SystemTime>,
}

impl ComplianceMetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: ComplianceMetrics::default(),
            config: MetricsConfig::default(),
            history: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: MetricsConfig) -> Self {
        Self {
            metrics: ComplianceMetrics::default(),
            config,
            history: Vec::new(),
        }
    }

    /// Collect comprehensive metrics from all compliance components
    pub fn collect_metrics(
        &mut self,
        audit_stats: &AuditStatistics,
        policy_stats: &PolicyStatistics,
        consent_stats: &ConsentStatistics,
        retention_stats: &RetentionStatistics,
        breach_stats: &BreachStatistics,
    ) {
        let now = SystemTime::now();

        // Update core metrics
        self.metrics.compliance_checks = audit_stats.total_entries;
        self.metrics.successful_checks = audit_stats.total_entries - audit_stats.high_risk_entries;
        self.metrics.policy_violations = policy_stats.total_violations;
        self.metrics.breaches_detected = breach_stats.total_breaches;
        self.metrics.consent_requests = consent_stats.total_consents;
        self.metrics.retention_actions = retention_stats.completed_deletions;
        self.metrics.audit_findings = audit_stats.completed_audits;
        self.metrics.last_updated = now;

        // Calculate overall compliance score
        self.metrics.compliance_score = self.calculate_compliance_score(
            audit_stats, policy_stats, consent_stats, retention_stats, breach_stats
        );

        // Calculate overall risk score
        self.metrics.risk_score = self.calculate_risk_score(
            audit_stats, policy_stats, breach_stats
        );

        // Update trends if enabled
        if self.config.enable_trends {
            self.update_trends();
        }

        // Create snapshot for history
        let snapshot = ComplianceSnapshot {
            timestamp: now,
            compliance_score: self.metrics.compliance_score,
            risk_score: self.metrics.risk_score,
            violations: self.metrics.policy_violations,
            breaches: self.metrics.breaches_detected,
            maturity_level: MaturityLevel::Defined, // Would be calculated from actual data
        };

        self.history.push(snapshot);
        self.cleanup_old_history();
    }

    /// Calculate overall compliance score
    fn calculate_compliance_score(
        &self,
        audit_stats: &AuditStatistics,
        policy_stats: &PolicyStatistics,
        consent_stats: &ConsentStatistics,
        retention_stats: &RetentionStatistics,
        breach_stats: &BreachStatistics,
    ) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Audit compliance (weight: 0.3)
        if audit_stats.total_entries > 0 {
            let audit_score = audit_stats.success_rate;
            score += audit_score * 0.3;
            weight_sum += 0.3;
        }

        // Policy compliance (weight: 0.25)
        if policy_stats.total_policies > 0 {
            let policy_score = if policy_stats.total_violations == 0 {
                1.0
            } else {
                1.0 - (policy_stats.total_violations as f64 / policy_stats.total_policies as f64).min(1.0)
            };
            score += policy_score * 0.25;
            weight_sum += 0.25;
        }

        // Consent compliance (weight: 0.2)
        let consent_score = consent_stats.consent_rate;
        score += consent_score * 0.2;
        weight_sum += 0.2;

        // Retention compliance (weight: 0.15)
        let retention_score = retention_stats.completion_rate;
        score += retention_score * 0.15;
        weight_sum += 0.15;

        // Breach response (weight: 0.1)
        let breach_score = breach_stats.resolution_rate;
        score += breach_score * 0.1;
        weight_sum += 0.1;

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.0
        }
    }

    /// Calculate overall risk score
    fn calculate_risk_score(
        &self,
        audit_stats: &AuditStatistics,
        policy_stats: &PolicyStatistics,
        breach_stats: &BreachStatistics,
    ) -> f64 {
        let mut risk = 0.0;

        // High-risk audit entries
        if audit_stats.total_entries > 0 {
            risk += (audit_stats.high_risk_entries as f64 / audit_stats.total_entries as f64) * 0.4;
        }

        // Open policy violations
        if policy_stats.total_violations > 0 {
            risk += (policy_stats.open_violations as f64 / policy_stats.total_violations as f64) * 0.3;
        }

        // Critical breaches
        if breach_stats.total_breaches > 0 {
            risk += (breach_stats.critical_breaches as f64 / breach_stats.total_breaches as f64) * 0.3;
        }

        risk.min(1.0)
    }

    /// Update trend analysis
    fn update_trends(&mut self) {
        if self.history.len() < self.config.min_trend_points {
            return;
        }

        let recent_snapshots: Vec<&ComplianceSnapshot> = self.history
            .iter()
            .rev()
            .take(self.config.min_trend_points)
            .collect();

        // Analyze compliance score trend
        self.metrics.trends.risk_trend = self.analyze_trend(
            &recent_snapshots.iter().map(|s| s.risk_score).collect::<Vec<f64>>()
        );

        // Analyze violation trend
        self.metrics.trends.violation_trend = self.analyze_trend(
            &recent_snapshots.iter().map(|s| s.violations as f64).collect::<Vec<f64>>()
        );

        // Analyze breach trend
        self.metrics.trends.breach_trend = self.analyze_trend(
            &recent_snapshots.iter().map(|s| s.breaches as f64).collect::<Vec<f64>>()
        );
    }

    /// Analyze trend direction from data points
    fn analyze_trend(&self, data: &[f64]) -> TrendDirection {
        if data.len() < 3 {
            return TrendDirection::Unknown;
        }

        let first_half = &data[0..data.len()/2];
        let second_half = &data[data.len()/2..];

        let first_avg: f64 = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg: f64 = second_half.iter().sum::<f64>() / second_half.len() as f64;

        let change_threshold = 0.05; // 5% change threshold

        if second_avg > first_avg * (1.0 + change_threshold) {
            TrendDirection::Declining // Higher values are worse for most metrics
        } else if second_avg < first_avg * (1.0 - change_threshold) {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        }
    }

    /// Generate compliance dashboard
    pub fn generate_dashboard(&self) -> ComplianceDashboard {
        let health_status = self.determine_health_status();
        let risk_level = self.determine_risk_level();

        ComplianceDashboard {
            summary: DashboardSummary {
                health_status,
                compliance_score: self.metrics.compliance_score,
                risk_level,
                active_issues: self.metrics.policy_violations + self.metrics.breaches_detected,
                overdue_items: 0, // Would be calculated from actual data
            },
            kpis: self.generate_kpis(),
            alerts: self.generate_alerts(),
            deadlines: self.generate_deadlines(),
            frameworks: self.generate_framework_status(),
            generated_at: SystemTime::now(),
        }
    }

    /// Determine overall health status
    fn determine_health_status(&self) -> HealthStatus {
        if self.metrics.compliance_score >= 0.9 && self.metrics.risk_score <= 0.2 {
            HealthStatus::Healthy
        } else if self.metrics.compliance_score >= 0.7 && self.metrics.risk_score <= 0.5 {
            HealthStatus::Warning
        } else if self.metrics.compliance_score >= 0.5 && self.metrics.risk_score <= 0.8 {
            HealthStatus::Critical
        } else {
            HealthStatus::Failing
        }
    }

    /// Determine risk level
    fn determine_risk_level(&self) -> RiskLevel {
        if self.metrics.risk_score <= 0.25 {
            RiskLevel::Low
        } else if self.metrics.risk_score <= 0.5 {
            RiskLevel::Medium
        } else if self.metrics.risk_score <= 0.75 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }

    /// Generate KPIs
    fn generate_kpis(&self) -> Vec<ComplianceKpi> {
        vec![
            ComplianceKpi {
                name: "Compliance Score".to_string(),
                current_value: self.metrics.compliance_score * 100.0,
                target_value: 95.0,
                previous_value: None, // Would be calculated from history
                unit: "%".to_string(),
                trend: TrendDirection::Stable,
                status: if self.metrics.compliance_score >= 0.95 {
                    KpiStatus::OnTarget
                } else {
                    KpiStatus::BelowTarget
                },
            },
            ComplianceKpi {
                name: "Risk Score".to_string(),
                current_value: self.metrics.risk_score * 100.0,
                target_value: 20.0,
                previous_value: None,
                unit: "%".to_string(),
                trend: self.metrics.trends.risk_trend.clone(),
                status: if self.metrics.risk_score <= 0.2 {
                    KpiStatus::OnTarget
                } else {
                    KpiStatus::AboveTarget
                },
            },
        ]
    }

    /// Generate alerts (placeholder)
    fn generate_alerts(&self) -> Vec<ComplianceAlert> {
        let mut alerts = Vec::new();

        if self.metrics.risk_score > 0.8 {
            alerts.push(ComplianceAlert {
                id: Uuid::new_v4(),
                alert_type: AlertType::SystemAnomaly,
                severity: ComplianceSeverity::Critical,
                message: "High risk score detected".to_string(),
                component: "Risk Management".to_string(),
                timestamp: SystemTime::now(),
                status: AlertStatus::Active,
            });
        }

        alerts
    }

    /// Generate deadlines (placeholder)
    fn generate_deadlines(&self) -> Vec<ComplianceDeadline> {
        Vec::new() // Would be populated from actual compliance requirements
    }

    /// Generate framework status (placeholder)
    fn generate_framework_status(&self) -> Vec<FrameworkStatus> {
        self.metrics.framework_metrics
            .iter()
            .map(|(framework, metrics)| FrameworkStatus {
                framework: framework.clone(),
                implementation_status: "Active".to_string(),
                compliance_percentage: metrics.implementation_completeness * 100.0,
                active_controls: 10, // Placeholder
                failed_controls: 0,  // Placeholder
                last_assessment: metrics.last_assessed,
            })
            .collect()
    }

    /// Clean up old history
    fn cleanup_old_history(&mut self) {
        let cutoff_time = SystemTime::now()
            .checked_sub(self.config.history_retention)
            .unwrap_or(SystemTime::UNIX_EPOCH);

        self.history.retain(|snapshot| snapshot.timestamp > cutoff_time);
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &ComplianceMetrics {
        &self.metrics
    }

    /// Get metrics history
    pub fn get_history(&self) -> &[ComplianceSnapshot] {
        &self.history
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = ComplianceMetricsCollector::new();
        assert_eq!(collector.metrics.compliance_checks, 0);
        assert_eq!(collector.history.len(), 0);
        assert!(collector.config.auto_collection);
    }

    #[test]
    fn test_compliance_score_calculation() {
        let collector = ComplianceMetricsCollector::new();

        let audit_stats = AuditStatistics {
            total_entries: 100,
            high_risk_entries: 10,
            active_audits: 2,
            completed_audits: 5,
            success_rate: 0.9,
        };

        let policy_stats = PolicyStatistics {
            total_policies: 10,
            active_policies: 8,
            total_violations: 2,
            open_violations: 1,
            violation_rate: 0.2,
        };

        let consent_stats = ConsentStatistics {
            total_consents: 100,
            granted_consents: 90,
            withdrawn_consents: 5,
            expired_consents: 5,
            pending_requests: 0,
            pending_withdrawals: 0,
            consent_rate: 0.9,
        };

        let retention_stats = RetentionStatistics {
            total_policies: 5,
            active_policies: 5,
            total_schedules: 20,
            due_schedules: 2,
            completed_deletions: 15,
            active_legal_holds: 1,
            unverified_deletions: 0,
            completion_rate: 0.75,
        };

        let breach_stats = BreachStatistics {
            total_breaches: 5,
            critical_breaches: 1,
            active_investigations: 2,
            overdue_notifications: 0,
            resolved_breaches: 3,
            resolution_rate: 0.6,
        };

        let score = collector.calculate_compliance_score(
            &audit_stats, &policy_stats, &consent_stats, &retention_stats, &breach_stats
        );

        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_trend_analysis() {
        let collector = ComplianceMetricsCollector::new();

        // Improving trend
        let improving_data = vec![0.8, 0.7, 0.6, 0.5, 0.4];
        assert_eq!(collector.analyze_trend(&improving_data), TrendDirection::Improving);

        // Declining trend
        let declining_data = vec![0.4, 0.5, 0.6, 0.7, 0.8];
        assert_eq!(collector.analyze_trend(&declining_data), TrendDirection::Declining);

        // Stable trend
        let stable_data = vec![0.5, 0.52, 0.48, 0.51, 0.49];
        assert_eq!(collector.analyze_trend(&stable_data), TrendDirection::Stable);
    }

    #[test]
    fn test_health_status_determination() {
        let mut collector = ComplianceMetricsCollector::new();

        // Healthy status
        collector.metrics.compliance_score = 0.95;
        collector.metrics.risk_score = 0.1;
        assert_eq!(collector.determine_health_status(), HealthStatus::Healthy);

        // Warning status
        collector.metrics.compliance_score = 0.8;
        collector.metrics.risk_score = 0.3;
        assert_eq!(collector.determine_health_status(), HealthStatus::Warning);

        // Critical status
        collector.metrics.compliance_score = 0.6;
        collector.metrics.risk_score = 0.6;
        assert_eq!(collector.determine_health_status(), HealthStatus::Critical);

        // Failing status
        collector.metrics.compliance_score = 0.3;
        collector.metrics.risk_score = 0.9;
        assert_eq!(collector.determine_health_status(), HealthStatus::Failing);
    }

    #[test]
    fn test_risk_level_determination() {
        let mut collector = ComplianceMetricsCollector::new();

        collector.metrics.risk_score = 0.1;
        assert_eq!(collector.determine_risk_level(), RiskLevel::Low);

        collector.metrics.risk_score = 0.4;
        assert_eq!(collector.determine_risk_level(), RiskLevel::Medium);

        collector.metrics.risk_score = 0.7;
        assert_eq!(collector.determine_risk_level(), RiskLevel::High);

        collector.metrics.risk_score = 0.9;
        assert_eq!(collector.determine_risk_level(), RiskLevel::Critical);
    }

    #[test]
    fn test_dashboard_generation() {
        let collector = ComplianceMetricsCollector::new();
        let dashboard = collector.generate_dashboard();

        assert!(dashboard.kpis.len() >= 2); // Should have at least compliance and risk KPIs
        assert_eq!(dashboard.summary.compliance_score, collector.metrics.compliance_score);
    }
}