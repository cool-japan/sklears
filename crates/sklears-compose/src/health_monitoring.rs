//! Health Monitoring and System Status Assessment
//!
//! This module provides comprehensive health monitoring capabilities including
//! system health checks, component status tracking, health scoring, and
//! automated health assessment for monitoring infrastructure.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, Instant};
use std::fmt;

use crate::monitoring_core::{SystemHealth, ComponentHealth, HealthStatus, HealthIssue, SeverityLevel};
use crate::metrics_collection::PerformanceMetric;
use crate::configuration_management::HealthCheckConfig;

/// Health checker for comprehensive system health monitoring
///
/// Monitors system components, evaluates health status, tracks health trends,
/// and provides automated health assessment and issue detection.
#[derive(Debug)]
pub struct HealthChecker {
    /// Registered health monitors
    monitors: HashMap<String, Box<dyn HealthMonitor>>,

    /// Component health states
    component_states: HashMap<String, ComponentHealthState>,

    /// Health check configuration
    config: HealthCheckConfiguration,

    /// Health statistics
    stats: HealthStatistics,

    /// System health evaluator
    evaluator: HealthEvaluator,

    /// Health history for trend analysis
    health_history: VecDeque<SystemHealthSnapshot>,

    /// Thread safety lock
    lock: Arc<RwLock<()>>,
}

impl HealthChecker {
    /// Create new health checker
    pub fn new(config: HealthCheckConfiguration) -> Self {
        Self {
            monitors: HashMap::new(),
            component_states: HashMap::new(),
            config: config.clone(),
            stats: HealthStatistics::new(),
            evaluator: HealthEvaluator::new(config.evaluation),
            health_history: VecDeque::with_capacity(config.history_size),
            lock: Arc::new(RwLock::new(())),
        }
    }

    /// Register health monitor for component
    pub fn register_monitor(&mut self, component_name: String, monitor: Box<dyn HealthMonitor>) -> SklResult<()> {
        let _lock = self.lock.write().unwrap();

        if self.monitors.contains_key(&component_name) {
            return Err(SklearsError::InvalidInput(
                format!("Monitor for component '{}' already registered", component_name)
            ));
        }

        self.monitors.insert(component_name.clone(), monitor);
        self.component_states.insert(component_name, ComponentHealthState::new());

        Ok(())
    }

    /// Remove health monitor
    pub fn remove_monitor(&mut self, component_name: &str) -> SklResult<()> {
        let _lock = self.lock.write().unwrap();

        self.monitors.remove(component_name);
        self.component_states.remove(component_name);

        Ok(())
    }

    /// Perform comprehensive health check
    pub fn check_health(&mut self) -> SklResult<SystemHealth> {
        let _lock = self.lock.write().unwrap();
        let start_time = Instant::now();

        let mut component_health = HashMap::new();
        let mut issues = Vec::new();
        let mut total_score = 0.0;
        let mut component_count = 0;

        // Check each component
        for (component_name, monitor) in &self.monitors {
            match self.check_component_health(component_name, monitor.as_ref()) {
                Ok(health) => {
                    total_score += health.score;
                    component_count += 1;

                    // Collect issues from this component
                    issues.extend(health.issues.clone());

                    component_health.insert(component_name.clone(), health);
                }
                Err(e) => {
                    log::error!("Health check failed for component '{}': {}", component_name, e);
                    self.stats.failed_checks += 1;

                    // Create failed health status
                    let failed_health = ComponentHealth {
                        component: component_name.clone(),
                        status: HealthStatus::Critical,
                        score: 0.0,
                        last_check: SystemTime::now(),
                        issues: vec![format!("Health check failed: {}", e)],
                    };

                    component_health.insert(component_name.clone(), failed_health);
                }
            }
        }

        // Calculate overall health
        let overall_score = if component_count > 0 {
            total_score / component_count as f64
        } else {
            1.0
        };

        let overall_status = self.evaluator.determine_status(overall_score, &issues);

        let system_health = SystemHealth {
            status: overall_status,
            components: component_health,
            score: overall_score,
            issues: issues,
        };

        // Update statistics
        let check_duration = start_time.elapsed();
        self.stats.total_checks += 1;
        self.stats.total_check_time += check_duration;
        self.stats.last_check = SystemTime::now();

        // Update health history
        self.update_health_history(&system_health);

        Ok(system_health)
    }

    /// Check health of specific component
    fn check_component_health(&mut self, component_name: &str, monitor: &dyn HealthMonitor) -> SklResult<ComponentHealth> {
        let state = self.component_states.get_mut(component_name)
            .ok_or_else(|| SklearsError::NotFound(format!("Component state for '{}' not found", component_name)))?;

        // Perform health check
        let health_result = monitor.check_health()?;

        // Update component state
        state.last_check = SystemTime::now();
        state.check_count += 1;

        // Calculate health score
        let score = self.evaluator.calculate_component_score(&health_result);

        // Determine status
        let status = self.evaluator.determine_component_status(score, &health_result.issues);

        // Update state based on status
        match status {
            HealthStatus::Healthy => state.healthy_checks += 1,
            HealthStatus::Warning => state.warning_checks += 1,
            HealthStatus::Critical => state.critical_checks += 1,
            HealthStatus::Unknown => state.unknown_checks += 1,
        }

        // Convert issues to string format
        let issue_strings: Vec<String> = health_result.issues.iter()
            .map(|issue| format!("{}: {}", issue.issue_type, issue.description))
            .collect();

        Ok(ComponentHealth {
            component: component_name.to_string(),
            status,
            score,
            last_check: SystemTime::now(),
            issues: issue_strings,
        })
    }

    /// Update health history for trend analysis
    fn update_health_history(&mut self, health: &SystemHealth) {
        let snapshot = SystemHealthSnapshot {
            timestamp: SystemTime::now(),
            status: health.status.clone(),
            score: health.score,
            component_count: health.components.len(),
            issue_count: health.issues.len(),
        };

        self.health_history.push_back(snapshot);

        // Limit history size
        while self.health_history.len() > self.config.history_size {
            self.health_history.pop_front();
        }
    }

    /// Get health trends
    pub fn get_health_trends(&self) -> HealthTrends {
        let _lock = self.lock.read().unwrap();

        if self.health_history.len() < 2 {
            return HealthTrends::default();
        }

        let recent_snapshots: Vec<&SystemHealthSnapshot> = self.health_history.iter()
            .rev()
            .take(self.config.trend_window_size)
            .collect();

        self.evaluator.analyze_trends(&recent_snapshots)
    }

    /// Get component health state
    pub fn get_component_state(&self, component_name: &str) -> Option<&ComponentHealthState> {
        let _lock = self.lock.read().unwrap();
        self.component_states.get(component_name)
    }

    /// Get health statistics
    pub fn statistics(&self) -> &HealthStatistics {
        &self.stats
    }

    /// Get registered components
    pub fn get_registered_components(&self) -> Vec<String> {
        let _lock = self.lock.read().unwrap();
        self.monitors.keys().cloned().collect()
    }

    /// Enable/disable component monitoring
    pub fn set_component_enabled(&mut self, component_name: &str, enabled: bool) -> SklResult<()> {
        let _lock = self.lock.write().unwrap();

        if let Some(state) = self.component_states.get_mut(component_name) {
            state.enabled = enabled;
            Ok(())
        } else {
            Err(SklearsError::NotFound(format!("Component '{}' not found", component_name)))
        }
    }

    /// Perform deep health analysis
    pub fn deep_health_analysis(&mut self) -> SklResult<DeepHealthAnalysis> {
        let current_health = self.check_health()?;
        let trends = self.get_health_trends();

        let analysis = DeepHealthAnalysis {
            current_health,
            trends,
            component_analysis: self.analyze_components(),
            recommendations: self.generate_recommendations(),
            risk_assessment: self.assess_risks(),
        };

        Ok(analysis)
    }

    /// Analyze individual components
    fn analyze_components(&self) -> Vec<ComponentAnalysis> {
        let _lock = self.lock.read().unwrap();

        self.component_states.iter()
            .map(|(name, state)| ComponentAnalysis {
                component_name: name.clone(),
                reliability_score: state.calculate_reliability(),
                availability_score: state.calculate_availability(),
                performance_score: state.calculate_performance(),
                issue_frequency: state.calculate_issue_frequency(),
                check_frequency: state.calculate_check_frequency(),
            })
            .collect()
    }

    /// Generate health recommendations
    fn generate_recommendations(&self) -> Vec<HealthRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze component states for recommendations
        for (name, state) in &self.component_states {
            if state.calculate_reliability() < 0.8 {
                recommendations.push(HealthRecommendation {
                    component: Some(name.clone()),
                    recommendation_type: RecommendationType::Reliability,
                    description: format!("Component '{}' has low reliability score", name),
                    priority: RecommendationPriority::High,
                    estimated_impact: 0.8,
                });
            }

            if state.critical_checks > state.healthy_checks {
                recommendations.push(HealthRecommendation {
                    component: Some(name.clone()),
                    recommendation_type: RecommendationType::Stability,
                    description: format!("Component '{}' has more critical than healthy checks", name),
                    priority: RecommendationPriority::Critical,
                    estimated_impact: 0.9,
                });
            }
        }

        recommendations
    }

    /// Assess system risks
    fn assess_risks(&self) -> RiskAssessment {
        let mut risks = Vec::new();

        // Analyze health history for risks
        if self.health_history.len() >= 5 {
            let recent_scores: Vec<f64> = self.health_history.iter()
                .rev()
                .take(5)
                .map(|s| s.score)
                .collect();

            let trend = recent_scores.windows(2)
                .map(|w| w[1] - w[0])
                .sum::<f64>();

            if trend < -0.1 {
                risks.push(HealthRisk {
                    risk_type: RiskType::DegradingHealth,
                    severity: RiskSeverity::High,
                    description: "System health is degrading over time".to_string(),
                    probability: 0.8,
                    impact: 0.7,
                });
            }
        }

        RiskAssessment {
            overall_risk_score: self.calculate_overall_risk_score(&risks),
            risks,
        }
    }

    /// Calculate overall risk score
    fn calculate_overall_risk_score(&self, risks: &[HealthRisk]) -> f64 {
        if risks.is_empty() {
            return 0.0;
        }

        let total_risk = risks.iter()
            .map(|r| r.probability * r.impact)
            .sum::<f64>();

        total_risk / risks.len() as f64
    }
}

/// Health monitor trait for component health checking
pub trait HealthMonitor: Send + Sync {
    /// Perform health check
    fn check_health(&self) -> SklResult<HealthCheckResult>;

    /// Get monitor name
    fn name(&self) -> &str;

    /// Get monitor description
    fn description(&self) -> &str;

    /// Check if monitor is enabled
    fn is_enabled(&self) -> bool;

    /// Get check interval
    fn check_interval(&self) -> Duration;
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Component name
    pub component: String,

    /// Health indicators
    pub indicators: Vec<HealthIndicator>,

    /// Health issues found
    pub issues: Vec<HealthIssue>,

    /// Check timestamp
    pub timestamp: SystemTime,

    /// Check duration
    pub duration: Duration,

    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Health indicator for specific aspect
#[derive(Debug, Clone)]
pub struct HealthIndicator {
    /// Indicator name
    pub name: String,

    /// Indicator value
    pub value: f64,

    /// Indicator status
    pub status: IndicatorStatus,

    /// Threshold configuration
    pub thresholds: IndicatorThresholds,

    /// Unit of measurement
    pub unit: String,
}

/// Indicator status
#[derive(Debug, Clone, PartialEq)]
pub enum IndicatorStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Indicator thresholds
#[derive(Debug, Clone)]
pub struct IndicatorThresholds {
    /// Warning threshold
    pub warning: f64,

    /// Critical threshold
    pub critical: f64,

    /// Threshold direction (true = higher is worse)
    pub higher_is_worse: bool,
}

/// Component health state tracking
#[derive(Debug, Clone)]
pub struct ComponentHealthState {
    /// Component enabled flag
    pub enabled: bool,

    /// Last health check timestamp
    pub last_check: SystemTime,

    /// Total number of checks performed
    pub check_count: u64,

    /// Number of healthy checks
    pub healthy_checks: u64,

    /// Number of warning checks
    pub warning_checks: u64,

    /// Number of critical checks
    pub critical_checks: u64,

    /// Number of unknown checks
    pub unknown_checks: u64,

    /// Total downtime duration
    pub total_downtime: Duration,

    /// Last known issues
    pub last_issues: Vec<String>,

    /// Performance metrics
    pub avg_check_duration: Duration,
}

impl ComponentHealthState {
    /// Create new component health state
    pub fn new() -> Self {
        Self {
            enabled: true,
            last_check: SystemTime::now(),
            check_count: 0,
            healthy_checks: 0,
            warning_checks: 0,
            critical_checks: 0,
            unknown_checks: 0,
            total_downtime: Duration::from_secs(0),
            last_issues: Vec::new(),
            avg_check_duration: Duration::from_millis(0),
        }
    }

    /// Calculate reliability score (0.0 to 1.0)
    pub fn calculate_reliability(&self) -> f64 {
        if self.check_count == 0 {
            return 1.0;
        }

        let successful_checks = self.healthy_checks + self.warning_checks;
        successful_checks as f64 / self.check_count as f64
    }

    /// Calculate availability score (0.0 to 1.0)
    pub fn calculate_availability(&self) -> f64 {
        if self.check_count == 0 {
            return 1.0;
        }

        let available_checks = self.check_count - self.critical_checks;
        available_checks as f64 / self.check_count as f64
    }

    /// Calculate performance score (0.0 to 1.0)
    pub fn calculate_performance(&self) -> f64 {
        if self.check_count == 0 {
            return 1.0;
        }

        // Simple performance score based on health distribution
        let score = (self.healthy_checks as f64 * 1.0 +
                    self.warning_checks as f64 * 0.7 +
                    self.critical_checks as f64 * 0.3 +
                    self.unknown_checks as f64 * 0.5) / self.check_count as f64;

        score
    }

    /// Calculate issue frequency (issues per hour)
    pub fn calculate_issue_frequency(&self) -> f64 {
        // Simplified calculation - would need issue timestamps for accuracy
        if self.check_count == 0 {
            return 0.0;
        }

        let issue_count = self.warning_checks + self.critical_checks;
        issue_count as f64 / self.check_count as f64 * 3600.0 // Assume hourly frequency
    }

    /// Calculate check frequency (checks per hour)
    pub fn calculate_check_frequency(&self) -> f64 {
        // Simplified calculation based on total checks
        self.check_count as f64 / 24.0 // Assume daily distribution
    }
}

impl Default for ComponentHealthState {
    fn default() -> Self {
        Self::new()
    }
}

/// Health evaluator for scoring and status determination
#[derive(Debug)]
pub struct HealthEvaluator {
    /// Evaluation configuration
    config: HealthEvaluationConfig,
}

impl HealthEvaluator {
    /// Create new health evaluator
    pub fn new(config: HealthEvaluationConfig) -> Self {
        Self { config }
    }

    /// Calculate component health score
    pub fn calculate_component_score(&self, result: &HealthCheckResult) -> f64 {
        if result.indicators.is_empty() {
            return if result.issues.is_empty() { 1.0 } else { 0.0 };
        }

        let indicator_scores: Vec<f64> = result.indicators.iter()
            .map(|indicator| self.calculate_indicator_score(indicator))
            .collect();

        let avg_score = indicator_scores.iter().sum::<f64>() / indicator_scores.len() as f64;

        // Apply issue penalty
        let issue_penalty = result.issues.len() as f64 * self.config.issue_penalty;
        (avg_score - issue_penalty).max(0.0)
    }

    /// Calculate individual indicator score
    fn calculate_indicator_score(&self, indicator: &HealthIndicator) -> f64 {
        match indicator.status {
            IndicatorStatus::Healthy => 1.0,
            IndicatorStatus::Warning => 0.7,
            IndicatorStatus::Critical => 0.3,
            IndicatorStatus::Unknown => 0.5,
        }
    }

    /// Determine component status from score and issues
    pub fn determine_component_status(&self, score: f64, issues: &[HealthIssue]) -> HealthStatus {
        // Check for critical issues
        if issues.iter().any(|issue| issue.severity == SeverityLevel::Critical) {
            return HealthStatus::Critical;
        }

        // Score-based determination
        if score >= self.config.healthy_threshold {
            HealthStatus::Healthy
        } else if score >= self.config.warning_threshold {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        }
    }

    /// Determine overall system status
    pub fn determine_status(&self, score: f64, issues: &[HealthIssue]) -> HealthStatus {
        self.determine_component_status(score, issues)
    }

    /// Analyze health trends
    pub fn analyze_trends(&self, snapshots: &[&SystemHealthSnapshot]) -> HealthTrends {
        if snapshots.len() < 2 {
            return HealthTrends::default();
        }

        let scores: Vec<f64> = snapshots.iter().map(|s| s.score).collect();

        // Calculate trend direction
        let first_half_avg = scores.iter().take(scores.len() / 2).sum::<f64>() / (scores.len() / 2) as f64;
        let second_half_avg = scores.iter().skip(scores.len() / 2).sum::<f64>() / (scores.len() - scores.len() / 2) as f64;

        let trend_direction = if second_half_avg > first_half_avg + 0.05 {
            TrendDirection::Improving
        } else if second_half_avg < first_half_avg - 0.05 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        // Calculate volatility
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter()
            .map(|s| (s - mean_score).powi(2))
            .sum::<f64>() / scores.len() as f64;
        let volatility = variance.sqrt();

        HealthTrends {
            direction: trend_direction,
            volatility,
            average_score: mean_score,
            score_range: ScoreRange {
                min: scores.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                max: scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            },
        }
    }
}

/// Health evaluation configuration
#[derive(Debug, Clone)]
pub struct HealthEvaluationConfig {
    /// Threshold for healthy status
    pub healthy_threshold: f64,

    /// Threshold for warning status
    pub warning_threshold: f64,

    /// Penalty per issue
    pub issue_penalty: f64,

    /// Weight for different indicator types
    pub indicator_weights: HashMap<String, f64>,
}

impl Default for HealthEvaluationConfig {
    fn default() -> Self {
        Self {
            healthy_threshold: 0.8,
            warning_threshold: 0.6,
            issue_penalty: 0.1,
            indicator_weights: HashMap::new(),
        }
    }
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfiguration {
    /// Enable health checking
    pub enabled: bool,

    /// Default check interval
    pub default_check_interval: Duration,

    /// Maximum check duration
    pub max_check_duration: Duration,

    /// Number of health snapshots to keep
    pub history_size: usize,

    /// Window size for trend analysis
    pub trend_window_size: usize,

    /// Health evaluation configuration
    pub evaluation: HealthEvaluationConfig,

    /// Enable automatic remediation
    pub enable_auto_remediation: bool,
}

impl Default for HealthCheckConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            default_check_interval: Duration::from_secs(60),
            max_check_duration: Duration::from_secs(30),
            history_size: 100,
            trend_window_size: 10,
            evaluation: HealthEvaluationConfig::default(),
            enable_auto_remediation: false,
        }
    }
}

/// System health snapshot for history tracking
#[derive(Debug, Clone)]
pub struct SystemHealthSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,

    /// Overall health status
    pub status: HealthStatus,

    /// Overall health score
    pub score: f64,

    /// Number of components
    pub component_count: usize,

    /// Number of active issues
    pub issue_count: usize,
}

/// Health trends analysis
#[derive(Debug, Clone)]
pub struct HealthTrends {
    /// Trend direction
    pub direction: TrendDirection,

    /// Score volatility
    pub volatility: f64,

    /// Average score over period
    pub average_score: f64,

    /// Score range
    pub score_range: ScoreRange,
}

impl Default for HealthTrends {
    fn default() -> Self {
        Self {
            direction: TrendDirection::Unknown,
            volatility: 0.0,
            average_score: 1.0,
            score_range: ScoreRange { min: 1.0, max: 1.0 },
        }
    }
}

/// Trend direction enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Score range
#[derive(Debug, Clone)]
pub struct ScoreRange {
    /// Minimum score
    pub min: f64,

    /// Maximum score
    pub max: f64,
}

/// Deep health analysis result
#[derive(Debug, Clone)]
pub struct DeepHealthAnalysis {
    /// Current system health
    pub current_health: SystemHealth,

    /// Health trends
    pub trends: HealthTrends,

    /// Component-specific analysis
    pub component_analysis: Vec<ComponentAnalysis>,

    /// Health recommendations
    pub recommendations: Vec<HealthRecommendation>,

    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Component analysis details
#[derive(Debug, Clone)]
pub struct ComponentAnalysis {
    /// Component name
    pub component_name: String,

    /// Reliability score (0.0 to 1.0)
    pub reliability_score: f64,

    /// Availability score (0.0 to 1.0)
    pub availability_score: f64,

    /// Performance score (0.0 to 1.0)
    pub performance_score: f64,

    /// Issue frequency (issues per hour)
    pub issue_frequency: f64,

    /// Check frequency (checks per hour)
    pub check_frequency: f64,
}

/// Health recommendation
#[derive(Debug, Clone)]
pub struct HealthRecommendation {
    /// Target component (None for system-wide)
    pub component: Option<String>,

    /// Recommendation type
    pub recommendation_type: RecommendationType,

    /// Description
    pub description: String,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Estimated impact (0.0 to 1.0)
    pub estimated_impact: f64,
}

/// Recommendation types
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationType {
    Performance,
    Reliability,
    Availability,
    Security,
    Scaling,
    Monitoring,
    Stability,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Risk assessment
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Overall risk score (0.0 to 1.0)
    pub overall_risk_score: f64,

    /// Identified risks
    pub risks: Vec<HealthRisk>,
}

/// Health risk
#[derive(Debug, Clone)]
pub struct HealthRisk {
    /// Risk type
    pub risk_type: RiskType,

    /// Risk severity
    pub severity: RiskSeverity,

    /// Risk description
    pub description: String,

    /// Probability (0.0 to 1.0)
    pub probability: f64,

    /// Impact (0.0 to 1.0)
    pub impact: f64,
}

/// Risk types
#[derive(Debug, Clone, PartialEq)]
pub enum RiskType {
    DegradingHealth,
    ComponentFailure,
    ResourceExhaustion,
    PerformanceDegradation,
    SecurityThreat,
    DataLoss,
}

/// Risk severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Health statistics
#[derive(Debug, Clone)]
pub struct HealthStatistics {
    /// Total health checks performed
    pub total_checks: u64,

    /// Number of failed checks
    pub failed_checks: u64,

    /// Total time spent on health checks
    pub total_check_time: Duration,

    /// Last health check timestamp
    pub last_check: SystemTime,

    /// Average check duration
    pub avg_check_duration: Duration,
}

impl HealthStatistics {
    fn new() -> Self {
        Self {
            total_checks: 0,
            failed_checks: 0,
            total_check_time: Duration::from_millis(0),
            last_check: SystemTime::now(),
            avg_check_duration: Duration::from_millis(0),
        }
    }

    /// Calculate check success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_checks > 0 {
            (self.total_checks - self.failed_checks) as f64 / self.total_checks as f64
        } else {
            1.0
        }
    }

    /// Update average check duration
    pub fn update_avg_duration(&mut self) {
        if self.total_checks > 0 {
            self.avg_check_duration = self.total_check_time / self.total_checks as u32;
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    // Mock health monitor for testing
    struct MockHealthMonitor {
        name: String,
        healthy: bool,
    }

    impl HealthMonitor for MockHealthMonitor {
        fn check_health(&self) -> SklResult<HealthCheckResult> {
            let status = if self.healthy {
                IndicatorStatus::Healthy
            } else {
                IndicatorStatus::Critical
            };

            let indicator = HealthIndicator {
                name: "test_indicator".to_string(),
                value: if self.healthy { 1.0 } else { 0.0 },
                status,
                thresholds: IndicatorThresholds {
                    warning: 0.7,
                    critical: 0.3,
                    higher_is_worse: false,
                },
                unit: "score".to_string(),
            };

            Ok(HealthCheckResult {
                component: self.name.clone(),
                indicators: vec![indicator],
                issues: if self.healthy { Vec::new() } else {
                    vec![HealthIssue {
                        issue_id: "test_issue".to_string(),
                        issue_type: "test".to_string(),
                        description: "Test issue".to_string(),
                        severity: SeverityLevel::Critical,
                        first_occurrence: SystemTime::now(),
                        count: 1,
                    }]
                },
                timestamp: SystemTime::now(),
                duration: Duration::from_millis(10),
                metrics: HashMap::new(),
            })
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Mock health monitor"
        }

        fn is_enabled(&self) -> bool {
            true
        }

        fn check_interval(&self) -> Duration {
            Duration::from_secs(60)
        }
    }

    #[test]
    fn test_health_checker_creation() {
        let config = HealthCheckConfiguration::default();
        let checker = HealthChecker::new(config);

        assert_eq!(checker.monitors.len(), 0);
        assert_eq!(checker.component_states.len(), 0);
    }

    #[test]
    fn test_monitor_registration() {
        let config = HealthCheckConfiguration::default();
        let mut checker = HealthChecker::new(config);

        let monitor = Box::new(MockHealthMonitor {
            name: "test_component".to_string(),
            healthy: true,
        });

        assert!(checker.register_monitor("test_component".to_string(), monitor).is_ok());
        assert_eq!(checker.monitors.len(), 1);
        assert_eq!(checker.component_states.len(), 1);

        // Test duplicate registration
        let duplicate_monitor = Box::new(MockHealthMonitor {
            name: "test_component".to_string(),
            healthy: true,
        });

        assert!(checker.register_monitor("test_component".to_string(), duplicate_monitor).is_err());
    }

    #[test]
    fn test_health_check() {
        let config = HealthCheckConfiguration::default();
        let mut checker = HealthChecker::new(config);

        // Register healthy monitor
        let healthy_monitor = Box::new(MockHealthMonitor {
            name: "healthy_component".to_string(),
            healthy: true,
        });
        checker.register_monitor("healthy_component".to_string(), healthy_monitor).unwrap();

        // Register unhealthy monitor
        let unhealthy_monitor = Box::new(MockHealthMonitor {
            name: "unhealthy_component".to_string(),
            healthy: false,
        });
        checker.register_monitor("unhealthy_component".to_string(), unhealthy_monitor).unwrap();

        // Perform health check
        let health = checker.check_health().unwrap();

        assert_eq!(health.components.len(), 2);
        assert!(health.components.contains_key("healthy_component"));
        assert!(health.components.contains_key("unhealthy_component"));

        // Check that unhealthy component affects overall health
        assert!(health.score < 1.0);
        assert!(!health.issues.is_empty());
    }

    #[test]
    fn test_component_health_state() {
        let mut state = ComponentHealthState::new();

        assert_eq!(state.check_count, 0);
        assert_eq!(state.calculate_reliability(), 1.0);
        assert_eq!(state.calculate_availability(), 1.0);

        // Simulate some checks
        state.check_count = 10;
        state.healthy_checks = 8;
        state.warning_checks = 1;
        state.critical_checks = 1;

        assert_eq!(state.calculate_reliability(), 0.9); // (8 + 1) / 10
        assert_eq!(state.calculate_availability(), 0.9); // (10 - 1) / 10
    }

    #[test]
    fn test_health_evaluator() {
        let config = HealthEvaluationConfig::default();
        let evaluator = HealthEvaluator::new(config);

        // Test status determination
        assert_eq!(evaluator.determine_component_status(0.9, &[]), HealthStatus::Healthy);
        assert_eq!(evaluator.determine_component_status(0.7, &[]), HealthStatus::Warning);
        assert_eq!(evaluator.determine_component_status(0.5, &[]), HealthStatus::Critical);

        // Test with critical issue
        let critical_issue = HealthIssue {
            issue_id: "test".to_string(),
            issue_type: "test".to_string(),
            description: "test".to_string(),
            severity: SeverityLevel::Critical,
            first_occurrence: SystemTime::now(),
            count: 1,
        };

        assert_eq!(evaluator.determine_component_status(0.9, &[critical_issue]), HealthStatus::Critical);
    }

    #[test]
    fn test_indicator_status_score() {
        let config = HealthEvaluationConfig::default();
        let evaluator = HealthEvaluator::new(config);

        let healthy_indicator = HealthIndicator {
            name: "test".to_string(),
            value: 1.0,
            status: IndicatorStatus::Healthy,
            thresholds: IndicatorThresholds {
                warning: 0.7,
                critical: 0.3,
                higher_is_worse: false,
            },
            unit: "score".to_string(),
        };

        assert_eq!(evaluator.calculate_indicator_score(&healthy_indicator), 1.0);

        let warning_indicator = HealthIndicator {
            status: IndicatorStatus::Warning,
            ..healthy_indicator.clone()
        };

        assert_eq!(evaluator.calculate_indicator_score(&warning_indicator), 0.7);
    }

    #[test]
    fn test_health_trends() {
        let snapshots = vec![
            SystemHealthSnapshot {
                timestamp: SystemTime::now(),
                status: HealthStatus::Healthy,
                score: 0.9,
                component_count: 2,
                issue_count: 0,
            },
            SystemHealthSnapshot {
                timestamp: SystemTime::now(),
                status: HealthStatus::Healthy,
                score: 0.8,
                component_count: 2,
                issue_count: 1,
            },
            SystemHealthSnapshot {
                timestamp: SystemTime::now(),
                status: HealthStatus::Warning,
                score: 0.7,
                component_count: 2,
                issue_count: 2,
            },
        ];

        let snapshot_refs: Vec<&SystemHealthSnapshot> = snapshots.iter().collect();

        let config = HealthEvaluationConfig::default();
        let evaluator = HealthEvaluator::new(config);
        let trends = evaluator.analyze_trends(&snapshot_refs);

        assert_eq!(trends.direction, TrendDirection::Degrading);
        assert_eq!(trends.average_score, 0.8);
        assert!(trends.volatility > 0.0);
    }
}