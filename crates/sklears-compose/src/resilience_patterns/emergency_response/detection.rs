//! Emergency Detection System
//!
//! This module provides comprehensive emergency detection capabilities including
//! system health monitoring, detection rules, emergency event classification,
//! and real-time threat assessment.

use sklears_core::{
    error::{Result as SklResult, SklearsError},
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

/// Emergency detection system
///
/// Monitors system health metrics and detects emergency conditions
/// based on configurable detection rules and thresholds.
#[derive(Debug)]
pub struct EmergencyDetector {
    /// Detection rules
    rules: Arc<RwLock<Vec<EmergencyDetectionRule>>>,
    /// Detection history
    history: Arc<RwLock<VecDeque<EmergencyEvent>>>,
    /// Detection state
    state: Arc<RwLock<DetectionState>>,
}

impl EmergencyDetector {
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            state: Arc::new(RwLock::new(DetectionState::new())),
        }
    }

    pub fn initialize(&self) -> SklResult<()> {
        self.setup_default_detection_rules()?;
        Ok(())
    }

    pub fn detect_emergency(&self, metrics: &SystemHealthMetrics) -> SklResult<Option<EmergencyEvent>> {
        let rules = self.rules.read()
            .map_err(|_| SklearsError::Other("Failed to acquire rules lock".into()))?;

        for rule in rules.iter() {
            if rule.enabled && self.evaluate_rule(rule, metrics)? {
                let event = self.create_emergency_event(rule, metrics)?;

                // Add to history
                {
                    let mut history = self.history.write()
                        .map_err(|_| SklearsError::Other("Failed to acquire history lock".into()))?;
                    history.push_back(event.clone());
                    if history.len() > 1000 {
                        history.pop_front();
                    }
                }

                return Ok(Some(event));
            }
        }

        Ok(None)
    }

    fn setup_default_detection_rules(&self) -> SklResult<()> {
        let mut rules = self.rules.write()
            .map_err(|_| SklearsError::Other("Failed to acquire rules lock".into()))?;

        // High error rate rule
        rules.push(EmergencyDetectionRule {
            rule_id: "high_error_rate".to_string(),
            name: "High Error Rate Emergency".to_string(),
            description: "Detect when error rate exceeds critical threshold".to_string(),
            rule_type: DetectionRuleType::Threshold,
            metric_path: "error_rate".to_string(),
            condition: DetectionCondition::GreaterThan(0.1), // 10% error rate
            severity: EmergencySeverity::Critical,
            emergency_type: EmergencyType::SystemFailure,
            enabled: true,
            cooldown: Duration::from_secs(300), // 5 minutes
            last_triggered: None,
        });

        // System unavailability rule
        rules.push(EmergencyDetectionRule {
            rule_id: "system_unavailable".to_string(),
            name: "System Unavailability Emergency".to_string(),
            description: "Detect when system availability drops critically".to_string(),
            rule_type: DetectionRuleType::Threshold,
            metric_path: "availability".to_string(),
            condition: DetectionCondition::LessThan(0.95), // Below 95% availability
            severity: EmergencySeverity::Catastrophic,
            emergency_type: EmergencyType::ServiceOutage,
            enabled: true,
            cooldown: Duration::from_secs(60), // 1 minute
            last_triggered: None,
        });

        // Resource exhaustion rule
        rules.push(EmergencyDetectionRule {
            rule_id: "resource_exhaustion".to_string(),
            name: "Resource Exhaustion Emergency".to_string(),
            description: "Detect critical resource exhaustion".to_string(),
            rule_type: DetectionRuleType::Composite,
            metric_path: "resources.cpu_usage".to_string(),
            condition: DetectionCondition::GreaterThan(0.95), // 95% CPU usage
            severity: EmergencySeverity::High,
            emergency_type: EmergencyType::ResourceExhaustion,
            enabled: true,
            cooldown: Duration::from_secs(120), // 2 minutes
            last_triggered: None,
        });

        // Security incident rule
        rules.push(EmergencyDetectionRule {
            rule_id: "security_incident".to_string(),
            name: "Security Incident Emergency".to_string(),
            description: "Detect potential security incidents".to_string(),
            rule_type: DetectionRuleType::Anomaly,
            metric_path: "security.failed_authentication_rate".to_string(),
            condition: DetectionCondition::GreaterThan(0.5), // 50% failed auth rate
            severity: EmergencySeverity::Critical,
            emergency_type: EmergencyType::SecurityIncident,
            enabled: true,
            cooldown: Duration::from_secs(60), // 1 minute
            last_triggered: None,
        });

        Ok(())
    }

    fn evaluate_rule(&self, rule: &EmergencyDetectionRule, metrics: &SystemHealthMetrics) -> SklResult<bool> {
        // Check cooldown
        if let Some(last_triggered) = rule.last_triggered {
            if SystemTime::now().duration_since(last_triggered).unwrap_or(Duration::from_secs(0)) < rule.cooldown {
                return Ok(false);
            }
        }

        // Extract metric value
        let metric_value = self.extract_metric_value(&rule.metric_path, metrics)?;

        // Evaluate condition
        let triggered = match &rule.condition {
            DetectionCondition::GreaterThan(threshold) => metric_value > *threshold,
            DetectionCondition::LessThan(threshold) => metric_value < *threshold,
            DetectionCondition::Equals(threshold) => (metric_value - threshold).abs() < f64::EPSILON,
            DetectionCondition::Range(min, max) => metric_value >= *min && metric_value <= *max,
            DetectionCondition::OutsideRange(min, max) => metric_value < *min || metric_value > *max,
        };

        Ok(triggered)
    }

    fn extract_metric_value(&self, metric_path: &str, metrics: &SystemHealthMetrics) -> SklResult<f64> {
        match metric_path {
            "error_rate" => Ok(metrics.error_rate),
            "availability" => Ok(metrics.availability),
            "resources.cpu_usage" => Ok(metrics.resources.cpu_usage),
            "resources.memory_usage" => Ok(metrics.resources.memory_usage),
            "latency.p99" => Ok(metrics.latency.p99.as_millis() as f64),
            "security.failed_authentication_rate" => Ok(metrics.security.failed_authentication_rate),
            _ => Err(SklearsError::InvalidInput(format!("Unknown metric path: {}", metric_path))),
        }
    }

    fn create_emergency_event(&self, rule: &EmergencyDetectionRule, metrics: &SystemHealthMetrics) -> SklResult<EmergencyEvent> {
        let current_value = self.extract_metric_value(&rule.metric_path, metrics)?;

        Ok(EmergencyEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            emergency_type: rule.emergency_type.clone(),
            severity: rule.severity,
            title: rule.name.clone(),
            description: format!("{}: {} = {:.4}", rule.description, rule.metric_path, current_value),
            source: "emergency_detector".to_string(),
            timestamp: SystemTime::now(),
            affected_systems: vec!["primary_system".to_string()], // Would be more specific in real implementation
            estimated_impact: EmergencyImpact {
                user_impact: UserImpact::High,
                business_impact: BusinessImpact::High,
                system_impact: SystemImpact::Critical,
                financial_impact: Some(10000.0), // $10k estimated impact
            },
            estimated_impact_duration: Some(Duration::from_secs(7200)), // 2 hours
            detected_by: rule.rule_id.clone(),
            context: HashMap::new(),
            related_events: vec![],
            urgency: Urgency::High,
            requires_immediate_action: rule.severity >= EmergencySeverity::Critical,
        })
    }

    /// Add a custom detection rule
    pub fn add_detection_rule(&self, rule: EmergencyDetectionRule) -> SklResult<()> {
        let mut rules = self.rules.write()
            .map_err(|_| SklearsError::Other("Failed to acquire rules lock".into()))?;
        rules.push(rule);
        Ok(())
    }

    /// Get detection history
    pub fn get_detection_history(&self, limit: Option<usize>) -> SklResult<Vec<EmergencyEvent>> {
        let history = self.history.read()
            .map_err(|_| SklearsError::Other("Failed to acquire history lock".into()))?;

        let events: Vec<EmergencyEvent> = if let Some(limit) = limit {
            history.iter().rev().take(limit).cloned().collect()
        } else {
            history.iter().cloned().collect()
        };

        Ok(events)
    }

    /// Get current detection state
    pub fn get_detection_state(&self) -> SklResult<DetectionState> {
        let state = self.state.read()
            .map_err(|_| SklearsError::Other("Failed to acquire state lock".into()))?;
        Ok(state.clone())
    }
}

/// Emergency detection rule
#[derive(Debug, Clone)]
pub struct EmergencyDetectionRule {
    pub rule_id: String,
    pub name: String,
    pub description: String,
    pub rule_type: DetectionRuleType,
    pub metric_path: String,
    pub condition: DetectionCondition,
    pub severity: EmergencySeverity,
    pub emergency_type: EmergencyType,
    pub enabled: bool,
    pub cooldown: Duration,
    pub last_triggered: Option<SystemTime>,
}

/// Detection rule types
#[derive(Debug, Clone, PartialEq)]
pub enum DetectionRuleType {
    Threshold,
    Trend,
    Anomaly,
    Composite,
    Pattern,
}

/// Detection conditions
#[derive(Debug, Clone)]
pub enum DetectionCondition {
    GreaterThan(f64),
    LessThan(f64),
    Equals(f64),
    Range(f64, f64),
    OutsideRange(f64, f64),
}

/// Detection state
#[derive(Debug, Clone)]
pub struct DetectionState {
    pub active_rules: usize,
    pub total_detections: u64,
    pub last_detection: Option<SystemTime>,
    pub current_threat_level: ThreatLevel,
}

impl DetectionState {
    pub fn new() -> Self {
        Self {
            active_rules: 0,
            total_detections: 0,
            last_detection: None,
            current_threat_level: ThreatLevel::Normal,
        }
    }
}

/// Threat level assessment
#[derive(Debug, Clone, PartialEq)]
pub enum ThreatLevel {
    Normal,
    Elevated,
    High,
    Critical,
    Emergency,
}

/// System health metrics
///
/// Comprehensive system health and performance metrics
/// used for emergency detection and assessment.
#[derive(Debug, Clone)]
pub struct SystemHealthMetrics {
    pub timestamp: SystemTime,
    pub availability: f64,
    pub error_rate: f64,
    pub latency: LatencyMetrics,
    pub resources: ResourceUtilizationMetrics,
    pub security: SecurityMetrics,
    pub business_metrics: BusinessHealthMetrics,
}

/// Latency performance metrics
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    pub mean: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilizationMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_utilization: f64,
    pub storage_utilization: f64,
}

/// Security-related metrics
#[derive(Debug, Clone)]
pub struct SecurityMetrics {
    pub failed_authentication_rate: f64,
    pub suspicious_activity_score: f64,
    pub breach_indicators: u32,
}

/// Business health metrics
#[derive(Debug, Clone)]
pub struct BusinessHealthMetrics {
    pub user_satisfaction: f64,
    pub conversion_rate: f64,
    pub revenue_impact: f64,
}

/// Emergency event structure
///
/// Represents a detected emergency condition with full
/// context and impact assessment information.
#[derive(Debug, Clone)]
pub struct EmergencyEvent {
    pub event_id: String,
    pub emergency_type: EmergencyType,
    pub severity: EmergencySeverity,
    pub title: String,
    pub description: String,
    pub source: String,
    pub timestamp: SystemTime,
    pub affected_systems: Vec<String>,
    pub estimated_impact: EmergencyImpact,
    pub estimated_impact_duration: Option<Duration>,
    pub detected_by: String,
    pub context: HashMap<String, String>,
    pub related_events: Vec<String>,
    pub urgency: Urgency,
    pub requires_immediate_action: bool,
}

/// Emergency types
#[derive(Debug, Clone, PartialEq)]
pub enum EmergencyType {
    SystemFailure,
    ServiceOutage,
    SecurityIncident,
    DataBreach,
    ResourceExhaustion,
    NetworkFailure,
    DatabaseFailure,
    ThirdPartyFailure,
    NaturalDisaster,
    HumanError,
    Unknown,
}

/// Emergency severity levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum EmergencySeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Catastrophic = 5,
}

/// Emergency impact assessment
#[derive(Debug, Clone)]
pub struct EmergencyImpact {
    pub user_impact: UserImpact,
    pub business_impact: BusinessImpact,
    pub system_impact: SystemImpact,
    pub financial_impact: Option<f64>,
}

/// User impact levels
#[derive(Debug, Clone, PartialEq)]
pub enum UserImpact {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Business impact levels
#[derive(Debug, Clone, PartialEq)]
pub enum BusinessImpact {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// System impact levels
#[derive(Debug, Clone, PartialEq)]
pub enum SystemImpact {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Urgency levels for emergency response
#[derive(Debug, Clone, PartialEq)]
pub enum Urgency {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

impl SystemHealthMetrics {
    /// Create new system health metrics
    pub fn new() -> Self {
        Self {
            timestamp: SystemTime::now(),
            availability: 1.0,
            error_rate: 0.0,
            latency: LatencyMetrics {
                mean: Duration::from_millis(50),
                p95: Duration::from_millis(100),
                p99: Duration::from_millis(200),
                max: Duration::from_millis(500),
            },
            resources: ResourceUtilizationMetrics {
                cpu_usage: 0.3,
                memory_usage: 0.4,
                network_utilization: 0.2,
                storage_utilization: 0.5,
            },
            security: SecurityMetrics {
                failed_authentication_rate: 0.01,
                suspicious_activity_score: 0.1,
                breach_indicators: 0,
            },
            business_metrics: BusinessHealthMetrics {
                user_satisfaction: 0.9,
                conversion_rate: 0.15,
                revenue_impact: 1.0,
            },
        }
    }

    /// Calculate overall health score
    pub fn calculate_health_score(&self) -> f64 {
        let availability_score = self.availability;
        let error_score = 1.0 - self.error_rate.min(1.0);
        let latency_score = 1.0 - (self.latency.p99.as_millis() as f64 / 1000.0).min(1.0);
        let resource_score = 1.0 - (self.resources.cpu_usage + self.resources.memory_usage) / 2.0;
        let security_score = 1.0 - self.security.failed_authentication_rate.min(1.0);

        (availability_score + error_score + latency_score + resource_score + security_score) / 5.0
    }
}

impl Default for SystemHealthMetrics {
    fn default() -> Self {
        Self::new()
    }
}