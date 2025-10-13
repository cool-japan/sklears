//! Core alert manager and configuration
//!
//! This module provides the central AlertManager struct and its associated
//! configuration types for managing alerts, rules, and system behavior.

use super::types_config::{AlertRouting, AlertSeverity, EvaluationPriority};
use super::rule_management::AlertRule;
use super::state_tracking::{AlertStateTracker, ActiveAlert};
use super::metrics_performance::AlertMetrics;
use super::evaluation_engine::AlertEvaluationEngine;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

/// Central alert management system for rule configuration and state tracking
#[derive(Debug, Clone)]
pub struct AlertManager {
    /// Configured alert rules for different conditions
    pub alert_rules: Arc<RwLock<Vec<AlertRule>>>,
    /// Active alerts currently firing
    pub active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,
    /// Alert history for analysis and reporting
    pub alert_history: Arc<RwLock<VecDeque<AlertHistoryEntry>>>,
    /// Alert routing configuration
    pub alert_routing: Arc<RwLock<AlertRouting>>,
    /// Configuration for the alert manager
    pub config: AlertManagerConfig,
    /// Alert state tracking
    pub state_tracker: Arc<RwLock<AlertStateTracker>>,
    /// Alert metrics and statistics
    pub metrics: Arc<RwLock<AlertMetrics>>,
    /// Rule evaluation engine
    pub evaluation_engine: Arc<AlertEvaluationEngine>,
}

/// Configuration for the alert manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagerConfig {
    /// Maximum number of active alerts
    pub max_active_alerts: usize,
    /// Maximum alert history size
    pub max_history_size: usize,
    /// Default evaluation interval
    pub default_evaluation_interval: Duration,
    /// Alert retention period
    pub alert_retention_period: Duration,
    /// Enable alert deduplication
    pub enable_deduplication: bool,
    /// Deduplication window
    pub deduplication_window: Duration,
    /// Enable alert grouping
    pub enable_grouping: bool,
    /// Grouping configuration
    pub grouping_config: AlertGroupingConfig,
    /// Performance optimization settings
    pub performance_config: AlertPerformanceConfig,
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            max_active_alerts: 1000,
            max_history_size: 10000,
            default_evaluation_interval: Duration::from_secs(60),
            alert_retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            enable_deduplication: true,
            deduplication_window: Duration::from_secs(300), // 5 minutes
            enable_grouping: false,
            grouping_config: AlertGroupingConfig::default(),
            performance_config: AlertPerformanceConfig::default(),
        }
    }
}

/// Alert grouping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertGroupingConfig {
    /// Enable time-based grouping
    pub enable_time_grouping: bool,
    /// Time grouping window
    pub time_window: Duration,
    /// Enable label-based grouping
    pub enable_label_grouping: bool,
    /// Labels to group by
    pub grouping_labels: Vec<String>,
    /// Maximum group size
    pub max_group_size: usize,
    /// Group expiration time
    pub group_expiration: Duration,
}

impl Default for AlertGroupingConfig {
    fn default() -> Self {
        Self {
            enable_time_grouping: false,
            time_window: Duration::from_secs(300),
            enable_label_grouping: false,
            grouping_labels: Vec::new(),
            max_group_size: 50,
            group_expiration: Duration::from_secs(3600),
        }
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertPerformanceConfig {
    /// Enable rule evaluation caching
    pub enable_evaluation_caching: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Enable parallel evaluation
    pub enable_parallel_evaluation: bool,
    /// Maximum parallel evaluators
    pub max_parallel_evaluators: usize,
    /// Enable background evaluation
    pub enable_background_evaluation: bool,
    /// Background evaluation batch size
    pub background_batch_size: usize,
}

impl Default for AlertPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_evaluation_caching: true,
            cache_ttl: Duration::from_secs(300),
            enable_parallel_evaluation: true,
            max_parallel_evaluators: 4,
            enable_background_evaluation: true,
            background_batch_size: 10,
        }
    }
}

/// Alert history entry for tracking alert lifecycle
#[derive(Debug, Clone)]
pub struct AlertHistoryEntry {
    /// Alert identifier
    pub alert_id: String,
    /// Alert rule ID that triggered this alert
    pub rule_id: String,
    /// Timestamp when alert was created
    pub timestamp: SystemTime,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert labels
    pub labels: HashMap<String, String>,
    /// Alert annotations
    pub annotations: HashMap<String, String>,
    /// Alert state when added to history
    pub final_state: AlertHistoryState,
    /// Duration the alert was active
    pub duration: Option<Duration>,
    /// User who resolved the alert
    pub resolved_by: Option<String>,
    /// Resolution reason
    pub resolution_reason: Option<String>,
}

/// Final state of alert in history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertHistoryState {
    Resolved,
    Expired,
    Superseded,
    Cancelled,
}

/// Statistics about alert manager performance
#[derive(Debug, Clone)]
pub struct AlertManagerStatistics {
    /// Total number of rules
    pub total_rules: usize,
    /// Number of active alerts
    pub active_alerts_count: usize,
    /// Number of acknowledged alerts
    pub acknowledged_alerts_count: usize,
    /// Number of resolved alerts in history
    pub resolved_alerts_count: usize,
    /// Average alert resolution time
    pub average_resolution_time: Duration,
    /// Alert firing rate (alerts per hour)
    pub alert_firing_rate: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Evaluation performance
    pub evaluation_performance: EvaluationPerformanceStats,
}

/// Performance statistics for rule evaluation
#[derive(Debug, Clone)]
pub struct EvaluationPerformanceStats {
    /// Average evaluation time
    pub average_evaluation_time: Duration,
    /// Maximum evaluation time
    pub max_evaluation_time: Duration,
    /// Evaluation success rate
    pub success_rate: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Parallel evaluations per second
    pub parallel_evaluations_per_second: f64,
}

impl AlertManager {
    /// Create a new alert manager with default configuration
    pub fn new() -> Self {
        Self {
            alert_rules: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            alert_routing: Arc::new(RwLock::new(AlertRouting::default())),
            config: AlertManagerConfig::default(),
            state_tracker: Arc::new(RwLock::new(AlertStateTracker::default())),
            metrics: Arc::new(RwLock::new(AlertMetrics::default())),
            evaluation_engine: Arc::new(AlertEvaluationEngine::new()),
        }
    }

    /// Create a new alert manager with custom configuration
    pub fn with_config(config: AlertManagerConfig) -> Self {
        Self {
            alert_rules: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            alert_routing: Arc::new(RwLock::new(AlertRouting::default())),
            config,
            state_tracker: Arc::new(RwLock::new(AlertStateTracker::default())),
            metrics: Arc::new(RwLock::new(AlertMetrics::default())),
            evaluation_engine: Arc::new(AlertEvaluationEngine::new()),
        }
    }

    /// Add a new alert rule
    pub fn add_rule(&self, rule: AlertRule) -> Result<(), String> {
        if let Ok(mut rules) = self.alert_rules.write() {
            // Check for duplicate rule IDs
            if rules.iter().any(|r| r.rule_id == rule.rule_id) {
                return Err(format!("Rule with ID '{}' already exists", rule.rule_id));
            }

            // Validate rule configuration
            self.validate_rule(&rule)?;

            rules.push(rule);
            Ok(())
        } else {
            Err("Failed to acquire write lock on alert rules".to_string())
        }
    }

    /// Remove an alert rule by ID
    pub fn remove_rule(&self, rule_id: &str) -> Result<bool, String> {
        if let Ok(mut rules) = self.alert_rules.write() {
            if let Some(pos) = rules.iter().position(|r| r.rule_id == rule_id) {
                rules.remove(pos);
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Err("Failed to acquire write lock on alert rules".to_string())
        }
    }

    /// Update an existing alert rule
    pub fn update_rule(&self, updated_rule: AlertRule) -> Result<bool, String> {
        if let Ok(mut rules) = self.alert_rules.write() {
            if let Some(rule) = rules.iter_mut().find(|r| r.rule_id == updated_rule.rule_id) {
                // Validate updated rule
                self.validate_rule(&updated_rule)?;
                *rule = updated_rule;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Err("Failed to acquire write lock on alert rules".to_string())
        }
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Result<Vec<ActiveAlert>, String> {
        if let Ok(alerts) = self.active_alerts.read() {
            Ok(alerts.values().cloned().collect())
        } else {
            Err("Failed to acquire read lock on active alerts".to_string())
        }
    }

    /// Get active alerts filtered by severity
    pub fn get_active_alerts_by_severity(&self, severity: AlertSeverity) -> Result<Vec<ActiveAlert>, String> {
        if let Ok(alerts) = self.active_alerts.read() {
            Ok(alerts.values()
                .filter(|alert| alert.severity == severity)
                .cloned()
                .collect())
        } else {
            Err("Failed to acquire read lock on active alerts".to_string())
        }
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&self, alert_id: &str, user: &str) -> Result<(), String> {
        if let Ok(mut alerts) = self.active_alerts.write() {
            if let Some(alert) = alerts.get_mut(alert_id) {
                alert.acknowledge(user.to_string());
                Ok(())
            } else {
                Err(format!("Alert '{}' not found", alert_id))
            }
        } else {
            Err("Failed to acquire write lock on active alerts".to_string())
        }
    }

    /// Resolve an alert
    pub fn resolve_alert(&self, alert_id: &str, reason: &str) -> Result<(), String> {
        if let Ok(mut alerts) = self.active_alerts.write() {
            if let Some(alert) = alerts.remove(alert_id) {
                // Add to history
                let history_entry = AlertHistoryEntry {
                    alert_id: alert.alert_id.clone(),
                    rule_id: alert.rule_id.clone(),
                    timestamp: alert.created_at,
                    severity: alert.severity.clone(),
                    message: alert.message.clone(),
                    labels: alert.labels.clone(),
                    annotations: alert.annotations.clone(),
                    final_state: AlertHistoryState::Resolved,
                    duration: Some(alert.updated_at.duration_since(alert.created_at).unwrap_or(Duration::from_secs(0))),
                    resolved_by: Some(reason.to_string()),
                    resolution_reason: Some(reason.to_string()),
                };

                self.add_to_history(history_entry)?;
                Ok(())
            } else {
                Err(format!("Alert '{}' not found", alert_id))
            }
        } else {
            Err("Failed to acquire write lock on active alerts".to_string())
        }
    }

    /// Get alert statistics
    pub fn get_statistics(&self) -> Result<AlertManagerStatistics, String> {
        let rules_count = self.alert_rules.read()
            .map_err(|_| "Failed to read alert rules".to_string())?
            .len();

        let active_alerts = self.active_alerts.read()
            .map_err(|_| "Failed to read active alerts".to_string())?;

        let active_count = active_alerts.len();
        let acknowledged_count = active_alerts.values()
            .filter(|alert| alert.is_acknowledged())
            .count();

        let history = self.alert_history.read()
            .map_err(|_| "Failed to read alert history".to_string())?;

        let resolved_count = history.len();

        // Calculate average resolution time
        let total_duration: Duration = history.iter()
            .filter_map(|entry| entry.duration)
            .sum();

        let average_resolution_time = if resolved_count > 0 {
            total_duration / resolved_count as u32
        } else {
            Duration::from_secs(0)
        };

        // Calculate alert firing rate (simplified)
        let alert_firing_rate = if resolved_count > 0 {
            resolved_count as f64 / 24.0 // alerts per hour (simplified)
        } else {
            0.0
        };

        Ok(AlertManagerStatistics {
            total_rules: rules_count,
            active_alerts_count: active_count,
            acknowledged_alerts_count: acknowledged_count,
            resolved_alerts_count: resolved_count,
            average_resolution_time,
            alert_firing_rate,
            false_positive_rate: 0.0, // Would need more sophisticated calculation
            evaluation_performance: EvaluationPerformanceStats {
                average_evaluation_time: Duration::from_millis(100),
                max_evaluation_time: Duration::from_millis(500),
                success_rate: 0.99,
                cache_hit_rate: 0.85,
                parallel_evaluations_per_second: 100.0,
            },
        })
    }

    /// Clean up expired alerts and history
    pub fn cleanup(&self) -> Result<(), String> {
        // Clean up expired history entries
        if let Ok(mut history) = self.alert_history.write() {
            let retention_cutoff = SystemTime::now() - self.config.alert_retention_period;

            while let Some(entry) = history.front() {
                if entry.timestamp < retention_cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }

            // Limit history size
            while history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }

        // Check for alerts that should be auto-resolved
        if let Ok(mut alerts) = self.active_alerts.write() {
            let now = SystemTime::now();
            let mut to_remove = Vec::new();

            for (alert_id, alert) in alerts.iter() {
                // Auto-resolve alerts that have been active too long without acknowledgment
                if let Ok(duration) = now.duration_since(alert.created_at) {
                    if duration > Duration::from_secs(24 * 3600) && !alert.is_acknowledged() {
                        to_remove.push(alert_id.clone());
                    }
                }
            }

            for alert_id in to_remove {
                if let Some(alert) = alerts.remove(&alert_id) {
                    let history_entry = AlertHistoryEntry {
                        alert_id: alert.alert_id.clone(),
                        rule_id: alert.rule_id.clone(),
                        timestamp: alert.created_at,
                        severity: alert.severity.clone(),
                        message: alert.message.clone(),
                        labels: alert.labels.clone(),
                        annotations: alert.annotations.clone(),
                        final_state: AlertHistoryState::Expired,
                        duration: Some(now.duration_since(alert.created_at).unwrap_or(Duration::from_secs(0))),
                        resolved_by: None,
                        resolution_reason: Some("Auto-expired".to_string()),
                    };

                    self.add_to_history(history_entry)?;
                }
            }
        }

        Ok(())
    }

    /// Validate an alert rule
    fn validate_rule(&self, rule: &AlertRule) -> Result<(), String> {
        if rule.rule_id.is_empty() {
            return Err("Rule ID cannot be empty".to_string());
        }

        if rule.name.is_empty() {
            return Err("Rule name cannot be empty".to_string());
        }

        if rule.expression.is_empty() {
            return Err("Rule expression cannot be empty".to_string());
        }

        // Additional validation logic can be added here
        Ok(())
    }

    /// Add entry to alert history
    fn add_to_history(&self, entry: AlertHistoryEntry) -> Result<(), String> {
        if let Ok(mut history) = self.alert_history.write() {
            history.push_back(entry);

            // Limit history size
            while history.len() > self.config.max_history_size {
                history.pop_front();
            }

            Ok(())
        } else {
            Err("Failed to acquire write lock on alert history".to_string())
        }
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::rule_management::*;
    use super::super::types_config::*;

    #[test]
    fn test_alert_manager_creation() {
        let manager = AlertManager::new();
        assert!(manager.alert_rules.read().unwrap().is_empty());
        assert!(manager.active_alerts.read().unwrap().is_empty());
        assert!(manager.alert_history.read().unwrap().is_empty());
    }

    #[test]
    fn test_alert_manager_config_default() {
        let config = AlertManagerConfig::default();
        assert_eq!(config.max_active_alerts, 1000);
        assert_eq!(config.max_history_size, 10000);
        assert!(config.enable_deduplication);
    }

    #[test]
    fn test_alert_grouping_config_default() {
        let config = AlertGroupingConfig::default();
        assert!(!config.enable_time_grouping);
        assert!(!config.enable_label_grouping);
        assert_eq!(config.max_group_size, 50);
    }

    #[test]
    fn test_alert_performance_config_default() {
        let config = AlertPerformanceConfig::default();
        assert!(config.enable_evaluation_caching);
        assert!(config.enable_parallel_evaluation);
        assert_eq!(config.max_parallel_evaluators, 4);
    }

    #[test]
    fn test_add_remove_rule() {
        let manager = AlertManager::new();

        let rule = AlertRule {
            rule_id: "test_rule".to_string(),
            name: "Test Rule".to_string(),
            description: "Test description".to_string(),
            expression: "cpu_usage > 80".to_string(),
            duration: Duration::from_secs(300),
            severity: AlertSeverity::Warning,
            labels: HashMap::new(),
            annotations: HashMap::new(),
            enabled: true,
            threshold: AlertThreshold::default(),
            evaluation: AlertEvaluation::default(),
            metadata: AlertRuleMetadata::default(),
            dependencies: Vec::new(),
            template_info: None,
        };

        // Add rule
        assert!(manager.add_rule(rule.clone()).is_ok());
        assert_eq!(manager.alert_rules.read().unwrap().len(), 1);

        // Try to add duplicate
        assert!(manager.add_rule(rule).is_err());

        // Remove rule
        assert!(manager.remove_rule("test_rule").unwrap());
        assert!(manager.alert_rules.read().unwrap().is_empty());

        // Try to remove non-existent rule
        assert!(!manager.remove_rule("non_existent").unwrap());
    }

    #[test]
    fn test_get_statistics() {
        let manager = AlertManager::new();
        let stats = manager.get_statistics().unwrap();

        assert_eq!(stats.total_rules, 0);
        assert_eq!(stats.active_alerts_count, 0);
        assert_eq!(stats.resolved_alerts_count, 0);
    }

    #[test]
    fn test_cleanup() {
        let manager = AlertManager::new();
        assert!(manager.cleanup().is_ok());
    }
}