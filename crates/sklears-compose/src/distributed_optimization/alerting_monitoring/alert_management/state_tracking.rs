//! Alert state tracking and management
//!
//! This module provides alert state tracking functionality including
//! active alert management, state transitions, and alert lifecycle tracking.

use super::types_config::{AlertSeverity, PerformanceSnapshot};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Alert state tracker for managing alert states and transitions
#[derive(Debug, Clone)]
pub struct AlertStateTracker {
    /// Current alert states
    pub alert_states: HashMap<String, AlertState>,
    /// State transition history
    pub state_transitions: VecDeque<StateTransition>,
    /// State statistics
    pub state_statistics: StateStatistics,
    /// Tracker configuration
    pub config: StateTrackerConfig,
    /// Performance tracking
    pub performance: StateTrackerPerformance,
}

impl Default for AlertStateTracker {
    fn default() -> Self {
        Self {
            alert_states: HashMap::new(),
            state_transitions: VecDeque::new(),
            state_statistics: StateStatistics::default(),
            config: StateTrackerConfig::default(),
            performance: StateTrackerPerformance::default(),
        }
    }
}

/// Configuration for state tracker
#[derive(Debug, Clone)]
pub struct StateTrackerConfig {
    /// Maximum transition history size
    pub max_transition_history: usize,
    /// State retention period
    pub state_retention_period: Duration,
    /// Enable detailed transition logging
    pub enable_detailed_logging: bool,
    /// Performance monitoring interval
    pub performance_monitoring_interval: Duration,
}

impl Default for StateTrackerConfig {
    fn default() -> Self {
        Self {
            max_transition_history: 10000,
            state_retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            enable_detailed_logging: true,
            performance_monitoring_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Alert states
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertState {
    /// Alert is firing
    Firing,
    /// Alert has been acknowledged
    Acknowledged,
    /// Alert is resolved
    Resolved,
    /// Alert is silenced
    Silenced,
    /// Alert is escalated
    Escalated,
    /// Alert is expired
    Expired,
    /// Alert is suppressed by dependency
    Suppressed,
    /// Alert is in testing state
    Testing,
}

impl Default for AlertState {
    fn default() -> Self {
        Self::Firing
    }
}

/// State transition record
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Alert ID
    pub alert_id: String,
    /// Previous state
    pub from_state: AlertState,
    /// New state
    pub to_state: AlertState,
    /// Transition timestamp
    pub timestamp: SystemTime,
    /// User who triggered the transition
    pub triggered_by: Option<String>,
    /// Reason for transition
    pub reason: Option<String>,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Transition duration (time spent in previous state)
    pub duration_in_previous_state: Duration,
}

/// State statistics for analysis
#[derive(Debug, Clone)]
pub struct StateStatistics {
    /// Count of alerts by state
    pub state_counts: HashMap<AlertState, u64>,
    /// Average time in each state
    pub average_state_durations: HashMap<AlertState, Duration>,
    /// State transition frequencies
    pub transition_frequencies: HashMap<(AlertState, AlertState), u64>,
    /// Alert frequency trends
    pub frequency_trends: AlertFrequencyTrends,
}

impl Default for StateStatistics {
    fn default() -> Self {
        Self {
            state_counts: HashMap::new(),
            average_state_durations: HashMap::new(),
            transition_frequencies: HashMap::new(),
            frequency_trends: AlertFrequencyTrends::default(),
        }
    }
}

/// Alert frequency analysis
#[derive(Debug, Clone)]
pub struct AlertFrequencyTrends {
    /// Hourly alert counts for trend analysis
    pub hourly_counts: VecDeque<u64>,
    /// Daily alert counts
    pub daily_counts: VecDeque<u64>,
    /// Weekly alert counts
    pub weekly_counts: VecDeque<u64>,
    /// Frequency trend direction
    pub trend_direction: FrequencyTrend,
    /// Seasonality detection
    pub seasonality_detected: bool,
    /// Peak hours analysis
    pub peak_hours: Vec<u8>,
}

impl Default for AlertFrequencyTrends {
    fn default() -> Self {
        Self {
            hourly_counts: VecDeque::new(),
            daily_counts: VecDeque::new(),
            weekly_counts: VecDeque::new(),
            trend_direction: FrequencyTrend::Unknown,
            seasonality_detected: false,
            peak_hours: Vec::new(),
        }
    }
}

/// Frequency trend directions
#[derive(Debug, Clone)]
pub enum FrequencyTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Unknown,
}

/// Performance tracking for state tracker
#[derive(Debug, Clone)]
pub struct StateTrackerPerformance {
    /// Update latency
    pub update_latency: Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// Query performance
    pub query_performance: HashMap<String, Duration>,
    /// Cache hit rates
    pub cache_hit_rates: HashMap<String, f64>,
}

impl Default for StateTrackerPerformance {
    fn default() -> Self {
        Self {
            update_latency: Duration::from_millis(1),
            memory_usage: 0,
            query_performance: HashMap::new(),
            cache_hit_rates: HashMap::new(),
        }
    }
}

/// Active alert representation
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// Unique alert identifier
    pub alert_id: String,
    /// Rule ID that triggered this alert
    pub rule_id: String,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert labels
    pub labels: HashMap<String, String>,
    /// Alert annotations
    pub annotations: HashMap<String, String>,
    /// Current alert state
    pub state: AlertState,
    /// When the alert was created
    pub created_at: SystemTime,
    /// When the alert was last updated
    pub updated_at: SystemTime,
    /// User who acknowledged the alert
    pub acknowledged_by: Option<String>,
    /// Acknowledgment timestamp
    pub acknowledged_at: Option<SystemTime>,
    /// Alert value that triggered the condition
    pub trigger_value: Option<f64>,
    /// State transition history for this alert
    pub state_history: Vec<StateTransition>,
    /// Alert fingerprint for deduplication
    pub fingerprint: String,
    /// Related alerts (grouping)
    pub related_alerts: Vec<String>,
}

impl ActiveAlert {
    /// Create a new active alert
    pub fn new(
        alert_id: String,
        rule_id: String,
        message: String,
        severity: AlertSeverity,
        fingerprint: String,
    ) -> Self {
        Self {
            alert_id,
            rule_id,
            message,
            severity,
            labels: HashMap::new(),
            annotations: HashMap::new(),
            state: AlertState::Firing,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            acknowledged_by: None,
            acknowledged_at: None,
            trigger_value: None,
            state_history: Vec::new(),
            fingerprint,
            related_alerts: Vec::new(),
        }
    }

    /// Acknowledge the alert
    pub fn acknowledge(&mut self, user: String) {
        self.acknowledged_by = Some(user);
        self.acknowledged_at = Some(SystemTime::now());
        self.transition_to_state(AlertState::Acknowledged, Some("Manual acknowledgment".to_string()));
    }

    /// Resolve the alert
    pub fn resolve(&mut self, reason: String) {
        self.transition_to_state(AlertState::Resolved, Some(reason));
    }

    /// Silence the alert
    pub fn silence(&mut self, reason: String) {
        self.transition_to_state(AlertState::Silenced, Some(reason));
    }

    /// Escalate the alert
    pub fn escalate(&mut self, reason: String) {
        self.transition_to_state(AlertState::Escalated, Some(reason));
    }

    /// Transition to a new state
    pub fn transition_to_state(&mut self, new_state: AlertState, reason: Option<String>) {
        if self.state != new_state {
            let transition = StateTransition {
                alert_id: self.alert_id.clone(),
                from_state: self.state.clone(),
                to_state: new_state.clone(),
                timestamp: SystemTime::now(),
                triggered_by: None, // Would be set by caller
                reason,
                context: HashMap::new(),
                duration_in_previous_state: SystemTime::now()
                    .duration_since(self.updated_at)
                    .unwrap_or(Duration::from_secs(0)),
            };

            self.state_history.push(transition);
            self.state = new_state;
            self.updated_at = SystemTime::now();
        }
    }

    /// Check if alert is acknowledged
    pub fn is_acknowledged(&self) -> bool {
        matches!(self.state, AlertState::Acknowledged)
    }

    /// Check if alert is resolved
    pub fn is_resolved(&self) -> bool {
        matches!(self.state, AlertState::Resolved)
    }

    /// Check if alert is silenced
    pub fn is_silenced(&self) -> bool {
        matches!(self.state, AlertState::Silenced)
    }

    /// Get alert age
    pub fn get_age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.created_at)
            .unwrap_or(Duration::from_secs(0))
    }

    /// Get time since last update
    pub fn get_time_since_update(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.updated_at)
            .unwrap_or(Duration::from_secs(0))
    }

    /// Get current state duration
    pub fn get_current_state_duration(&self) -> Duration {
        let state_start = self.state_history
            .last()
            .map(|t| t.timestamp)
            .unwrap_or(self.created_at);

        SystemTime::now()
            .duration_since(state_start)
            .unwrap_or(Duration::from_secs(0))
    }

    /// Add related alert
    pub fn add_related_alert(&mut self, alert_id: String) {
        if !self.related_alerts.contains(&alert_id) {
            self.related_alerts.push(alert_id);
        }
    }

    /// Remove related alert
    pub fn remove_related_alert(&mut self, alert_id: &str) {
        self.related_alerts.retain(|id| id != alert_id);
    }
}

impl AlertStateTracker {
    /// Create a new state tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Update alert state
    pub fn update_alert_state(&mut self, alert_id: String, new_state: AlertState, user: Option<String>, reason: Option<String>) -> Result<(), String> {
        if let Some(current_state) = self.alert_states.get(&alert_id) {
            if *current_state != new_state {
                // Record transition
                let transition = StateTransition {
                    alert_id: alert_id.clone(),
                    from_state: current_state.clone(),
                    to_state: new_state.clone(),
                    timestamp: SystemTime::now(),
                    triggered_by: user,
                    reason,
                    context: HashMap::new(),
                    duration_in_previous_state: Duration::from_secs(0), // Would calculate actual duration
                };

                self.state_transitions.push_back(transition);

                // Limit transition history size
                while self.state_transitions.len() > self.config.max_transition_history {
                    self.state_transitions.pop_front();
                }

                // Update state
                self.alert_states.insert(alert_id, new_state.clone());

                // Update statistics
                self.update_statistics(&new_state);
            }
            Ok(())
        } else {
            Err(format!("Alert '{}' not found in state tracker", alert_id))
        }
    }

    /// Add new alert to tracking
    pub fn add_alert(&mut self, alert_id: String, initial_state: AlertState) {
        self.alert_states.insert(alert_id.clone(), initial_state.clone());
        self.update_statistics(&initial_state);
    }

    /// Remove alert from tracking
    pub fn remove_alert(&mut self, alert_id: &str) -> bool {
        self.alert_states.remove(alert_id).is_some()
    }

    /// Get current state of an alert
    pub fn get_alert_state(&self, alert_id: &str) -> Option<&AlertState> {
        self.alert_states.get(alert_id)
    }

    /// Get alerts by state
    pub fn get_alerts_by_state(&self, state: &AlertState) -> Vec<String> {
        self.alert_states
            .iter()
            .filter(|(_, s)| *s == state)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get state transition history for an alert
    pub fn get_alert_transitions(&self, alert_id: &str) -> Vec<&StateTransition> {
        self.state_transitions
            .iter()
            .filter(|t| t.alert_id == alert_id)
            .collect()
    }

    /// Get recent transitions
    pub fn get_recent_transitions(&self, since: SystemTime) -> Vec<&StateTransition> {
        self.state_transitions
            .iter()
            .filter(|t| t.timestamp >= since)
            .collect()
    }

    /// Update statistics
    fn update_statistics(&mut self, state: &AlertState) {
        *self.state_statistics.state_counts.entry(state.clone()).or_insert(0) += 1;
    }

    /// Analyze frequency trends
    pub fn analyze_frequency_trends(&mut self) {
        // Simplified trend analysis
        let recent_count = self.state_transitions
            .iter()
            .filter(|t| {
                SystemTime::now()
                    .duration_since(t.timestamp)
                    .unwrap_or(Duration::MAX) < Duration::from_secs(3600)
            })
            .count() as u64;

        self.state_statistics.frequency_trends.hourly_counts.push_back(recent_count);

        // Keep only recent data
        while self.state_statistics.frequency_trends.hourly_counts.len() > 168 { // 1 week of hourly data
            self.state_statistics.frequency_trends.hourly_counts.pop_front();
        }

        // Simple trend detection
        if self.state_statistics.frequency_trends.hourly_counts.len() >= 3 {
            let len = self.state_statistics.frequency_trends.hourly_counts.len();
            let recent_avg = self.state_statistics.frequency_trends.hourly_counts
                .iter()
                .skip(len - 3)
                .sum::<u64>() as f64 / 3.0;

            let older_avg = if len >= 6 {
                self.state_statistics.frequency_trends.hourly_counts
                    .iter()
                    .skip(len - 6)
                    .take(3)
                    .sum::<u64>() as f64 / 3.0
            } else {
                recent_avg
            };

            self.state_statistics.frequency_trends.trend_direction = if recent_avg > older_avg * 1.2 {
                FrequencyTrend::Increasing
            } else if recent_avg < older_avg * 0.8 {
                FrequencyTrend::Decreasing
            } else {
                FrequencyTrend::Stable
            };
        }
    }

    /// Clean up old data
    pub fn cleanup(&mut self) {
        let cutoff = SystemTime::now() - self.config.state_retention_period;

        // Remove old transitions
        self.state_transitions.retain(|t| t.timestamp >= cutoff);

        // Clean up old frequency data
        while self.state_statistics.frequency_trends.hourly_counts.len() > 168 {
            self.state_statistics.frequency_trends.hourly_counts.pop_front();
        }
        while self.state_statistics.frequency_trends.daily_counts.len() > 30 {
            self.state_statistics.frequency_trends.daily_counts.pop_front();
        }
        while self.state_statistics.frequency_trends.weekly_counts.len() > 12 {
            self.state_statistics.frequency_trends.weekly_counts.pop_front();
        }
    }

    /// Get summary statistics
    pub fn get_summary_statistics(&self) -> StateTrackerSummary {
        StateTrackerSummary {
            total_alerts_tracked: self.alert_states.len(),
            active_alerts: self.get_alerts_by_state(&AlertState::Firing).len(),
            acknowledged_alerts: self.get_alerts_by_state(&AlertState::Acknowledged).len(),
            resolved_alerts: self.get_alerts_by_state(&AlertState::Resolved).len(),
            total_transitions: self.state_transitions.len(),
            average_transition_rate: self.calculate_average_transition_rate(),
            most_common_state: self.get_most_common_state(),
            trend_direction: self.state_statistics.frequency_trends.trend_direction.clone(),
        }
    }

    /// Calculate average transition rate
    fn calculate_average_transition_rate(&self) -> f64 {
        if self.state_transitions.is_empty() {
            return 0.0;
        }

        let time_span = if let (Some(first), Some(last)) = (
            self.state_transitions.front(),
            self.state_transitions.back()
        ) {
            last.timestamp.duration_since(first.timestamp).unwrap_or(Duration::from_secs(1)).as_secs() as f64
        } else {
            1.0
        };

        self.state_transitions.len() as f64 / (time_span / 3600.0) // transitions per hour
    }

    /// Get most common state
    fn get_most_common_state(&self) -> Option<AlertState> {
        self.state_statistics.state_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(state, _)| state.clone())
    }
}

/// Summary statistics for state tracker
#[derive(Debug, Clone)]
pub struct StateTrackerSummary {
    /// Total number of alerts being tracked
    pub total_alerts_tracked: usize,
    /// Number of active (firing) alerts
    pub active_alerts: usize,
    /// Number of acknowledged alerts
    pub acknowledged_alerts: usize,
    /// Number of resolved alerts
    pub resolved_alerts: usize,
    /// Total number of state transitions recorded
    pub total_transitions: usize,
    /// Average transition rate (transitions per hour)
    pub average_transition_rate: f64,
    /// Most common alert state
    pub most_common_state: Option<AlertState>,
    /// Current frequency trend direction
    pub trend_direction: FrequencyTrend,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_alert_creation() {
        let alert = ActiveAlert::new(
            "test_alert_1".to_string(),
            "test_rule_1".to_string(),
            "Test alert message".to_string(),
            AlertSeverity::Warning,
            "fingerprint_123".to_string(),
        );

        assert_eq!(alert.alert_id, "test_alert_1");
        assert_eq!(alert.state, AlertState::Firing);
        assert!(!alert.is_acknowledged());
    }

    #[test]
    fn test_alert_acknowledgment() {
        let mut alert = ActiveAlert::new(
            "test_alert_1".to_string(),
            "test_rule_1".to_string(),
            "Test alert message".to_string(),
            AlertSeverity::Warning,
            "fingerprint_123".to_string(),
        );

        alert.acknowledge("test_user".to_string());

        assert!(alert.is_acknowledged());
        assert_eq!(alert.acknowledged_by, Some("test_user".to_string()));
        assert!(alert.acknowledged_at.is_some());
        assert_eq!(alert.state_history.len(), 1);
    }

    #[test]
    fn test_alert_state_transitions() {
        let mut alert = ActiveAlert::new(
            "test_alert_1".to_string(),
            "test_rule_1".to_string(),
            "Test alert message".to_string(),
            AlertSeverity::Warning,
            "fingerprint_123".to_string(),
        );

        alert.acknowledge("user1".to_string());
        alert.resolve("Issue fixed".to_string());

        assert!(alert.is_resolved());
        assert_eq!(alert.state_history.len(), 2);

        let transitions = &alert.state_history;
        assert_eq!(transitions[0].from_state, AlertState::Firing);
        assert_eq!(transitions[0].to_state, AlertState::Acknowledged);
        assert_eq!(transitions[1].from_state, AlertState::Acknowledged);
        assert_eq!(transitions[1].to_state, AlertState::Resolved);
    }

    #[test]
    fn test_state_tracker_basic_operations() {
        let mut tracker = AlertStateTracker::new();

        // Add alert
        tracker.add_alert("alert_1".to_string(), AlertState::Firing);
        assert_eq!(tracker.get_alert_state("alert_1"), Some(&AlertState::Firing));

        // Update state
        tracker.update_alert_state(
            "alert_1".to_string(),
            AlertState::Acknowledged,
            Some("user1".to_string()),
            Some("Manual ack".to_string())
        ).unwrap();

        assert_eq!(tracker.get_alert_state("alert_1"), Some(&AlertState::Acknowledged));
        assert_eq!(tracker.state_transitions.len(), 1);

        // Remove alert
        assert!(tracker.remove_alert("alert_1"));
        assert!(tracker.get_alert_state("alert_1").is_none());
    }

    #[test]
    fn test_alerts_by_state() {
        let mut tracker = AlertStateTracker::new();

        tracker.add_alert("alert_1".to_string(), AlertState::Firing);
        tracker.add_alert("alert_2".to_string(), AlertState::Firing);
        tracker.add_alert("alert_3".to_string(), AlertState::Acknowledged);

        let firing_alerts = tracker.get_alerts_by_state(&AlertState::Firing);
        let acked_alerts = tracker.get_alerts_by_state(&AlertState::Acknowledged);

        assert_eq!(firing_alerts.len(), 2);
        assert_eq!(acked_alerts.len(), 1);
        assert!(firing_alerts.contains(&"alert_1".to_string()));
        assert!(firing_alerts.contains(&"alert_2".to_string()));
        assert!(acked_alerts.contains(&"alert_3".to_string()));
    }

    #[test]
    fn test_related_alerts() {
        let mut alert = ActiveAlert::new(
            "test_alert_1".to_string(),
            "test_rule_1".to_string(),
            "Test alert message".to_string(),
            AlertSeverity::Warning,
            "fingerprint_123".to_string(),
        );

        alert.add_related_alert("related_1".to_string());
        alert.add_related_alert("related_2".to_string());

        assert_eq!(alert.related_alerts.len(), 2);
        assert!(alert.related_alerts.contains(&"related_1".to_string()));

        alert.remove_related_alert("related_1");
        assert_eq!(alert.related_alerts.len(), 1);
        assert!(!alert.related_alerts.contains(&"related_1".to_string()));
    }

    #[test]
    fn test_state_tracker_summary() {
        let mut tracker = AlertStateTracker::new();

        tracker.add_alert("alert_1".to_string(), AlertState::Firing);
        tracker.add_alert("alert_2".to_string(), AlertState::Acknowledged);
        tracker.add_alert("alert_3".to_string(), AlertState::Resolved);

        let summary = tracker.get_summary_statistics();

        assert_eq!(summary.total_alerts_tracked, 3);
        assert_eq!(summary.active_alerts, 1);
        assert_eq!(summary.acknowledged_alerts, 1);
        assert_eq!(summary.resolved_alerts, 1);
    }

    #[test]
    fn test_alert_age_calculation() {
        let alert = ActiveAlert::new(
            "test_alert_1".to_string(),
            "test_rule_1".to_string(),
            "Test alert message".to_string(),
            AlertSeverity::Warning,
            "fingerprint_123".to_string(),
        );

        let age = alert.get_age();
        assert!(age < Duration::from_secs(1)); // Should be very recent
    }

    #[test]
    fn test_frequency_trend_analysis() {
        let mut tracker = AlertStateTracker::new();

        // Add some transition data
        for i in 0..5 {
            tracker.add_alert(format!("alert_{}", i), AlertState::Firing);
        }

        tracker.analyze_frequency_trends();

        // Should have recorded the transitions
        assert!(!tracker.state_statistics.frequency_trends.hourly_counts.is_empty());
    }
}