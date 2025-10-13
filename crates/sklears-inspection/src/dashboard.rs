//! Dashboard Integration
//!
//! This module provides web-based dashboard functionality for real-time model monitoring,
//! explanation visualization, and collaborative model interpretation.

use crate::{Float, SklResult, SklearsError};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "serde")]
use chrono::{DateTime, Utc};

/// Configuration for dashboard functionality
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DashboardConfig {
    /// Dashboard title
    pub title: String,
    /// Refresh interval in milliseconds
    pub refresh_interval: u32,
    /// Maximum number of data points to keep in memory
    pub max_data_points: usize,
    /// Enable real-time updates
    pub real_time_updates: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Dashboard layout configuration
    pub layout: DashboardLayout,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            title: "Model Inspection Dashboard".to_string(),
            refresh_interval: 5000,
            max_data_points: 1000,
            real_time_updates: true,
            alert_thresholds: AlertThresholds::default(),
            layout: DashboardLayout::default(),
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AlertThresholds {
    /// Performance drift threshold
    pub performance_drift: Float,
    /// Feature drift threshold
    pub feature_drift: Float,
    /// Prediction confidence threshold
    pub confidence_threshold: Float,
    /// Model agreement threshold
    pub agreement_threshold: Float,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            performance_drift: 0.05,
            feature_drift: 0.1,
            confidence_threshold: 0.8,
            agreement_threshold: 0.9,
        }
    }
}

/// Dashboard layout configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DashboardLayout {
    /// Grid layout (rows, columns)
    pub grid: (usize, usize),
    /// Widget configurations
    pub widgets: Vec<WidgetConfig>,
    /// Theme settings
    pub theme: DashboardTheme,
}

impl Default for DashboardLayout {
    fn default() -> Self {
        Self {
            grid: (3, 3),
            widgets: vec![
                WidgetConfig::new("performance", WidgetType::Performance, (0, 0), (1, 2)),
                WidgetConfig::new(
                    "feature_importance",
                    WidgetType::FeatureImportance,
                    (0, 2),
                    (1, 1),
                ),
                WidgetConfig::new(
                    "model_comparison",
                    WidgetType::ModelComparison,
                    (1, 0),
                    (1, 1),
                ),
                WidgetConfig::new(
                    "drift_detection",
                    WidgetType::DriftDetection,
                    (1, 1),
                    (1, 1),
                ),
                WidgetConfig::new("alerts", WidgetType::Alerts, (1, 2), (1, 1)),
                WidgetConfig::new("explanations", WidgetType::Explanations, (2, 0), (1, 3)),
            ],
            theme: DashboardTheme::default(),
        }
    }
}

/// Widget configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WidgetConfig {
    /// Widget identifier
    pub id: String,
    /// Widget type
    pub widget_type: WidgetType,
    /// Position in grid (row, col)
    pub position: (usize, usize),
    /// Size in grid units (rows, cols)
    pub size: (usize, usize),
    /// Widget-specific settings
    pub settings: HashMap<String, String>,
}

impl WidgetConfig {
    /// Create a new widget configuration
    pub fn new(
        id: &str,
        widget_type: WidgetType,
        position: (usize, usize),
        size: (usize, usize),
    ) -> Self {
        Self {
            id: id.to_string(),
            widget_type,
            position,
            size,
            settings: HashMap::new(),
        }
    }
}

/// Available widget types
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum WidgetType {
    /// Performance metrics over time
    Performance,
    /// Feature importance visualization
    FeatureImportance,
    /// Model comparison charts
    ModelComparison,
    /// Data/model drift detection
    DriftDetection,
    /// Alert notifications
    Alerts,
    /// Explanation visualizations
    Explanations,
    /// Custom widget
    Custom,
}

/// Dashboard theme settings
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DashboardTheme {
    /// Primary color
    pub primary_color: String,
    /// Secondary color
    pub secondary_color: String,
    /// Background color
    pub background_color: String,
    /// Text color
    pub text_color: String,
    /// Font family
    pub font_family: String,
}

impl Default for DashboardTheme {
    fn default() -> Self {
        Self {
            primary_color: "#3498db".to_string(),
            secondary_color: "#2ecc71".to_string(),
            background_color: "#ecf0f1".to_string(),
            text_color: "#2c3e50".to_string(),
            font_family: "Arial, sans-serif".to_string(),
        }
    }
}

/// Main dashboard struct
pub struct Dashboard {
    config: DashboardConfig,
    data_store: Arc<Mutex<DashboardDataStore>>,
    alert_manager: AlertManager,
    update_listeners: Vec<Box<dyn Fn(&DashboardUpdate) + Send + Sync>>,
}

impl Dashboard {
    /// Create a new dashboard
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config: config.clone(),
            data_store: Arc::new(Mutex::new(DashboardDataStore::new(config.max_data_points))),
            alert_manager: AlertManager::new(config.alert_thresholds),
            update_listeners: Vec::new(),
        }
    }

    /// Add a data point to the dashboard
    pub fn add_data_point(&mut self, data_point: DashboardDataPoint) -> SklResult<()> {
        let mut store = self.data_store.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire data store lock".to_string())
        })?;

        store.add_data_point(data_point.clone());

        // Check for alerts
        if let Some(alert) = self.alert_manager.check_alerts(&data_point) {
            self.notify_listeners(&DashboardUpdate::Alert(alert));
        }

        // Notify listeners of data update
        self.notify_listeners(&DashboardUpdate::DataUpdate(data_point));

        Ok(())
    }

    /// Get current dashboard state
    pub fn get_dashboard_state(&self) -> SklResult<DashboardState> {
        let store = self.data_store.lock().map_err(|_| {
            SklearsError::InvalidInput("Failed to acquire data store lock".to_string())
        })?;

        Ok(DashboardState {
            config: self.config.clone(),
            recent_data: store.get_recent_data(100),
            active_alerts: self.alert_manager.get_active_alerts(),
            #[cfg(feature = "serde")]
            last_updated: Utc::now(),
        })
    }

    /// Generate HTML dashboard
    pub fn generate_html(&self) -> SklResult<String> {
        let state = self.get_dashboard_state()?;
        let html = self.render_dashboard_html(&state);
        Ok(html)
    }

    /// Add update listener
    pub fn add_listener<F>(&mut self, listener: F)
    where
        F: Fn(&DashboardUpdate) + Send + Sync + 'static,
    {
        self.update_listeners.push(Box::new(listener));
    }

    fn notify_listeners(&self, update: &DashboardUpdate) {
        for listener in &self.update_listeners {
            listener(update);
        }
    }

    fn render_dashboard_html(&self, state: &DashboardState) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{
            font-family: {};
            background-color: {};
            color: {};
            margin: 0;
            padding: 20px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat({}, 1fr);
            grid-template-rows: repeat({}, 200px);
            gap: 20px;
            height: 100vh;
        }}
        .widget {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            overflow: hidden;
        }}
        .widget-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: {};
        }}
        .alert {{
            background-color: #e74c3c;
            color: white;
            padding: 10px;
            border-radius: 4px;
            margin: 5px 0;
        }}
        .metric {{
            font-size: 24px;
            font-weight: bold;
            color: {};
        }}
    </style>
    <script>
        // Real-time update functionality
        setInterval(function() {{
            if (window.dashboardRealtimeUpdates) {{
                location.reload();
            }}
        }}, {});
        window.dashboardRealtimeUpdates = {};
    </script>
</head>
<body>
    <h1>{}</h1>
    <div class="dashboard-grid">
        {}
    </div>
</body>
</html>"#,
            state.config.title,
            state.config.layout.theme.font_family,
            state.config.layout.theme.background_color,
            state.config.layout.theme.text_color,
            state.config.layout.grid.1,
            state.config.layout.grid.0,
            state.config.layout.theme.primary_color,
            state.config.layout.theme.secondary_color,
            state.config.refresh_interval,
            state.config.real_time_updates,
            state.config.title,
            self.render_widgets(state)
        )
    }

    fn render_widgets(&self, state: &DashboardState) -> String {
        state
            .config
            .layout
            .widgets
            .iter()
            .map(|widget| self.render_widget(widget, state))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn render_widget(&self, widget: &WidgetConfig, state: &DashboardState) -> String {
        let content = match widget.widget_type {
            WidgetType::Performance => self.render_performance_widget(state),
            WidgetType::FeatureImportance => self.render_feature_importance_widget(state),
            WidgetType::ModelComparison => self.render_model_comparison_widget(state),
            WidgetType::DriftDetection => self.render_drift_detection_widget(state),
            WidgetType::Alerts => self.render_alerts_widget(state),
            WidgetType::Explanations => self.render_explanations_widget(state),
            WidgetType::Custom => self.render_custom_widget(widget, state),
        };

        format!(
            r#"<div class="widget" style="grid-column: {} / span {}; grid-row: {} / span {};">
                <div class="widget-title">{}</div>
                {}
            </div>"#,
            widget.position.1 + 1,
            widget.size.1,
            widget.position.0 + 1,
            widget.size.0,
            widget.id.replace('_', " ").to_uppercase(),
            content
        )
    }

    fn render_performance_widget(&self, state: &DashboardState) -> String {
        if let Some(latest) = state.recent_data.last() {
            format!(
                r#"<div class="metric">Accuracy: {:.2}%</div>
                <div>Last updated: {}</div>"#,
                latest.metrics.get("accuracy").unwrap_or(&0.0) * 100.0,
                "just now"
            )
        } else {
            "<div>No performance data available</div>".to_string()
        }
    }

    fn render_feature_importance_widget(&self, _state: &DashboardState) -> String {
        "<div>Feature importance visualization would go here</div>".to_string()
    }

    fn render_model_comparison_widget(&self, _state: &DashboardState) -> String {
        "<div>Model comparison charts would go here</div>".to_string()
    }

    fn render_drift_detection_widget(&self, _state: &DashboardState) -> String {
        "<div>Drift detection metrics would go here</div>".to_string()
    }

    fn render_alerts_widget(&self, state: &DashboardState) -> String {
        if state.active_alerts.is_empty() {
            "<div>No active alerts</div>".to_string()
        } else {
            state
                .active_alerts
                .iter()
                .map(|alert| {
                    format!(
                        r#"<div class="alert">{}: {}</div>"#,
                        alert.alert_type, alert.message
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    fn render_explanations_widget(&self, _state: &DashboardState) -> String {
        "<div>Explanation visualizations would go here</div>".to_string()
    }

    fn render_custom_widget(&self, widget: &WidgetConfig, _state: &DashboardState) -> String {
        format!("<div>Custom widget: {}</div>", widget.id)
    }
}

/// Dashboard data store
pub struct DashboardDataStore {
    data_points: Vec<DashboardDataPoint>,
    max_points: usize,
}

impl DashboardDataStore {
    pub fn new(max_points: usize) -> Self {
        Self {
            data_points: Vec::new(),
            max_points,
        }
    }

    pub fn add_data_point(&mut self, data_point: DashboardDataPoint) {
        self.data_points.push(data_point);
        if self.data_points.len() > self.max_points {
            self.data_points.remove(0);
        }
    }

    pub fn get_recent_data(&self, count: usize) -> Vec<DashboardDataPoint> {
        let start = if self.data_points.len() > count {
            self.data_points.len() - count
        } else {
            0
        };
        self.data_points[start..].to_vec()
    }
}

/// Dashboard data point
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DashboardDataPoint {
    /// Timestamp
    #[cfg(feature = "serde")]
    pub timestamp: DateTime<Utc>,
    /// Model identifier
    pub model_id: String,
    /// Performance metrics
    pub metrics: HashMap<String, Float>,
    /// Feature statistics
    pub feature_stats: HashMap<String, Float>,
    /// Prediction confidence
    pub confidence_scores: Vec<Float>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl DashboardDataPoint {
    /// Create a new dashboard data point
    pub fn new(model_id: String) -> Self {
        Self {
            #[cfg(feature = "serde")]
            timestamp: Utc::now(),
            model_id,
            metrics: HashMap::new(),
            feature_stats: HashMap::new(),
            confidence_scores: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, name: String, value: Float) {
        self.metrics.insert(name, value);
    }

    /// Add feature statistic
    pub fn add_feature_stat(&mut self, feature: String, value: Float) {
        self.feature_stats.insert(feature, value);
    }
}

/// Dashboard state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DashboardState {
    /// Dashboard configuration
    pub config: DashboardConfig,
    /// Recent data points
    pub recent_data: Vec<DashboardDataPoint>,
    /// Active alerts
    pub active_alerts: Vec<Alert>,
    /// Last update timestamp
    #[cfg(feature = "serde")]
    pub last_updated: DateTime<Utc>,
}

/// Alert manager
pub struct AlertManager {
    thresholds: AlertThresholds,
    active_alerts: Vec<Alert>,
}

impl AlertManager {
    pub fn new(thresholds: AlertThresholds) -> Self {
        Self {
            thresholds,
            active_alerts: Vec::new(),
        }
    }

    pub fn check_alerts(&mut self, data_point: &DashboardDataPoint) -> Option<Alert> {
        // Check performance drift
        if let Some(accuracy) = data_point.metrics.get("accuracy") {
            if *accuracy < (1.0 - self.thresholds.performance_drift) {
                let alert = Alert {
                    id: format!("perf_drift_{}", data_point.model_id),
                    alert_type: AlertType::PerformanceDrift,
                    message: format!(
                        "Model {} accuracy dropped to {:.2}%",
                        data_point.model_id,
                        accuracy * 100.0
                    ),
                    severity: AlertSeverity::High,
                    #[cfg(feature = "serde")]
                    timestamp: Utc::now(),
                };
                self.active_alerts.push(alert.clone());
                return Some(alert);
            }
        }

        // Check confidence threshold
        let avg_confidence = if !data_point.confidence_scores.is_empty() {
            data_point.confidence_scores.iter().sum::<Float>()
                / data_point.confidence_scores.len() as Float
        } else {
            1.0
        };

        if avg_confidence < self.thresholds.confidence_threshold {
            let alert = Alert {
                id: format!("low_conf_{}", data_point.model_id),
                alert_type: AlertType::LowConfidence,
                message: format!("Low prediction confidence: {:.2}%", avg_confidence * 100.0),
                severity: AlertSeverity::Medium,
                #[cfg(feature = "serde")]
                timestamp: Utc::now(),
            };
            self.active_alerts.push(alert.clone());
            return Some(alert);
        }

        None
    }

    pub fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.clone()
    }

    pub fn clear_alert(&mut self, alert_id: &str) {
        self.active_alerts.retain(|alert| alert.id != alert_id);
    }
}

/// Alert information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Alert {
    /// Alert identifier
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Timestamp
    #[cfg(feature = "serde")]
    pub timestamp: DateTime<Utc>,
}

/// Alert types
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AlertType {
    /// Performance degradation detected
    PerformanceDrift,
    /// Feature distribution drift
    FeatureDrift,
    /// Low prediction confidence
    LowConfidence,
    /// Model disagreement
    ModelDisagreement,
    /// System error
    SystemError,
}

impl std::fmt::Display for AlertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertType::PerformanceDrift => write!(f, "Performance Drift"),
            AlertType::FeatureDrift => write!(f, "Feature Drift"),
            AlertType::LowConfidence => write!(f, "Low Confidence"),
            AlertType::ModelDisagreement => write!(f, "Model Disagreement"),
            AlertType::SystemError => write!(f, "System Error"),
        }
    }
}

/// Alert severity levels
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AlertSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Dashboard update types
#[derive(Debug, Clone)]
pub enum DashboardUpdate {
    /// New data point added
    DataUpdate(DashboardDataPoint),
    /// New alert triggered
    Alert(Alert),
    /// Configuration changed
    ConfigUpdate(DashboardConfig),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let dashboard = Dashboard::new(config);

        assert_eq!(dashboard.config.title, "Model Inspection Dashboard");
        assert_eq!(dashboard.config.refresh_interval, 5000);
    }

    #[test]
    fn test_data_point_creation() {
        let mut data_point = DashboardDataPoint::new("test_model".to_string());
        data_point.add_metric("accuracy".to_string(), 0.95);
        data_point.add_feature_stat("feature_0".to_string(), 0.5);

        assert_eq!(data_point.model_id, "test_model");
        assert_eq!(data_point.metrics.get("accuracy"), Some(&0.95));
        assert_eq!(data_point.feature_stats.get("feature_0"), Some(&0.5));
    }

    #[test]
    fn test_alert_generation() {
        let thresholds = AlertThresholds::default();
        let mut alert_manager = AlertManager::new(thresholds);

        let mut data_point = DashboardDataPoint::new("test_model".to_string());
        data_point.add_metric("accuracy".to_string(), 0.80); // Below drift threshold

        let alert = alert_manager.check_alerts(&data_point);
        assert!(alert.is_some());

        if let Some(alert) = alert {
            assert!(matches!(alert.alert_type, AlertType::PerformanceDrift));
        }
    }

    #[test]
    fn test_dashboard_state() {
        let config = DashboardConfig::default();
        let dashboard = Dashboard::new(config);
        let state = dashboard.get_dashboard_state().unwrap();

        assert_eq!(state.config.title, "Model Inspection Dashboard");
        assert!(state.recent_data.is_empty());
        assert!(state.active_alerts.is_empty());
    }

    #[test]
    fn test_widget_configuration() {
        let widget = WidgetConfig::new("test_widget", WidgetType::Performance, (0, 0), (1, 1));

        assert_eq!(widget.id, "test_widget");
        assert!(matches!(widget.widget_type, WidgetType::Performance));
        assert_eq!(widget.position, (0, 0));
        assert_eq!(widget.size, (1, 1));
    }

    #[test]
    fn test_html_generation() {
        let config = DashboardConfig::default();
        let dashboard = Dashboard::new(config);
        let html = dashboard.generate_html().unwrap();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Model Inspection Dashboard"));
        assert!(html.contains("dashboard-grid"));
    }

    #[test]
    fn test_data_store() {
        let mut store = DashboardDataStore::new(2);

        let data1 = DashboardDataPoint::new("model1".to_string());
        let data2 = DashboardDataPoint::new("model2".to_string());
        let data3 = DashboardDataPoint::new("model3".to_string());

        store.add_data_point(data1);
        store.add_data_point(data2);
        assert_eq!(store.data_points.len(), 2);

        store.add_data_point(data3);
        assert_eq!(store.data_points.len(), 2); // Should maintain max size
        assert_eq!(store.data_points[0].model_id, "model2"); // First one should be removed
    }

    #[test]
    fn test_alert_clearing() {
        let thresholds = AlertThresholds::default();
        let mut alert_manager = AlertManager::new(thresholds);

        let alert = Alert {
            id: "test_alert".to_string(),
            alert_type: AlertType::SystemError,
            message: "Test alert".to_string(),
            severity: AlertSeverity::Medium,
            #[cfg(feature = "serde")]
            timestamp: Utc::now(),
        };

        alert_manager.active_alerts.push(alert);
        assert_eq!(alert_manager.get_active_alerts().len(), 1);

        alert_manager.clear_alert("test_alert");
        assert_eq!(alert_manager.get_active_alerts().len(), 0);
    }
}
