use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};
use thiserror::Error;

// Re-export types from other modules that dashboard_core depends on
use super::layout_engine::{DashboardLayout};
use super::widget_system::{DashboardWidget};
use super::access_control::{DashboardPermissions};
use super::real_time_updates::{RefreshSettings};
use super::theme_styling::{DashboardThemeManager};
use super::performance_monitor::{DashboardPerformanceMonitor};

/// Main dashboard management system orchestrator
/// Coordinates all dashboard subsystems and provides core CRUD operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardManagementSystem {
    /// Dashboard registry and storage
    pub dashboards: Arc<RwLock<HashMap<String, Dashboard>>>,
    /// Dashboard templates for rapid deployment
    pub dashboard_templates: Arc<RwLock<HashMap<String, DashboardTemplate>>>,
    /// Real-time update system
    pub real_time_updates: Arc<RwLock<super::real_time_updates::RealTimeUpdates>>,
    /// Widget management system
    pub widget_manager: Arc<RwLock<super::widget_system::WidgetManager>>,
    /// Layout engine for responsive design
    pub layout_engine: Arc<RwLock<super::layout_engine::LayoutEngine>>,
    /// Permission and access control
    pub access_control: Arc<RwLock<super::access_control::DashboardAccessControl>>,
    /// Performance monitoring
    pub performance_monitor: Arc<RwLock<DashboardPerformanceMonitor>>,
    /// Theme and styling management
    pub theme_manager: Arc<RwLock<DashboardThemeManager>>,
}

/// Core dashboard structure containing all dashboard components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    /// Unique dashboard identifier
    pub dashboard_id: String,
    /// Human-readable dashboard name
    pub dashboard_name: String,
    /// Dashboard description
    pub description: String,
    /// Dashboard layout configuration
    pub layout: DashboardLayout,
    /// Widgets contained in this dashboard
    pub widgets: Vec<DashboardWidget>,
    /// Permission and security settings
    pub permissions: DashboardPermissions,
    /// Refresh and update settings
    pub refresh_settings: RefreshSettings,
    /// Dashboard metadata
    pub metadata: DashboardMetadata,
    /// Dashboard state management
    pub state: DashboardState,
    /// Version control information
    pub version_info: VersionInfo,
}

/// Dashboard metadata for tracking creation, modification, and categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
    /// Created by user
    pub created_by: String,
    /// Last modified by user
    pub modified_by: String,
    /// Dashboard tags
    pub tags: Vec<String>,
    /// Dashboard category
    pub category: Option<String>,
    /// Dashboard version
    pub version: String,
    /// Dashboard status
    pub status: DashboardStatus,
}

/// Dashboard lifecycle status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardStatus {
    /// Draft status
    Draft,
    /// Published status
    Published,
    /// Archived status
    Archived,
    /// Deprecated status
    Deprecated,
    /// Custom status
    Custom(String),
}

/// Dashboard state management and synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    /// Current dashboard state
    pub current_state: DashboardStateType,
    /// State persistence
    pub persistent_state: HashMap<String, String>,
    /// State synchronization
    pub synchronization: StateSynchronization,
    /// State history
    pub state_history: Vec<DashboardStateEntry>,
}

/// Dashboard operational state types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardStateType {
    /// Dashboard is loading
    Loading,
    /// Dashboard is ready
    Ready,
    /// Dashboard has error
    Error,
    /// Dashboard is refreshing
    Refreshing,
    /// Dashboard is in edit mode
    EditMode,
    /// Dashboard is in view mode
    ViewMode,
    /// Custom state
    Custom(String),
}

/// State synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSynchronization {
    /// Enable state sync
    pub enabled: bool,
    /// Synchronization scope
    pub scope: SynchronizationScope,
    /// Conflict resolution
    pub conflict_resolution: StateConflictResolution,
}

/// Synchronization scope enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynchronizationScope {
    /// User-specific sync
    User,
    /// Session-specific sync
    Session,
    /// Global sync
    Global,
    /// Custom scope
    Custom(String),
}

/// State conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateConflictResolution {
    /// Server wins
    ServerWins,
    /// Client wins
    ClientWins,
    /// Merge states
    Merge,
    /// Manual resolution
    Manual,
    /// Custom resolution
    Custom(String),
}

/// Dashboard state history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStateEntry {
    /// State timestamp
    pub timestamp: DateTime<Utc>,
    /// State type
    pub state_type: DashboardStateType,
    /// State data
    pub state_data: HashMap<String, String>,
    /// User who triggered state change
    pub triggered_by: String,
}

/// Version control information for dashboards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Current version
    pub current_version: String,
    /// Version history
    pub version_history: Vec<VersionEntry>,
    /// Version control settings
    pub version_control: VersionControl,
}

/// Version history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    /// Version number
    pub version: String,
    /// Version timestamp
    pub timestamp: DateTime<Utc>,
    /// Version author
    pub author: String,
    /// Version description
    pub description: String,
    /// Version changes
    pub changes: Vec<VersionChange>,
}

/// Individual version change record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionChange {
    /// Change type
    pub change_type: ChangeType,
    /// Changed component
    pub component: String,
    /// Change description
    pub description: String,
    /// Change data
    pub change_data: Option<String>,
}

/// Version change type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// Added component
    Added,
    /// Modified component
    Modified,
    /// Removed component
    Removed,
    /// Moved component
    Moved,
    /// Renamed component
    Renamed,
    /// Custom change
    Custom(String),
}

/// Version control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionControl {
    /// Enable version control
    pub enabled: bool,
    /// Auto-versioning
    pub auto_versioning: bool,
    /// Version retention
    pub retention_policy: VersionRetention,
    /// Branch support
    pub branch_support: bool,
}

/// Version retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionRetention {
    /// Maximum versions to keep
    pub max_versions: Option<usize>,
    /// Retention duration
    pub retention_duration: Option<Duration>,
    /// Retention strategy
    pub strategy: RetentionStrategy,
}

/// Retention strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionStrategy {
    /// Keep all versions
    KeepAll,
    /// Keep latest N versions
    KeepLatest(usize),
    /// Keep versions within duration
    KeepDuration(Duration),
    /// Custom retention strategy
    Custom(String),
}

/// Dashboard template for rapid dashboard creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTemplate {
    /// Template identifier
    pub template_id: String,
    /// Template name
    pub template_name: String,
    /// Template description
    pub description: String,
    /// Template category
    pub category: String,
    /// Template dashboard
    pub dashboard: Dashboard,
    /// Template metadata
    pub metadata: DashboardTemplateMetadata,
}

/// Dashboard template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTemplateMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Template author
    pub author: String,
    /// Template version
    pub version: String,
    /// Template tags
    pub tags: Vec<String>,
    /// Usage count
    pub usage_count: u64,
    /// Template rating
    pub rating: Option<f64>,
}

/// Comprehensive error handling for dashboard operations
#[derive(Debug, Error)]
pub enum DashboardError {
    #[error("Dashboard not found: {0}")]
    DashboardNotFound(String),
    #[error("Widget not found: {0}")]
    WidgetNotFound(String),
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Layout error: {0}")]
    LayoutError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Performance error: {0}")]
    PerformanceError(String),
    #[error("Theme error: {0}")]
    ThemeError(String),
    #[error("Version control error: {0}")]
    VersionControlError(String),
    #[error("State management error: {0}")]
    StateManagementError(String),
    #[error("Synchronization error: {0}")]
    SynchronizationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Core dashboard management system implementation
impl DashboardManagementSystem {
    /// Create a new dashboard management system
    pub fn new() -> Self {
        Self {
            dashboards: Arc::new(RwLock::new(HashMap::new())),
            dashboard_templates: Arc::new(RwLock::new(HashMap::new())),
            real_time_updates: Arc::new(RwLock::new(super::real_time_updates::RealTimeUpdates::default())),
            widget_manager: Arc::new(RwLock::new(super::widget_system::WidgetManager::new())),
            layout_engine: Arc::new(RwLock::new(super::layout_engine::LayoutEngine::new())),
            access_control: Arc::new(RwLock::new(super::access_control::DashboardAccessControl::new())),
            performance_monitor: Arc::new(RwLock::new(DashboardPerformanceMonitor::new())),
            theme_manager: Arc::new(RwLock::new(DashboardThemeManager::new())),
        }
    }

    /// Create a new dashboard
    pub fn create_dashboard(&self, dashboard: Dashboard) -> Result<(), DashboardError> {
        // Validate dashboard before creation
        self.validate_dashboard(&dashboard)?;

        let mut dashboards = self.dashboards.write().unwrap();

        // Check for duplicate dashboard ID
        if dashboards.contains_key(&dashboard.dashboard_id) {
            return Err(DashboardError::ConfigurationError(
                format!("Dashboard with ID '{}' already exists", dashboard.dashboard_id)
            ));
        }

        dashboards.insert(dashboard.dashboard_id.clone(), dashboard);
        Ok(())
    }

    /// Get dashboard by ID
    pub fn get_dashboard(&self, dashboard_id: &str) -> Result<Dashboard, DashboardError> {
        let dashboards = self.dashboards.read().unwrap();
        dashboards.get(dashboard_id)
            .cloned()
            .ok_or_else(|| DashboardError::DashboardNotFound(dashboard_id.to_string()))
    }

    /// Update dashboard
    pub fn update_dashboard(&self, dashboard: Dashboard) -> Result<(), DashboardError> {
        // Validate dashboard before update
        self.validate_dashboard(&dashboard)?;

        let mut dashboards = self.dashboards.write().unwrap();

        // Check if dashboard exists
        if !dashboards.contains_key(&dashboard.dashboard_id) {
            return Err(DashboardError::DashboardNotFound(dashboard.dashboard_id.clone()));
        }

        dashboards.insert(dashboard.dashboard_id.clone(), dashboard);
        Ok(())
    }

    /// Delete dashboard
    pub fn delete_dashboard(&self, dashboard_id: &str) -> Result<(), DashboardError> {
        let mut dashboards = self.dashboards.write().unwrap();
        dashboards.remove(dashboard_id)
            .ok_or_else(|| DashboardError::DashboardNotFound(dashboard_id.to_string()))?;
        Ok(())
    }

    /// List all dashboards
    pub fn list_dashboards(&self) -> Result<Vec<String>, DashboardError> {
        let dashboards = self.dashboards.read().unwrap();
        Ok(dashboards.keys().cloned().collect())
    }

    /// Get dashboard count
    pub fn get_dashboard_count(&self) -> Result<usize, DashboardError> {
        let dashboards = self.dashboards.read().unwrap();
        Ok(dashboards.len())
    }

    /// Create dashboard template
    pub fn create_template(&self, template: DashboardTemplate) -> Result<(), DashboardError> {
        let mut templates = self.dashboard_templates.write().unwrap();

        // Check for duplicate template ID
        if templates.contains_key(&template.template_id) {
            return Err(DashboardError::ConfigurationError(
                format!("Template with ID '{}' already exists", template.template_id)
            ));
        }

        templates.insert(template.template_id.clone(), template);
        Ok(())
    }

    /// Get dashboard template
    pub fn get_template(&self, template_id: &str) -> Result<DashboardTemplate, DashboardError> {
        let templates = self.dashboard_templates.read().unwrap();
        templates.get(template_id)
            .cloned()
            .ok_or_else(|| DashboardError::TemplateNotFound(template_id.to_string()))
    }

    /// Create dashboard from template
    pub fn create_dashboard_from_template(&self, template_id: &str, dashboard_id: String, dashboard_name: String) -> Result<Dashboard, DashboardError> {
        let template = self.get_template(template_id)?;

        let mut dashboard = template.dashboard.clone();
        dashboard.dashboard_id = dashboard_id;
        dashboard.dashboard_name = dashboard_name;

        // Update metadata
        dashboard.metadata.created_at = Utc::now();
        dashboard.metadata.modified_at = Utc::now();
        dashboard.metadata.status = DashboardStatus::Draft;

        // Reset state
        dashboard.state.current_state = DashboardStateType::Ready;
        dashboard.state.state_history.clear();

        // Reset version info
        dashboard.version_info.current_version = "1.0.0".to_string();
        dashboard.version_info.version_history.clear();

        Ok(dashboard)
    }

    /// Validate dashboard configuration
    fn validate_dashboard(&self, dashboard: &Dashboard) -> Result<(), DashboardError> {
        // Validate dashboard ID
        if dashboard.dashboard_id.is_empty() {
            return Err(DashboardError::ConfigurationError("Dashboard ID cannot be empty".to_string()));
        }

        // Validate dashboard name
        if dashboard.dashboard_name.is_empty() {
            return Err(DashboardError::ConfigurationError("Dashboard name cannot be empty".to_string()));
        }

        // Validate widgets
        for widget in &dashboard.widgets {
            if widget.widget_id.is_empty() {
                return Err(DashboardError::ConfigurationError("Widget ID cannot be empty".to_string()));
            }
        }

        Ok(())
    }

    /// Update dashboard state
    pub fn update_dashboard_state(&self, dashboard_id: &str, new_state: DashboardStateType, triggered_by: String) -> Result<(), DashboardError> {
        let mut dashboards = self.dashboards.write().unwrap();

        if let Some(dashboard) = dashboards.get_mut(dashboard_id) {
            // Add state history entry
            let state_entry = DashboardStateEntry {
                timestamp: Utc::now(),
                state_type: dashboard.state.current_state.clone(),
                state_data: dashboard.state.persistent_state.clone(),
                triggered_by: triggered_by.clone(),
            };
            dashboard.state.state_history.push(state_entry);

            // Update current state
            dashboard.state.current_state = new_state;
            dashboard.metadata.modified_at = Utc::now();
            dashboard.metadata.modified_by = triggered_by;

            Ok(())
        } else {
            Err(DashboardError::DashboardNotFound(dashboard_id.to_string()))
        }
    }

    /// Get dashboard by status
    pub fn get_dashboards_by_status(&self, status: DashboardStatus) -> Result<Vec<Dashboard>, DashboardError> {
        let dashboards = self.dashboards.read().unwrap();
        Ok(dashboards.values()
            .filter(|d| std::mem::discriminant(&d.metadata.status) == std::mem::discriminant(&status))
            .cloned()
            .collect())
    }
}

impl Default for DashboardManagementSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DashboardMetadata {
    fn default() -> Self {
        Self {
            created_at: Utc::now(),
            modified_at: Utc::now(),
            created_by: "system".to_string(),
            modified_by: "system".to_string(),
            tags: Vec::new(),
            category: None,
            version: "1.0.0".to_string(),
            status: DashboardStatus::Draft,
        }
    }
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            current_state: DashboardStateType::Ready,
            persistent_state: HashMap::new(),
            synchronization: StateSynchronization::default(),
            state_history: Vec::new(),
        }
    }
}

impl Default for StateSynchronization {
    fn default() -> Self {
        Self {
            enabled: true,
            scope: SynchronizationScope::User,
            conflict_resolution: StateConflictResolution::ServerWins,
        }
    }
}

impl Default for VersionInfo {
    fn default() -> Self {
        Self {
            current_version: "1.0.0".to_string(),
            version_history: Vec::new(),
            version_control: VersionControl::default(),
        }
    }
}

impl Default for VersionControl {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_versioning: false,
            retention_policy: VersionRetention::default(),
            branch_support: false,
        }
    }
}

impl Default for VersionRetention {
    fn default() -> Self {
        Self {
            max_versions: Some(10),
            retention_duration: Some(Duration::days(90)),
            strategy: RetentionStrategy::KeepLatest(10),
        }
    }
}