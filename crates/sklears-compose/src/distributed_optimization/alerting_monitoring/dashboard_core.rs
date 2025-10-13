use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use super::dashboard_layout::DashboardTheme;
use super::dashboard_datasources::DataSourceConfiguration;
use super::dashboard_widgets::Widget;

/// Dashboard types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardType {
    System,
    Application,
    Business,
    Infrastructure,
    Security,
    Performance,
    Custom(String),
}

/// Dashboard priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DashboardPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Dashboard visibility levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisibilityLevel {
    Public,
    Internal,
    Private,
    Restricted(Vec<String>),
}

/// Time range for dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeRange {
    Last5Minutes,
    Last15Minutes,
    Last30Minutes,
    LastHour,
    Last3Hours,
    Last6Hours,
    Last12Hours,
    Last24Hours,
    Last7Days,
    Last30Days,
    Custom { start: SystemTime, end: SystemTime },
}

/// Dashboard filter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardFilterType {
    Dropdown,
    MultiSelect,
    DateRange,
    TimeRange,
    Text,
    Number,
    Slider,
    Radio,
    Checkbox,
    Custom(String),
}

/// Filter positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterPosition {
    Top,
    Bottom,
    Left,
    Right,
    Inline,
    Floating,
    Custom(f64, f64),
}

/// Variable types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    String,
    Number,
    Boolean,
    Date,
    Time,
    DateTime,
    Array,
    Object,
    Custom(String),
}

/// Variable sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableSource {
    Static(String),
    Query(String),
    Environment,
    User,
    Session,
    Custom(String),
}

/// Variable scopes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableScope {
    Dashboard,
    Widget(String),
    Global,
    Session,
    User,
}

/// Share types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShareType {
    PublicLink,
    EmbeddedIframe,
    Snapshot,
    LiveStream,
    API,
    Custom(String),
}

/// Time range values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeRangeValue {
    Absolute { start: SystemTime, end: SystemTime },
    Relative { duration: Duration },
    Dynamic(String),
}

/// Timezone handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimezoneHandling {
    UTC,
    Local,
    Specific(String),
    UserPreference,
}

/// Dashboard filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardFilter {
    pub filter_id: String,
    pub name: String,
    pub type_: DashboardFilterType,
    pub data_source: String,
    pub field: String,
    pub default_value: Option<String>,
    pub multi_select: bool,
    pub required: bool,
    pub visible: bool,
    pub position: FilterPosition,
}

/// Dashboard variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardVariable {
    pub variable_id: String,
    pub name: String,
    pub description: Option<String>,
    pub variable_type: VariableType,
    pub default_value: String,
    pub value_source: VariableSource,
    pub scope: VariableScope,
}

/// Time range configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRangeConfiguration {
    pub default_range: TimeRange,
    pub available_ranges: Vec<TimeRangePreset>,
    pub custom_range_enabled: bool,
    pub auto_refresh: bool,
    pub timezone_handling: TimezoneHandling,
}

/// Time range presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRangePreset {
    pub name: String,
    pub display_name: String,
    pub range: TimeRangeValue,
    pub relative: bool,
}

/// Dashboard refresh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardRefreshConfig {
    pub auto_refresh: bool,
    pub refresh_interval: Duration,
    pub refresh_on_focus: bool,
    pub refresh_on_data_change: bool,
    pub partial_refresh: bool,
    pub background_refresh: bool,
}

/// Sharing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingConfiguration {
    pub enabled: bool,
    pub share_types: Vec<ShareType>,
    pub expiration_config: ExpirationConfig,
    pub password_protection: bool,
    pub watermark_enabled: bool,
    pub download_enabled: bool,
}

/// Expiration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpirationConfig {
    pub enabled: bool,
    pub default_expiration: Duration,
    pub max_expiration: Duration,
    pub auto_extend: bool,
}

/// Dashboard permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPermissions {
    pub owner: String,
    pub view_permissions: Vec<String>,
    pub edit_permissions: Vec<String>,
    pub admin_permissions: Vec<String>,
    pub share_permissions: Vec<String>,
    pub delete_permissions: Vec<String>,
    pub inherit_permissions: bool,
}

/// Dashboard metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetadata {
    pub created_at: SystemTime,
    pub created_by: String,
    pub modified_at: SystemTime,
    pub modified_by: String,
    pub version: String,
    pub tags: Vec<String>,
    pub category: String,
    pub starred_by: Vec<String>,
    pub usage_statistics: DashboardUsageStatistics,
    pub performance_metrics: DashboardPerformanceMetrics,
}

/// Dashboard usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardUsageStatistics {
    pub view_count: u64,
    pub unique_visitors: u64,
    pub avg_session_duration: Duration,
    pub last_accessed: Option<SystemTime>,
    pub peak_concurrent_users: u32,
    pub access_patterns: Vec<AccessPattern>,
    pub peak_usage_times: Vec<PeakUsageTime>,
}

/// Access pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    pub user_id: String,
    pub access_frequency: f64,
    pub preferred_widgets: Vec<String>,
    pub interaction_patterns: Vec<String>,
}

/// Peak usage time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakUsageTime {
    pub time_range: String,
    pub concurrent_users: u32,
    pub load_metrics: HashMap<String, f64>,
}

/// Dashboard performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPerformanceMetrics {
    pub load_time: Duration,
    pub render_time: Duration,
    pub data_fetch_time: Duration,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub resource_usage: HashMap<String, f64>,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfiguration {
    pub max_dashboards: Option<u32>,
    pub max_widgets_per_dashboard: Option<u32>,
    pub default_theme: String,
    pub cache_settings: DashboardCacheSettings,
    pub security_settings: DashboardSecuritySettings,
    pub performance_settings: DashboardPerformanceSettings,
}

/// Dashboard cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardCacheSettings {
    pub enabled: bool,
    pub default_ttl: Duration,
    pub max_cache_size: u64,
    pub cache_strategy: String,
}

/// Dashboard security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSecuritySettings {
    pub csrf_protection: bool,
    pub xss_protection: bool,
    pub content_security_policy: String,
    pub session_timeout: Duration,
    pub max_login_attempts: u32,
    pub password_policy: PasswordPolicy,
}

/// Password policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordPolicy {
    pub min_length: u32,
    pub require_uppercase: bool,
    pub require_lowercase: bool,
    pub require_numbers: bool,
    pub require_special_chars: bool,
    pub password_history: u32,
    pub password_expiration: Duration,
}

/// Dashboard performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPerformanceSettings {
    pub lazy_loading: bool,
    pub virtual_scrolling: bool,
    pub progressive_rendering: bool,
    pub data_pagination: bool,
    pub image_optimization: bool,
    pub compression_enabled: bool,
    pub cdn_enabled: bool,
    pub worker_threads: u32,
}

/// Main dashboard definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub dashboard_id: String,
    pub name: String,
    pub description: Option<String>,
    pub dashboard_type: DashboardType,
    pub priority: DashboardPriority,
    pub visibility: VisibilityLevel,
    pub layout: super::dashboard_layout::DashboardLayout,
    pub theme: String,
    pub widgets: Vec<Widget>,
    pub data_sources: Vec<String>,
    pub filters: Vec<DashboardFilter>,
    pub variables: Vec<DashboardVariable>,
    pub time_range: TimeRangeConfiguration,
    pub refresh_config: DashboardRefreshConfig,
    pub sharing_config: SharingConfiguration,
    pub permissions: DashboardPermissions,
    pub metadata: DashboardMetadata,
}

/// Main dashboard manager
pub struct DashboardManager {
    /// Available dashboards
    pub dashboards: Arc<RwLock<HashMap<String, Dashboard>>>,
    /// Dashboard themes
    pub themes: Arc<RwLock<HashMap<String, DashboardTheme>>>,
    /// Data sources
    pub data_sources: Arc<RwLock<HashMap<String, DataSourceConfiguration>>>,
    /// Manager configuration
    pub config: DashboardConfiguration,
    /// Real-time data handler
    pub real_time_handler: Arc<RwLock<RealTimeHandler>>,
    /// Rendering engine
    pub rendering_engine: Arc<RwLock<super::dashboard_performance::RenderingEngine>>,
    /// Cache manager
    pub cache_manager: Arc<RwLock<super::dashboard_performance::CacheManager>>,
    /// Security manager
    pub security_manager: Arc<RwLock<SecurityManager>>,
    /// Performance monitor
    pub performance_monitor: Arc<RwLock<super::dashboard_performance::PerformanceMonitor>>,
}

/// Real-time data handler
pub struct RealTimeHandler {
    /// Active connections
    pub connections: HashMap<String, Connection>,
    /// Data streams
    pub data_streams: HashMap<String, DataStream>,
    /// Update frequency
    pub update_frequency: Duration,
    /// Buffer size
    pub buffer_size: u32,
}

/// Connection information
#[derive(Debug, Clone)]
pub struct Connection {
    pub connection_id: String,
    pub dashboard_id: String,
    pub user_id: String,
    pub connected_at: SystemTime,
    pub last_activity: SystemTime,
    pub subscribed_widgets: Vec<String>,
}

/// Data stream
#[derive(Debug, Clone)]
pub struct DataStream {
    pub stream_id: String,
    pub data_source: String,
    pub update_frequency: Duration,
    pub last_update: SystemTime,
    pub subscribers: Vec<String>,
    pub buffer: VecDeque<DataPoint>,
}

/// Data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: SystemTime,
    pub value: String,
    pub metadata: HashMap<String, String>,
}

/// Security manager
pub struct SecurityManager {
    /// Active sessions
    pub sessions: HashMap<String, UserSession>,
    /// Access tokens
    pub tokens: HashMap<String, AccessToken>,
    /// Security policies
    pub policies: Vec<SecurityPolicy>,
    /// Audit log
    pub audit_log: VecDeque<AuditEntry>,
}

/// User session
#[derive(Debug, Clone)]
pub struct UserSession {
    pub session_id: String,
    pub user_id: String,
    pub created_at: SystemTime,
    pub last_activity: SystemTime,
    pub ip_address: String,
    pub user_agent: String,
    pub permissions: Vec<String>,
}

/// Access token
#[derive(Debug, Clone)]
pub struct AccessToken {
    pub token_id: String,
    pub user_id: String,
    pub scopes: Vec<String>,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub last_used: Option<SystemTime>,
}

/// Security policy
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    pub policy_id: String,
    pub name: String,
    pub rules: Vec<SecurityRule>,
    pub enabled: bool,
}

/// Security rule
#[derive(Debug, Clone)]
pub struct SecurityRule {
    pub rule_id: String,
    pub condition: String,
    pub action: SecurityAction,
    pub priority: u32,
}

/// Security actions
#[derive(Debug, Clone)]
pub enum SecurityAction {
    Allow,
    Deny,
    Require2FA,
    LogAndAllow,
    LogAndDeny,
    RateLimit(u32),
}

/// Audit entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub timestamp: SystemTime,
    pub user_id: String,
    pub action: String,
    pub resource: String,
    pub outcome: AuditOutcome,
    pub details: HashMap<String, String>,
}

/// Audit outcomes
#[derive(Debug, Clone)]
pub enum AuditOutcome {
    Success,
    Failure,
    Warning,
}

impl DashboardManager {
    /// Create a new dashboard manager
    pub fn new(config: DashboardConfiguration) -> Self {
        Self {
            dashboards: Arc::new(RwLock::new(HashMap::new())),
            themes: Arc::new(RwLock::new(HashMap::new())),
            data_sources: Arc::new(RwLock::new(HashMap::new())),
            config,
            real_time_handler: Arc::new(RwLock::new(RealTimeHandler::new())),
            rendering_engine: Arc::new(RwLock::new(super::dashboard_performance::RenderingEngine::new())),
            cache_manager: Arc::new(RwLock::new(super::dashboard_performance::CacheManager::new())),
            security_manager: Arc::new(RwLock::new(SecurityManager::new())),
            performance_monitor: Arc::new(RwLock::new(super::dashboard_performance::PerformanceMonitor::new())),
        }
    }

    /// Add a new dashboard
    pub fn add_dashboard(&self, dashboard: Dashboard) -> Result<(), DashboardError> {
        let mut dashboards = self.dashboards.write().unwrap();

        if let Some(max_dashboards) = self.config.max_dashboards {
            if dashboards.len() >= max_dashboards as usize {
                return Err(DashboardError::MaxDashboardsExceeded);
            }
        }

        dashboards.insert(dashboard.dashboard_id.clone(), dashboard);
        Ok(())
    }

    /// Remove a dashboard
    pub fn remove_dashboard(&self, dashboard_id: &str) -> Option<Dashboard> {
        let mut dashboards = self.dashboards.write().unwrap();
        dashboards.remove(dashboard_id)
    }

    /// Get a dashboard
    pub fn get_dashboard(&self, dashboard_id: &str) -> Option<Dashboard> {
        let dashboards = self.dashboards.read().unwrap();
        dashboards.get(dashboard_id).cloned()
    }

    /// List all dashboards
    pub fn list_dashboards(&self) -> Vec<Dashboard> {
        let dashboards = self.dashboards.read().unwrap();
        dashboards.values().cloned().collect()
    }

    /// Update dashboard
    pub fn update_dashboard(&self, dashboard: Dashboard) -> Result<(), DashboardError> {
        let mut dashboards = self.dashboards.write().unwrap();

        if !dashboards.contains_key(&dashboard.dashboard_id) {
            return Err(DashboardError::DashboardNotFound(dashboard.dashboard_id));
        }

        dashboards.insert(dashboard.dashboard_id.clone(), dashboard);
        Ok(())
    }
}

impl RealTimeHandler {
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
            data_streams: HashMap::new(),
            update_frequency: Duration::from_secs(1),
            buffer_size: 1000,
        }
    }

    pub fn add_connection(&mut self, connection: Connection) {
        self.connections.insert(connection.connection_id.clone(), connection);
    }

    pub fn remove_connection(&mut self, connection_id: &str) -> Option<Connection> {
        self.connections.remove(connection_id)
    }

    pub fn add_data_stream(&mut self, stream: DataStream) {
        self.data_streams.insert(stream.stream_id.clone(), stream);
    }

    pub fn remove_data_stream(&mut self, stream_id: &str) -> Option<DataStream> {
        self.data_streams.remove(stream_id)
    }
}

impl SecurityManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            tokens: HashMap::new(),
            policies: Vec::new(),
            audit_log: VecDeque::new(),
        }
    }

    pub fn create_session(&mut self, session: UserSession) {
        self.sessions.insert(session.session_id.clone(), session);
    }

    pub fn validate_session(&self, session_id: &str) -> bool {
        self.sessions.contains_key(session_id)
    }

    pub fn create_token(&mut self, token: AccessToken) {
        self.tokens.insert(token.token_id.clone(), token);
    }

    pub fn validate_token(&self, token_id: &str) -> bool {
        if let Some(token) = self.tokens.get(token_id) {
            token.expires_at > SystemTime::now()
        } else {
            false
        }
    }

    pub fn add_audit_entry(&mut self, entry: AuditEntry) {
        self.audit_log.push_back(entry);

        // Keep only recent entries (e.g., last 10000)
        while self.audit_log.len() > 10000 {
            self.audit_log.pop_front();
        }
    }
}

/// Dashboard errors
#[derive(Debug, Clone)]
pub enum DashboardError {
    DashboardNotFound(String),
    MaxDashboardsExceeded,
    InvalidConfiguration,
    PermissionDenied,
    ValidationError(String),
}

impl std::fmt::Display for DashboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DashboardNotFound(id) => write!(f, "Dashboard not found: {}", id),
            Self::MaxDashboardsExceeded => write!(f, "Maximum number of dashboards exceeded"),
            Self::InvalidConfiguration => write!(f, "Invalid dashboard configuration"),
            Self::PermissionDenied => write!(f, "Permission denied"),
            Self::ValidationError(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for DashboardError {}

impl Default for RealTimeHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self::new()
    }
}