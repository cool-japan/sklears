//! Dashboard interface modules
//!
//! This module provides a comprehensive dashboard management system broken down into focused modules:
//!
//! ## Module Overview
//!
//! ### Core Components
//! - [`dashboard_core`] - Core dashboard components, layout management, and widget systems
//! - [`permissions_access`] - Access control, permissions, and security management
//! - [`templates_reports`] - Template and report management with versioning support
//!
//! ### User Interface & Experience
//! - [`styling_themes`] - Styling, theming, and CSS generation systems
//! - [`web_interface`] - Web server configuration, API endpoints, and frontend integration
//! - [`realtime_updates`] - Real-time communication and notification systems
//!
//! ### Distribution & Export
//! - [`distribution_export`] - Content distribution channels and export functionality
//!
//! ### System Support
//! - [`error_types`] - Comprehensive error handling and recovery systems
//!
//! ## Architecture
//!
//! The dashboard interface follows a modular architecture where each module has
//! clear responsibilities and well-defined interfaces:
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │ dashboard_core  │    │permissions_access│    │templates_reports│
//! │                 │    │                 │    │                 │
//! │ • Dashboard     │    │ • Permissions   │    │ • Templates     │
//! │ • Layout        │    │ • Access Control│    │ • Reports       │
//! │ • Widgets       │    │ • Security      │    │ • Versioning    │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!          │                       │                       │
//!          └───────────────────────┼───────────────────────┘
//!                                  │
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │ styling_themes  │    │ web_interface   │    │realtime_updates │
//! │                 │    │                 │    │                 │
//! │ • Themes        │    │ • Web Server    │    │ • WebSockets    │
//! │ • CSS Generator │    │ • API Endpoints │    │ • Notifications │
//! │ • Style System  │    │ • Authentication│    │ • Event System  │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!          │                       │                       │
//!          └───────────────────────┼───────────────────────┘
//!                                  │
//! ┌─────────────────┐    ┌─────────────────┐
//! │distribution_export│   │  error_types    │
//! │                 │    │                 │
//! │ • Distribution  │    │ • Error Types   │
//! │ • Export        │    │ • Error Handler │
//! │ • Scheduling    │    │ • Recovery      │
//! └─────────────────┘    └─────────────────┘
//! ```
//!
//! ## Usage Examples
//!
//! ### Basic Dashboard Creation
//! ```rust,ignore
//! use dashboard_interface::{Dashboard, DashboardManager, DashboardWidget};
//!
//! let mut manager = DashboardManager::new();
//! let dashboard = Dashboard::new("dashboard-1".to_string(), "My Dashboard".to_string());
//! manager.add_dashboard(dashboard);
//! ```
//!
//! ### Permission Management
//! ```rust,ignore
//! use dashboard_interface::{DashboardPermissions, Permission};
//!
//! let mut permissions = DashboardPermissions::new("owner-id".to_string());
//! permissions.add_viewer("user-1".to_string());
//! permissions.add_editor("user-2".to_string());
//! ```
//!
//! ### Template Management
//! ```rust,ignore
//! use dashboard_interface::{TemplateManager, DashboardTemplate};
//!
//! let mut template_manager = TemplateManager::new();
//! // Add templates and manage versions
//! ```
//!
//! ### Real-time Updates
//! ```rust,ignore
//! use dashboard_interface::{RealTimeUpdates, WebSocketConfig};
//!
//! let mut updates = RealTimeUpdates::new();
//! updates.enable_push_notifications();
//! ```

pub mod dashboard_core;
pub mod permissions_access;
pub mod templates_reports;
pub mod styling_themes;
pub mod distribution_export;
pub mod web_interface;
pub mod realtime_updates;
pub mod error_types;

// Re-export core dashboard components
pub use dashboard_core::{
    DashboardManager, Dashboard, DashboardLayout, DashboardLayoutType, DashboardGridConfig,
    DashboardResponsiveConfig, DashboardBreakpoint, DashboardLayoutAdjustments,
    DashboardWidget, WidgetType, WidgetPosition, WidgetSize, WidgetConfiguration,
    WidgetStyling, WidgetInteractions, DrillDownConfig, WidgetFilter, WidgetFilterType,
    FilterLogic, RefreshSettings
};

// Re-export permissions and access control components
pub use permissions_access::{
    DashboardPermissions, AccessControls, TimeRestriction, Permission, UserRole,
    AccessControlEntry, PrincipalType, AccessCondition, AccessConditionType,
    AccessOperator, PermissionMatrix, PermissionRelationship, RelationshipType,
    InheritanceRule, InheritanceMode, SecurityPolicy, PasswordPolicy, SessionPolicy,
    AccessAttemptPolicy, AuditPolicy, AuditEventType
};

// Re-export template and report components
pub use templates_reports::{
    TemplateManager, DashboardTemplate, WidgetTemplate, CustomizationOptions,
    ReportTemplate, TemplateStructure, TemplateSection, SectionType, ContentType,
    SectionLayoutProperties, Position, PositionUnit, Size, SizeUnit, Padding, Margin,
    Alignment, HorizontalAlignment, VerticalAlignment, ConditionalDisplay,
    LayoutConfig, PageSize, PageUnit, PageOrientation, ResponsiveLayout,
    ResponsiveBreakpoint, LayoutAdjustments, ScalingStrategy, GridSystem, GridType,
    TemplateMetadata, CompatibilityInfo, ParameterDefinition, ParameterType,
    ValidationRule, ValidationRuleType, LocalizationConfig, TemplateCategory,
    TemplateVersioning, TemplateVersion, VersioningRule, VersioningTrigger,
    VersionIncrementType
};

// Re-export styling and theme components
pub use styling_themes::{
    StyleManager, Theme, ComponentStyle, StyleVariable, StyleVariableType,
    VariableScope, CssGenerator, CssOutputFormat, CompressionLevel, ThemeMetadata,
    ThemeInheritanceRule, InheritanceMode, ColorPalette, ColorGroup, SemanticColors,
    ColorAccessibility, WcagLevel, TypographySystem, FontFamily, FontWeight,
    FontStyle, TypeScale, FontLoadingStrategy, FontLoadingOptions, SpacingSystem,
    SemanticSpacing, AnimationSystem, AnimationPreset
};

// Re-export distribution and export components
pub use distribution_export::{
    DistributionManager, DistributionChannel, DistributionChannelType,
    DistributionConfiguration, DistributionAuthentication, AuthenticationType,
    RetryPolicy, RateLimiting, Recipient, RecipientContactInfo, RecipientPreferences,
    FrequencyPreference, DistributionScheduling, ScheduledDistribution,
    RecurringDistribution, RecurrencePattern, DistributionContent, ContentType,
    ContentSource, ExportFormat, ContentMetadata, DistributionStatus,
    DistributionTracking, DeliveryLog, DeliveryStatus, DeliveryMetrics,
    DistributionAnalytics, DistributionPerformanceMetrics, DistributionThroughput,
    DistributionLatency, DistributionErrorMetrics
};

// Re-export web interface components
pub use web_interface::{
    WebInterface, WebServerConfig, CorsConfig, ApiEndpoint, HttpMethod, HttpStatus,
    EndpointRateLimit, RateLimitScope, RequestValidation, ValidationRule as WebValidationRule,
    ValidationRuleType as WebValidationRuleType, ResponseFormat, EndpointDocumentation,
    RequestExample, ResponseExample, ParameterDoc, ParameterLocation,
    WebAuthentication, WebAuthenticationType, SessionManagement, SessionStorage,
    SessionSecuritySettings, SameSitePolicy, Authorization, WebPermission,
    UserInterface, FrontendFramework, AccessibilityCompliance,
    WcagLevel as WebWcagLevel, Internationalization, UiThemeConfig,
    MiddlewareConfig, MiddlewareType, MiddlewareDefinition
};

// Re-export real-time update components
pub use realtime_updates::{
    RealTimeUpdates, WebSocketConfig, AuthenticationMethod, OAuth2Config,
    ReconnectionPolicy, ProtocolConfig, ConnectionManagement, ConnectionPooling,
    LoadBalancing, LoadBalancingStrategy, UpdateStrategy, UpdateStrategyType,
    UpdateTrigger, UpdateTriggerType, UpdateScope, UpdateThrottling,
    ThrottlingStrategy, UpdatePriority, PushNotifications, NotificationType,
    NotificationDeliveryChannel, DeliveryChannelType, RateLimiting as NotificationRateLimiting,
    RateLimitScope as NotificationRateLimitScope, DeliveryPreferences, DeliveryTiming,
    NotificationRetryPolicy, NotificationTemplate, TemplateVariable,
    TemplateFormatting, ContentFormat, RealTimeEvent
};

// Re-export error handling components
pub use error_types::{
    DashboardError, ErrorContext, ErrorSeverity, ErrorInfo, ErrorRecoveryAction,
    ErrorHandler, DefaultErrorHandler
};

/// Main dashboard interface system combining all modules
///
/// This is the primary entry point for the dashboard interface system that provides
/// access to all functionality through a unified interface.
#[derive(Debug, Clone)]
pub struct DashboardInterfaceSystem {
    /// Dashboard manager for core functionality
    pub dashboard_manager: std::sync::Arc<std::sync::RwLock<DashboardManager>>,
    /// Template manager for template operations
    pub template_manager: std::sync::Arc<std::sync::RwLock<TemplateManager>>,
    /// Style manager for theming and styling
    pub style_manager: std::sync::Arc<std::sync::RwLock<StyleManager>>,
    /// Distribution manager for content delivery
    pub distribution_manager: std::sync::Arc<std::sync::RwLock<DistributionManager>>,
    /// Web interface for API and frontend
    pub web_interface: std::sync::Arc<std::sync::RwLock<WebInterface>>,
    /// Real-time updates system
    pub realtime_updates: std::sync::Arc<std::sync::RwLock<RealTimeUpdates>>,
    /// Error handler for system-wide error management
    pub error_handler: std::sync::Arc<std::sync::RwLock<DefaultErrorHandler>>,
}

impl Default for DashboardInterfaceSystem {
    fn default() -> Self {
        Self {
            dashboard_manager: std::sync::Arc::new(std::sync::RwLock::new(DashboardManager::default())),
            template_manager: std::sync::Arc::new(std::sync::RwLock::new(TemplateManager::default())),
            style_manager: std::sync::Arc::new(std::sync::RwLock::new(StyleManager::default())),
            distribution_manager: std::sync::Arc::new(std::sync::RwLock::new(DistributionManager::default())),
            web_interface: std::sync::Arc::new(std::sync::RwLock::new(WebInterface::default())),
            realtime_updates: std::sync::Arc::new(std::sync::RwLock::new(RealTimeUpdates::default())),
            error_handler: std::sync::Arc::new(std::sync::RwLock::new(DefaultErrorHandler::default())),
        }
    }
}

impl DashboardInterfaceSystem {
    /// Create a new dashboard interface system
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new dashboard
    pub fn create_dashboard(&self, dashboard_id: String, dashboard_name: String) -> Result<(), DashboardError> {
        let dashboard = Dashboard::new(dashboard_id, dashboard_name);
        let mut manager = self.dashboard_manager.write().unwrap();
        manager.add_dashboard(dashboard);
        Ok(())
    }

    /// Get a dashboard by ID
    pub fn get_dashboard(&self, dashboard_id: &str) -> Option<Dashboard> {
        let manager = self.dashboard_manager.read().unwrap();
        manager.get_dashboard(dashboard_id).cloned()
    }

    /// Add a template
    pub fn add_template(&self, template: DashboardTemplate) -> Result<(), DashboardError> {
        let mut manager = self.template_manager.write().unwrap();
        manager.add_template(template);
        Ok(())
    }

    /// Set active theme
    pub fn set_active_theme(&self, theme_id: String) -> Result<(), DashboardError> {
        let mut style_manager = self.style_manager.write().unwrap();
        style_manager.set_active_theme(theme_id)
            .map_err(|e| DashboardError::ThemeNotFound(e))
    }

    /// Schedule distribution
    pub fn schedule_distribution(&self, distribution: ScheduledDistribution) -> Result<(), DashboardError> {
        let mut manager = self.distribution_manager.write().unwrap();
        manager.schedule_distribution(distribution);
        Ok(())
    }

    /// Enable real-time updates
    pub fn enable_realtime_updates(&self) -> Result<(), DashboardError> {
        let mut updates = self.realtime_updates.write().unwrap();
        updates.enable_push_notifications();
        Ok(())
    }

    /// Add API endpoint
    pub fn add_api_endpoint(&self, endpoint: ApiEndpoint) -> Result<(), DashboardError> {
        let mut web_interface = self.web_interface.write().unwrap();
        web_interface.add_endpoint(endpoint);
        Ok(())
    }

    /// Handle error with context
    pub fn handle_error(&self, error: DashboardError, context: ErrorContext) -> Result<ErrorRecoveryAction, DashboardError> {
        let error_info = ErrorInfo {
            severity: ErrorSeverity::Medium,
            context,
            recovery_suggestions: vec![],
            related_errors: vec![],
            error,
        };

        let handler = self.error_handler.read().unwrap();
        handler.handle_error(&error_info)
    }
}

/// Utility functions for dashboard operations
pub mod utils {
    use super::*;

    /// Create a default dashboard with basic configuration
    pub fn create_default_dashboard(id: String, name: String) -> Dashboard {
        Dashboard::new(id, name)
    }

    /// Create a default template manager with common templates
    pub fn create_default_template_manager() -> TemplateManager {
        TemplateManager::new()
    }

    /// Create a default style manager with basic themes
    pub fn create_default_style_manager() -> StyleManager {
        StyleManager::new()
    }

    /// Validate dashboard configuration
    pub fn validate_dashboard_config(dashboard: &Dashboard) -> Result<(), DashboardError> {
        if dashboard.dashboard_id.is_empty() {
            return Err(DashboardError::ValidationError("Dashboard ID cannot be empty".to_string()));
        }

        if dashboard.dashboard_name.is_empty() {
            return Err(DashboardError::ValidationError("Dashboard name cannot be empty".to_string()));
        }

        Ok(())
    }

    /// Check if user has permission for dashboard operation
    pub fn check_dashboard_permission(
        permissions: &DashboardPermissions,
        user_id: &str,
        required_permission: &Permission,
    ) -> bool {
        match required_permission {
            Permission::View => permissions.can_view(user_id),
            Permission::Edit => permissions.can_edit(user_id),
            Permission::Delete => permissions.can_delete(user_id),
            _ => false,
        }
    }
}