use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

// Re-export types that widget system depends on
use super::theme_styling::WidgetAnimations;

/// Widget management system for dashboard widgets
/// Handles widget creation, lifecycle, and interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetManager {
    /// Widget registry for all available widget types
    pub widget_registry: HashMap<String, WidgetDefinition>,
    /// Widget factory for creating widget instances
    pub widget_factory: WidgetFactory,
    /// Widget lifecycle manager for state transitions
    pub lifecycle_manager: WidgetLifecycleManager,
}

/// Core dashboard widget structure
/// Represents a single widget instance in a dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardWidget {
    /// Unique widget identifier
    pub widget_id: String,
    /// Widget type and functionality
    pub widget_type: WidgetType,
    /// Widget position in the dashboard grid
    pub position: WidgetPosition,
    /// Widget size configuration
    pub size: WidgetSize,
    /// Data source for the widget
    pub data_source: String,
    /// Widget configuration and settings
    pub configuration: WidgetConfiguration,
    /// Widget state information
    pub state: WidgetState,
    /// Widget performance metrics
    pub performance: WidgetPerformance,
    /// Widget template reference
    pub template: Option<WidgetTemplate>,
}

/// Enumeration of available widget types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    /// Chart widget for data visualization
    Chart,
    /// Table widget for tabular data display
    Table,
    /// Metric widget for single values and KPIs
    Metric,
    /// Text widget for content display
    Text,
    /// Image widget for image display
    Image,
    /// Map widget for geographical data
    Map,
    /// Filter widget for data filtering
    Filter,
    /// Control widget for user interactions
    Control,
    /// Iframe widget for embedded content
    Iframe,
    /// Video widget for video content
    Video,
    /// List widget for item listings
    List,
    /// Progress widget for progress indicators
    Progress,
    /// Calendar widget for date/time display
    Calendar,
    /// Search widget for search functionality
    Search,
    /// Custom widget implementation
    Custom(String),
}

/// Widget position configuration in dashboard grid
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPosition {
    /// X coordinate in grid units
    pub x: usize,
    /// Y coordinate in grid units
    pub y: usize,
    /// Z-index for layering (higher values appear on top)
    pub z_index: Option<i32>,
    /// Position constraints
    pub constraints: PositionConstraints,
}

/// Position constraints for widget placement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionConstraints {
    /// Minimum x position
    pub min_x: Option<usize>,
    /// Maximum x position
    pub max_x: Option<usize>,
    /// Minimum y position
    pub min_y: Option<usize>,
    /// Maximum y position
    pub max_y: Option<usize>,
    /// Snap to grid enabled
    pub snap_to_grid: bool,
    /// Allow overlap with other widgets
    pub allow_overlap: bool,
}

/// Widget size configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSize {
    /// Width in grid units
    pub width: usize,
    /// Height in grid units
    pub height: usize,
    /// Minimum width constraint
    pub min_width: Option<usize>,
    /// Maximum width constraint
    pub max_width: Option<usize>,
    /// Minimum height constraint
    pub min_height: Option<usize>,
    /// Maximum height constraint
    pub max_height: Option<usize>,
    /// Auto-resize configuration
    pub auto_resize: AutoResizeConfig,
}

/// Auto-resize configuration for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoResizeConfig {
    /// Enable auto-resize
    pub enabled: bool,
    /// Resize trigger events
    pub triggers: Vec<ResizeTrigger>,
    /// Resize constraints
    pub constraints: ResizeConstraints,
}

/// Triggers for auto-resize functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResizeTrigger {
    /// Resize on content change
    ContentChange,
    /// Resize on data update
    DataUpdate,
    /// Resize on viewport change
    ViewportChange,
    /// Resize on user interaction
    UserInteraction,
    /// Custom resize trigger
    Custom(String),
}

/// Constraints for resize operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeConstraints {
    /// Aspect ratio preservation
    pub preserve_aspect_ratio: bool,
    /// Minimum aspect ratio
    pub min_aspect_ratio: Option<f64>,
    /// Maximum aspect ratio
    pub max_aspect_ratio: Option<f64>,
    /// Resize animation duration
    pub animation_duration: Duration,
}

/// Comprehensive widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfiguration {
    /// Widget title
    pub title: String,
    /// Widget description
    pub description: String,
    /// Widget properties and settings
    pub properties: HashMap<String, String>,
    /// Widget styling configuration
    pub styling: WidgetStyling,
    /// Widget interactions configuration
    pub interactions: WidgetInteractions,
    /// Widget data configuration
    pub data_config: WidgetDataConfig,
    /// Widget refresh configuration
    pub refresh_config: WidgetRefreshConfig,
    /// Widget security configuration
    pub security: WidgetSecurity,
}

/// Widget styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetStyling {
    /// Widget theme identifier
    pub theme: String,
    /// Custom CSS styles
    pub custom_styles: HashMap<String, String>,
    /// Responsive styles for different breakpoints
    pub responsive_styles: HashMap<String, HashMap<String, String>>,
    /// State-based styles (hover, active, focus, disabled)
    pub state_styles: HashMap<String, HashMap<String, String>>,
    /// Animation settings
    pub animations: WidgetAnimations,
    /// Border configuration
    pub border: Option<BorderConfig>,
    /// Shadow configuration
    pub shadow: Option<ShadowConfig>,
}

/// Border configuration for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderConfig {
    /// Border width in pixels
    pub width: f64,
    /// Border color
    pub color: String,
    /// Border style
    pub style: BorderStyle,
    /// Border radius for rounded corners
    pub radius: Option<f64>,
}

/// Border style enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BorderStyle {
    /// Solid border
    Solid,
    /// Dashed border
    Dashed,
    /// Dotted border
    Dotted,
    /// Double border
    Double,
    /// No border
    None,
    /// Custom border style
    Custom(String),
}

/// Shadow configuration for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfig {
    /// Horizontal offset
    pub offset_x: f64,
    /// Vertical offset
    pub offset_y: f64,
    /// Blur radius
    pub blur_radius: f64,
    /// Spread radius
    pub spread_radius: f64,
    /// Shadow color
    pub color: String,
    /// Shadow opacity
    pub opacity: f64,
}

/// Widget security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSecurity {
    /// Content Security Policy
    pub csp: Option<String>,
    /// Allowed origins for iframe content
    pub allowed_origins: Vec<String>,
    /// Sandbox configuration
    pub sandbox: SandboxConfig,
    /// Script execution permissions
    pub script_permissions: ScriptPermissions,
}

/// Sandbox configuration for widget isolation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Enable sandboxing
    pub enabled: bool,
    /// Allowed sandbox features
    pub allowed_features: Vec<SandboxFeature>,
    /// Sandbox restrictions
    pub restrictions: Vec<String>,
}

/// Sandbox feature enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxFeature {
    /// Allow forms
    Forms,
    /// Allow modals
    Modals,
    /// Allow orientation lock
    OrientationLock,
    /// Allow pointer lock
    PointerLock,
    /// Allow popups
    Popups,
    /// Allow same origin
    SameOrigin,
    /// Allow scripts
    Scripts,
    /// Allow top navigation
    TopNavigation,
    /// Custom feature
    Custom(String),
}

/// Script execution permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptPermissions {
    /// Allow inline scripts
    pub allow_inline: bool,
    /// Allow external scripts
    pub allow_external: bool,
    /// Allowed script sources
    pub allowed_sources: Vec<String>,
    /// Script nonce for CSP
    pub nonce: Option<String>,
}

/// Widget interactions configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetInteractions {
    /// Enable click-through behavior
    pub click_through: bool,
    /// Drill-down configuration
    pub drill_down: Option<DrillDownConfig>,
    /// Filter interactions
    pub filters: Vec<WidgetFilter>,
    /// Cross-filtering configuration
    pub cross_filtering: CrossFilteringConfig,
    /// Selection interactions
    pub selection: SelectionConfig,
    /// Hover interactions
    pub hover: HoverConfig,
    /// Context menu configuration
    pub context_menu: Option<ContextMenuConfig>,
}

/// Drill-down interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrillDownConfig {
    /// Target dashboard for drill-down
    pub target_dashboard: String,
    /// Parameter mapping between widgets
    pub parameter_mapping: HashMap<String, String>,
    /// Drill-down behavior
    pub behavior: DrillDownBehavior,
    /// Drill-down conditions
    pub conditions: Vec<DrillDownCondition>,
}

/// Drill-down behavior options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DrillDownBehavior {
    /// Navigate to target dashboard
    Navigate,
    /// Open target dashboard in new tab
    NewTab,
    /// Open target dashboard in modal
    Modal,
    /// Replace current widget content
    Replace,
    /// Expand widget inline
    ExpandInline,
    /// Custom drill-down behavior
    Custom(String),
}

/// Drill-down condition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrillDownCondition {
    /// Field to check
    pub field: String,
    /// Condition operator
    pub operator: String,
    /// Condition value
    pub value: String,
    /// Condition logic (AND/OR)
    pub logic: String,
}

/// Widget filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetFilter {
    /// Filter identifier
    pub filter_id: String,
    /// Filter type
    pub filter_type: WidgetFilterType,
    /// Target widgets for this filter
    pub target_widgets: Vec<String>,
    /// Filter logic
    pub filter_logic: FilterLogic,
    /// Filter configuration
    pub filter_config: FilterConfig,
    /// Filter UI configuration
    pub ui_config: FilterUIConfig,
}

/// Widget filter type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetFilterType {
    /// Include filter (show matching data)
    Include,
    /// Exclude filter (hide matching data)
    Exclude,
    /// Highlight filter (emphasize matching data)
    Highlight,
    /// Transform filter (modify matching data)
    Transform,
    /// Custom filter implementation
    Custom(String),
}

/// Filter logic enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterLogic {
    /// AND logic (all conditions must match)
    And,
    /// OR logic (any condition can match)
    Or,
    /// NOT logic (conditions must not match)
    Not,
    /// XOR logic (exactly one condition must match)
    Xor,
    /// Custom logic implementation
    Custom(String),
}

/// Filter configuration details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Fields to filter on
    pub fields: Vec<String>,
    /// Filter operators
    pub operators: Vec<FilterOperator>,
    /// Filter values
    pub values: Vec<FilterValue>,
    /// Filter persistence across sessions
    pub persistent: bool,
    /// Case sensitivity
    pub case_sensitive: bool,
    /// Partial matching enabled
    pub partial_match: bool,
}

/// Filter operator enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    /// Equal operator
    Equal,
    /// Not equal operator
    NotEqual,
    /// Greater than operator
    GreaterThan,
    /// Less than operator
    LessThan,
    /// Greater than or equal operator
    GreaterThanOrEqual,
    /// Less than or equal operator
    LessThanOrEqual,
    /// Contains operator
    Contains,
    /// Starts with operator
    StartsWith,
    /// Ends with operator
    EndsWith,
    /// In list operator
    InList,
    /// Not in list operator
    NotInList,
    /// Between operator
    Between,
    /// Regular expression operator
    Regex,
    /// Custom operator
    Custom(String),
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Date value
    Date(DateTime<Utc>),
    /// List of string values
    StringList(Vec<String>),
    /// Range of numeric values
    NumericRange(f64, f64),
    /// Date range
    DateRange(DateTime<Utc>, DateTime<Utc>),
    /// Custom value type
    Custom(String),
}

/// Filter UI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterUIConfig {
    /// Display filter controls
    pub show_controls: bool,
    /// Filter control position
    pub control_position: FilterControlPosition,
    /// Filter control styling
    pub control_styling: HashMap<String, String>,
    /// Filter feedback configuration
    pub feedback: FilterFeedbackConfig,
}

/// Filter control position options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterControlPosition {
    /// Top of widget
    Top,
    /// Bottom of widget
    Bottom,
    /// Left side of widget
    Left,
    /// Right side of widget
    Right,
    /// Overlay on widget
    Overlay,
    /// External to widget
    External,
}

/// Filter feedback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterFeedbackConfig {
    /// Show filter count
    pub show_count: bool,
    /// Show filter status
    pub show_status: bool,
    /// Highlight filtered data
    pub highlight_filtered: bool,
    /// Animation for filter changes
    pub animate_changes: bool,
}

/// Cross-filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossFilteringConfig {
    /// Enable cross-filtering
    pub enabled: bool,
    /// Participating widgets
    pub participating_widgets: Vec<String>,
    /// Filter propagation rules
    pub propagation_rules: Vec<PropagationRule>,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolution,
    /// Cross-filter coordination
    pub coordination: CrossFilterCoordination,
}

/// Filter propagation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationRule {
    /// Source widget identifier
    pub source_widget: String,
    /// Target widget identifiers
    pub target_widgets: Vec<String>,
    /// Propagation condition
    pub condition: PropagationCondition,
    /// Propagation action
    pub action: PropagationAction,
    /// Rule priority
    pub priority: u32,
}

/// Propagation condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropagationCondition {
    /// Always propagate
    Always,
    /// Conditional propagation
    Conditional(String),
    /// Manual propagation (user-triggered)
    Manual,
    /// Time-based propagation
    TimeBased(Duration),
    /// Custom condition
    Custom(String),
}

/// Propagation action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropagationAction {
    /// Apply filter to targets
    Filter,
    /// Highlight data in targets
    Highlight,
    /// Update target widget data
    Update,
    /// Refresh target widgets
    Refresh,
    /// Custom action
    Custom(String),
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// First filter wins
    FirstWins,
    /// Last filter wins
    LastWins,
    /// Union of all filters
    Union,
    /// Intersection of all filters
    Intersection,
    /// Priority-based resolution
    Priority,
    /// User choice resolution
    UserChoice,
    /// Custom resolution strategy
    Custom(String),
}

/// Cross-filter coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossFilterCoordination {
    /// Coordination mode
    pub mode: CoordinationMode,
    /// Debounce delay for filter updates
    pub debounce_delay: Duration,
    /// Maximum propagation depth
    pub max_propagation_depth: Option<usize>,
    /// Coordination timeout
    pub timeout: Duration,
}

/// Cross-filter coordination mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMode {
    /// Immediate propagation
    Immediate,
    /// Batched propagation
    Batched,
    /// Delayed propagation
    Delayed,
    /// Manual coordination
    Manual,
    /// Custom coordination mode
    Custom(String),
}

/// Selection interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Enable selection
    pub enabled: bool,
    /// Selection mode
    pub selection_mode: SelectionMode,
    /// Selection styling
    pub selection_styling: SelectionStyling,
    /// Selection persistence
    pub persistent: bool,
    /// Multi-selection configuration
    pub multi_selection: MultiSelectionConfig,
}

/// Selection mode options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMode {
    /// Single item selection
    Single,
    /// Multiple item selection
    Multiple,
    /// Range selection
    Range,
    /// Lasso selection
    Lasso,
    /// Custom selection mode
    Custom(String),
}

/// Selection styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionStyling {
    /// Selection color
    pub color: String,
    /// Selection opacity
    pub opacity: f64,
    /// Selection border configuration
    pub border: Option<BorderConfig>,
    /// Selection animation
    pub animation: Option<super::theme_styling::WidgetAnimations>,
    /// Selected state styling
    pub selected_state: HashMap<String, String>,
}

/// Multi-selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSelectionConfig {
    /// Maximum number of selections
    pub max_selections: Option<usize>,
    /// Selection aggregation mode
    pub aggregation_mode: SelectionAggregationMode,
    /// Selection combination logic
    pub combination_logic: SelectionCombinationLogic,
}

/// Selection aggregation mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionAggregationMode {
    /// No aggregation
    None,
    /// Count aggregation
    Count,
    /// Sum aggregation
    Sum,
    /// Average aggregation
    Average,
    /// Custom aggregation
    Custom(String),
}

/// Selection combination logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionCombinationLogic {
    /// Union of selections
    Union,
    /// Intersection of selections
    Intersection,
    /// Difference of selections
    Difference,
    /// Custom combination logic
    Custom(String),
}

/// Hover interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverConfig {
    /// Enable hover effects
    pub enabled: bool,
    /// Hover delay before activation
    pub hover_delay: Duration,
    /// Hover styling changes
    pub hover_styling: HashMap<String, String>,
    /// Tooltip configuration
    pub tooltip: Option<TooltipConfig>,
    /// Hover actions
    pub actions: Vec<HoverAction>,
}

/// Tooltip configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipConfig {
    /// Enable tooltip
    pub enabled: bool,
    /// Tooltip content template
    pub content_template: String,
    /// Tooltip position
    pub position: TooltipPosition,
    /// Tooltip styling
    pub styling: HashMap<String, String>,
    /// Tooltip delay
    pub delay: Duration,
}

/// Tooltip position options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipPosition {
    /// Above the element
    Top,
    /// Below the element
    Bottom,
    /// Left of the element
    Left,
    /// Right of the element
    Right,
    /// Follow mouse cursor
    Follow,
    /// Custom position
    Custom(String),
}

/// Hover action configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverAction {
    pub action_type: HoverActionType,
    pub target: String,
    pub parameters: HashMap<String, String>,
}

/// Hover action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HoverActionType {
    /// Highlight related data
    Highlight,
    /// Show additional information
    ShowInfo,
    /// Update other widgets
    UpdateWidgets,
    /// Trigger animation
    Animate,
    /// Custom action
    Custom(String),
}

/// Context menu configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMenuConfig {
    /// Enable context menu
    pub enabled: bool,
    /// Menu items
    pub items: Vec<ContextMenuItem>,
    /// Menu styling
    pub styling: HashMap<String, String>,
    /// Menu position
    pub position: ContextMenuPosition,
}

/// Context menu item configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMenuItem {
    /// Menu item ID
    pub id: String,
    /// Menu item label
    pub label: String,
    /// Menu item icon
    pub icon: Option<String>,
    /// Menu item action
    pub action: String,
    /// Menu item enabled state
    pub enabled: bool,
    /// Submenu items
    pub submenu: Option<Vec<ContextMenuItem>>,
}

/// Context menu position options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextMenuPosition {
    /// At cursor position
    Cursor,
    /// At widget center
    Center,
    /// Fixed position
    Fixed(f64, f64),
    /// Custom position logic
    Custom(String),
}

/// Widget data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetDataConfig {
    /// Data source connection
    pub data_source: String,
    /// Data query or filter
    pub data_query: String,
    /// Data transformation pipeline
    pub data_transform: Option<String>,
    /// Data aggregation configuration
    pub data_aggregation: Option<DataAggregationConfig>,
    /// Data validation configuration
    pub data_validation: DataValidationConfig,
    /// Data caching configuration
    pub caching: DataCachingConfig,
}

/// Data aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAggregationConfig {
    /// Primary aggregation type
    pub aggregation_type: AggregationType,
    /// Fields to group by
    pub group_by: Vec<String>,
    /// Fields to aggregate
    pub aggregate_fields: Vec<AggregateField>,
    /// Having clause for aggregation
    pub having_clause: Option<String>,
    /// Sort order for aggregated data
    pub sort_order: Vec<SortField>,
}

/// Aggregation type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Sum,
    Average,
    Count,
    CountDistinct,
    Min,
    Max,
    Median,
    StdDev,
    Variance,
    Percentile(f64),
    Custom(String),
}

/// Aggregate field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateField {
    /// Source field name
    pub field: String,
    /// Aggregation function to apply
    pub function: AggregationType,
    /// Output field name
    pub output_name: String,
    /// Field-specific options
    pub options: HashMap<String, String>,
}

/// Sort field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortField {
    /// Field name to sort by
    pub field: String,
    /// Sort direction
    pub direction: SortDirection,
    /// Sort priority (lower numbers sort first)
    pub priority: u32,
}

/// Sort direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortDirection {
    /// Ascending order
    Ascending,
    /// Descending order
    Descending,
}

/// Data validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    /// Enable data validation
    pub enabled: bool,
    /// Validation rules
    pub rules: Vec<ValidationRule>,
    /// Error handling strategy
    pub error_handling: ValidationErrorHandling,
    /// Validation timing
    pub timing: ValidationTiming,
}

/// Data validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Error message for rule violation
    pub error_message: String,
    /// Rule severity level
    pub severity: ValidationSeverity,
    /// Fields to validate
    pub fields: Vec<String>,
}

/// Validation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Warning level (non-blocking)
    Warning,
    /// Error level (blocking)
    Error,
    /// Critical level (stops execution)
    Critical,
    /// Info level (informational only)
    Info,
}

/// Validation error handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationErrorHandling {
    /// Ignore validation errors
    Ignore,
    /// Log errors to console
    Log,
    /// Show error message to user
    ShowError,
    /// Hide widget on error
    HideWidget,
    /// Show fallback content
    ShowFallback,
    /// Custom error handling
    Custom(String),
}

/// Validation timing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationTiming {
    /// Validate on data load
    OnLoad,
    /// Validate on data change
    OnChange,
    /// Validate on user interaction
    OnInteraction,
    /// Validate periodically
    Periodic(Duration),
    /// Custom validation timing
    Custom(String),
}

/// Data caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCachingConfig {
    /// Enable data caching
    pub enabled: bool,
    /// Cache duration
    pub cache_duration: Duration,
    /// Cache invalidation strategy
    pub invalidation_strategy: CacheInvalidationStrategy,
    /// Cache size limit
    pub size_limit: Option<usize>,
}

/// Cache invalidation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheInvalidationStrategy {
    /// Time-based invalidation
    TimeBased,
    /// Manual invalidation
    Manual,
    /// Data-change-based invalidation
    DataChange,
    /// Custom invalidation logic
    Custom(String),
}

/// Widget refresh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetRefreshConfig {
    /// Auto refresh enabled
    pub auto_refresh: bool,
    /// Refresh interval
    pub refresh_interval: Duration,
    /// Manual refresh enabled
    pub manual_refresh: bool,
    /// Incremental refresh enabled
    pub incremental_refresh: bool,
    /// Refresh triggers
    pub refresh_triggers: Vec<RefreshTrigger>,
    /// Refresh strategy
    pub refresh_strategy: RefreshStrategy,
}

/// Refresh trigger types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefreshTrigger {
    /// Time-based trigger
    Time(Duration),
    /// Data change trigger
    DataChange,
    /// User interaction trigger
    UserInteraction,
    /// External event trigger
    ExternalEvent(String),
    /// Dependency change trigger
    DependencyChange(String),
    /// Custom trigger
    Custom(String),
}

/// Refresh strategy options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefreshStrategy {
    /// Full refresh (reload all data)
    Full,
    /// Incremental refresh (only changed data)
    Incremental,
    /// Smart refresh (adaptive based on data size)
    Smart,
    /// Custom refresh strategy
    Custom(String),
}

/// Widget state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetState {
    /// Current widget state
    pub current_state: WidgetStateType,
    /// State history
    pub state_history: Vec<StateHistoryEntry>,
    /// Allowed state transitions
    pub allowed_transitions: Vec<StateTransition>,
    /// State persistence enabled
    pub persistent_state: bool,
    /// State metadata
    pub metadata: HashMap<String, String>,
}

/// Widget state types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetStateType {
    /// Widget is loading
    Loading,
    /// Widget is ready and displayed
    Ready,
    /// Widget has an error
    Error,
    /// Widget is refreshing data
    Refreshing,
    /// Widget is minimized
    Minimized,
    /// Widget is maximized
    Maximized,
    /// Widget is hidden
    Hidden,
    /// Widget is in edit mode
    EditMode,
    /// Widget is disabled
    Disabled,
    /// Custom state
    Custom(String),
}

/// State history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateHistoryEntry {
    /// Previous state
    pub state: WidgetStateType,
    /// Timestamp of state change
    pub timestamp: DateTime<Utc>,
    /// Duration in this state
    pub duration: Duration,
    /// Reason for state change
    pub reason: String,
    /// User who triggered the state change
    pub triggered_by: Option<String>,
}

/// State transition definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Source state
    pub from_state: WidgetStateType,
    /// Target state
    pub to_state: WidgetStateType,
    /// Transition condition
    pub condition: TransitionCondition,
    /// Transition action
    pub action: Option<String>,
    /// Transition validation
    pub validation: Option<String>,
}

/// State transition conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionCondition {
    /// Automatic transition
    Automatic,
    /// Manual trigger required
    Manual,
    /// Time-based transition
    TimeBased(Duration),
    /// Event-based transition
    EventBased(String),
    /// Data-based transition
    DataBased(String),
    /// Custom condition
    Custom(String),
}

/// Widget performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPerformance {
    /// Widget load time
    pub load_time: Duration,
    /// Widget render time
    pub render_time: Duration,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Data fetch time
    pub data_fetch_time: Duration,
    /// Update frequency (updates per second)
    pub update_frequency: f64,
    /// Error rate (percentage)
    pub error_rate: f64,
    /// Cache hit rate (percentage)
    pub cache_hit_rate: f64,
    /// User interaction metrics
    pub interaction_metrics: InteractionMetrics,
}

/// User interaction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionMetrics {
    /// Total number of interactions
    pub total_interactions: u64,
    /// Average interaction duration
    pub avg_interaction_duration: Duration,
    /// Most frequent interaction type
    pub most_frequent_interaction: Option<String>,
    /// Last interaction timestamp
    pub last_interaction: Option<DateTime<Utc>>,
}

/// Widget template for rapid widget creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetTemplate {
    /// Template identifier
    pub template_id: String,
    /// Widget type this template creates
    pub widget_type: WidgetType,
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Default widget configuration
    pub default_configuration: WidgetConfiguration,
    /// Parameter bindings for customization
    pub parameter_bindings: HashMap<String, String>,
    /// Template constraints
    pub constraints: TemplateConstraints,
    /// Template metadata
    pub metadata: TemplateMetadata,
}

/// Template constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConstraints {
    /// Required data sources
    pub required_data_sources: Vec<String>,
    /// Minimum widget size
    pub min_size: Option<WidgetSize>,
    /// Maximum widget size
    pub max_size: Option<WidgetSize>,
    /// Required permissions
    pub required_permissions: Vec<String>,
    /// Supported devices
    pub supported_devices: Vec<DeviceType>,
}

/// Device type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    /// Desktop device
    Desktop,
    /// Tablet device
    Tablet,
    /// Mobile device
    Mobile,
    /// All devices
    All,
    /// Custom device type
    Custom(String),
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template description
    pub description: String,
    /// Template author
    pub author: String,
    /// Template version
    pub version: String,
    /// Template tags
    pub tags: Vec<String>,
    /// Template preview image URL
    pub preview_image: Option<String>,
    /// Template compatibility information
    pub compatibility: TemplateCompatibility,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Template compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateCompatibility {
    /// Minimum platform version
    pub min_platform_version: String,
    /// Required features
    pub required_features: Vec<String>,
    /// Supported devices
    pub supported_devices: Vec<DeviceType>,
    /// Browser compatibility
    pub browser_compatibility: HashMap<String, String>,
}

/// Widget definition for the widget registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetDefinition {
    /// Widget type identifier
    pub widget_type: WidgetType,
    /// Widget capabilities
    pub capabilities: WidgetCapabilities,
    /// Default configuration
    pub default_config: WidgetConfiguration,
    /// Widget metadata
    pub metadata: WidgetDefinitionMetadata,
    /// Widget schema definition
    pub schema: Option<WidgetSchema>,
}

/// Widget capabilities definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetCapabilities {
    /// Supports real-time updates
    pub real_time_updates: bool,
    /// Supports interactivity
    pub interactive: bool,
    /// Supports filtering
    pub filtering: bool,
    /// Supports data export
    pub export: bool,
    /// Supports drill-down
    pub drill_down: bool,
    /// Supports cross-filtering
    pub cross_filtering: bool,
    /// Supports custom styling
    pub custom_styling: bool,
    /// Supports animations
    pub animations: bool,
    /// Custom capabilities
    pub custom_capabilities: Vec<String>,
}

/// Widget definition metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetDefinitionMetadata {
    /// Widget name
    pub name: String,
    /// Widget description
    pub description: String,
    /// Widget icon identifier
    pub icon: Option<String>,
    /// Widget category
    pub category: String,
    /// Widget version
    pub version: String,
    /// Widget author
    pub author: String,
    /// Widget documentation URL
    pub documentation_url: Option<String>,
}

/// Widget schema definition for configuration validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSchema {
    /// Schema version
    pub version: String,
    /// Configuration properties schema
    pub properties: HashMap<String, PropertySchema>,
    /// Required properties
    pub required: Vec<String>,
    /// Additional properties allowed
    pub additional_properties: bool,
}

/// Property schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    /// Property type
    pub property_type: PropertyType,
    /// Property description
    pub description: String,
    /// Default value
    pub default: Option<String>,
    /// Validation constraints
    pub constraints: Vec<PropertyConstraint>,
}

/// Property type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyType {
    /// String property
    String,
    /// Number property
    Number,
    /// Boolean property
    Boolean,
    /// Array property
    Array,
    /// Object property
    Object,
    /// Enum property
    Enum(Vec<String>),
}

/// Property constraint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyConstraint {
    /// Constraint type
    pub constraint_type: String,
    /// Constraint value
    pub value: String,
    /// Error message
    pub message: String,
}

/// Widget factory for creating widget instances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetFactory {
    /// Factory configuration
    pub config: FactoryConfig,
    /// Widget builders
    pub builders: HashMap<String, WidgetBuilder>,
    /// Factory performance metrics
    pub metrics: FactoryMetrics,
}

/// Factory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryConfig {
    /// Default widget settings
    pub default_settings: HashMap<String, String>,
    /// Factory caching enabled
    pub caching_enabled: bool,
    /// Factory validation enabled
    pub validation_enabled: bool,
    /// Concurrent build limit
    pub max_concurrent_builds: usize,
    /// Build timeout
    pub build_timeout: Duration,
}

/// Widget builder definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetBuilder {
    /// Builder identifier
    pub builder_id: String,
    /// Supported widget types
    pub supported_types: Vec<WidgetType>,
    /// Builder configuration
    pub configuration: BuilderConfiguration,
    /// Builder priority
    pub priority: u32,
}

/// Builder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderConfiguration {
    /// Build parameters
    pub parameters: HashMap<String, String>,
    /// Build validation rules
    pub validation_rules: Vec<String>,
    /// Build optimization enabled
    pub optimization: bool,
    /// Build parallelization enabled
    pub parallelization: bool,
}

/// Factory performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryMetrics {
    /// Total widgets created
    pub total_created: u64,
    /// Average creation time
    pub avg_creation_time: Duration,
    /// Creation success rate
    pub success_rate: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Widget lifecycle manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetLifecycleManager {
    /// Lifecycle hooks
    pub hooks: HashMap<String, LifecycleHook>,
    /// State transitions
    pub state_transitions: Vec<LifecycleTransition>,
    /// Lifecycle policies
    pub policies: Vec<LifecyclePolicy>,
}

/// Lifecycle hook definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleHook {
    /// Hook name
    pub name: String,
    /// Hook execution point
    pub execution_point: ExecutionPoint,
    /// Hook handler
    pub handler: String,
    /// Hook priority
    pub priority: u32,
    /// Hook enabled
    pub enabled: bool,
}

/// Lifecycle execution points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionPoint {
    /// Before widget creation
    BeforeCreate,
    /// After widget creation
    AfterCreate,
    /// Before widget update
    BeforeUpdate,
    /// After widget update
    AfterUpdate,
    /// Before widget deletion
    BeforeDelete,
    /// After widget deletion
    AfterDelete,
    /// Before state change
    BeforeStateChange,
    /// After state change
    AfterStateChange,
    /// Custom execution point
    Custom(String),
}

/// Lifecycle transition definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleTransition {
    /// Source state
    pub from_state: String,
    /// Target state
    pub to_state: String,
    /// Transition handler
    pub handler: Option<String>,
    /// Transition validation
    pub validation: Option<String>,
    /// Transition priority
    pub priority: u32,
}

/// Lifecycle policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecyclePolicy {
    /// Policy name
    pub name: String,
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    /// Policy enforcement level
    pub enforcement_level: EnforcementLevel,
}

/// Policy rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: String,
    /// Rule priority
    pub priority: u32,
}

/// Policy enforcement levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory (warnings only)
    Advisory,
    /// Enforced (blocks violation)
    Enforced,
    /// Strict (terminates on violation)
    Strict,
}

/// Widget manager implementation
impl WidgetManager {
    /// Create a new widget manager
    pub fn new() -> Self {
        Self {
            widget_registry: HashMap::new(),
            widget_factory: WidgetFactory::default(),
            lifecycle_manager: WidgetLifecycleManager::default(),
        }
    }

    /// Register a new widget definition
    pub fn register_widget(&mut self, definition: WidgetDefinition) -> Result<(), WidgetError> {
        let widget_type_key = format!("{:?}", definition.widget_type);
        self.widget_registry.insert(widget_type_key, definition);
        Ok(())
    }

    /// Create a new widget instance
    pub fn create_widget(&self, widget_type: WidgetType, config: WidgetConfiguration) -> Result<DashboardWidget, WidgetError> {
        let widget_type_key = format!("{:?}", widget_type);
        let definition = self.widget_registry.get(&widget_type_key)
            .ok_or_else(|| WidgetError::WidgetTypeNotFound(widget_type_key))?;

        // Create widget with default values
        Ok(DashboardWidget {
            widget_id: format!("widget_{}", uuid::Uuid::new_v4()),
            widget_type,
            position: WidgetPosition::default(),
            size: WidgetSize::default(),
            data_source: "default".to_string(),
            configuration: config,
            state: WidgetState::default(),
            performance: WidgetPerformance::default(),
            template: None,
        })
    }

    /// Get widget definition
    pub fn get_widget_definition(&self, widget_type: &WidgetType) -> Option<&WidgetDefinition> {
        let widget_type_key = format!("{:?}", widget_type);
        self.widget_registry.get(&widget_type_key)
    }

    /// List all registered widget types
    pub fn list_widget_types(&self) -> Vec<WidgetType> {
        self.widget_registry.values()
            .map(|def| def.widget_type.clone())
            .collect()
    }
}

/// Widget-related error types
#[derive(Debug, thiserror::Error)]
pub enum WidgetError {
    #[error("Widget type not found: {0}")]
    WidgetTypeNotFound(String),
    #[error("Widget not found: {0}")]
    WidgetNotFound(String),
    #[error("Widget configuration error: {0}")]
    ConfigurationError(String),
    #[error("Widget validation error: {0}")]
    ValidationError(String),
    #[error("Widget state error: {0}")]
    StateError(String),
    #[error("Widget interaction error: {0}")]
    InteractionError(String),
    #[error("Widget performance error: {0}")]
    PerformanceError(String),
}

// Default implementations

impl Default for WidgetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WidgetPosition {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            z_index: None,
            constraints: PositionConstraints::default(),
        }
    }
}

impl Default for PositionConstraints {
    fn default() -> Self {
        Self {
            min_x: None,
            max_x: None,
            min_y: None,
            max_y: None,
            snap_to_grid: true,
            allow_overlap: false,
        }
    }
}

impl Default for WidgetSize {
    fn default() -> Self {
        Self {
            width: 4,
            height: 3,
            min_width: Some(1),
            max_width: None,
            min_height: Some(1),
            max_height: None,
            auto_resize: AutoResizeConfig::default(),
        }
    }
}

impl Default for AutoResizeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            triggers: Vec::new(),
            constraints: ResizeConstraints::default(),
        }
    }
}

impl Default for ResizeConstraints {
    fn default() -> Self {
        Self {
            preserve_aspect_ratio: false,
            min_aspect_ratio: None,
            max_aspect_ratio: None,
            animation_duration: Duration::from_millis(200),
        }
    }
}

impl Default for WidgetState {
    fn default() -> Self {
        Self {
            current_state: WidgetStateType::Ready,
            state_history: Vec::new(),
            allowed_transitions: Vec::new(),
            persistent_state: false,
            metadata: HashMap::new(),
        }
    }
}

impl Default for WidgetPerformance {
    fn default() -> Self {
        Self {
            load_time: Duration::from_millis(0),
            render_time: Duration::from_millis(0),
            memory_usage: 0,
            data_fetch_time: Duration::from_millis(0),
            update_frequency: 0.0,
            error_rate: 0.0,
            cache_hit_rate: 0.0,
            interaction_metrics: InteractionMetrics::default(),
        }
    }
}

impl Default for InteractionMetrics {
    fn default() -> Self {
        Self {
            total_interactions: 0,
            avg_interaction_duration: Duration::from_millis(0),
            most_frequent_interaction: None,
            last_interaction: None,
        }
    }
}

impl Default for WidgetFactory {
    fn default() -> Self {
        Self {
            config: FactoryConfig::default(),
            builders: HashMap::new(),
            metrics: FactoryMetrics::default(),
        }
    }
}

impl Default for FactoryConfig {
    fn default() -> Self {
        Self {
            default_settings: HashMap::new(),
            caching_enabled: true,
            validation_enabled: true,
            max_concurrent_builds: 10,
            build_timeout: Duration::from_secs(30),
        }
    }
}

impl Default for FactoryMetrics {
    fn default() -> Self {
        Self {
            total_created: 0,
            avg_creation_time: Duration::from_millis(0),
            success_rate: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

impl Default for WidgetLifecycleManager {
    fn default() -> Self {
        Self {
            hooks: HashMap::new(),
            state_transitions: Vec::new(),
            policies: Vec::new(),
        }
    }
}