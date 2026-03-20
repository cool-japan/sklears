//! Comprehensive interaction controllers for visualization components
//!
//! This module provides specialized controllers for various interaction types:
//! - Zoom controller for zooming operations
//! - Pan controller for panning operations
//! - Brush controller for selection brushing
//! - Selection controller for item selection
//! - Tooltip controller for tooltip management
//! - Drag controller for drag and drop operations
//! - Resize controller for element resizing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::Duration;

/// Interaction controller for managing specific interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionController {
    /// Controller identifier
    pub controller_id: String,
    /// Controller type
    pub controller_type: InteractionControllerType,
    /// Controller state
    pub state: ControllerState,
    /// Controller configuration
    pub configuration: ControllerConfiguration,
    /// Interaction behaviors
    pub behaviors: Vec<InteractionBehavior>,
    /// Performance metrics
    pub performance_metrics: ControllerPerformanceMetrics,
}

/// Interaction controller types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionControllerType {
    /// Zoom controller
    Zoom(ZoomController),
    /// Pan controller
    Pan(PanController),
    /// Brush controller
    Brush(BrushController),
    /// Selection controller
    Selection(SelectionController),
    /// Tooltip controller
    Tooltip(TooltipController),
    /// Drag controller
    Drag(DragController),
    /// Resize controller
    Resize(ResizeController),
    /// Custom controller
    Custom(String),
}

/// Zoom controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomController {
    /// Zoom mode
    pub zoom_mode: ZoomMode,
    /// Zoom constraints
    pub constraints: ZoomConstraints,
    /// Zoom animation
    pub animation: ZoomAnimation,
    /// Zoom behavior
    pub behavior: ZoomBehavior,
}

/// Zoom modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZoomMode {
    /// Wheel zoom
    Wheel,
    /// Pinch zoom
    Pinch,
    /// Box zoom
    Box,
    /// Double-click zoom
    DoubleClick,
    /// Programmatic zoom
    Programmatic,
    /// Custom zoom mode
    Custom(String),
}

/// Zoom constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomConstraints {
    /// Minimum zoom level
    pub min_zoom: f64,
    /// Maximum zoom level
    pub max_zoom: f64,
    /// Zoom step size
    pub zoom_step: f64,
    /// Constrain to bounds
    pub constrain_to_bounds: bool,
    /// Zoom center constraints
    pub center_constraints: CenterConstraints,
}

/// Center constraints for zoom
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CenterConstraints {
    /// Allow center change
    pub allow_center_change: bool,
    /// Center bounds
    pub center_bounds: Option<BoundingBox>,
    /// Snap to center
    pub snap_to_center: bool,
}

/// Bounding box definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Minimum X coordinate
    pub min_x: f64,
    /// Minimum Y coordinate
    pub min_y: f64,
    /// Maximum X coordinate
    pub max_x: f64,
    /// Maximum Y coordinate
    pub max_y: f64,
}

/// Zoom animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomAnimation {
    /// Animation enabled
    pub enabled: bool,
    /// Animation duration
    pub duration: Duration,
    /// Animation easing
    pub easing: EasingFunction,
    /// Animation frame rate
    pub frame_rate: u32,
}

/// Easing functions for animations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    /// Linear easing
    Linear,
    /// Ease in
    EaseIn,
    /// Ease out
    EaseOut,
    /// Ease in out
    EaseInOut,
    /// Cubic bezier
    CubicBezier(f64, f64, f64, f64),
    /// Custom easing function
    Custom(String),
}

/// Zoom behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomBehavior {
    /// Zoom sensitivity
    pub sensitivity: f64,
    /// Momentum scrolling
    pub momentum: bool,
    /// Inertia configuration
    pub inertia: InertiaConfig,
    /// Zoom indicators
    pub indicators: ZoomIndicators,
}

/// Inertia configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InertiaConfig {
    /// Inertia enabled
    pub enabled: bool,
    /// Friction coefficient
    pub friction: f64,
    /// Minimum velocity
    pub min_velocity: f64,
    /// Maximum velocity
    pub max_velocity: f64,
}

/// Zoom indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomIndicators {
    /// Show zoom level
    pub show_zoom_level: bool,
    /// Show zoom controls
    pub show_zoom_controls: bool,
    /// Show zoom area
    pub show_zoom_area: bool,
    /// Indicator style
    pub indicator_style: IndicatorStyle,
}

/// Indicator style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorStyle {
    /// Indicator color
    pub color: String,
    /// Indicator opacity
    pub opacity: f64,
    /// Indicator size
    pub size: f64,
    /// Indicator position
    pub position: IndicatorPosition,
}

/// Indicator position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorPosition {
    /// Top left
    TopLeft,
    /// Top right
    TopRight,
    /// Bottom left
    BottomLeft,
    /// Bottom right
    BottomRight,
    /// Center
    Center,
    /// Custom position
    Custom(f64, f64),
}

/// Pan controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanController {
    /// Pan mode
    pub pan_mode: PanMode,
    /// Pan constraints
    pub constraints: PanConstraints,
    /// Pan animation
    pub animation: PanAnimation,
    /// Pan behavior
    pub behavior: PanBehavior,
}

/// Pan modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanMode {
    /// Mouse pan
    Mouse,
    /// Touch pan
    Touch,
    /// Keyboard pan
    Keyboard,
    /// Automatic pan
    Automatic,
    /// Custom pan mode
    Custom(String),
}

/// Pan constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanConstraints {
    pub bounds: Option<BoundingBox>,
    pub constrain_to_data: bool,
    pub pan_axes: PanAxes,
    pub edge_behavior: EdgeBehavior,
}

/// Pan axes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanAxes {
    /// Allow horizontal pan
    pub horizontal: bool,
    /// Allow vertical pan
    pub vertical: bool,
    /// Lock aspect ratio
    pub lock_aspect_ratio: bool,
}

/// Edge behavior for panning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeBehavior {
    /// Stop at edge
    Stop,
    /// Bounce at edge
    Bounce,
    /// Elastic at edge
    Elastic,
    /// Wrap around
    WrapAround,
    /// Custom edge behavior
    Custom(String),
}

/// Pan animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanAnimation {
    /// Animation enabled
    pub enabled: bool,
    /// Animation duration
    pub duration: Duration,
    /// Animation easing
    pub easing: EasingFunction,
    /// Smooth panning
    pub smooth_panning: bool,
}

/// Pan behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanBehavior {
    /// Pan sensitivity
    pub sensitivity: f64,
    /// Momentum panning
    pub momentum: bool,
    /// Inertia configuration
    pub inertia: InertiaConfig,
    /// Pan indicators
    pub indicators: PanIndicators,
}

/// Pan indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanIndicators {
    /// Show pan bounds
    pub show_bounds: bool,
    /// Show pan center
    pub show_center: bool,
    /// Show pan guide
    pub show_guide: bool,
    /// Indicator style
    pub indicator_style: IndicatorStyle,
}

/// Brush controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushController {
    /// Brush mode
    pub brush_mode: BrushMode,
    /// Brush constraints
    pub constraints: BrushConstraints,
    /// Brush appearance
    pub appearance: BrushAppearance,
    /// Brush behavior
    pub behavior: BrushBehavior,
}

/// Brush modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrushMode {
    /// Rectangle brush
    Rectangle,
    /// Circle brush
    Circle,
    /// Lasso brush
    Lasso,
    /// Polygon brush
    Polygon,
    /// Custom brush mode
    Custom(String),
}

/// Brush constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushConstraints {
    /// Minimum brush size
    pub min_size: f64,
    /// Maximum brush size
    pub max_size: Option<f64>,
    /// Snap to grid
    pub snap_to_grid: bool,
    /// Grid size for snapping
    pub grid_size: Option<f64>,
    /// Constrain to data bounds
    pub constrain_to_data: bool,
}

/// Brush appearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushAppearance {
    /// Brush stroke color
    pub stroke_color: String,
    /// Brush fill color
    pub fill_color: String,
    /// Brush opacity
    pub opacity: f64,
    /// Brush stroke width
    pub stroke_width: f64,
    /// Brush dash pattern
    pub dash_pattern: Option<Vec<f64>>,
}

/// Brush behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushBehavior {
    /// Multi-selection
    pub multi_selection: bool,
    /// Clear on outside click
    pub clear_on_outside_click: bool,
    /// Brush handles
    pub handles: BrushHandles,
    /// Selection feedback
    pub selection_feedback: SelectionFeedback,
}

/// Brush handles configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushHandles {
    /// Show handles
    pub show_handles: bool,
    /// Handle size
    pub handle_size: f64,
    /// Handle shape
    pub handle_shape: HandleShape,
    /// Handle style
    pub handle_style: HandleStyle,
}

/// Handle shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandleShape {
    /// Square handle
    Square,
    /// Circle handle
    Circle,
    /// Diamond handle
    Diamond,
    /// Triangle handle
    Triangle,
    /// Custom handle shape
    Custom(String),
}

/// Handle style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandleStyle {
    /// Handle color
    pub color: String,
    /// Handle stroke color
    pub stroke_color: String,
    /// Handle opacity
    pub opacity: f64,
}

/// Selection feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionFeedback {
    /// Visual feedback
    pub visual_feedback: VisualFeedback,
    /// Audio feedback
    pub audio_feedback: AudioFeedback,
    /// Haptic feedback
    pub haptic_feedback: HapticFeedback,
}

/// Visual feedback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualFeedback {
    /// Highlight selected items
    pub highlight_selected: bool,
    /// Selection animation
    pub selection_animation: Option<SelectionAnimation>,
    /// Selection overlay
    pub selection_overlay: Option<SelectionOverlay>,
}

/// Selection animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionAnimation {
    /// Animation type
    pub animation_type: SelectionAnimationType,
    /// Animation duration
    pub duration: Duration,
    /// Animation easing
    pub easing: EasingFunction,
}

/// Selection animation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionAnimationType {
    /// Fade animation
    Fade,
    /// Scale animation
    Scale,
    /// Pulse animation
    Pulse,
    /// Glow animation
    Glow,
    /// Custom animation
    Custom(String),
}

/// Selection overlay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionOverlay {
    /// Overlay color
    pub color: String,
    /// Overlay opacity
    pub opacity: f64,
    /// Overlay pattern
    pub pattern: Option<String>,
}

/// Audio feedback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeedback {
    /// Audio enabled
    pub enabled: bool,
    /// Selection sound
    pub selection_sound: Option<String>,
    /// Deselection sound
    pub deselection_sound: Option<String>,
    /// Audio volume
    pub volume: f64,
}

/// Haptic feedback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticFeedback {
    /// Haptic enabled
    pub enabled: bool,
    /// Haptic pattern
    pub pattern: HapticPattern,
    /// Haptic intensity
    pub intensity: f64,
}

/// Haptic patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HapticPattern {
    /// Light haptic
    Light,
    /// Medium haptic
    Medium,
    /// Heavy haptic
    Heavy,
    /// Custom pattern
    Custom(String),
}

/// Selection controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionController {
    /// Selection mode
    pub selection_mode: SelectionMode,
    /// Selection constraints
    pub constraints: SelectionConstraints,
    /// Selection behavior
    pub behavior: SelectionBehavior,
    /// Selection state management
    pub state_management: SelectionStateManagement,
}

/// Selection modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMode {
    /// Single selection
    Single,
    /// Multiple selection
    Multiple,
    /// Range selection
    Range,
    /// Toggle selection
    Toggle,
    /// Custom selection mode
    Custom(String),
}

/// Selection constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConstraints {
    /// Maximum selections
    pub max_selections: Option<usize>,
    /// Minimum selections
    pub min_selections: Option<usize>,
    /// Selectable items filter
    pub selectable_filter: Option<String>,
    /// Selection validation
    pub validation: SelectionValidation,
}

/// Selection validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionValidation {
    /// Validation rules
    pub rules: Vec<SelectionValidationRule>,
    /// Validation mode
    pub mode: SelectionValidationMode,
    /// Error handling
    pub error_handling: SelectionErrorHandling,
}

/// Selection validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionValidationRule {
    /// Rule name
    pub rule_name: String,
    /// Rule condition
    pub condition: String,
    /// Error message
    pub error_message: String,
}

/// Selection validation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionValidationMode {
    /// Strict validation
    Strict,
    /// Lenient validation
    Lenient,
    /// Warning only
    WarningOnly,
}

/// Selection error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionErrorHandling {
    /// Prevent selection
    Prevent,
    /// Allow with warning
    AllowWithWarning,
    /// Auto correct
    AutoCorrect,
    /// Custom handling
    Custom(String),
}

/// Selection behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionBehavior {
    /// Clear on background click
    pub clear_on_background_click: bool,
    /// Selection persistence
    pub persistence: SelectionPersistence,
    /// Selection synchronization
    pub synchronization: SelectionSynchronization,
    /// Selection callbacks
    pub callbacks: SelectionCallbacks,
}

/// Selection persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionPersistence {
    /// Persist across sessions
    pub persist_across_sessions: bool,
    /// Storage mechanism
    pub storage: SelectionStorage,
    /// Persistence key
    pub persistence_key: String,
}

/// Selection storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStorage {
    /// Local storage
    LocalStorage,
    /// Session storage
    SessionStorage,
    /// Memory storage
    Memory,
    /// Custom storage
    Custom(String),
}

/// Selection synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionSynchronization {
    /// Sync with other components
    pub sync_with_components: bool,
    /// Sync channels
    pub sync_channels: Vec<String>,
    /// Sync mode
    pub sync_mode: SelectionSyncMode,
}

/// Selection synchronization modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionSyncMode {
    /// Bidirectional sync
    Bidirectional,
    /// Unidirectional sync
    Unidirectional,
    /// Master-slave sync
    MasterSlave,
    /// Custom sync mode
    Custom(String),
}

/// Selection callbacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCallbacks {
    /// On selection change
    pub on_selection_change: Option<String>,
    /// On selection start
    pub on_selection_start: Option<String>,
    /// On selection end
    pub on_selection_end: Option<String>,
    /// On selection clear
    pub on_selection_clear: Option<String>,
}

/// Selection state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionStateManagement {
    /// State tracking
    pub state_tracking: StateTracking,
    /// Undo/redo support
    pub undo_redo: UndoRedoSupport,
    /// State serialization
    pub serialization: StateSerializationConfig,
}

/// State tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTracking {
    /// Track selection history
    pub track_history: bool,
    /// History size limit
    pub history_size: usize,
    /// State change detection
    pub change_detection: ChangeDetection,
}

/// Change detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeDetection {
    /// Detection strategy
    pub strategy: ChangeDetectionStrategy,
    /// Detection frequency
    pub frequency: ChangeDetectionFrequency,
}

/// Change detection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeDetectionStrategy {
    /// Deep comparison
    Deep,
    /// Shallow comparison
    Shallow,
    /// Reference comparison
    Reference,
    /// Custom strategy
    Custom(String),
}

/// Change detection frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeDetectionFrequency {
    /// Immediate detection
    Immediate,
    /// Batched detection
    Batched(Duration),
    /// On-demand detection
    OnDemand,
}

/// Undo/redo support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoRedoSupport {
    /// Undo/redo enabled
    pub enabled: bool,
    /// Undo stack size
    pub undo_stack_size: usize,
    /// Redo stack size
    pub redo_stack_size: usize,
    /// Command patterns
    pub command_patterns: Vec<CommandPattern>,
}

/// Command pattern for undo/redo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandPattern {
    /// Command name
    pub command_name: String,
    /// Execute function
    pub execute: String,
    /// Undo function
    pub undo: String,
    /// Redo function
    pub redo: Option<String>,
}

/// State serialization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSerializationConfig {
    /// Serialization format
    pub format: SerializationFormat,
    /// Compression enabled
    pub compression: bool,
    /// Encryption enabled
    pub encryption: bool,
    /// Custom serializers
    pub custom_serializers: HashMap<String, String>,
}

/// Serialization formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// JSON format
    JSON,
    /// Binary format
    Binary,
    /// MessagePack format
    MessagePack,
    /// Custom format
    Custom(String),
}

/// Tooltip controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipController {
    /// Tooltip trigger
    pub trigger: TooltipTrigger,
    /// Tooltip positioning
    pub positioning: TooltipPositioning,
    /// Tooltip appearance
    pub appearance: TooltipAppearance,
    /// Tooltip behavior
    pub behavior: TooltipBehavior,
}

/// Tooltip triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipTrigger {
    /// Hover trigger
    Hover,
    /// Click trigger
    Click,
    /// Focus trigger
    Focus,
    /// Manual trigger
    Manual,
    /// Custom trigger
    Custom(String),
}

/// Tooltip positioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipPositioning {
    /// Position strategy
    pub strategy: TooltipPositionStrategy,
    /// Offset configuration
    pub offset: TooltipOffset,
    /// Collision detection
    pub collision_detection: CollisionDetection,
}

/// Tooltip position strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipPositionStrategy {
    /// Fixed position
    Fixed,
    /// Absolute position
    Absolute,
    /// Relative position
    Relative,
    /// Smart positioning
    Smart,
    /// Custom strategy
    Custom(String),
}

/// Tooltip offset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipOffset {
    /// X offset
    pub x: f64,
    /// Y offset
    pub y: f64,
    /// Dynamic offset
    pub dynamic: bool,
}

/// Collision detection for tooltips
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionDetection {
    /// Detection enabled
    pub enabled: bool,
    /// Detection boundaries
    pub boundaries: CollisionBoundaries,
    /// Flip behavior
    pub flip_behavior: FlipBehavior,
}

/// Collision boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionBoundaries {
    /// Viewport boundaries
    pub viewport: bool,
    /// Parent boundaries
    pub parent: bool,
    /// Custom boundaries
    pub custom: Option<BoundingBox>,
}

/// Flip behavior for collision avoidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipBehavior {
    /// Horizontal flip
    pub horizontal: bool,
    /// Vertical flip
    pub vertical: bool,
    /// Flip priority
    pub priority: FlipPriority,
}

/// Flip priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlipPriority {
    /// Horizontal first
    HorizontalFirst,
    /// Vertical first
    VerticalFirst,
    /// Best fit
    BestFit,
    /// Custom priority
    Custom(String),
}

/// Tooltip appearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipAppearance {
    /// Tooltip theme
    pub theme: TooltipTheme,
    /// Background style
    pub background: TooltipBackground,
    /// Border style
    pub border: TooltipBorder,
    /// Shadow style
    pub shadow: Option<TooltipShadow>,
    /// Arrow configuration
    pub arrow: Option<TooltipArrow>,
}

/// Tooltip theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipTheme {
    /// Light theme
    Light,
    /// Dark theme
    Dark,
    /// Auto theme
    Auto,
    /// Custom theme
    Custom(String),
}

/// Tooltip background
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipBackground {
    /// Background color
    pub color: String,
    /// Background opacity
    pub opacity: f64,
    /// Background image
    pub image: Option<String>,
}

/// Tooltip border
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipBorder {
    /// Border width
    pub width: f64,
    /// Border color
    pub color: String,
    /// Border radius
    pub radius: f64,
    /// Border style
    pub style: BorderStyle,
}

/// Border styles
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
    /// Custom border style
    Custom(String),
}

/// Tooltip shadow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipShadow {
    /// Shadow color
    pub color: String,
    /// Shadow offset X
    pub offset_x: f64,
    /// Shadow offset Y
    pub offset_y: f64,
    /// Shadow blur
    pub blur: f64,
    /// Shadow spread
    pub spread: f64,
}

/// Tooltip arrow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipArrow {
    /// Arrow enabled
    pub enabled: bool,
    /// Arrow size
    pub size: f64,
    /// Arrow color
    pub color: String,
    /// Arrow position
    pub position: ArrowPosition,
}

/// Arrow position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArrowPosition {
    /// Top arrow
    Top,
    /// Bottom arrow
    Bottom,
    /// Left arrow
    Left,
    /// Right arrow
    Right,
    /// Auto position
    Auto,
    /// Custom position
    Custom(f64, f64),
}

/// Tooltip behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipBehavior {
    /// Show delay
    pub show_delay: Duration,
    /// Hide delay
    pub hide_delay: Duration,
    /// Animation configuration
    pub animation: TooltipAnimation,
    /// Interactive tooltip
    pub interactive: bool,
    /// Content management
    pub content_management: TooltipContentManagement,
}

/// Tooltip animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipAnimation {
    /// Show animation
    pub show_animation: TooltipAnimationType,
    /// Hide animation
    pub hide_animation: TooltipAnimationType,
    /// Animation duration
    pub duration: Duration,
    /// Animation easing
    pub easing: EasingFunction,
}

/// Tooltip animation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipAnimationType {
    /// Fade animation
    Fade,
    /// Scale animation
    Scale,
    /// Slide animation
    Slide,
    None,
    /// Custom animation
    Custom(String),
}

/// Tooltip content management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipContentManagement {
    /// Dynamic content
    pub dynamic_content: bool,
    /// Content caching
    pub content_caching: bool,
    /// Content templates
    pub templates: HashMap<String, String>,
    /// Content sanitization
    pub sanitization: ContentSanitization,
}

/// Content sanitization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSanitization {
    /// HTML sanitization
    pub html_sanitization: bool,
    /// Script removal
    pub script_removal: bool,
    /// Allowed tags
    pub allowed_tags: Vec<String>,
    /// Custom sanitizers
    pub custom_sanitizers: Vec<String>,
}

/// Drag controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragController {
    /// Drag mode
    pub drag_mode: DragMode,
    /// Drag constraints
    pub constraints: DragConstraints,
    /// Drag feedback
    pub feedback: DragFeedback,
    /// Drop zones
    pub drop_zones: Vec<DropZone>,
}

/// Drag modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DragMode {
    /// Move drag
    Move,
    /// Copy drag
    Copy,
    /// Link drag
    Link,
    /// Custom drag mode
    Custom(String),
}

/// Drag constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragConstraints {
    /// Drag bounds
    pub bounds: Option<BoundingBox>,
    /// Drag axes
    pub axes: DragAxes,
    /// Snap to grid
    pub snap_to_grid: bool,
    /// Grid size
    pub grid_size: Option<f64>,
}

/// Drag axes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragAxes {
    /// Allow horizontal drag
    pub horizontal: bool,
    /// Allow vertical drag
    pub vertical: bool,
    /// Lock aspect ratio
    pub lock_aspect_ratio: bool,
}

/// Drag feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragFeedback {
    /// Visual feedback
    pub visual: DragVisualFeedback,
    /// Audio feedback
    pub audio: AudioFeedback,
    /// Haptic feedback
    pub haptic: HapticFeedback,
}

/// Drag visual feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragVisualFeedback {
    /// Drag ghost
    pub ghost: DragGhost,
    /// Drag indicators
    pub indicators: DragIndicators,
    /// Drop preview
    pub drop_preview: DropPreview,
}

/// Drag ghost configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragGhost {
    /// Ghost enabled
    pub enabled: bool,
    /// Ghost opacity
    pub opacity: f64,
    /// Ghost scale
    pub scale: f64,
    /// Ghost offset
    pub offset: (f64, f64),
}

/// Drag indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragIndicators {
    /// Show drag cursor
    pub show_cursor: bool,
    /// Show drag path
    pub show_path: bool,
    /// Show drop zones
    pub show_drop_zones: bool,
    /// Indicator style
    pub style: DragIndicatorStyle,
}

/// Drag indicator style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragIndicatorStyle {
    /// Cursor style
    pub cursor: String,
    /// Path color
    pub path_color: String,
    /// Drop zone color
    pub drop_zone_color: String,
    /// Indicator opacity
    pub opacity: f64,
}

/// Drop preview configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropPreview {
    /// Preview enabled
    pub enabled: bool,
    /// Preview style
    pub style: DropPreviewStyle,
    /// Preview animation
    pub animation: Option<DropPreviewAnimation>,
}

/// Drop preview style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropPreviewStyle {
    /// Preview color
    pub color: String,
    /// Preview opacity
    pub opacity: f64,
    /// Preview border
    pub border: Option<String>,
}

/// Drop preview animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropPreviewAnimation {
    /// Animation type
    pub animation_type: DropPreviewAnimationType,
    /// Animation duration
    pub duration: Duration,
    /// Animation loop
    pub loop_animation: bool,
}

/// Drop preview animation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropPreviewAnimationType {
    /// Pulse animation
    Pulse,
    /// Glow animation
    Glow,
    /// Ripple animation
    Ripple,
    /// Custom animation
    Custom(String),
}

/// Drop zone definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropZone {
    /// Zone identifier
    pub zone_id: String,
    /// Zone bounds
    pub bounds: BoundingBox,
    /// Zone type
    pub zone_type: DropZoneType,
    /// Zone behavior
    pub behavior: DropZoneBehavior,
    /// Zone appearance
    pub appearance: DropZoneAppearance,
}

/// Drop zone types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropZoneType {
    /// Accept all drops
    Accept,
    /// Reject all drops
    Reject,
    /// Conditional drops
    Conditional(String),
    /// Transform drops
    Transform(String),
}

/// Drop zone behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropZoneBehavior {
    /// Drop action
    pub drop_action: DropAction,
    /// Validation rules
    pub validation: Vec<DropValidationRule>,
    /// Callbacks
    pub callbacks: DropZoneCallbacks,
}

/// Drop actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropAction {
    /// Move action
    Move,
    /// Copy action
    Copy,
    /// Link action
    Link,
    /// Custom action
    Custom(String),
}

/// Drop validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropValidationRule {
    /// Rule name
    pub rule_name: String,
    /// Rule condition
    pub condition: String,
    /// Error message
    pub error_message: String,
}

/// Drop zone callbacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropZoneCallbacks {
    /// On drag enter
    pub on_drag_enter: Option<String>,
    /// On drag over
    pub on_drag_over: Option<String>,
    /// On drag leave
    pub on_drag_leave: Option<String>,
    /// On drop
    pub on_drop: Option<String>,
}

/// Drop zone appearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropZoneAppearance {
    /// Default style
    pub default_style: DropZoneStyle,
    /// Hover style
    pub hover_style: DropZoneStyle,
    /// Active style
    pub active_style: DropZoneStyle,
    /// Invalid style
    pub invalid_style: DropZoneStyle,
}

/// Drop zone style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropZoneStyle {
    /// Background color
    pub background_color: String,
    /// Border color
    pub border_color: String,
    /// Border width
    pub border_width: f64,
    /// Border style
    pub border_style: BorderStyle,
    /// Opacity
    pub opacity: f64,
}

/// Resize controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeController {
    /// Resize mode
    pub resize_mode: ResizeMode,
    /// Resize constraints
    pub constraints: ResizeConstraints,
    /// Resize handles
    pub handles: ResizeHandles,
    /// Resize behavior
    pub behavior: ResizeBehavior,
}

/// Resize modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResizeMode {
    /// Manual resize
    Manual,
    /// Automatic resize
    Automatic,
    /// Proportional resize
    Proportional,
    /// Custom resize mode
    Custom(String),
}

/// Resize constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeConstraints {
    /// Minimum size
    pub min_size: (f64, f64),
    /// Maximum size
    pub max_size: Option<(f64, f64)>,
    /// Aspect ratio constraints
    pub aspect_ratio: AspectRatioConstraints,
    /// Snap to grid
    pub snap_to_grid: bool,
    /// Grid size
    pub grid_size: Option<f64>,
}

/// Aspect ratio constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AspectRatioConstraints {
    /// Lock aspect ratio
    pub lock_aspect_ratio: bool,
    /// Target aspect ratio
    pub target_ratio: Option<f64>,
    /// Ratio tolerance
    pub tolerance: f64,
}

/// Resize handles configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeHandles {
    /// Show handles
    pub show_handles: bool,
    /// Handle positions
    pub positions: Vec<HandlePosition>,
    /// Handle appearance
    pub appearance: ResizeHandleAppearance,
    /// Handle behavior
    pub behavior: ResizeHandleBehavior,
}

/// Handle positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandlePosition {
    /// Top left corner
    TopLeft,
    /// Top center
    TopCenter,
    /// Top right corner
    TopRight,
    /// Middle left
    MiddleLeft,
    /// Middle right
    MiddleRight,
    /// Bottom left corner
    BottomLeft,
    /// Bottom center
    BottomCenter,
    /// Bottom right corner
    BottomRight,
    /// Custom position
    Custom(f64, f64),
}

/// Resize handle appearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeHandleAppearance {
    /// Handle size
    pub size: f64,
    /// Handle shape
    pub shape: HandleShape,
    /// Handle color
    pub color: String,
    /// Handle border
    pub border: Option<HandleBorder>,
}

/// Handle border
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandleBorder {
    /// Border width
    pub width: f64,
    /// Border color
    pub color: String,
    /// Border style
    pub style: BorderStyle,
}

/// Resize handle behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeHandleBehavior {
    /// Handle sensitivity
    pub sensitivity: f64,
    /// Cursor style
    pub cursor_style: HashMap<HandlePosition, String>,
    /// Handle activation
    pub activation: HandleActivation,
}

/// Handle activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandleActivation {
    /// Activation method
    pub method: ActivationMethod,
    /// Activation area
    pub area: ActivationArea,
    /// Auto hide
    pub auto_hide: bool,
}

/// Activation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationMethod {
    /// Hover activation
    Hover,
    /// Click activation
    Click,
    /// Double click activation
    DoubleClick,
    /// Always active
    Always,
    /// Custom activation
    Custom(String),
}

/// Activation area
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationArea {
    /// Area size
    pub size: f64,
    /// Area shape
    pub shape: ActivationAreaShape,
    /// Area offset
    pub offset: (f64, f64),
}

/// Activation area shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationAreaShape {
    /// Rectangle area
    Rectangle,
    /// Circle area
    Circle,
    /// Custom area shape
    Custom(String),
}

/// Resize behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeBehavior {
    /// Live resize
    pub live_resize: bool,
    /// Resize animation
    pub animation: Option<ResizeAnimation>,
    /// Resize callbacks
    pub callbacks: ResizeCallbacks,
    /// Collision detection
    pub collision_detection: ResizeCollisionDetection,
}

/// Resize animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeAnimation {
    /// Animation enabled
    pub enabled: bool,
    /// Animation duration
    pub duration: Duration,
    /// Animation easing
    pub easing: EasingFunction,
    /// Animation type
    pub animation_type: ResizeAnimationType,
}

/// Resize animation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResizeAnimationType {
    /// Smooth resize
    Smooth,
    /// Elastic resize
    Elastic,
    /// Bounce resize
    Bounce,
    /// Custom animation
    Custom(String),
}

/// Resize callbacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeCallbacks {
    /// On resize start
    pub on_resize_start: Option<String>,
    /// On resize
    pub on_resize: Option<String>,
    /// On resize end
    pub on_resize_end: Option<String>,
    /// On resize cancel
    pub on_resize_cancel: Option<String>,
}

/// Resize collision detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeCollisionDetection {
    /// Detection enabled
    pub enabled: bool,
    /// Collision boundaries
    pub boundaries: CollisionBoundaries,
    /// Collision response
    pub response: CollisionResponse,
}

/// Collision response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollisionResponse {
    /// Stop resize
    Stop,
    /// Constrain resize
    Constrain,
    /// Push other elements
    Push,
    /// Custom response
    Custom(String),
}

/// Controller state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControllerState {
    /// Inactive controller
    Inactive,
    /// Active controller
    Active,
    /// Busy controller
    Busy,
    /// Error controller
    Error,
    /// Disabled controller
    Disabled,
}

/// Controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfiguration {
    /// Configuration parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Configuration schema
    pub schema: Option<String>,
    /// Configuration validation
    pub validation: bool,
    /// Configuration versioning
    pub version: String,
}

/// Interaction behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionBehavior {
    /// Behavior name
    pub name: String,
    /// Behavior type
    pub behavior_type: BehaviorType,
    /// Behavior conditions
    pub conditions: Vec<String>,
    /// Behavior actions
    pub actions: Vec<String>,
}

/// Behavior types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BehaviorType {
    /// Event-driven behavior
    EventDriven,
    /// State-driven behavior
    StateDriven,
    /// Time-driven behavior
    TimeDriven,
    /// Custom behavior
    Custom(String),
}

/// Controller performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerPerformanceMetrics {
    /// Response time metrics
    pub response_time: ResponseTimeMetrics,
    /// Resource usage metrics
    pub resource_usage: ControllerResourceUsage,
    /// Error metrics
    pub error_metrics: ControllerErrorMetrics,
    /// Throughput metrics
    pub throughput_metrics: ThroughputMetrics,
}

/// Response time metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeMetrics {
    /// Average response time
    pub average: Duration,
    /// Median response time
    pub median: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// Maximum response time
    pub max: Duration,
}

/// Controller resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Network bandwidth usage
    pub network_usage: f64,
    /// GPU usage percentage
    pub gpu_usage: f64,
}

/// Controller error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerErrorMetrics {
    /// Error rate
    pub error_rate: f64,
    /// Error count
    pub error_count: u64,
    /// Error types
    pub error_types: HashMap<String, u64>,
    /// Recovery time
    pub recovery_time: Duration,
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub operations_per_second: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Average throughput
    pub average_throughput: f64,
    /// Current throughput
    pub current_throughput: f64,
}

// Default implementations

impl Default for InteractionController {
    fn default() -> Self {
        Self {
            controller_id: "default".to_string(),
            controller_type: InteractionControllerType::Custom("default".to_string()),
            state: ControllerState::Inactive,
            configuration: ControllerConfiguration::default(),
            behaviors: vec![],
            performance_metrics: ControllerPerformanceMetrics::default(),
        }
    }
}

impl Default for ControllerConfiguration {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            schema: None,
            validation: false,
            version: "1.0.0".to_string(),
        }
    }
}

impl Default for ControllerPerformanceMetrics {
    fn default() -> Self {
        Self {
            response_time: ResponseTimeMetrics::default(),
            resource_usage: ControllerResourceUsage::default(),
            error_metrics: ControllerErrorMetrics::default(),
            throughput_metrics: ThroughputMetrics::default(),
        }
    }
}

impl Default for ResponseTimeMetrics {
    fn default() -> Self {
        Self {
            average: Duration::from_millis(1),
            median: Duration::from_millis(1),
            p95: Duration::from_millis(5),
            p99: Duration::from_millis(10),
            max: Duration::from_millis(50),
        }
    }
}

impl Default for ControllerResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            network_usage: 0.0,
            gpu_usage: 0.0,
        }
    }
}

impl Default for ControllerErrorMetrics {
    fn default() -> Self {
        Self {
            error_rate: 0.0,
            error_count: 0,
            error_types: HashMap::new(),
            recovery_time: Duration::from_millis(100),
        }
    }
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            operations_per_second: 0.0,
            peak_throughput: 0.0,
            average_throughput: 0.0,
            current_throughput: 0.0,
        }
    }
}