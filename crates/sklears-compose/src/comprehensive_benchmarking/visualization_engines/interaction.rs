use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_types::*;

/// Margin configuration for external
/// spacing around chart elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Margin {
    /// Top margin
    top: f64,
    /// Right margin
    right: f64,
    /// Bottom margin
    bottom: f64,
    /// Left margin
    left: f64,
}

/// Padding configuration for internal
/// spacing within chart elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Padding {
    /// Top padding
    top: f64,
    /// Right padding
    right: f64,
    /// Bottom padding
    bottom: f64,
    /// Left padding
    left: f64,
}

/// Interaction configuration for user
/// engagement and chart interactivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionConfig {
    /// Hover effects configuration
    hover_effects: HoverEffects,
    /// Click actions configuration
    click_actions: ClickActions,
    /// Zoom configuration
    zoom_config: ZoomConfig,
    /// Pan configuration
    pan_config: PanConfig,
    /// Selection configuration
    selection_config: SelectionConfig,
}

/// Hover effects configuration for
/// mouse interaction feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverEffects {
    enabled: bool,
    highlight_color: Color,
    tooltip: TooltipConfig,
    animation: HoverAnimation,
}

/// Tooltip configuration for
/// contextual information display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipConfig {
    /// Whether tooltip is enabled
    enabled: bool,
    /// Tooltip content template
    template: String,
    /// Tooltip styling
    styling: TooltipStyling,
    /// Tooltip positioning strategy
    positioning: TooltipPositioning,
}

/// Tooltip styling configuration for
/// tooltip appearance customization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipStyling {
    /// Background color
    background: Color,
    /// Border styling
    border: Border,
    /// Text styling
    text_styling: TextStyling,
    /// Optional shadow effect
    shadow: Option<Shadow>,
}

/// Tooltip positioning enumeration for
/// tooltip placement strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipPositioning {
    /// Automatic positioning
    Auto,
    /// Fixed position
    Fixed(f64, f64),
    /// Follow mouse cursor
    Follow,
    /// Custom positioning logic
    Custom(String),
}

/// Hover animation configuration for
/// smooth hover transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverAnimation {
    enabled: bool,
    duration: Duration,
    easing: EasingFunction,
}

/// Easing function enumeration for
/// animation transition control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    /// Linear easing
    Linear,
    /// Ease-in transition
    EaseIn,
    /// Ease-out transition
    EaseOut,
    /// Ease-in-out transition
    EaseInOut,
    /// Bounce effect
    Bounce,
    /// Elastic effect
    Elastic,
    /// Custom easing function
    Custom(String),
}

/// Click actions configuration for
/// interactive chart behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickActions {
    /// Whether click actions are enabled
    enabled: bool,
    /// List of available click actions
    actions: Vec<ClickAction>,
}

/// Individual click action configuration
/// for specific user interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickAction {
    /// Type of click action
    action_type: ClickActionType,
    /// Target for the action
    target: String,
    /// Action parameters
    parameters: HashMap<String, String>,
}

/// Click action type enumeration for
/// different interaction behaviors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClickActionType {
    /// Drill down to detailed view
    DrillDown,
    /// Apply data filter
    Filter,
    /// Navigate to different view
    Navigate,
    /// Highlight related elements
    Highlight,
    /// Custom action implementation
    Custom(String),
}

/// Zoom configuration for chart
/// magnification and navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomConfig {
    /// Whether zoom is enabled
    enabled: bool,
    /// Type of zoom interaction
    zoom_type: ZoomType,
    /// Zoom limits and constraints
    zoom_limits: ZoomLimits,
    /// Zoom animation settings
    zoom_animation: ZoomAnimation,
}

/// Zoom type enumeration for
/// different zoom interaction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZoomType {
    /// Mouse wheel zoom
    Wheel,
    /// Touch pinch zoom
    Pinch,
    /// Selection-based zoom
    Selection,
    /// Programmatic zoom
    Programmatic,
    /// Custom zoom implementation
    Custom(String),
}

/// Zoom limits configuration for
/// zoom range constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomLimits {
    min_zoom: f64,
    max_zoom: f64,
    zoom_step: f64,
}

/// Zoom animation configuration for
/// smooth zoom transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomAnimation {
    /// Whether animation is enabled
    enabled: bool,
    /// Animation duration
    duration: Duration,
    /// Easing function
    easing: EasingFunction,
}

/// Pan configuration for chart
/// navigation and exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanConfig {
    /// Whether pan is enabled
    enabled: bool,
    /// Type of pan interaction
    pan_type: PanType,
    /// Pan limits and constraints
    pan_limits: PanLimits,
    /// Pan animation settings
    pan_animation: PanAnimation,
}

/// Pan type enumeration for
/// different pan interaction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanType {
    /// Mouse drag pan
    Drag,
    /// Touch pan
    Touch,
    /// Programmatic pan
    Programmatic,
    /// Custom pan implementation
    Custom(String),
}

/// Pan limits configuration for
/// pan range constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanLimits {
    constrain_to_data: bool,
    custom_bounds: Option<(f64, f64, f64, f64)>,
}

/// Pan animation configuration for
/// smooth pan transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanAnimation {
    /// Whether animation is enabled
    enabled: bool,
    /// Animation duration
    duration: Duration,
    /// Easing function
    easing: EasingFunction,
}

/// Selection configuration for data
/// element selection and highlighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Whether selection is enabled
    enabled: bool,
    /// Type of selection
    selection_type: SelectionType,
    /// Selection styling
    selection_styling: SelectionStyling,
    /// Whether multiple selection is allowed
    multi_select: bool,
}

/// Selection type enumeration for
/// different selection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionType {
    /// Single element selection
    Single,
    /// Multiple element selection
    Multiple,
    /// Range selection
    Range,
    /// Lasso selection
    Lasso,
    /// Custom selection implementation
    Custom(String),
}

/// Selection styling configuration for
/// visual selection feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionStyling {
    /// Selection highlight color
    selection_color: Color,
    /// Selection opacity
    selection_opacity: f64,
    /// Selection border
    selection_border: Border,
}

/// Animation configuration for dynamic
/// visual effects and transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    /// Whether animation is enabled
    enabled: bool,
    /// Entrance animation settings
    entrance_animation: EntranceAnimation,
    /// Transition animations
    transition_animations: Vec<TransitionAnimation>,
    /// Exit animation settings
    exit_animation: ExitAnimation,
}

/// Entrance animation configuration for
/// chart appearance transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntranceAnimation {
    /// Type of entrance animation
    animation_type: AnimationType,
    /// Animation duration
    duration: Duration,
    /// Animation delay
    delay: Duration,
    /// Easing function
    easing: EasingFunction,
}

/// Animation type enumeration for
/// different animation styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationType {
    /// Fade in animation
    FadeIn,
    /// Slide in animation
    SlideIn,
    /// Scale in animation
    ScaleIn,
    /// Rotate in animation
    RotateIn,
    /// Bounce in animation
    BounceIn,
    /// Custom animation implementation
    Custom(String),
}

/// Transition animation configuration for
/// data change animations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionAnimation {
    /// Animation trigger condition
    trigger: AnimationTrigger,
    /// Type of transition animation
    animation_type: AnimationType,
    /// Animation duration
    duration: Duration,
    /// Easing function
    easing: EasingFunction,
}

/// Animation trigger enumeration for
/// animation activation conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationTrigger {
    /// Data change trigger
    DataChange,
    /// User interaction trigger
    UserInteraction,
    /// Time interval trigger
    TimeInterval,
    /// Custom trigger implementation
    Custom(String),
}

/// Exit animation configuration for
/// chart disappearance transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitAnimation {
    /// Type of exit animation
    animation_type: AnimationType,
    /// Animation duration
    duration: Duration,
    /// Easing function
    easing: EasingFunction,
}

