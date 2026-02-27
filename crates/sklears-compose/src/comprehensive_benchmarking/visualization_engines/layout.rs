use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_types::*;


/// Theme variant configuration for
/// context-specific styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeVariant {
    /// Variant name identifier
    variant_name: String,
    /// Color overrides for this variant
    color_overrides: HashMap<String, Color>,
    /// Style property overrides
    style_overrides: HashMap<String, String>,
    /// Component style overrides
    component_overrides: HashMap<String, ComponentStyle>,
}

/// Component style configuration for
/// individual component customization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStyle {
    /// Base style properties
    properties: HashMap<String, String>,
    /// Pseudo-class style definitions
    pseudo_classes: HashMap<String, HashMap<String, String>>,
}

/// Chart layout configuration for
/// structural organization of chart elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartLayoutConfig {
    /// Title configuration
    title: TitleConfig,
    /// Legend configuration
    legend: LegendConfig,
    /// Axes configuration
    axes: AxesConfig,
    /// Grid configuration
    grid: GridConfig,
    /// Spacing configuration
    spacing: SpacingConfig,
}

/// Title configuration for chart
/// heading and labeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitleConfig {
    /// Whether title is enabled
    enabled: bool,
    /// Title text content
    text: String,
    /// Title position
    position: TitlePosition,
    /// Title styling
    styling: TextStyling,
}

/// Title position enumeration for
/// flexible title placement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TitlePosition {
    /// Top center position
    Top,
    /// Bottom center position
    Bottom,
    /// Left center position
    Left,
    /// Right center position
    Right,
    /// Centered position
    Center,
    /// Custom position with coordinates
    Custom(f64, f64),
}

/// Text styling configuration for
/// consistent text appearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStyling {
    /// Font family name
    font_family: String,
    /// Font size
    font_size: f64,
    /// Font weight
    font_weight: u16,
    /// Text color
    color: Color,
    /// Text alignment
    alignment: TextAlignment,
}

/// Text alignment enumeration for
/// text positioning control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextAlignment {
    /// Left-aligned text
    Left,
    /// Center-aligned text
    Center,
    /// Right-aligned text
    Right,
    /// Justified text
    Justify,
    /// Custom text alignment
    Custom(String),
}

/// Legend configuration for chart
/// data series identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendConfig {
    /// Whether legend is enabled
    enabled: bool,
    /// Legend position
    position: LegendPosition,
    /// Legend orientation
    orientation: LegendOrientation,
    /// Legend styling
    styling: LegendStyling,
    /// Whether legend is interactive
    interactive: bool,
}

/// Legend position enumeration for
/// flexible legend placement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendPosition {
    /// Top position
    Top,
    /// Bottom position
    Bottom,
    /// Left position
    Left,
    /// Right position
    Right,
    /// Top-left corner
    TopLeft,
    /// Top-right corner
    TopRight,
    /// Bottom-left corner
    BottomLeft,
    /// Bottom-right corner
    BottomRight,
    /// Custom position with coordinates
    Custom(f64, f64),
}

/// Legend orientation enumeration for
/// legend layout control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendOrientation {
    /// Horizontal legend layout
    Horizontal,
    /// Vertical legend layout
    Vertical,
}

/// Legend styling configuration for
/// legend appearance customization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendStyling {
    /// Background color
    background: Color,
    /// Border styling
    border: Border,
    /// Text styling
    text_styling: TextStyling,
    /// Spacing between legend items
    item_spacing: f64,
}

/// Axes configuration for chart
/// coordinate system setup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxesConfig {
    /// X-axis configuration
    x_axis: AxisConfig,
    /// Y-axis configuration
    y_axis: AxisConfig,
    /// Optional secondary Y-axis
    secondary_y_axis: Option<AxisConfig>,
}

/// Individual axis configuration for
/// axis appearance and behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    /// Whether axis is enabled
    enabled: bool,
    /// Axis title configuration
    title: AxisTitleConfig,
    /// Axis labels configuration
    labels: AxisLabelsConfig,
    /// Axis ticks configuration
    ticks: AxisTicksConfig,
    /// Axis line configuration
    line: AxisLineConfig,
    /// Axis scale configuration
    scale: AxisScaleConfig,
}

/// Axis title configuration for
/// axis labeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisTitleConfig {
    /// Whether title is enabled
    enabled: bool,
    /// Title text content
    text: String,
    /// Title styling
    styling: TextStyling,
    /// Title rotation angle
    rotation: f64,
}

/// Axis labels configuration for
/// data point labeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisLabelsConfig {
    /// Whether labels are enabled
    enabled: bool,
    /// Label styling
    styling: TextStyling,
    /// Label rotation angle
    rotation: f64,
    /// Label format configuration
    format: LabelFormat,
    /// Whether to skip overlapping labels
    skip_overlapping: bool,
}

/// Label format enumeration for
/// different label formatting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LabelFormat {
    /// Automatic label formatting
    Auto,
    /// Number formatting
    Number(NumberFormat),
    /// Date formatting
    Date(DateFormat),
    /// Custom format string
    Custom(String),
}

/// Number format configuration for
/// numeric label formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberFormat {
    /// Decimal separator character
    decimal_separator: String,
    /// Thousands separator character
    thousands_separator: String,
    /// Number of decimal places
    decimal_places: usize,
    /// Currency symbol for monetary values
    currency_symbol: Option<String>,
}

/// Date format configuration for
/// temporal label formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateFormat {
    /// Date pattern string
    date_pattern: String,
    /// Time pattern string
    time_pattern: String,
    /// Whether to display timezone
    timezone_display: bool,
}

/// Axis ticks configuration for
/// axis marking and measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisTicksConfig {
    /// Whether ticks are enabled
    enabled: bool,
    /// Major tick configuration
    major_ticks: TickConfig,
    /// Minor tick configuration
    minor_ticks: TickConfig,
    /// Tick interval specification
    tick_interval: TickInterval,
}

/// Individual tick configuration for
/// tick appearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickConfig {
    /// Whether tick is enabled
    enabled: bool,
    /// Tick length
    length: f64,
    /// Tick width
    width: f64,
    /// Tick color
    color: Color,
}

/// Tick interval enumeration for
/// tick spacing control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TickInterval {
    /// Automatic tick interval
    Auto,
    /// Fixed interval value
    Fixed(f64),
    /// Fixed number of ticks
    Count(usize),
    /// Custom tick positions
    Custom(Vec<f64>),
}

/// Axis line configuration for
/// axis visual representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisLineConfig {
    /// Whether axis line is enabled
    enabled: bool,
    /// Line width
    width: f64,
    /// Line color
    color: Color,
    /// Line style
    style: BorderStyle,
}

/// Axis scale configuration for
/// data mapping and transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisScaleConfig {
    /// Type of scale
    scale_type: ScaleType,
    /// Scale domain
    domain: ScaleDomain,
    /// Scale range
    range: ScaleRange,
    /// Optional scale transformation
    transform: Option<ScaleTransform>,
}

/// Scale type enumeration for
/// different data scaling methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleType {
    /// Linear scale
    Linear,
    /// Logarithmic scale
    Log,
    /// Power scale
    Power,
    /// Time scale
    Time,
    /// Ordinal scale
    Ordinal,
    /// Custom scale implementation
    Custom(String),
}

/// Scale domain enumeration for
/// input data range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleDomain {
    /// Automatic domain from data
    Auto,
    /// Fixed domain range
    Fixed(f64, f64),
    /// Domain from data extent
    Data,
    /// Custom domain values
    Custom(Vec<f64>),
}

/// Scale range enumeration for
/// output range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleRange {
    /// Automatic range
    Auto,
    /// Fixed range
    Fixed(f64, f64),
    /// Custom range values
    Custom(Vec<f64>),
}

/// Scale transform enumeration for
/// data transformation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleTransform {
    /// No transformation
    None,
    /// Logarithmic transformation
    Log,
    /// Square root transformation
    Sqrt,
    /// Square transformation
    Square,
    /// Custom transformation
    Custom(String),
}

/// Grid configuration for chart
/// background grid lines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Whether grid is enabled
    enabled: bool,
    /// Major grid line configuration
    major_grid: GridLineConfig,
    /// Minor grid line configuration
    minor_grid: GridLineConfig,
}

/// Grid line configuration for
/// individual grid line appearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridLineConfig {
    /// Whether grid line is enabled
    enabled: bool,
    /// Line width
    width: f64,
    /// Line color
    color: Color,
    /// Line style
    style: BorderStyle,
    /// Line opacity
    opacity: f64,
}

/// Spacing configuration for chart
/// element positioning and layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacingConfig {
    /// Margin configuration
    margin: Margin,
    /// Padding configuration
    padding: Padding,
    /// Element spacing
    element_spacing: f64,
}

