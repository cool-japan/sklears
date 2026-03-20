use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Data visualization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    Static,
    Interactive,
    RealTime,
    Historical,
    Comparative,
    Predictive,
    Hybrid,
}

/// Widget types for dashboards
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetType {
    LineChart,
    BarChart,
    PieChart,
    ScatterPlot,
    Heatmap,
    Table,
    Gauge,
    Counter,
    Alert,
    Map,
    Timeline,
    TreeMap,
    Sankey,
    Sunburst,
    Radar,
    Candlestick,
    Histogram,
    BoxPlot,
    Text,
    Image,
    Video,
    Custom(String),
}

/// Chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfiguration {
    pub chart_type: WidgetType,
    pub title: String,
    pub subtitle: Option<String>,
    pub axes_config: AxesConfiguration,
    pub legend_config: LegendConfiguration,
    pub color_scheme: ColorScheme,
    pub animation_config: AnimationConfiguration,
    pub interaction_config: InteractionConfiguration,
    pub tooltip_config: TooltipConfiguration,
    pub zoom_config: ZoomConfiguration,
    pub export_config: ExportConfiguration,
}

/// Axes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxesConfiguration {
    pub x_axis: AxisConfig,
    pub y_axis: AxisConfig,
    pub y2_axis: Option<AxisConfig>,
    pub z_axis: Option<AxisConfig>,
}

/// Individual axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    pub title: String,
    pub scale_type: ScaleType,
    pub range: Option<(f64, f64)>,
    pub tick_format: TickFormat,
    pub grid_lines: GridLineConfig,
    pub label_rotation: f64,
    pub reverse: bool,
    pub logarithmic: bool,
}

/// Scale types for axes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleType {
    Linear,
    Logarithmic,
    Categorical,
    Time,
    Ordinal,
    Custom(String),
}

/// Tick formatting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TickFormat {
    Number(NumberFormat),
    Time(TimeFormat),
    Percentage,
    Currency(CurrencyFormat),
    Scientific,
    Custom(String),
}

/// Number formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberFormat {
    pub decimal_places: u32,
    pub thousands_separator: String,
    pub decimal_separator: String,
    pub prefix: Option<String>,
    pub suffix: Option<String>,
}

/// Time formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeFormat {
    pub format_string: String,
    pub timezone: String,
    pub locale: String,
}

/// Currency formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyFormat {
    pub currency_code: String,
    pub symbol: String,
    pub position: CurrencyPosition,
    pub decimal_places: u32,
}

/// Currency symbol position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurrencyPosition {
    Before,
    After,
}

/// Grid line configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridLineConfig {
    pub enabled: bool,
    pub color: String,
    pub width: f64,
    pub style: LineStyle,
    pub opacity: f64,
}

/// Line styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
    Custom(String),
}

/// Legend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendConfiguration {
    pub enabled: bool,
    pub position: LegendPosition,
    pub alignment: LegendAlignment,
    pub orientation: LegendOrientation,
    pub max_columns: Option<u32>,
    pub font_size: f64,
    pub font_family: String,
    pub background_color: String,
    pub border_config: BorderConfiguration,
}

/// Legend positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendPosition {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Custom(f64, f64),
}

/// Legend alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendAlignment {
    Start,
    Center,
    End,
}

/// Legend orientation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendOrientation {
    Horizontal,
    Vertical,
}

/// Border configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderConfiguration {
    pub enabled: bool,
    pub color: String,
    pub width: f64,
    pub radius: f64,
    pub style: LineStyle,
}

/// Color scheme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    pub scheme_type: ColorSchemeType,
    pub colors: Vec<String>,
    pub gradient_config: Option<GradientConfiguration>,
    pub accessibility_mode: AccessibilityMode,
}

/// Color scheme types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorSchemeType {
    Categorical,
    Sequential,
    Diverging,
    Qualitative,
    Custom,
}

/// Gradient configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConfiguration {
    pub direction: GradientDirection,
    pub stops: Vec<GradientStop>,
    pub interpolation: ColorInterpolation,
}

/// Gradient directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientDirection {
    Horizontal,
    Vertical,
    Radial,
    Angular(f64),
}

/// Gradient stops
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStop {
    pub offset: f64,
    pub color: String,
    pub opacity: f64,
}

/// Color interpolation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorInterpolation {
    RGB,
    HSV,
    Lab,
    LCH,
    XYZ,
}

/// Accessibility modes for color schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessibilityMode {
    None,
    Colorblind,
    HighContrast,
    Grayscale,
    Custom(String),
}

/// Animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfiguration {
    pub enabled: bool,
    pub duration: Duration,
    pub easing: EasingFunction,
    pub delay: Duration,
    pub loop_animation: bool,
    pub auto_play: bool,
    pub trigger: AnimationTrigger,
}

/// Easing functions for animations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
    Custom(String),
}

/// Animation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationTrigger {
    OnLoad,
    OnDataUpdate,
    OnHover,
    OnClick,
    Manual,
    Timer(Duration),
}

/// Interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionConfiguration {
    pub hover_enabled: bool,
    pub click_enabled: bool,
    pub selection_enabled: bool,
    pub zoom_enabled: bool,
    pub pan_enabled: bool,
    pub brush_enabled: bool,
    pub crossfilter_enabled: bool,
    pub drill_down_enabled: bool,
    pub custom_interactions: Vec<CustomInteraction>,
}

/// Custom interaction definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomInteraction {
    pub interaction_id: String,
    pub trigger: InteractionTrigger,
    pub action: InteractionAction,
    pub target: InteractionTarget,
    pub parameters: HashMap<String, String>,
}

/// Interaction triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionTrigger {
    Click,
    DoubleClick,
    Hover,
    MouseDown,
    MouseUp,
    KeyPress(String),
    Gesture(GestureType),
    Custom(String),
}

/// Gesture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GestureType {
    Swipe,
    Pinch,
    Rotate,
    Tap,
    LongPress,
}

/// Interaction actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionAction {
    Navigate(String),
    Filter(FilterAction),
    Highlight,
    Select,
    Zoom(ZoomAction),
    ShowTooltip,
    HideTooltip,
    UpdateData,
    Custom(String),
}

/// Filter actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterAction {
    pub filter_type: FilterType,
    pub field: String,
    pub value: String,
    pub operator: FilterOperator,
}

/// Filter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    Include,
    Exclude,
    Range,
    Text,
    Date,
    Custom(String),
}

/// Filter operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    StartsWith,
    EndsWith,
    In,
    NotIn,
    Between,
}

/// Zoom actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZoomAction {
    ZoomIn(f64),
    ZoomOut(f64),
    ZoomToFit,
    ZoomToSelection,
    ResetZoom,
}

/// Interaction targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionTarget {
    Self_,
    OtherWidget(String),
    Dashboard,
    Application,
    External(String),
}

/// Tooltip configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipConfiguration {
    pub enabled: bool,
    pub trigger: TooltipTrigger,
    pub position: TooltipPosition,
    pub content_template: String,
    pub styling: TooltipStyling,
    pub animation: TooltipAnimation,
    pub delay: Duration,
    pub duration: Option<Duration>,
}

/// Tooltip triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipTrigger {
    Hover,
    Click,
    Focus,
    Manual,
}

/// Tooltip positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipPosition {
    Auto,
    Top,
    Bottom,
    Left,
    Right,
    Custom(f64, f64),
}

/// Tooltip styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipStyling {
    pub background_color: String,
    pub text_color: String,
    pub font_size: f64,
    pub font_family: String,
    pub border: BorderConfiguration,
    pub shadow: ShadowConfiguration,
    pub padding: PaddingConfiguration,
    pub max_width: Option<f64>,
}

/// Shadow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfiguration {
    pub enabled: bool,
    pub offset_x: f64,
    pub offset_y: f64,
    pub blur_radius: f64,
    pub color: String,
    pub opacity: f64,
}

/// Padding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingConfiguration {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

/// Tooltip animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipAnimation {
    pub show_animation: AnimationConfig,
    pub hide_animation: AnimationConfig,
}

/// Animation config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    pub duration: Duration,
    pub easing: EasingFunction,
    pub direction: AnimationDirection,
}

/// Animation directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationDirection {
    FadeIn,
    FadeOut,
    SlideIn(SlideDirection),
    SlideOut(SlideDirection),
    ScaleIn,
    ScaleOut,
    Custom(String),
}

/// Slide directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlideDirection {
    Up,
    Down,
    Left,
    Right,
}

/// Zoom configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomConfiguration {
    pub enabled: bool,
    pub min_zoom: f64,
    pub max_zoom: f64,
    pub zoom_step: f64,
    pub zoom_controls: ZoomControls,
    pub mouse_wheel_zoom: bool,
    pub touch_zoom: bool,
    pub keyboard_zoom: bool,
}

/// Zoom controls configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomControls {
    pub show_controls: bool,
    pub position: ControlPosition,
    pub style: ControlStyle,
    pub custom_controls: Vec<CustomControl>,
}

/// Control positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Custom(f64, f64),
}

/// Control styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlStyle {
    Buttons,
    Slider,
    Wheel,
    Custom(String),
}

/// Custom control definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomControl {
    pub control_id: String,
    pub control_type: CustomControlType,
    pub label: String,
    pub action: ControlAction,
    pub styling: ControlStyling,
}

/// Custom control types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CustomControlType {
    Button,
    Slider,
    Dropdown,
    Checkbox,
    RadioButton,
    TextInput,
    DatePicker,
    ColorPicker,
    Custom(String),
}

/// Control actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlAction {
    ZoomIn,
    ZoomOut,
    Reset,
    Filter(FilterAction),
    Navigate(String),
    UpdateSetting(String, String),
    Custom(String),
}

/// Control styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlStyling {
    pub background_color: String,
    pub text_color: String,
    pub border: BorderConfiguration,
    pub font_size: f64,
    pub font_family: String,
    pub padding: PaddingConfiguration,
    pub margin: PaddingConfiguration,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfiguration {
    pub enabled: bool,
    pub formats: Vec<ExportFormat>,
    pub quality_settings: QualitySettings,
    pub size_settings: SizeSettings,
    pub watermark: Option<WatermarkConfiguration>,
    pub metadata_inclusion: MetadataInclusion,
}

/// Export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    PNG,
    JPEG,
    SVG,
    PDF,
    CSV,
    Excel,
    JSON,
    XML,
    Custom(String),
}

/// Quality settings for exports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    pub image_quality: f64,
    pub dpi: u32,
    pub compression: CompressionSettings,
}

/// Compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    pub enabled: bool,
    pub algorithm: CompressionAlgorithm,
    pub level: u32,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    GZIP,
    LZ4,
    Deflate,
    Custom(String),
}

/// Size settings for exports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeSettings {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub scale_factor: f64,
    pub maintain_aspect_ratio: bool,
}

/// Watermark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatermarkConfiguration {
    pub text: Option<String>,
    pub image_url: Option<String>,
    pub position: WatermarkPosition,
    pub opacity: f64,
    pub size: WatermarkSize,
    pub rotation: f64,
}

/// Watermark positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatermarkPosition {
    TopLeft,
    TopCenter,
    TopRight,
    MiddleLeft,
    MiddleCenter,
    MiddleRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
    Custom(f64, f64),
}

/// Watermark sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatermarkSize {
    Small,
    Medium,
    Large,
    Custom(f64, f64),
}

/// Metadata inclusion settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataInclusion {
    pub include_title: bool,
    pub include_description: bool,
    pub include_timestamp: bool,
    pub include_data_source: bool,
    pub include_filters: bool,
    pub custom_metadata: HashMap<String, String>,
}