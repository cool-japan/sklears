use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_types::*;


/// Visualization template configuration providing
/// standardized styling and layout patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationTemplate {
    template_id: String,
    chart_type: ChartType,
    style_definitions: StyleDefinitions,
    layout_config: ChartLayoutConfig,
    interaction_config: InteractionConfig,
    animation_config: AnimationConfig,
}

/// Comprehensive style definitions providing
/// professional theming and visual consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleDefinitions {
    /// Color palette for chart elements
    color_palette: ColorPalette,
    /// Typography configuration for text elements
    typography: Typography,
    /// Visual effects for enhanced appearance
    visual_effects: VisualEffects,
    /// Theme variants for different contexts
    theme_variants: HashMap<String, ThemeVariant>,
}

/// Color palette configuration providing
/// systematic color management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    /// Primary colors for main chart elements
    primary_colors: Vec<Color>,
    /// Secondary colors for supporting elements
    secondary_colors: Vec<Color>,
    /// Accent colors for highlights
    accent_colors: Vec<Color>,
    /// Neutral colors for backgrounds
    neutral_colors: Vec<Color>,
    /// Semantic colors for status indication
    semantic_colors: SemanticColors,
}

/// Color definition with multiple representation
/// formats for flexibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Color {
    /// Human-readable color name
    name: String,
    /// Hexadecimal color representation
    hex: String,
    /// RGB color representation
    rgb: (u8, u8, u8),
    /// HSL color representation
    hsl: (f64, f64, f64),
    /// Alpha transparency value
    alpha: f64,
}

/// Semantic colors for consistent status
/// and meaning representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticColors {
    /// Success state color
    success: Color,
    /// Warning state color
    warning: Color,
    /// Error state color
    error: Color,
    /// Information state color
    info: Color,
    /// Neutral state color
    neutral: Color,
}

/// Typography configuration for consistent
/// text rendering across visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Typography {
    /// Available font families
    font_families: Vec<FontFamily>,
    /// Font size scale system
    font_sizes: FontSizes,
    /// Font weight scale system
    font_weights: FontWeights,
    /// Line height scale system
    line_heights: LineHeights,
}

/// Font family configuration with
/// fallback and licensing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontFamily {
    /// Primary font name
    name: String,
    /// Fallback font list
    fallbacks: Vec<String>,
    /// Web font URL for loading
    web_font_url: Option<String>,
    /// Font license information
    license: String,
}

/// Font size scale system for
/// consistent typography hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSizes {
    /// Small text size
    small: f64,
    /// Medium text size
    medium: f64,
    /// Large text size
    large: f64,
    /// Extra large text size
    extra_large: f64,
    /// Custom text sizes
    custom_sizes: HashMap<String, f64>,
}

/// Font weight scale system for
/// typography emphasis control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontWeights {
    /// Light font weight
    light: u16,
    /// Normal font weight
    normal: u16,
    /// Bold font weight
    bold: u16,
    /// Extra bold font weight
    extra_bold: u16,
}

/// Line height scale system for
/// readable text layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineHeights {
    /// Tight line height for compact text
    tight: f64,
    /// Normal line height for readable text
    normal: f64,
    /// Loose line height for expanded text
    loose: f64,
}

/// Visual effects configuration for
/// enhanced chart appearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEffects {
    /// Shadow effects for depth
    shadows: Vec<Shadow>,
    /// Gradient effects for smooth transitions
    gradients: Vec<Gradient>,
    /// Border effects for definition
    borders: Vec<Border>,
    /// Transparency configuration
    transparency: TransparencyConfig,
}

/// Shadow effect configuration for
/// visual depth and hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shadow {
    /// Shadow name identifier
    name: String,
    /// Horizontal offset
    offset_x: f64,
    /// Vertical offset
    offset_y: f64,
    /// Blur radius for soft shadows
    blur_radius: f64,
    /// Spread radius for shadow expansion
    spread_radius: f64,
    /// Shadow color
    color: Color,
}

/// Gradient effect configuration for
/// smooth color transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    /// Gradient name identifier
    name: String,
    /// Type of gradient
    gradient_type: GradientType,
    /// Color stops for gradient definition
    stops: Vec<GradientStop>,
    /// Direction of gradient flow
    direction: GradientDirection,
}

/// Gradient type enumeration for
/// different gradient styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientType {
    /// Linear gradient
    Linear,
    /// Radial gradient
    Radial,
    /// Conic gradient
    Conic,
    /// Custom gradient implementation
    Custom(String),
}

/// Gradient stop configuration for
/// color position definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStop {
    /// Position along gradient (0.0 to 1.0)
    position: f64,
    /// Color at this position
    color: Color,
}

/// Gradient direction enumeration for
/// gradient orientation control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientDirection {
    /// Left to right gradient
    ToRight,
    /// Right to left gradient
    ToLeft,
    /// Bottom to top gradient
    ToTop,
    /// Top to bottom gradient
    ToBottom,
    /// Bottom-left to top-right gradient
    ToTopRight,
    /// Bottom-right to top-left gradient
    ToTopLeft,
    /// Top-left to bottom-right gradient
    ToBottomRight,
    /// Top-right to bottom-left gradient
    ToBottomLeft,
    /// Gradient at specific angle
    Angle(f64),
    /// Custom gradient direction
    Custom(String),
}

/// Border effect configuration for
/// element definition and separation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Border {
    /// Border name identifier
    name: String,
    /// Border width
    width: f64,
    /// Border style
    style: BorderStyle,
    /// Border color
    color: Color,
    /// Border radius for rounded corners
    radius: BorderRadius,
}

/// Border style enumeration for
/// different border appearances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BorderStyle {
    /// Solid border line
    Solid,
    /// Dashed border line
    Dashed,
    /// Dotted border line
    Dotted,
    /// Double border line
    Double,
    /// Grooved border effect
    Groove,
    /// Ridged border effect
    Ridge,
    /// Inset border effect
    Inset,
    /// Outset border effect
    Outset,
    /// Custom border style
    Custom(String),
}

/// Border radius configuration for
/// rounded corner control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderRadius {
    /// Top-left corner radius
    top_left: f64,
    /// Top-right corner radius
    top_right: f64,
    /// Bottom-right corner radius
    bottom_right: f64,
    /// Bottom-left corner radius
    bottom_left: f64,
}

/// Transparency configuration for
/// alpha blending control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransparencyConfig {
    /// Default opacity level
    default_opacity: f64,
    /// Hover state opacity
    hover_opacity: f64,
    /// Disabled state opacity
    disabled_opacity: f64,
    /// Overlay opacity
    overlay_opacity: f64,
}
