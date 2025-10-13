//! Chart factory and creation system
//!
//! This module provides comprehensive chart creation and factory capabilities including:
//! - Chart type definitions and template system
//! - Chart building and configuration management
//! - Data handling and transformation pipelines
//! - Chart layout and styling configuration
//! - Animation and interaction setup

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::chart_types::{ChartType, ChartData, DataFormat, InteractiveFeature, ExportFormat};

/// Chart factory for creating different chart types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartFactory {
    /// Chart type registry
    pub chart_types: HashMap<String, ChartTypeDefinition>,
    /// Chart templates
    pub chart_templates: HashMap<String, ChartTemplate>,
    /// Factory configuration
    pub configuration: ChartFactoryConfig,
}

/// Chart type definition with capabilities and requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartTypeDefinition {
    /// Chart type identifier
    pub type_id: String,
    /// Chart type name
    pub type_name: String,
    /// Supported data formats
    pub supported_data_formats: Vec<DataFormat>,
    /// Required properties
    pub required_properties: Vec<String>,
    /// Optional properties
    pub optional_properties: Vec<String>,
    /// Default configuration
    pub default_config: HashMap<String, String>,
}

/// Chart template for predefined configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartTemplate {
    /// Template identifier
    pub template_id: String,
    /// Template name
    pub template_name: String,
    /// Chart type
    pub chart_type: ChartType,
    /// Template configuration
    pub configuration: HashMap<String, String>,
    /// Template style
    pub style: HashMap<String, String>,
    /// Template description
    pub description: String,
}

/// Comprehensive chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chart {
    /// Chart identifier
    pub chart_id: String,
    /// Chart type
    pub chart_type: ChartType,
    /// Chart data
    pub data: ChartData,
    /// Chart configuration
    pub configuration: ChartConfiguration,
    /// Chart metadata
    pub metadata: ChartMetadata,
}

/// Chart configuration encompassing all aspects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfiguration {
    /// Layout configuration
    pub layout: ChartLayoutConfiguration,
    /// Style configuration
    pub style: ChartStyleConfiguration,
    /// Interaction configuration
    pub interaction: ChartInteractionConfiguration,
    /// Animation configuration
    pub animation: ChartAnimationConfiguration,
}

/// Chart layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartLayoutConfiguration {
    /// Chart dimensions
    pub dimensions: ChartDimensions,
    /// Chart margins
    pub margins: ChartMargins,
    /// Chart padding
    pub padding: ChartPadding,
    /// Chart alignment
    pub alignment: ChartAlignment,
}

/// Chart dimensions specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDimensions {
    /// Chart width
    pub width: u32,
    /// Chart height
    pub height: u32,
    /// Aspect ratio lock
    pub aspect_ratio_locked: bool,
    /// Responsive sizing
    pub responsive: bool,
}

/// Chart margins configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartMargins {
    /// Top margin
    pub top: u32,
    /// Right margin
    pub right: u32,
    /// Bottom margin
    pub bottom: u32,
    /// Left margin
    pub left: u32,
}

/// Chart padding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartPadding {
    /// Top padding
    pub top: u32,
    /// Right padding
    pub right: u32,
    /// Bottom padding
    pub bottom: u32,
    /// Left padding
    pub left: u32,
}

/// Chart alignment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAlignment {
    /// Horizontal alignment
    pub horizontal: HorizontalAlignment,
    /// Vertical alignment
    pub vertical: VerticalAlignment,
}

/// Horizontal alignment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HorizontalAlignment {
    /// Left alignment
    Left,
    /// Center alignment
    Center,
    /// Right alignment
    Right,
}

/// Vertical alignment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerticalAlignment {
    /// Top alignment
    Top,
    /// Middle alignment
    Middle,
    /// Bottom alignment
    Bottom,
}

/// Chart style configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartStyleConfiguration {
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Font configuration
    pub fonts: FontConfiguration,
    /// Theme configuration
    pub theme: ThemeConfiguration,
    /// Border configuration
    pub borders: BorderConfiguration,
}

/// Color scheme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    /// Primary colors
    pub primary_colors: Vec<String>,
    /// Secondary colors
    pub secondary_colors: Vec<String>,
    /// Background colors
    pub background_colors: Vec<String>,
    /// Accent colors
    pub accent_colors: Vec<String>,
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfiguration {
    /// Default font family
    pub default_family: String,
    /// Title font configuration
    pub title_font: FontSizeConfiguration,
    /// Label font configuration
    pub label_font: FontSizeConfiguration,
    /// Body font configuration
    pub body_font: FontSizeConfiguration,
}

/// Font size configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSizeConfiguration {
    /// Font family
    pub family: String,
    /// Font size
    pub size: u32,
    /// Font weight
    pub weight: FontWeightConfiguration,
    /// Font style
    pub style: String,
}

/// Font weight configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeightConfiguration {
    /// Normal weight
    Normal,
    /// Bold weight
    Bold,
    /// Light weight
    Light,
    /// Custom weight
    Custom(u32),
}

/// Theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfiguration {
    /// Theme name
    pub name: String,
    /// Dark mode support
    pub dark_mode: bool,
    /// Custom properties
    pub custom_properties: HashMap<String, String>,
}

/// Border configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderConfiguration {
    /// Border style
    pub style: BorderStyle,
    /// Border width
    pub width: u32,
    /// Border color
    pub color: String,
    /// Border radius
    pub radius: u32,
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
    /// No border
    None,
}

/// Chart interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartInteractionConfiguration {
    /// Interactive features
    pub features: Vec<InteractiveFeature>,
    /// Interaction types
    pub interaction_types: Vec<InteractionType>,
    /// Interaction sensitivity
    pub sensitivity: InteractionSensitivity,
    /// Gesture configuration
    pub gestures: GestureConfiguration,
    /// Keyboard shortcuts
    pub keyboard_shortcuts: KeyboardShortcuts,
}

/// Interaction type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Click interaction
    Click,
    /// Hover interaction
    Hover,
    /// Drag interaction
    Drag,
    /// Scroll interaction
    Scroll,
    /// Touch interaction
    Touch,
    /// Custom interaction
    Custom(String),
}

/// Interaction sensitivity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionSensitivity {
    /// Click sensitivity
    pub click_sensitivity: f64,
    /// Hover sensitivity
    pub hover_sensitivity: f64,
    /// Drag sensitivity
    pub drag_sensitivity: f64,
}

/// Gesture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureConfiguration {
    /// Enabled gestures
    pub enabled_gestures: Vec<String>,
    /// Custom gestures
    pub custom_gestures: Vec<CustomGesture>,
}

/// Custom gesture definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomGesture {
    /// Gesture name
    pub name: String,
    /// Gesture pattern
    pub pattern: String,
    /// Gesture action
    pub action: String,
}

/// Keyboard shortcuts configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyboardShortcuts {
    /// Enabled shortcuts
    pub enabled: bool,
    /// Custom shortcuts
    pub custom_shortcuts: Vec<CustomShortcut>,
}

/// Custom keyboard shortcut
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomShortcut {
    /// Key combination
    pub key_combination: String,
    /// Action
    pub action: String,
    /// Description
    pub description: String,
}

/// Chart animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAnimationConfiguration {
    /// Animation enabled
    pub enabled: bool,
    /// Animation duration
    pub duration: Duration,
    /// Animation easing
    pub easing: AnimationEasing,
    /// Animation timing
    pub timing: AnimationTiming,
    /// Loop animation
    pub loop_animation: bool,
}

/// Animation easing types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationEasing {
    /// Linear easing
    Linear,
    /// Ease in
    EaseIn,
    /// Ease out
    EaseOut,
    /// Ease in out
    EaseInOut,
    /// Custom easing
    Custom(String),
}

/// Animation timing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationTiming {
    /// Start delay
    pub start_delay: Duration,
    /// End delay
    pub end_delay: Duration,
    /// Frame rate
    pub frame_rate: u32,
}

/// Chart metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartMetadata {
    /// Chart title
    pub title: String,
    /// Chart description
    pub description: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Author information
    pub author: String,
    /// Version
    pub version: String,
    /// Tags
    pub tags: Vec<String>,
}

/// Chart validation mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartValidationMode {
    /// Strict validation
    Strict,
    /// Lenient validation
    Lenient,
    /// Custom validation
    Custom(String),
}

/// Chart factory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartFactoryConfig {
    /// Default chart type
    pub default_chart_type: ChartType,
    /// Chart validation mode
    pub validation_mode: ChartValidationMode,
    /// Performance settings
    pub performance_settings: FactoryPerformanceSettings,
    /// Caching enabled
    pub caching_enabled: bool,
}

/// Factory performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryPerformanceSettings {
    /// Parallel chart creation
    pub parallel_creation: bool,
    /// Cache chart templates
    pub cache_templates: bool,
    /// Optimize for memory
    pub memory_optimization: bool,
}

impl Default for ChartFactory {
    fn default() -> Self {
        Self {
            chart_types: HashMap::new(),
            chart_templates: HashMap::new(),
            configuration: ChartFactoryConfig::default(),
        }
    }
}

impl Default for ChartFactoryConfig {
    fn default() -> Self {
        Self {
            default_chart_type: ChartType::Line,
            validation_mode: ChartValidationMode::Strict,
            performance_settings: FactoryPerformanceSettings::default(),
            caching_enabled: true,
        }
    }
}

impl Default for FactoryPerformanceSettings {
    fn default() -> Self {
        Self {
            parallel_creation: true,
            cache_templates: true,
            memory_optimization: true,
        }
    }
}