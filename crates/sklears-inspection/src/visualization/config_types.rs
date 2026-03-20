//! Configuration Types and Data Structures for Visualization Framework
//!
//! This module contains all configuration structures, enums, and data types used
//! throughout the visualization system. It provides comprehensive configuration
//! options for plots, mobile responsiveness, 3D visualization, and color schemes.
//!
//! ## Key Components
//!
//! - **Plot Configuration**: Basic plot settings and appearance options
//! - **Mobile Configuration**: Responsive design settings for mobile devices
//! - **3D Configuration**: Camera, projection, and animation settings for 3D plots
//! - **Data Structures**: Plot data containers for different visualization types

use crate::Float;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for visualization plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotConfig {
    /// Plot title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Plot width in pixels
    pub width: usize,
    /// Plot height in pixels
    pub height: usize,
    /// Color scheme for the plot
    pub color_scheme: ColorScheme,
    /// Whether to show grid
    pub show_grid: bool,
    /// Whether plot is interactive
    pub interactive: bool,
    /// Mobile-responsive configuration
    pub mobile_config: MobileConfig,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: "Visualization".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            width: 800,
            height: 600,
            color_scheme: ColorScheme::Default,
            show_grid: true,
            interactive: true,
            mobile_config: MobileConfig::default(),
        }
    }
}

/// Color schemes for visualizations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Default
    Default,
    /// Viridis
    Viridis,
    /// Plasma
    Plasma,
    /// Magma
    Magma,
    /// Inferno
    Inferno,
    /// Blues
    Blues,
    /// Reds
    Reds,
    /// Greens
    Greens,
}

/// Mobile-responsive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileConfig {
    /// Enable mobile-responsive behavior
    pub enabled: bool,
    /// Mobile breakpoint in pixels
    pub mobile_breakpoint: usize,
    /// Tablet breakpoint in pixels
    pub tablet_breakpoint: usize,
    /// Mobile plot width (percentage or pixels)
    pub mobile_width: String,
    /// Mobile plot height
    pub mobile_height: usize,
    /// Whether to stack plots vertically on mobile
    pub stack_on_mobile: bool,
    /// Whether to simplify plots on mobile
    pub simplify_on_mobile: bool,
    /// Touch-friendly controls
    pub touch_friendly: bool,
    /// Minimum font size for mobile
    pub min_font_size: usize,
}

impl Default for MobileConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mobile_breakpoint: 768,
            tablet_breakpoint: 1024,
            mobile_width: "100%".to_string(),
            mobile_height: 300,
            stack_on_mobile: true,
            simplify_on_mobile: true,
            touch_friendly: true,
            min_font_size: 14,
        }
    }
}

/// Interactive feature importance plot data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportancePlot {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Importance values
    pub importance_values: Vec<Float>,
    /// Standard deviations (if available)
    pub std_values: Option<Vec<Float>>,
    /// Plot configuration
    pub config: PlotConfig,
    /// Plot type
    pub plot_type: FeatureImportanceType,
}

/// Types of feature importance plots
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeatureImportanceType {
    /// Bar
    Bar,
    /// Horizontal
    Horizontal,
    /// Radial
    Radial,
    /// TreeMap
    TreeMap,
}

/// SHAP visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapPlot {
    /// SHAP values for each feature and instance
    pub shap_values: Array2<Float>,
    /// Feature values
    pub feature_values: Array2<Float>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Instance names/indices
    pub instance_names: Vec<String>,
    /// Plot configuration
    pub config: PlotConfig,
    /// SHAP plot type
    pub plot_type: ShapPlotType,
}

/// Types of SHAP plots
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShapPlotType {
    /// Waterfall
    Waterfall,
    /// ForceLayout
    ForceLayout,
    /// Summary
    Summary,
    /// Dependence
    Dependence,
    /// Beeswarm
    Beeswarm,
    /// DecisionPlot
    DecisionPlot,
    /// 3D scatter plot for feature interactions
    Scatter3D,
    /// 3D surface plot for feature dependencies
    Surface3D,
}

/// 3D visualization data for complex feature interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plot3D {
    /// X-axis values
    pub x_values: Array1<Float>,
    /// Y-axis values
    pub y_values: Array1<Float>,
    /// Z-axis values
    pub z_values: Array1<Float>,
    /// Optional color mapping values
    pub color_values: Option<Array1<Float>>,
    /// Size values for scatter plots
    pub size_values: Option<Array1<Float>>,
    /// Labels for each point
    pub point_labels: Option<Vec<String>>,
    /// Feature names for axes
    pub axis_labels: (String, String, String),
    /// Plot configuration
    pub config: PlotConfig,
    /// 3D plot type
    pub plot_type: Plot3DType,
    /// Camera settings
    pub camera_settings: Camera3D,
}

/// Types of 3D plots
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Plot3DType {
    /// 3D scatter plot
    Scatter,
    /// 3D surface plot
    Surface,
    /// 3D mesh plot
    Mesh,
    /// 3D contour plot
    Contour,
    /// 3D volume plot
    Volume,
    /// 3D network graph
    Network,
}

/// 3D camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera3D {
    /// Camera position
    pub eye: (Float, Float, Float),
    /// Camera center (look-at point)
    pub center: (Float, Float, Float),
    /// Camera up vector
    pub up: (Float, Float, Float),
    /// Projection type
    pub projection: ProjectionType,
    /// Auto-rotation settings
    pub auto_rotate: AutoRotate3D,
}

/// 3D projection types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProjectionType {
    /// Perspective
    Perspective,
    /// Orthographic
    Orthographic,
}

/// Auto-rotation settings for 3D plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoRotate3D {
    /// Enable auto-rotation
    pub enabled: bool,
    /// Rotation speed (degrees per second)
    pub speed: Float,
    /// Rotation axis
    pub axis: (Float, Float, Float),
    /// Pause on interaction
    pub pause_on_interaction: bool,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            eye: (1.25, 1.25, 1.25),
            center: (0.0, 0.0, 0.0),
            up: (0.0, 0.0, 1.0),
            projection: ProjectionType::Perspective,
            auto_rotate: AutoRotate3D::default(),
        }
    }
}

impl Default for AutoRotate3D {
    fn default() -> Self {
        Self {
            enabled: false,
            speed: 30.0,           // 30 degrees per second
            axis: (0.0, 0.0, 1.0), // Rotate around Z-axis
            pause_on_interaction: true,
        }
    }
}

/// 3D surface data for feature interaction visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Surface3D {
    /// X-axis grid values
    pub x_grid: Array2<Float>,
    /// Y-axis grid values
    pub y_grid: Array2<Float>,
    /// Z-surface values
    pub z_surface: Array2<Float>,
    /// Color mapping for surface
    pub color_map: Option<Array2<Float>>,
    /// Surface opacity
    pub opacity: Float,
    /// Contour lines
    pub contours: Option<ContourLines3D>,
    /// Axis labels
    pub axis_labels: (String, String, String),
    /// Plot configuration
    pub config: PlotConfig,
    /// Camera settings
    pub camera_settings: Camera3D,
}

/// 3D contour line configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourLines3D {
    /// Show contour lines on X plane
    pub show_x: bool,
    /// Show contour lines on Y plane
    pub show_y: bool,
    /// Show contour lines on Z plane
    pub show_z: bool,
    /// Number of contour levels
    pub levels: usize,
    /// Contour line color
    pub color: String,
    /// Contour line width
    pub width: Float,
}

impl Default for ContourLines3D {
    fn default() -> Self {
        Self {
            show_x: false,
            show_y: false,
            show_z: true,
            levels: 10,
            color: "black".to_string(),
            width: 1.0,
        }
    }
}

/// Partial dependence plot data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialDependencePlot {
    /// Feature values for x-axis
    pub feature_values: Array1<Float>,
    /// Partial dependence values
    pub pd_values: Array1<Float>,
    /// ICE curves (if available)
    pub ice_curves: Option<Array2<Float>>,
    /// Feature name
    pub feature_name: String,
    /// Plot configuration
    pub config: PlotConfig,
    /// Whether to show individual curves
    pub show_ice: bool,
}

/// Comparative visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativePlot {
    /// Data for different models/methods
    pub model_data: HashMap<String, Array2<Float>>,
    /// Labels for comparison
    pub labels: Vec<String>,
    /// Plot configuration
    pub config: PlotConfig,
    /// Comparison type
    pub comparison_type: ComparisonType,
}

/// Types of comparative plots
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonType {
    /// SideBySide
    SideBySide,
    /// Overlay
    Overlay,
    /// Difference
    Difference,
    /// Ratio
    Ratio,
    /// Heatmap
    Heatmap,
    /// Statistical
    Statistical,
}

/// 3D animation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Animation3D {
    /// Enable animation
    pub enabled: bool,
    /// Animation duration in milliseconds
    pub duration: u64,
    /// Animation easing type
    pub easing: EasingType,
    /// Loop animation
    pub loop_animation: bool,
    /// Animation frames per second
    pub fps: u32,
}

/// Animation easing types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EasingType {
    /// Linear
    Linear,
    /// EaseIn
    EaseIn,
    /// EaseOut
    EaseOut,
    /// EaseInOut
    EaseInOut,
    /// Bounce
    Bounce,
    /// Elastic
    Elastic,
}

impl Default for Animation3D {
    fn default() -> Self {
        Self {
            enabled: false,
            duration: 2000,
            easing: EasingType::EaseInOut,
            loop_animation: false,
            fps: 30,
        }
    }
}

/// Plot interaction handler for user interactions
#[derive(Debug, Clone)]
pub struct PlotInteractionHandler {
    /// Enable zoom functionality
    pub zoom_enabled: bool,
    /// Enable pan functionality
    pub pan_enabled: bool,
    /// Enable selection functionality
    pub selection_enabled: bool,
    /// Enable hover tooltips
    pub hover_enabled: bool,
    /// Custom interaction callbacks
    pub custom_handlers: HashMap<String, String>,
}

impl Default for PlotInteractionHandler {
    fn default() -> Self {
        Self {
            zoom_enabled: true,
            pan_enabled: true,
            selection_enabled: true,
            hover_enabled: true,
            custom_handlers: HashMap::new(),
        }
    }
}

/// Configuration builder for fluent API
pub struct PlotConfigBuilder {
    config: PlotConfig,
}

impl PlotConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: PlotConfig::default(),
        }
    }

    pub fn title(mut self, title: &str) -> Self {
        self.config.title = title.to_string();
        self
    }

    pub fn x_label(mut self, label: &str) -> Self {
        self.config.x_label = label.to_string();
        self
    }

    pub fn y_label(mut self, label: &str) -> Self {
        self.config.y_label = label.to_string();
        self
    }

    pub fn size(mut self, width: usize, height: usize) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    pub fn color_scheme(mut self, scheme: ColorScheme) -> Self {
        self.config.color_scheme = scheme;
        self
    }

    pub fn grid(mut self, show: bool) -> Self {
        self.config.show_grid = show;
        self
    }

    pub fn interactive(mut self, enabled: bool) -> Self {
        self.config.interactive = enabled;
        self
    }

    pub fn mobile_config(mut self, mobile: MobileConfig) -> Self {
        self.config.mobile_config = mobile;
        self
    }

    pub fn build(self) -> PlotConfig {
        self.config
    }
}

impl Default for PlotConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.title, "Visualization");
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert!(config.show_grid);
        assert!(config.interactive);
    }

    #[test]
    fn test_mobile_config_default() {
        let mobile = MobileConfig::default();
        assert!(mobile.enabled);
        assert_eq!(mobile.mobile_breakpoint, 768);
        assert_eq!(mobile.tablet_breakpoint, 1024);
        assert_eq!(mobile.mobile_width, "100%");
        assert!(mobile.stack_on_mobile);
    }

    #[test]
    fn test_camera3d_default() {
        let camera = Camera3D::default();
        assert_eq!(camera.eye, (1.25, 1.25, 1.25));
        assert_eq!(camera.center, (0.0, 0.0, 0.0));
        assert_eq!(camera.up, (0.0, 0.0, 1.0));
        assert!(!camera.auto_rotate.enabled);
    }

    #[test]
    fn test_plot_config_builder() {
        let config = PlotConfigBuilder::new()
            .title("Test Plot")
            .x_label("X Axis")
            .y_label("Y Axis")
            .size(1000, 800)
            .color_scheme(ColorScheme::Viridis)
            .grid(false)
            .interactive(true)
            .build();

        assert_eq!(config.title, "Test Plot");
        assert_eq!(config.x_label, "X Axis");
        assert_eq!(config.y_label, "Y Axis");
        assert_eq!(config.width, 1000);
        assert_eq!(config.height, 800);
        assert!(!config.show_grid);
        assert!(config.interactive);
    }

    #[test]
    fn test_animation3d_default() {
        let animation = Animation3D::default();
        assert!(!animation.enabled);
        assert_eq!(animation.duration, 2000);
        assert_eq!(animation.fps, 30);
        assert!(!animation.loop_animation);
    }

    #[test]
    fn test_contour_lines_3d_default() {
        let contours = ContourLines3D::default();
        assert!(!contours.show_x);
        assert!(!contours.show_y);
        assert!(contours.show_z);
        assert_eq!(contours.levels, 10);
        assert_eq!(contours.color, "black");
        assert_eq!(contours.width, 1.0);
    }
}
