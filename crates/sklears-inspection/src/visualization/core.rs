//! Core types and configuration for visualizations
//!
//! This module contains fundamental types used across all visualization modules,
//! including plot configuration, color schemes, and mobile responsiveness settings.

use crate::Float;
use serde::{Deserialize, Serialize};

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

/// Types of comparative plots
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ComparisonType {
    /// SideBySide
    SideBySide,
    /// Overlay
    Overlay,
    /// Difference
    Difference,
    /// Ratio
    Ratio,
    /// Ranking
    Ranking,
    /// Heatmap
    Heatmap,
    /// Statistical
    Statistical,
}

/// Device types for responsive optimization
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Desktop
    Desktop,
    /// Tablet
    Tablet,
    /// Mobile
    Mobile,
}
