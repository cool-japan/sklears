//! 3D plotting functionality
//!
//! This module provides comprehensive 3D visualization capabilities including
//! scatter plots, surface plots, and interactive 3D features.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::config_types::{
    Camera3D, ContourLines3D, PlotConfig, Plot3D, Plot3DType,
    Surface3D, Animation3D, EasingType,
};

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

/// Animation settings for 3D plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Animation3D {
    /// Enable animation
    pub enabled: bool,
    /// Animation duration in seconds
    pub duration: Float,
    /// Animation easing type
    pub easing: EasingType,
    /// Loop animation
    pub loop_animation: bool,
    /// Auto-play animation
    pub auto_play: bool,
}

/// Animation easing types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
            duration: 2.0,
            easing: EasingType::EaseInOut,
            loop_animation: false,
            auto_play: false,
        }
    }
}

/// Create 3D plot for feature interactions
///
/// # Arguments
///
/// * `x_values` - X-axis coordinate values
/// * `y_values` - Y-axis coordinate values
/// * `z_values` - Z-axis coordinate values
/// * `color_values` - Optional color mapping values
/// * `size_values` - Optional size values for scatter plots
/// * `point_labels` - Optional labels for each point
/// * `axis_labels` - Labels for X, Y, Z axes
/// * `config` - Plot configuration
/// * `plot_type` - Type of 3D plot
///
/// # Returns
///
/// Result containing 3D plot data
pub fn create_3d_plot(
    x_values: &ArrayView1<Float>,
    y_values: &ArrayView1<Float>,
    z_values: &ArrayView1<Float>,
    color_values: Option<&ArrayView1<Float>>,
    size_values: Option<&ArrayView1<Float>>,
    point_labels: Option<&[String]>,
    axis_labels: (String, String, String),
    config: &PlotConfig,
    plot_type: Plot3DType,
) -> SklResult<Plot3D> {
    let n_points = x_values.len();

    if y_values.len() != n_points || z_values.len() != n_points {
        return Err(crate::SklearsError::InvalidInput(
            "All coordinate arrays must have the same length".to_string(),
        ));
    }

    if let Some(colors) = color_values {
        if colors.len() != n_points {
            return Err(crate::SklearsError::InvalidInput(
                "Color values length does not match coordinate arrays".to_string(),
            ));
        }
    }

    if let Some(sizes) = size_values {
        if sizes.len() != n_points {
            return Err(crate::SklearsError::InvalidInput(
                "Size values length does not match coordinate arrays".to_string(),
            ));
        }
    }

    if let Some(labels) = point_labels {
        if labels.len() != n_points {
            return Err(crate::SklearsError::InvalidInput(
                "Point labels length does not match coordinate arrays".to_string(),
            ));
        }
    }

    Ok(Plot3D {
        x_values: x_values.to_owned(),
        y_values: y_values.to_owned(),
        z_values: z_values.to_owned(),
        color_values: color_values.map(|c| c.to_owned()),
        size_values: size_values.map(|s| s.to_owned()),
        point_labels: point_labels.map(|l| l.to_vec()),
        axis_labels,
        config: config.clone(),
        plot_type,
        camera_settings: Camera3D::default(),
    })
}

/// Create 3D surface plot for feature dependencies
///
/// # Arguments
///
/// * `x_grid` - X-axis grid values
/// * `y_grid` - Y-axis grid values
/// * `z_surface` - Z-surface values
/// * `color_map` - Optional color mapping for surface
/// * `axis_labels` - Labels for X, Y, Z axes
/// * `config` - Plot configuration
/// * `opacity` - Surface opacity (0.0 to 1.0)
///
/// # Returns
///
/// Result containing 3D surface plot data
pub fn create_3d_surface_plot(
    x_grid: &ArrayView2<Float>,
    y_grid: &ArrayView2<Float>,
    z_surface: &ArrayView2<Float>,
    color_map: Option<&ArrayView2<Float>>,
    axis_labels: (String, String, String),
    config: &PlotConfig,
    opacity: Float,
) -> SklResult<Surface3D> {
    let grid_shape = x_grid.dim();

    if y_grid.dim() != grid_shape || z_surface.dim() != grid_shape {
        return Err(crate::SklearsError::InvalidInput(
            "All grid arrays must have the same dimensions".to_string(),
        ));
    }

    if let Some(colors) = color_map {
        if colors.dim() != grid_shape {
            return Err(crate::SklearsError::InvalidInput(
                "Color map dimensions do not match grid dimensions".to_string(),
            ));
        }
    }

    if !(0.0..=1.0).contains(&opacity) {
        return Err(crate::SklearsError::InvalidInput(
            "Opacity must be between 0.0 and 1.0".to_string(),
        ));
    }

    Ok(Surface3D {
        x_grid: x_grid.to_owned(),
        y_grid: y_grid.to_owned(),
        z_surface: z_surface.to_owned(),
        color_map: color_map.map(|c| c.to_owned()),
        opacity,
        contours: None,
        axis_labels,
        config: config.clone(),
        camera_settings: Camera3D::default(),
    })
}

/// Create 3D SHAP plot for feature interactions
///
/// # Arguments
///
/// * `shap_values` - SHAP values array
/// * `feature_indices` - Indices of features to plot (must select exactly 3)
/// * `axis_labels` - Labels for X, Y, Z axes
/// * `config` - Plot configuration
///
/// # Returns
///
/// Result containing 3D SHAP plot data
pub fn create_3d_shap_plot(
    shap_values: &ArrayView2<Float>,
    feature_indices: &[usize; 3],
    axis_labels: (String, String, String),
    config: &PlotConfig,
) -> SklResult<Plot3D> {
    let n_samples = shap_values.nrows();
    let n_features = shap_values.ncols();

    // Validate feature indices
    for &idx in feature_indices {
        if idx >= n_features {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Feature index {} is out of range (max: {})",
                idx,
                n_features - 1
            )));
        }
    }

    // Extract the three features for 3D plotting
    let x_values = shap_values.column(feature_indices[0]).to_owned();
    let y_values = shap_values.column(feature_indices[1]).to_owned();
    let z_values = shap_values.column(feature_indices[2]).to_owned();

    Ok(Plot3D {
        x_values,
        y_values,
        z_values,
        color_values: None,
        size_values: None,
        point_labels: None,
        axis_labels,
        config: config.clone(),
        plot_type: Plot3DType::Scatter,
        camera_settings: Camera3D::default(),
    })
}
