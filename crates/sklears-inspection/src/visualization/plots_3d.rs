//! 3D plotting functionality
//!
//! This module provides comprehensive 3D visualization capabilities including
//! scatter plots, surface plots, and interactive 3D features.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};

use super::config_types::{AutoRotate3D, Camera3D, PlotConfig, ProjectionType, Surface3D};

// Re-export types from config_types to maintain API compatibility
pub use super::config_types::{Plot3D, Plot3DType};

/// Create comprehensive 3D plot visualization
///
/// Generates 3D visualization for complex feature interaction analysis with support
/// for scatter plots, surface plots, mesh plots, and other 3D visualization types.
/// Includes camera controls, color mapping, and interactive capabilities.
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
                "Color values array length must match coordinate arrays".to_string(),
            ));
        }
    }

    if let Some(sizes) = size_values {
        if sizes.len() != n_points {
            return Err(crate::SklearsError::InvalidInput(
                "Size values array length must match coordinate arrays".to_string(),
            ));
        }
    }

    if let Some(labels) = point_labels {
        if labels.len() != n_points {
            return Err(crate::SklearsError::InvalidInput(
                "Point labels array length must match coordinate arrays".to_string(),
            ));
        }
    }

    // Create default camera settings
    let camera_settings = Camera3D {
        eye: (10.0, 10.0, 10.0),
        center: (0.0, 0.0, 0.0),
        up: (0.0, 0.0, 1.0),
        projection: ProjectionType::Perspective,
        auto_rotate: AutoRotate3D {
            enabled: false,
            speed: 1.0,
            axis: (0.0, 0.0, 1.0),
            pause_on_interaction: true,
        },
    };

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
        camera_settings,
    })
}

/// Create 3D surface plot for feature interaction visualization
///
/// Generates 3D surface visualization for understanding complex feature interactions
/// and model behavior across continuous feature spaces.
///
/// # Arguments
///
/// * `x_grid` - X-axis grid values (2D array)
/// * `y_grid` - Y-axis grid values (2D array)
/// * `z_surface` - Z-surface values (2D array)
/// * `color_map` - Optional color mapping values (2D array)
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
                "Color map dimensions must match grid dimensions".to_string(),
            ));
        }
    }

    if !(0.0..=1.0).contains(&opacity) {
        return Err(crate::SklearsError::InvalidInput(
            "Opacity must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Create default camera settings
    let camera_settings = Camera3D {
        eye: (15.0, 15.0, 15.0),
        center: (0.0, 0.0, 0.0),
        up: (0.0, 0.0, 1.0),
        projection: ProjectionType::Perspective,
        auto_rotate: AutoRotate3D {
            enabled: false,
            speed: 1.0,
            axis: (0.0, 0.0, 1.0),
            pause_on_interaction: true,
        },
    };

    Ok(Surface3D {
        x_grid: x_grid.to_owned(),
        y_grid: y_grid.to_owned(),
        z_surface: z_surface.to_owned(),
        color_map: color_map.map(|c| c.to_owned()),
        opacity,
        contours: None,
        axis_labels,
        config: config.clone(),
        camera_settings,
    })
}

/// Create 3D SHAP visualization for three-feature interactions
///
/// Generates specialized 3D visualization for SHAP values showing how three
/// specific features interact in their contribution to model predictions.
///
/// # Arguments
///
/// * `shap_values` - SHAP values matrix (samples × features)
/// * `feature_indices` - Three feature indices to visualize [x, y, z]
/// * `axis_labels` - Labels for the three features
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
                "Feature index {} is out of bounds (max: {})",
                idx,
                n_features - 1
            )));
        }
    }

    // Extract SHAP values for the three features
    let x_values = shap_values.column(feature_indices[0]).to_owned();
    let y_values = shap_values.column(feature_indices[1]).to_owned();
    let z_values = shap_values.column(feature_indices[2]).to_owned();

    // Use absolute SHAP values for color mapping
    let color_values = Some(
        x_values
            .iter()
            .zip(y_values.iter())
            .zip(z_values.iter())
            .map(|((&x, &y), &z)| x.abs() + y.abs() + z.abs())
            .collect::<Array1<Float>>(),
    );

    let color_view = color_values.as_ref().map(|c| c.view());
    create_3d_plot(
        &x_values.view(),
        &y_values.view(),
        &z_values.view(),
        color_view.as_ref(),
        None,
        None,
        axis_labels,
        config,
        Plot3DType::Scatter,
    )
}
