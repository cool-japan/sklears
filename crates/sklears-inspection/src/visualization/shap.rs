//! SHAP (SHapley Additive exPlanations) visualization functionality
//!
//! This module provides various SHAP visualization types including waterfall plots,
//! force layouts, summary plots, and 3D SHAP interaction visualizations.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use super::config_types::{Camera3D, PlotConfig};
use super::plots_3d::{Plot3D, Plot3DType};

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

/// Create interactive SHAP plot
///
/// # Arguments
///
/// * `shap_values` - SHAP values matrix (instances x features)
/// * `feature_values` - Feature values matrix
/// * `feature_names` - Optional feature names
/// * `instance_names` - Optional instance names
/// * `config` - Plot configuration
/// * `plot_type` - Type of SHAP plot
///
/// # Returns
///
/// Result containing SHAP plot data
pub fn create_shap_plot(
    shap_values: &ArrayView2<Float>,
    feature_values: &ArrayView2<Float>,
    feature_names: Option<&[String]>,
    instance_names: Option<&[String]>,
    config: &PlotConfig,
    plot_type: ShapPlotType,
) -> SklResult<ShapPlot> {
    let (n_instances, n_features) = shap_values.dim();

    if feature_values.dim() != (n_instances, n_features) {
        return Err(crate::SklearsError::InvalidInput(
            "SHAP values and feature values dimensions do not match".to_string(),
        ));
    }

    let feature_names = if let Some(names) = feature_names {
        if names.len() != n_features {
            return Err(crate::SklearsError::InvalidInput(
                "Feature names length does not match number of features".to_string(),
            ));
        }
        names.to_vec()
    } else {
        (0..n_features).map(|i| format!("Feature_{}", i)).collect()
    };

    let instance_names = if let Some(names) = instance_names {
        if names.len() != n_instances {
            return Err(crate::SklearsError::InvalidInput(
                "Instance names length does not match number of instances".to_string(),
            ));
        }
        names.to_vec()
    } else {
        (0..n_instances)
            .map(|i| format!("Instance_{}", i))
            .collect()
    };

    Ok(ShapPlot {
        shap_values: shap_values.to_owned(),
        feature_values: feature_values.to_owned(),
        feature_names,
        instance_names,
        config: config.clone(),
        plot_type,
    })
}

/// Create 3D SHAP interaction plot
///
/// # Arguments
///
/// * `shap_values` - SHAP values matrix (instances x features)
/// * `feature_values` - Feature values matrix
/// * `feature_indices` - Indices of three features to plot (x, y, z)
/// * `config` - Plot configuration
///
/// # Returns
///
/// Result containing 3D SHAP plot data
pub fn create_3d_shap_plot(
    shap_values: &ArrayView2<Float>,
    feature_values: &ArrayView2<Float>,
    feature_indices: (usize, usize, usize),
    config: &PlotConfig,
) -> SklResult<Plot3D> {
    let (n_instances, n_features) = shap_values.dim();

    if feature_values.dim() != (n_instances, n_features) {
        return Err(crate::SklearsError::InvalidInput(
            "SHAP values and feature values dimensions do not match".to_string(),
        ));
    }

    let (x_idx, y_idx, z_idx) = feature_indices;
    if x_idx >= n_features || y_idx >= n_features || z_idx >= n_features {
        return Err(crate::SklearsError::InvalidInput(
            "Feature indices are out of bounds".to_string(),
        ));
    }

    let x_values = feature_values.column(x_idx).to_owned();
    let y_values = feature_values.column(y_idx).to_owned();
    let z_values = feature_values.column(z_idx).to_owned();
    let color_values = Some(shap_values.column(z_idx).to_owned());

    let axis_labels = (
        format!("Feature_{}", x_idx),
        format!("Feature_{}", y_idx),
        format!("Feature_{}", z_idx),
    );

    Ok(Plot3D {
        x_values,
        y_values,
        z_values,
        color_values,
        size_values: None,
        point_labels: None,
        axis_labels,
        config: config.clone(),
        plot_type: Plot3DType::Scatter,
        camera_settings: Camera3D::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_shap_plot_creation() {
        let shap_values = array![[0.1, 0.2, -0.1], [0.3, -0.1, 0.2]];
        let feature_values = array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]];
        let config = PlotConfig::default();

        let plot = create_shap_plot(
            &shap_values.view(),
            &feature_values.view(),
            None,
            None,
            &config,
            ShapPlotType::Summary,
        )
        .unwrap();

        assert_eq!(plot.shap_values.shape(), &[2, 3]);
        assert_eq!(plot.feature_names.len(), 3);
        assert_eq!(plot.instance_names.len(), 2);
    }

    #[test]
    fn test_shap_plot_dimension_mismatch() {
        let shap_values = array![[0.1, 0.2], [0.3, -0.1]];
        let feature_values = array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]];
        let config = PlotConfig::default();

        let result = create_shap_plot(
            &shap_values.view(),
            &feature_values.view(),
            None,
            None,
            &config,
            ShapPlotType::Summary,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_3d_shap_plot_creation() {
        let shap_values = array![[0.1, 0.2, -0.1], [0.3, -0.1, 0.2]];
        let feature_values = array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]];
        let config = PlotConfig::default();

        let plot = create_3d_shap_plot(
            &shap_values.view(),
            &feature_values.view(),
            (0, 1, 2),
            &config,
        )
        .unwrap();

        assert_eq!(plot.x_values.len(), 2);
        assert_eq!(plot.y_values.len(), 2);
        assert_eq!(plot.z_values.len(), 2);
        assert!(plot.color_values.is_some());
        assert!(matches!(plot.plot_type, Plot3DType::Scatter));
    }

    #[test]
    fn test_3d_shap_plot_invalid_indices() {
        let shap_values = array![[0.1, 0.2], [0.3, -0.1]];
        let feature_values = array![[1.0, 2.0], [1.5, 2.5]];
        let config = PlotConfig::default();

        let result = create_3d_shap_plot(
            &shap_values.view(),
            &feature_values.view(),
            (0, 1, 2), // Index 2 is out of bounds
            &config,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_shap_plot_type_3d_variants() {
        let scatter_3d = ShapPlotType::Scatter3D;
        let surface_3d = ShapPlotType::Surface3D;

        // Test that new 3D variants are different from existing ones
        assert_ne!(scatter_3d as u8, ShapPlotType::Summary as u8);
        assert_ne!(surface_3d as u8, ShapPlotType::Waterfall as u8);
    }
}
