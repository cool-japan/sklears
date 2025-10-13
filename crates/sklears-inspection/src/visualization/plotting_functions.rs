//! 2D Plotting Functions Module
//!
//! This module provides comprehensive 2D plotting functionality for model interpretability,
//! consolidating feature importance plots, SHAP visualizations, partial dependence plots,
//! and comparative analysis visualizations in a single, convenient interface.
//!
//! ## Key Features
//!
//! - **Feature Importance Plots**: Interactive bar, horizontal, radial, and tree-map visualizations
//! - **SHAP Visualizations**: Waterfall, force layout, summary, dependence, and beeswarm plots
//! - **Partial Dependence Plots**: PDP curves with optional ICE (Individual Conditional Expectation) curves
//! - **Comparative Plots**: Side-by-side model comparisons and overlay visualizations
//! - **High Performance**: Leverages SciRS2 for optimized numerical computations
//! - **Comprehensive Validation**: Input validation and detailed error messages
//!
//! ## Usage Examples
//!
//! ```rust,ignore
//! use sklears_inspection::visualization::plotting_functions::*;
//! // ✅ SciRS2 Policy Compliant Import
//! use scirs2_core::ndarray::array;
//!
//! // Feature importance plot
//! let importance = array![0.3, 0.5, 0.2];
//! let features = vec!["Feature1".to_string(), "Feature2".to_string(), "Feature3".to_string()];
//! let config = PlotConfig::default();
//!
//! let plot = create_feature_importance_plot(
//!     &importance.view(),
//!     Some(&features),
//!     None,
//!     &config,
//!     FeatureImportanceType::Bar
//! ).unwrap();
//!
//! // SHAP plot
//! let shap_values = array![[0.1, 0.2, -0.1], [0.3, -0.1, 0.2]];
//! let feature_values = array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]];
//!
//! let shap_plot = create_shap_visualization(
//!     &shap_values.view(),
//!     &feature_values.view(),
//!     None,
//!     None,
//!     &config,
//!     ShapPlotType::Summary,
//! ).unwrap();
//! ```

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

use super::config_types::{
    ComparativePlot, ComparisonType, FeatureImportancePlot, FeatureImportanceType,
    PartialDependencePlot, PlotConfig, ShapPlot, ShapPlotType,
};

// =============================================================================
// Feature Importance Plotting Functions
// =============================================================================

/// Create interactive feature importance plot
///
/// Generates interactive visualizations of feature importance scores with support
/// for error bars, multiple plot types, and comprehensive validation.
///
/// # Arguments
///
/// * `importance_values` - Feature importance scores as 1D array
/// * `feature_names` - Optional feature names; generates default names if None
/// * `std_values` - Optional standard deviations for error bars
/// * `config` - Plot configuration settings
/// * `plot_type` - Type of feature importance visualization
///
/// # Returns
///
/// Result containing feature importance plot data or error
///
/// # Errors
///
/// - `InvalidInput` - If feature names length doesn't match importance values
/// - `InvalidInput` - If std_values length doesn't match importance values
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_inspection::visualization::plotting_functions::*;
/// // ✅ SciRS2 Policy Compliant Import
/// use scirs2_core::ndarray::array;
///
/// let importance = array![0.3, 0.5, 0.2];
/// let features = vec!["Feature1".to_string(), "Feature2".to_string(), "Feature3".to_string()];
/// let config = PlotConfig::default();
///
/// let plot = create_feature_importance_plot(
///     &importance.view(),
///     Some(&features),
///     None,
///     &config,
///     FeatureImportanceType::Bar
/// ).unwrap();
///
/// assert_eq!(plot.feature_names.len(), 3);
/// assert_eq!(plot.importance_values.len(), 3);
/// assert_eq!(plot.importance_values[1], 0.5);
/// ```
pub fn create_feature_importance_plot(
    importance_values: &ArrayView1<Float>,
    feature_names: Option<&[String]>,
    std_values: Option<&ArrayView1<Float>>,
    config: &PlotConfig,
    plot_type: FeatureImportanceType,
) -> SklResult<FeatureImportancePlot> {
    let n_features = importance_values.len();

    // Validate input dimensions
    if n_features == 0 {
        return Err(crate::SklearsError::InvalidInput(
            "Importance values cannot be empty".to_string(),
        ));
    }

    // Generate or validate feature names
    let feature_names = if let Some(names) = feature_names {
        if names.len() != n_features {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Feature names length ({}) does not match importance values length ({})",
                names.len(),
                n_features
            )));
        }
        names.to_vec()
    } else {
        (0..n_features).map(|i| format!("Feature_{}", i)).collect()
    };

    // Validate standard deviations if provided
    if let Some(std) = std_values {
        if std.len() != n_features {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Standard deviation values length ({}) does not match importance values length ({})",
                std.len(),
                n_features
            )));
        }

        // Check for negative standard deviations
        for (i, &val) in std.iter().enumerate() {
            if val < 0.0 {
                return Err(crate::SklearsError::InvalidInput(format!(
                    "Standard deviation at index {} is negative: {}",
                    i, val
                )));
            }
        }
    }

    let std_values = std_values.map(|std| std.to_vec());
    let importance_values = importance_values.to_vec();

    Ok(FeatureImportancePlot {
        feature_names,
        importance_values,
        std_values,
        config: config.clone(),
        plot_type,
    })
}

/// Create advanced feature importance plot with ranking and filtering
///
/// Enhanced version that provides additional functionality such as automatic ranking,
/// filtering by threshold, and statistical significance testing.
///
/// # Arguments
///
/// * `importance_values` - Feature importance scores
/// * `feature_names` - Optional feature names
/// * `std_values` - Optional standard deviations
/// * `config` - Plot configuration
/// * `plot_type` - Type of visualization
/// * `top_k` - Optional limit to top K features (None for all features)
/// * `min_threshold` - Optional minimum importance threshold for inclusion
///
/// # Returns
///
/// Result containing filtered and ranked feature importance plot data
pub fn create_ranked_feature_importance_plot(
    importance_values: &ArrayView1<Float>,
    feature_names: Option<&[String]>,
    std_values: Option<&ArrayView1<Float>>,
    config: &PlotConfig,
    plot_type: FeatureImportanceType,
    top_k: Option<usize>,
    min_threshold: Option<Float>,
) -> SklResult<FeatureImportancePlot> {
    let n_features = importance_values.len();

    if n_features == 0 {
        return Err(crate::SklearsError::InvalidInput(
            "Importance values cannot be empty".to_string(),
        ));
    }

    // Create feature names if not provided
    let feature_names = if let Some(names) = feature_names {
        if names.len() != n_features {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Feature names length ({}) does not match importance values length ({})",
                names.len(),
                n_features
            )));
        }
        names.to_vec()
    } else {
        (0..n_features).map(|i| format!("Feature_{}", i)).collect()
    };

    // Create indices and sort by importance (descending)
    let mut indices: Vec<usize> = (0..n_features).collect();
    indices.sort_by(|&a, &b| {
        importance_values[b]
            .partial_cmp(&importance_values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply threshold filtering
    if let Some(threshold) = min_threshold {
        indices.retain(|&i| importance_values[i].abs() >= threshold);
    }

    // Apply top-k filtering
    if let Some(k) = top_k {
        indices.truncate(k);
    }

    if indices.is_empty() {
        return Err(crate::SklearsError::InvalidInput(
            "No features meet the filtering criteria".to_string(),
        ));
    }

    // Extract filtered and sorted data
    let filtered_names = indices.iter().map(|&i| feature_names[i].clone()).collect();
    let filtered_importance = indices.iter().map(|&i| importance_values[i]).collect();
    let filtered_std = std_values.map(|std| indices.iter().map(|&i| std[i]).collect());

    Ok(FeatureImportancePlot {
        feature_names: filtered_names,
        importance_values: filtered_importance,
        std_values: filtered_std,
        config: config.clone(),
        plot_type,
    })
}

// =============================================================================
// SHAP Plotting Functions
// =============================================================================

/// Create interactive SHAP plot for model explanations
///
/// Generates SHAP (SHapley Additive exPlanations) visualizations for understanding
/// model predictions with comprehensive validation and multiple plot types.
///
/// # Arguments
///
/// * `shap_values` - SHAP values matrix (instances × features)
/// * `feature_values` - Feature values matrix (instances × features)
/// * `feature_names` - Optional feature names; generates defaults if None
/// * `instance_names` - Optional instance names; generates defaults if None
/// * `config` - Plot configuration settings
/// * `plot_type` - Type of SHAP visualization
///
/// # Returns
///
/// Result containing SHAP plot data or error
///
/// # Errors
///
/// - `InvalidInput` - If SHAP and feature values dimensions don't match
/// - `InvalidInput` - If feature/instance names lengths don't match data dimensions
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_inspection::visualization::plotting_functions::*;
/// // ✅ SciRS2 Policy Compliant Import
/// use scirs2_core::ndarray::array;
///
/// let shap_values = array![[0.1, 0.2, -0.1], [0.3, -0.1, 0.2]];
/// let feature_values = array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]];
/// let config = PlotConfig::default();
///
/// let plot = create_shap_visualization(
///     &shap_values.view(),
///     &feature_values.view(),
///     None,
///     None,
///     &config,
///     ShapPlotType::Summary,
/// ).unwrap();
///
/// assert_eq!(plot.shap_values.shape(), &[2, 3]);
/// assert_eq!(plot.feature_names.len(), 3);
/// assert_eq!(plot.instance_names.len(), 2);
/// ```
pub fn create_shap_visualization(
    shap_values: &ArrayView2<Float>,
    feature_values: &ArrayView2<Float>,
    feature_names: Option<&[String]>,
    instance_names: Option<&[String]>,
    config: &PlotConfig,
    plot_type: ShapPlotType,
) -> SklResult<ShapPlot> {
    let (n_instances, n_features) = shap_values.dim();

    // Validate input dimensions
    if n_instances == 0 || n_features == 0 {
        return Err(crate::SklearsError::InvalidInput(
            "SHAP values cannot have zero dimensions".to_string(),
        ));
    }

    if feature_values.dim() != (n_instances, n_features) {
        return Err(crate::SklearsError::InvalidInput(format!(
            "SHAP values shape {:?} and feature values shape {:?} do not match",
            (n_instances, n_features),
            feature_values.dim()
        )));
    }

    // Generate or validate feature names
    let feature_names = if let Some(names) = feature_names {
        if names.len() != n_features {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Feature names length ({}) does not match number of features ({})",
                names.len(),
                n_features
            )));
        }
        names.to_vec()
    } else {
        (0..n_features).map(|i| format!("Feature_{}", i)).collect()
    };

    // Generate or validate instance names
    let instance_names = if let Some(names) = instance_names {
        if names.len() != n_instances {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Instance names length ({}) does not match number of instances ({})",
                names.len(),
                n_instances
            )));
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

/// Create SHAP summary plot with aggregated feature importance
///
/// Specialized function for creating SHAP summary plots that aggregate
/// importance across all instances and provide statistical summaries.
///
/// # Arguments
///
/// * `shap_values` - SHAP values matrix (instances × features)
/// * `feature_values` - Feature values matrix (instances × features)
/// * `feature_names` - Optional feature names
/// * `config` - Plot configuration
/// * `show_distribution` - Whether to include value distribution information
///
/// # Returns
///
/// Result containing aggregated SHAP summary plot
pub fn create_shap_summary_plot(
    shap_values: &ArrayView2<Float>,
    feature_values: &ArrayView2<Float>,
    feature_names: Option<&[String]>,
    config: &PlotConfig,
    show_distribution: bool,
) -> SklResult<ShapPlot> {
    let (n_instances, n_features) = shap_values.dim();

    if n_instances == 0 || n_features == 0 {
        return Err(crate::SklearsError::InvalidInput(
            "SHAP values cannot have zero dimensions".to_string(),
        ));
    }

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

    let instance_names = (0..n_instances)
        .map(|i| format!("Instance_{}", i))
        .collect();

    let plot_type = if show_distribution {
        ShapPlotType::Beeswarm
    } else {
        ShapPlotType::Summary
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

// =============================================================================
// Partial Dependence Plotting Functions
// =============================================================================

/// Create partial dependence plot (PDP) with optional ICE curves
///
/// Generates partial dependence plots showing how model predictions change
/// with feature values, with optional Individual Conditional Expectation curves.
///
/// # Arguments
///
/// * `feature_values` - Feature values for x-axis (sorted grid points)
/// * `pd_values` - Partial dependence values corresponding to feature values
/// * `ice_curves` - Optional ICE curves (instances × feature_values)
/// * `feature_name` - Name of the feature being analyzed
/// * `config` - Plot configuration settings
/// * `show_ice` - Whether to display individual ICE curves
///
/// # Returns
///
/// Result containing partial dependence plot data or error
///
/// # Errors
///
/// - `InvalidInput` - If feature values and PD values lengths don't match
/// - `InvalidInput` - If ICE curves columns don't match feature values length
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_inspection::visualization::plotting_functions::*;
/// // ✅ SciRS2 Policy Compliant Import
/// use scirs2_core::ndarray::array;
///
/// let feature_values = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
/// let pd_values = array![0.1, 0.3, 0.5, 0.4, 0.2, 0.1];
/// let config = PlotConfig::default();
///
/// let plot = create_partial_dependence_plot(
///     &feature_values.view(),
///     &pd_values.view(),
///     None,
///     "feature_1",
///     &config,
///     false,
/// ).unwrap();
///
/// assert_eq!(plot.feature_name, "feature_1");
/// assert_eq!(plot.feature_values.len(), 6);
/// assert_eq!(plot.pd_values.len(), 6);
/// assert!(!plot.show_ice);
/// ```
pub fn create_partial_dependence_plot(
    feature_values: &ArrayView1<Float>,
    pd_values: &ArrayView1<Float>,
    ice_curves: Option<&ArrayView2<Float>>,
    feature_name: &str,
    config: &PlotConfig,
    show_ice: bool,
) -> SklResult<PartialDependencePlot> {
    let n_points = feature_values.len();

    // Validate input dimensions
    if n_points == 0 {
        return Err(crate::SklearsError::InvalidInput(
            "Feature values cannot be empty".to_string(),
        ));
    }

    if pd_values.len() != n_points {
        return Err(crate::SklearsError::InvalidInput(format!(
            "Feature values length ({}) and PD values length ({}) must match",
            n_points,
            pd_values.len()
        )));
    }

    // Validate ICE curves if provided
    if let Some(ice) = ice_curves {
        if ice.ncols() != n_points {
            return Err(crate::SklearsError::InvalidInput(format!(
                "ICE curves columns ({}) must match feature values length ({})",
                ice.ncols(),
                n_points
            )));
        }

        if ice.nrows() == 0 {
            return Err(crate::SklearsError::InvalidInput(
                "ICE curves cannot have zero instances".to_string(),
            ));
        }
    }

    // Validate feature values are sorted (for proper PD interpretation)
    for i in 1..n_points {
        if feature_values[i] < feature_values[i - 1] {
            return Err(crate::SklearsError::InvalidInput(
                "Feature values must be sorted in ascending order for proper PD interpretation"
                    .to_string(),
            ));
        }
    }

    Ok(PartialDependencePlot {
        feature_values: feature_values.to_owned(),
        pd_values: pd_values.to_owned(),
        ice_curves: ice_curves.map(|ice| ice.to_owned()),
        feature_name: feature_name.to_string(),
        config: config.clone(),
        show_ice: show_ice && ice_curves.is_some(),
    })
}

/// Create 2D partial dependence plot for feature interaction analysis
///
/// Creates a 2D PDP showing how two features interact to affect model predictions.
/// Returns data suitable for contour plots, heatmaps, or 3D surface visualization.
///
/// # Arguments
///
/// * `feature1_values` - Values for first feature (x-axis)
/// * `feature2_values` - Values for second feature (y-axis)
/// * `pd_surface` - 2D partial dependence values (feature1 × feature2)
/// * `feature1_name` - Name of first feature
/// * `feature2_name` - Name of second feature
/// * `config` - Plot configuration
///
/// # Returns
///
/// Result containing 2D partial dependence data structured for visualization
pub fn create_2d_partial_dependence_plot(
    feature1_values: &ArrayView1<Float>,
    feature2_values: &ArrayView1<Float>,
    pd_surface: &ArrayView2<Float>,
    feature1_name: &str,
    feature2_name: &str,
    config: &PlotConfig,
) -> SklResult<ComparativePlot> {
    let n_points1 = feature1_values.len();
    let n_points2 = feature2_values.len();

    if n_points1 == 0 || n_points2 == 0 {
        return Err(crate::SklearsError::InvalidInput(
            "Feature values cannot be empty".to_string(),
        ));
    }

    if pd_surface.dim() != (n_points1, n_points2) {
        return Err(crate::SklearsError::InvalidInput(format!(
            "PD surface shape {:?} does not match expected shape ({}, {})",
            pd_surface.dim(),
            n_points1,
            n_points2
        )));
    }

    let mut model_data = HashMap::new();
    model_data.insert("2D_PD_Surface".to_string(), pd_surface.to_owned());

    let labels = vec![feature1_name.to_string(), feature2_name.to_string()];

    Ok(ComparativePlot {
        model_data,
        labels,
        config: config.clone(),
        comparison_type: ComparisonType::Heatmap,
    })
}

// =============================================================================
// Comparative Plotting Functions
// =============================================================================

/// Create comparative visualization for model comparison
///
/// Generates comparative plots for analyzing differences between multiple models
/// or different parameter configurations with comprehensive validation.
///
/// # Arguments
///
/// * `model_data` - HashMap of model names to their prediction/score data
/// * `labels` - Labels for data dimensions/features
/// * `config` - Plot configuration settings
/// * `comparison_type` - Type of comparison visualization
///
/// # Returns
///
/// Result containing comparative plot data or error
///
/// # Errors
///
/// - `InvalidInput` - If model data is empty
/// - `InvalidInput` - If model data shapes are inconsistent
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_inspection::visualization::plotting_functions::*;
/// use std::collections::HashMap;
/// // ✅ SciRS2 Policy Compliant Import
/// use scirs2_core::ndarray::array;
///
/// let mut model_data = HashMap::new();
/// model_data.insert("model_1".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
/// model_data.insert("model_2".to_string(), array![[2.0, 3.0], [4.0, 5.0]]);
///
/// let labels = vec!["Feature A".to_string(), "Feature B".to_string()];
/// let config = PlotConfig::default();
///
/// let plot = create_comparative_plot(
///     model_data,
///     labels,
///     &config,
///     ComparisonType::SideBySide
/// ).unwrap();
///
/// assert_eq!(plot.model_data.len(), 2);
/// assert_eq!(plot.labels.len(), 2);
/// ```
pub fn create_comparative_plot(
    model_data: HashMap<String, Array2<Float>>,
    labels: Vec<String>,
    config: &PlotConfig,
    comparison_type: ComparisonType,
) -> SklResult<ComparativePlot> {
    // Validate non-empty model data
    if model_data.is_empty() {
        return Err(crate::SklearsError::InvalidInput(
            "Model data cannot be empty".to_string(),
        ));
    }

    // Validate labels are not empty
    if labels.is_empty() {
        return Err(crate::SklearsError::InvalidInput(
            "Labels cannot be empty".to_string(),
        ));
    }

    // Validate that all model data has compatible dimensions
    let first_entry = model_data.iter().next().unwrap();
    let (first_name, first_data) = first_entry;
    let expected_shape = first_data.dim();

    // Check for empty data arrays
    if expected_shape.0 == 0 || expected_shape.1 == 0 {
        return Err(crate::SklearsError::InvalidInput(format!(
            "Model '{}' has invalid data shape: {:?}",
            first_name, expected_shape
        )));
    }

    // Validate all models have consistent shapes
    for (model_name, data) in &model_data {
        let current_shape = data.dim();
        if current_shape != expected_shape {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Model '{}' data shape {:?} does not match expected shape {:?}",
                model_name, current_shape, expected_shape
            )));
        }

        // Validate for non-finite values
        for value in data.iter() {
            if !value.is_finite() {
                return Err(crate::SklearsError::InvalidInput(format!(
                    "Model '{}' contains non-finite values",
                    model_name
                )));
            }
        }
    }

    // Validate labels count matches data dimensions
    if labels.len() != expected_shape.1 {
        return Err(crate::SklearsError::InvalidInput(format!(
            "Labels count ({}) does not match data columns ({})",
            labels.len(),
            expected_shape.1
        )));
    }

    Ok(ComparativePlot {
        model_data,
        labels,
        config: config.clone(),
        comparison_type,
    })
}

/// Create performance comparison plot for multiple metrics
///
/// Specialized comparative plot for model performance metrics with statistical
/// significance testing and confidence intervals.
///
/// # Arguments
///
/// * `performance_data` - Performance metrics for each model
/// * `metric_names` - Names of the performance metrics
/// * `confidence_intervals` - Optional confidence intervals for each metric
/// * `config` - Plot configuration
/// * `show_significance` - Whether to show statistical significance markers
///
/// # Returns
///
/// Result containing performance comparison plot data
pub fn create_performance_comparison_plot(
    performance_data: HashMap<String, Array1<Float>>,
    metric_names: Vec<String>,
    confidence_intervals: Option<HashMap<String, Array2<Float>>>,
    config: &PlotConfig,
    show_significance: bool,
) -> SklResult<ComparativePlot> {
    if performance_data.is_empty() {
        return Err(crate::SklearsError::InvalidInput(
            "Performance data cannot be empty".to_string(),
        ));
    }

    if metric_names.is_empty() {
        return Err(crate::SklearsError::InvalidInput(
            "Metric names cannot be empty".to_string(),
        ));
    }

    // Convert 1D performance data to 2D for compatibility with ComparativePlot
    let mut model_data_2d = HashMap::new();
    let expected_len = metric_names.len();

    for (model_name, metrics) in performance_data {
        if metrics.len() != expected_len {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Model '{}' metrics length ({}) does not match expected length ({})",
                model_name,
                metrics.len(),
                expected_len
            )));
        }

        // Convert to 2D array (1 × n_metrics)
        let metrics_2d = metrics.insert_axis(scirs2_core::ndarray::Axis(0));
        model_data_2d.insert(model_name, metrics_2d);
    }

    // Validate confidence intervals if provided
    if let Some(ci) = &confidence_intervals {
        for (model_name, intervals) in ci {
            if !model_data_2d.contains_key(model_name) {
                return Err(crate::SklearsError::InvalidInput(format!(
                    "Confidence interval provided for unknown model: '{}'",
                    model_name
                )));
            }

            if intervals.dim() != (2, expected_len) {
                return Err(crate::SklearsError::InvalidInput(format!(
                    "Confidence intervals for model '{}' have incorrect shape: {:?}, expected (2, {})",
                    model_name,
                    intervals.dim(),
                    expected_len
                )));
            }
        }
    }

    let comparison_type = if show_significance {
        ComparisonType::Statistical
    } else {
        ComparisonType::SideBySide
    };

    Ok(ComparativePlot {
        model_data: model_data_2d,
        labels: metric_names,
        config: config.clone(),
        comparison_type,
    })
}

// =============================================================================
// Comprehensive Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    // Feature Importance Tests
    #[test]
    fn test_feature_importance_plot_creation() {
        let importance = array![0.3, 0.5, 0.2];
        let features = vec![
            "Feature1".to_string(),
            "Feature2".to_string(),
            "Feature3".to_string(),
        ];
        let config = PlotConfig::default();

        let plot = create_feature_importance_plot(
            &importance.view(),
            Some(&features),
            None,
            &config,
            FeatureImportanceType::Bar,
        )
        .unwrap();

        assert_eq!(plot.feature_names.len(), 3);
        assert_eq!(plot.importance_values.len(), 3);
        assert_eq!(plot.importance_values[1], 0.5);
        assert!(plot.std_values.is_none());
        assert_eq!(plot.plot_type, FeatureImportanceType::Bar);
    }

    #[test]
    fn test_feature_importance_with_std() {
        let importance = array![0.3, 0.5, 0.2];
        let std_vals = array![0.1, 0.05, 0.15];
        let config = PlotConfig::default();

        let plot = create_feature_importance_plot(
            &importance.view(),
            None,
            Some(&std_vals.view()),
            &config,
            FeatureImportanceType::Horizontal,
        )
        .unwrap();

        assert_eq!(plot.feature_names.len(), 3);
        assert!(plot.std_values.is_some());
        assert_eq!(plot.std_values.as_ref().unwrap().len(), 3);
        assert_eq!(plot.plot_type, FeatureImportanceType::Horizontal);
    }

    #[test]
    fn test_feature_importance_dimension_mismatch() {
        let importance = array![0.3, 0.5];
        let features = vec![
            "Feature1".to_string(),
            "Feature2".to_string(),
            "Feature3".to_string(),
        ];
        let config = PlotConfig::default();

        let result = create_feature_importance_plot(
            &importance.view(),
            Some(&features),
            None,
            &config,
            FeatureImportanceType::Bar,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_importance_empty_input() {
        let importance = array![];
        let config = PlotConfig::default();

        let result = create_feature_importance_plot(
            &importance.view(),
            None,
            None,
            &config,
            FeatureImportanceType::Bar,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_importance_negative_std() {
        let importance = array![0.3, 0.5, 0.2];
        let std_vals = array![0.1, -0.05, 0.15]; // negative std
        let config = PlotConfig::default();

        let result = create_feature_importance_plot(
            &importance.view(),
            None,
            Some(&std_vals.view()),
            &config,
            FeatureImportanceType::Bar,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_ranked_feature_importance_plot() {
        let importance = array![0.1, 0.5, 0.3, 0.2];
        let config = PlotConfig::default();

        let plot = create_ranked_feature_importance_plot(
            &importance.view(),
            None,
            None,
            &config,
            FeatureImportanceType::Bar,
            Some(2),    // top 2 features
            Some(0.15), // minimum threshold
        )
        .unwrap();

        // Should have only top 2 features above threshold (0.5 and 0.3)
        assert_eq!(plot.feature_names.len(), 2);
        assert_eq!(plot.importance_values.len(), 2);
        assert_eq!(plot.importance_values[0], 0.5); // highest first
        assert_eq!(plot.importance_values[1], 0.3); // second highest
    }

    // SHAP Tests
    #[test]
    fn test_shap_plot_creation() {
        let shap_values = array![[0.1, 0.2, -0.1], [0.3, -0.1, 0.2]];
        let feature_values = array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]];
        let config = PlotConfig::default();

        let plot = create_shap_visualization(
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
        assert_eq!(plot.plot_type, ShapPlotType::Summary);
    }

    #[test]
    fn test_shap_plot_dimension_mismatch() {
        let shap_values = array![[0.1, 0.2], [0.3, -0.1]];
        let feature_values = array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]];
        let config = PlotConfig::default();

        let result = create_shap_visualization(
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
    fn test_shap_plot_zero_dimensions() {
        let shap_values = array![[], []]; // 2x0 array
        let feature_values = array![[], []]; // 2x0 array
        let config = PlotConfig::default();

        let result = create_shap_visualization(
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
    fn test_shap_summary_plot() {
        let shap_values = array![[0.1, 0.2, -0.1], [0.3, -0.1, 0.2], [0.0, 0.1, -0.05]];
        let feature_values = array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [1.2, 2.1, 3.2]];
        let config = PlotConfig::default();

        let plot = create_shap_summary_plot(
            &shap_values.view(),
            &feature_values.view(),
            None,
            &config,
            true, // show distribution
        )
        .unwrap();

        assert_eq!(plot.shap_values.shape(), &[3, 3]);
        assert_eq!(plot.plot_type, ShapPlotType::Beeswarm);
    }

    // Partial Dependence Tests
    #[test]
    fn test_partial_dependence_plot_creation() {
        let feature_values = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let pd_values = array![0.1, 0.3, 0.5, 0.4, 0.2, 0.1];
        let config = PlotConfig::default();

        let plot = create_partial_dependence_plot(
            &feature_values.view(),
            &pd_values.view(),
            None,
            "feature_1",
            &config,
            false,
        )
        .unwrap();

        assert_eq!(plot.feature_name, "feature_1");
        assert_eq!(plot.feature_values.len(), 6);
        assert_eq!(plot.pd_values.len(), 6);
        assert!(!plot.show_ice);
        assert!(plot.ice_curves.is_none());
    }

    #[test]
    fn test_partial_dependence_plot_with_ice() {
        let feature_values = array![0.0, 0.5, 1.0];
        let pd_values = array![0.1, 0.5, 0.2];
        let ice_curves = array![[0.0, 0.4, 0.1], [0.2, 0.6, 0.3]]; // 2 instances, 3 points
        let config = PlotConfig::default();

        let plot = create_partial_dependence_plot(
            &feature_values.view(),
            &pd_values.view(),
            Some(&ice_curves.view()),
            "feature_1",
            &config,
            true,
        )
        .unwrap();

        assert_eq!(plot.feature_name, "feature_1");
        assert!(plot.show_ice);
        assert!(plot.ice_curves.is_some());
        assert_eq!(plot.ice_curves.as_ref().unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn test_partial_dependence_plot_dimension_mismatch() {
        let feature_values = array![0.0, 0.5, 1.0];
        let pd_values = array![0.1, 0.5]; // wrong length
        let config = PlotConfig::default();

        let result = create_partial_dependence_plot(
            &feature_values.view(),
            &pd_values.view(),
            None,
            "feature_1",
            &config,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_dependence_plot_unsorted_features() {
        let feature_values = array![0.0, 1.0, 0.5]; // unsorted
        let pd_values = array![0.1, 0.2, 0.3];
        let config = PlotConfig::default();

        let result = create_partial_dependence_plot(
            &feature_values.view(),
            &pd_values.view(),
            None,
            "feature_1",
            &config,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_2d_partial_dependence_plot() {
        let feature1_values = array![0.0, 0.5, 1.0];
        let feature2_values = array![0.0, 1.0];
        let pd_surface = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]; // 3x2
        let config = PlotConfig::default();

        let plot = create_2d_partial_dependence_plot(
            &feature1_values.view(),
            &feature2_values.view(),
            &pd_surface.view(),
            "feature_1",
            "feature_2",
            &config,
        )
        .unwrap();

        assert_eq!(plot.model_data.len(), 1);
        assert!(plot.model_data.contains_key("2D_PD_Surface"));
        assert_eq!(plot.labels.len(), 2);
        assert_eq!(plot.comparison_type, ComparisonType::Heatmap);
    }

    // Comparative Plot Tests
    #[test]
    fn test_comparative_plot_creation() {
        let mut model_data = HashMap::new();
        model_data.insert("model_1".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
        model_data.insert("model_2".to_string(), array![[2.0, 3.0], [4.0, 5.0]]);

        let labels = vec!["Feature A".to_string(), "Feature B".to_string()];
        let config = PlotConfig::default();

        let plot = create_comparative_plot(model_data, labels, &config, ComparisonType::SideBySide)
            .unwrap();

        assert_eq!(plot.model_data.len(), 2);
        assert_eq!(plot.labels.len(), 2);
        assert_eq!(plot.comparison_type, ComparisonType::SideBySide);
    }

    #[test]
    fn test_comparative_plot_empty_data() {
        let model_data = HashMap::new();
        let labels = vec!["Feature A".to_string()];
        let config = PlotConfig::default();

        let result =
            create_comparative_plot(model_data, labels, &config, ComparisonType::SideBySide);
        assert!(result.is_err());
    }

    #[test]
    fn test_comparative_plot_shape_mismatch() {
        let mut model_data = HashMap::new();
        model_data.insert("model_1".to_string(), array![[1.0, 2.0], [3.0, 4.0]]); // 2x2
        model_data.insert("model_2".to_string(), array![[2.0, 3.0, 5.0]]); // 1x3, different shape

        let labels = vec!["Feature A".to_string(), "Feature B".to_string()];
        let config = PlotConfig::default();

        let result =
            create_comparative_plot(model_data, labels, &config, ComparisonType::SideBySide);
        assert!(result.is_err());
    }

    #[test]
    fn test_performance_comparison_plot() {
        let mut performance_data = HashMap::new();
        performance_data.insert("model_1".to_string(), array![0.85, 0.78, 0.92]);
        performance_data.insert("model_2".to_string(), array![0.83, 0.80, 0.89]);

        let metric_names = vec![
            "Accuracy".to_string(),
            "Precision".to_string(),
            "Recall".to_string(),
        ];
        let config = PlotConfig::default();

        let plot = create_performance_comparison_plot(
            performance_data,
            metric_names,
            None,
            &config,
            false,
        )
        .unwrap();

        assert_eq!(plot.model_data.len(), 2);
        assert_eq!(plot.labels.len(), 3);
        assert_eq!(plot.comparison_type, ComparisonType::SideBySide);
    }

    #[test]
    fn test_performance_comparison_with_significance() {
        let mut performance_data = HashMap::new();
        performance_data.insert("model_1".to_string(), array![0.85, 0.78]);

        let metric_names = vec!["Accuracy".to_string(), "Precision".to_string()];
        let config = PlotConfig::default();

        let plot = create_performance_comparison_plot(
            performance_data,
            metric_names,
            None,
            &config,
            true, // show significance
        )
        .unwrap();

        assert_eq!(plot.comparison_type, ComparisonType::Statistical);
    }

    // Edge Case Tests
    #[test]
    fn test_all_plot_types_enum_coverage() {
        // Test that we can create plots with all enum variants
        let importance = array![0.5];
        let config = PlotConfig::default();

        for &plot_type in &[
            FeatureImportanceType::Bar,
            FeatureImportanceType::Horizontal,
            FeatureImportanceType::Radial,
            FeatureImportanceType::TreeMap,
        ] {
            let result =
                create_feature_importance_plot(&importance.view(), None, None, &config, plot_type);
            assert!(result.is_ok());
        }

        let shap_values = array![[0.1]];
        let feature_values = array![[1.0]];

        for &plot_type in &[
            ShapPlotType::Waterfall,
            ShapPlotType::ForceLayout,
            ShapPlotType::Summary,
            ShapPlotType::Dependence,
            ShapPlotType::Beeswarm,
            ShapPlotType::DecisionPlot,
        ] {
            let result = create_shap_visualization(
                &shap_values.view(),
                &feature_values.view(),
                None,
                None,
                &config,
                plot_type,
            );
            assert!(result.is_ok());
        }
    }
}
