//! Comparative and partial dependence plot functionality
//!
//! This module provides visualization capabilities for comparing models,
//! partial dependence plots, and ICE (Individual Conditional Expectation) curves.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::core::{ComparisonType, PlotConfig};

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

/// Create partial dependence plot
///
/// # Arguments
///
/// * `feature_values` - Feature values for x-axis
/// * `pd_values` - Partial dependence values
/// * `feature_name` - Name of the feature
/// * `config` - Plot configuration
/// * `ice_curves` - Optional ICE curves
///
/// # Returns
///
/// Result containing partial dependence plot data
pub fn create_partial_dependence_plot(
    feature_values: &ArrayView1<Float>,
    pd_values: &ArrayView1<Float>,
    feature_name: String,
    config: &PlotConfig,
    ice_curves: Option<&ArrayView2<Float>>,
) -> SklResult<PartialDependencePlot> {
    if feature_values.len() != pd_values.len() {
        return Err(crate::SklearsError::InvalidInput(
            "Feature values and PD values must have the same length".to_string(),
        ));
    }

    if let Some(ice) = ice_curves {
        if ice.ncols() != feature_values.len() {
            return Err(crate::SklearsError::InvalidInput(
                "ICE curves columns must match feature values length".to_string(),
            ));
        }
    }

    Ok(PartialDependencePlot {
        feature_values: feature_values.to_owned(),
        pd_values: pd_values.to_owned(),
        ice_curves: ice_curves.map(|ice| ice.to_owned()),
        feature_name,
        config: config.clone(),
        show_ice: ice_curves.is_some(),
    })
}

/// Create comparative plot for model comparison
///
/// # Arguments
///
/// * `model_data` - Data from different models/methods
/// * `labels` - Labels for each data series
/// * `config` - Plot configuration
/// * `comparison_type` - Type of comparison visualization
///
/// # Returns
///
/// Result containing comparative plot data
pub fn create_comparative_plot(
    model_data: HashMap<String, Array2<Float>>,
    labels: Vec<String>,
    config: &PlotConfig,
    comparison_type: ComparisonType,
) -> SklResult<ComparativePlot> {
    if model_data.is_empty() {
        return Err(crate::SklearsError::InvalidInput(
            "Model data cannot be empty".to_string(),
        ));
    }

    // Validate all data arrays have compatible dimensions
    let first_shape = model_data.values().next().unwrap().dim();
    for (model_name, data) in &model_data {
        if data.dim() != first_shape {
            return Err(crate::SklearsError::InvalidInput(format!(
                "Model '{}' data shape {:?} does not match expected shape {:?}",
                model_name,
                data.dim(),
                first_shape
            )));
        }
    }

    Ok(ComparativePlot {
        model_data,
        labels,
        config: config.clone(),
        comparison_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_partial_dependence_plot_creation() {
        let feature_values = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let pd_values = array![0.1, 0.3, 0.5, 0.4, 0.2, 0.1];
        let config = PlotConfig::default();

        let plot = create_partial_dependence_plot(
            &feature_values.view(),
            &pd_values.view(),
            "feature_1".to_string(),
            &config,
            None,
        )
        .unwrap();

        assert_eq!(plot.feature_name, "feature_1");
        assert_eq!(plot.feature_values.len(), 6);
        assert_eq!(plot.pd_values.len(), 6);
        assert!(!plot.show_ice);
    }

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
    }
}
