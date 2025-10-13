//! Feature importance visualization functionality
//!
//! This module provides interactive feature importance plots with various visualization types.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

use super::core::PlotConfig;

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

///
/// let importance = array![0.3, 0.5, 0.2];
/// let features = vec!["Feature1".to_string(), "Feature2".to_string(), "Feature3".to_string()];
/// let config = PlotConfig::default();
///
/// let plot = create_interactive_feature_importance_plot(
///     &importance.view(),
///     Some(&features),
///     None,
///     &config,
///     FeatureImportanceType::Bar
/// ).unwrap();
/// assert_eq!(plot.feature_names.len(), 3);
/// assert_eq!(plot.importance_values.len(), 3);
/// ```
pub fn create_interactive_feature_importance_plot(
    importance_values: &ArrayView1<Float>,
    feature_names: Option<&[String]>,
    std_values: Option<&ArrayView1<Float>>,
    config: &PlotConfig,
    plot_type: FeatureImportanceType,
) -> SklResult<FeatureImportancePlot> {
    let n_features = importance_values.len();

    let feature_names = if let Some(names) = feature_names {
        if names.len() != n_features {
            return Err(crate::SklearsError::InvalidInput(
                "Feature names length does not match importance values".to_string(),
            ));
        }
        names.to_vec()
    } else {
        (0..n_features).map(|i| format!("Feature_{}", i)).collect()
    };

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

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature_importance_plot_creation() {
        let importance = array![0.3, 0.5, 0.2];
        let features = vec![
            "Feature1".to_string(),
            "Feature2".to_string(),
            "Feature3".to_string(),
        ];
        let config = PlotConfig::default();

        let plot = create_interactive_feature_importance_plot(
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
    }

    #[test]
    fn test_feature_importance_with_std() {
        let importance = array![0.3, 0.5, 0.2];
        let std_vals = array![0.1, 0.05, 0.15];
        let config = PlotConfig::default();

        let plot = create_interactive_feature_importance_plot(
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
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let importance = array![0.3, 0.5];
        let features = vec![
            "Feature1".to_string(),
            "Feature2".to_string(),
            "Feature3".to_string(),
        ];
        let config = PlotConfig::default();

        let result = create_interactive_feature_importance_plot(
            &importance.view(),
            Some(&features),
            None,
            &config,
            FeatureImportanceType::Bar,
        );
        assert!(result.is_err());
    }
}
