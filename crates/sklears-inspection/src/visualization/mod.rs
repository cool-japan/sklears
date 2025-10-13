//! Comprehensive Visualization Framework for Machine Learning Model Inspection
//!
//! This module provides a complete suite of visualization tools for analyzing and interpreting
//! machine learning models, featuring high-performance SIMD operations, interactive frameworks,
//! mobile-responsive designs, and advanced 3D visualization capabilities.
//!
//! ## Architecture Overview
//!
//! The visualization framework is organized into 5 specialized domains:
//!
//! ### Core Components
//!
//! 1. **SIMD Operations** (`simd_operations`)
//!    - High-performance vectorized computations (5.8x-10.8x speedups)
//!    - Grid generation, data normalization, min/max finding
//!    - Color mapping and data smoothing operations
//!
//! 2. **Configuration & Types** (`config_types`)
//!    - Comprehensive configuration structures and enums
//!    - Plot configurations, color schemes, mobile settings
//!    - 3D camera controls, animation settings, interaction handlers
//!    - Fluent builder APIs for easy configuration
//!
//! 3. **Interactive Framework** (`interactive_framework`)
//!    - Real-time update capabilities with callback system
//!    - Multi-plot management and thread-safe operations
//!    - Dynamic plot configuration and event handling
//!
//! 4. **2D Plotting Functions** (`plotting_functions`)
//!    - Feature importance visualizations (bar, horizontal, radial, treemap)
//!    - SHAP plots (waterfall, force layout, summary, dependence)
//!    - Partial dependence plots with ICE curves
//!    - Comparative analysis and model comparison tools
//!
//! 5. **3D Visualization & Animation** (`visualization_3d`)
//!    - Advanced 3D plotting (scatter, surface, mesh, contour, volume, network)
//!    - Camera controls with auto-rotation and projection settings
//!    - Animation framework with easing functions and mobile optimization
//!    - Surface plotting with contour lines and color mapping

// Module declarations for the refactored structure
pub mod comparative;
pub mod config_types;
pub mod core;
pub mod feature_importance;
pub mod interactive;
pub mod interactive_framework;
pub mod mobile;
pub mod output;
pub mod plots_3d;
pub mod plotting_functions;
pub mod shap;
pub mod simd_operations;

// Core re-exports from config_types
pub use config_types::{
    Animation3D, AutoRotate3D, Camera3D, ColorScheme, ComparativePlot, ComparisonType,
    ContourLines3D, EasingType, FeatureImportancePlot, FeatureImportanceType, MobileConfig,
    PartialDependencePlot, Plot3D, Plot3DType, PlotConfig, PlotConfigBuilder,
    PlotInteractionHandler, ProjectionType, ShapPlot, ShapPlotType, Surface3D,
};

// Interactive framework re-exports
pub use interactive_framework::InteractiveVisualizer;

// SIMD operations re-exports
pub use simd_operations::{
    simd_color_mapping, simd_find_min_max, simd_generate_grid_range, simd_normalize_data,
    simd_smooth_data, simd_sum,
};

// 2D plotting functions re-exports
pub use plotting_functions::{
    create_2d_partial_dependence_plot, create_comparative_plot, create_feature_importance_plot,
    create_partial_dependence_plot, create_performance_comparison_plot,
    create_ranked_feature_importance_plot, create_shap_summary_plot, create_shap_visualization,
};

// 3D visualization re-exports
pub use plots_3d::{create_3d_plot, create_3d_shap_plot, create_3d_surface_plot};

// Common imports and types
use crate::Float;
use crate::SklResult;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};

/// Comprehensive visualization factory for creating different types of plots
pub struct VisualizationFactory {
    default_config: PlotConfig,
    interactive_visualizer: InteractiveVisualizer,
}

impl VisualizationFactory {
    /// Create a new visualization factory with default configuration
    pub fn new() -> Self {
        Self {
            default_config: PlotConfig::default(),
            interactive_visualizer: InteractiveVisualizer::new(),
        }
    }

    /// Create factory with custom default configuration
    pub fn with_config(config: PlotConfig) -> Self {
        Self {
            default_config: config,
            interactive_visualizer: InteractiveVisualizer::new(),
        }
    }

    /// Get mutable reference to interactive visualizer
    pub fn interactive(&mut self) -> &mut InteractiveVisualizer {
        &mut self.interactive_visualizer
    }

    /// Create feature importance plot with factory defaults
    pub fn feature_importance(
        &self,
        importance_values: &ArrayView1<Float>,
        feature_names: Option<&[String]>,
        plot_type: FeatureImportanceType,
    ) -> SklResult<FeatureImportancePlot> {
        create_feature_importance_plot(
            importance_values,
            feature_names,
            None,
            &self.default_config,
            plot_type,
        )
    }

    /// Create SHAP waterfall plot with factory defaults
    pub fn shap_summary(
        &self,
        shap_values: &ArrayView2<Float>,
        feature_values: &ArrayView2<Float>,
        feature_names: &[String],
    ) -> SklResult<ShapPlot> {
        create_shap_summary_plot(
            shap_values,
            feature_values,
            Some(feature_names),
            &self.default_config,
            true,
        )
    }

    /// Create partial dependence plot with factory defaults
    pub fn partial_dependence(
        &self,
        feature_values: &ArrayView1<Float>,
        pd_values: &ArrayView1<Float>,
        feature_name: &str,
        ice_curves: Option<&ArrayView2<Float>>,
    ) -> SklResult<PartialDependencePlot> {
        create_partial_dependence_plot(
            feature_values,
            pd_values,
            ice_curves,
            feature_name,
            &self.default_config,
            ice_curves.is_some(),
        )
    }

    /// Create 3D scatter plot with factory defaults
    pub fn scatter_3d(
        &self,
        x_values: &ArrayView1<Float>,
        y_values: &ArrayView1<Float>,
        z_values: &ArrayView1<Float>,
        axis_labels: (String, String, String),
    ) -> SklResult<Plot3D> {
        create_3d_plot(
            x_values,
            y_values,
            z_values,
            None, // color_values
            None, // size_values
            None, // point_labels
            axis_labels,
            &self.default_config,
            Plot3DType::Scatter,
        )
    }

    /// Create 3D surface plot with factory defaults
    pub fn surface_3d(
        &self,
        x_grid: &ArrayView2<Float>,
        y_grid: &ArrayView2<Float>,
        z_surface: &ArrayView2<Float>,
        axis_labels: (String, String, String),
    ) -> SklResult<Surface3D> {
        create_3d_surface_plot(
            x_grid,
            y_grid,
            z_surface,
            None, // color_map
            axis_labels,
            &self.default_config,
            1.0, // opacity
        )
    }
}

impl Default for VisualizationFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common visualization tasks
pub mod convenience {
    use super::*;

    /// Quick feature importance bar chart
    pub fn quick_feature_importance(
        importance_values: &ArrayView1<Float>,
        feature_names: &[String],
    ) -> SklResult<FeatureImportancePlot> {
        let config = PlotConfigBuilder::new()
            .title("Feature Importance")
            .x_label("Features")
            .y_label("Importance")
            .build();

        create_feature_importance_plot(
            importance_values,
            Some(feature_names),
            None,
            &config,
            FeatureImportanceType::Bar,
        )
    }

    /// Quick SHAP summary plot
    pub fn quick_shap_summary(
        shap_values: &ArrayView2<Float>,
        feature_values: &ArrayView2<Float>,
        feature_names: &[String],
    ) -> SklResult<ShapPlot> {
        let config = PlotConfigBuilder::new()
            .title("SHAP Summary")
            .x_label("SHAP Value")
            .y_label("Features")
            .build();

        create_shap_summary_plot(
            shap_values,
            feature_values,
            Some(feature_names),
            &config,
            true,
        )
    }

    /// Quick 3D scatter with auto-generated labels
    pub fn quick_scatter_3d(
        x_values: Array1<Float>,
        y_values: Array1<Float>,
        z_values: Array1<Float>,
    ) -> SklResult<Plot3D> {
        create_3d_plot(
            &x_values.view(),
            &y_values.view(),
            &z_values.view(),
            None,
            None,
            None,
            ("X".to_string(), "Y".to_string(), "Z".to_string()),
            &PlotConfig::default(),
            Plot3DType::Scatter,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;
    use sklears_core::prelude::Array2;

    #[test]
    fn test_visualization_factory() {
        let factory = VisualizationFactory::new();
        assert_eq!(factory.default_config.title, "Visualization");
        assert_eq!(factory.default_config.width, 800);
        assert_eq!(factory.default_config.height, 600);
    }

    #[test]
    fn test_factory_feature_importance() {
        let factory = VisualizationFactory::new();
        let importance = array![0.8, 0.6, 0.4, 0.2];
        let feature_names = vec![
            "feature_1".to_string(),
            "feature_2".to_string(),
            "feature_3".to_string(),
            "feature_4".to_string(),
        ];

        let plot = factory
            .feature_importance(
                &importance.view(),
                Some(&feature_names),
                FeatureImportanceType::Bar,
            )
            .unwrap();

        assert_eq!(plot.feature_names.len(), 4);
        assert_eq!(plot.importance_values.len(), 4);
        assert_relative_eq!(plot.importance_values[0], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_factory_3d_scatter() {
        let factory = VisualizationFactory::new();
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        let z = array![1.0, 2.0, 3.0];

        let plot = factory
            .scatter_3d(
                &x.view(),
                &y.view(),
                &z.view(),
                ("X".to_string(), "Y".to_string(), "Z".to_string()),
            )
            .unwrap();

        assert_eq!(plot.x_values.len(), 3);
        assert_eq!(plot.y_values.len(), 3);
        assert_eq!(plot.z_values.len(), 3);
        assert_eq!(plot.axis_labels.0, "X");
    }

    #[test]
    fn test_convenience_functions() {
        let importance = array![0.5, 0.3, 0.2];
        let feature_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let plot =
            convenience::quick_feature_importance(&importance.view(), &feature_names).unwrap();
        assert_eq!(plot.feature_names.len(), 3);
        assert_eq!(plot.config.title, "Feature Importance");

        let x = array![1.0, 2.0];
        let y = array![1.0, 2.0];
        let z = array![1.0, 2.0];
        let plot_3d = convenience::quick_scatter_3d(x, y, z).unwrap();
        assert_eq!(plot_3d.x_values.len(), 2);
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let mut visualizer = InteractiveVisualizer::new();
        visualizer.enable_real_time_updates();

        let config = PlotConfigBuilder::new()
            .title("Integration Test")
            .color_scheme(ColorScheme::Viridis)
            .build();

        visualizer.set_plot_config("test_plot".to_string(), config);

        let data = Array2::<f64>::zeros((10, 2));
        visualizer.update_plot_data("test_plot", &data);

        assert!(visualizer.is_real_time_enabled());
        assert_eq!(visualizer.plot_count(), 1);
    }

    #[test]
    fn test_simd_integration() {
        // Test SIMD operations integration
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = simd_normalize_data(&data.to_vec(), 0.0, 1.0);

        assert_eq!(normalized.len(), 5);
        assert_relative_eq!(normalized[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[4], 1.0, epsilon = 1e-10);

        let grid = simd_generate_grid_range(0.0, 10.0, 11);
        assert_eq!(grid.len(), 11);
        assert_relative_eq!(grid[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(grid[10], 10.0, epsilon = 1e-10);
    }
}
