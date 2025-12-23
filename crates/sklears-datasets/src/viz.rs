//! Dataset visualization utilities
//!
//! This module provides simple visualization functions for generated datasets
//! when the `visualization` feature is enabled.

use scirs2_core::ndarray::{Array1, Array2};
use std::path::Path;
use thiserror::Error;

/// Error types for visualization operations
#[derive(Debug, Error)]
pub enum VisualizationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Plotting error: {0}")]
    Plotting(String),
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

pub type VisualizationResult<T> = Result<T, VisualizationError>;

/// Configuration for plot appearance
#[derive(Debug, Clone)]
pub struct PlotConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    pub show_legend: bool,
    pub marker_size: u32,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "Dataset Visualization".to_string(),
            xlabel: "Feature 1".to_string(),
            ylabel: "Feature 2".to_string(),
            show_legend: true,
            marker_size: 3,
        }
    }
}

#[cfg(feature = "visualization")]
use plotters::prelude::*;

#[cfg(feature = "visualization")]
/// Plot 2D classification dataset with class labels
pub fn plot_2d_classification<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();

    if features.ncols() < 2 {
        return Err(VisualizationError::InvalidDimensions(
            "Need at least 2 features for 2D plot".to_string(),
        ));
    }

    let root = BitMapBackend::new(path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    // Find data ranges
    let x_min = features
        .column(0)
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let x_max = features
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = features
        .column(1)
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = features
        .column(1)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    chart
        .configure_mesh()
        .x_desc(&config.xlabel)
        .y_desc(&config.ylabel)
        .draw()
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    // Group points by class
    let mut class_points: std::collections::HashMap<i32, Vec<(f64, f64)>> =
        std::collections::HashMap::new();

    for i in 0..features.nrows() {
        let x = features[[i, 0]];
        let y = features[[i, 1]];
        let class = targets[i];
        class_points.entry(class).or_default().push((x, y));
    }

    // Plot each class with different color
    let colors = [&RED, &BLUE, &GREEN, &YELLOW, &MAGENTA, &CYAN];

    for (idx, (class, points)) in class_points.iter().enumerate() {
        let color = colors[idx % colors.len()];
        chart
            .draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), config.marker_size, color.filled())),
            )
            .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?
            .label(format!("Class {}", class))
            .legend(move |(x, y)| Circle::new((x, y), config.marker_size, color.filled()));
    }

    if config.show_legend {
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()
            .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;
    }

    root.present()
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    Ok(())
}

#[cfg(feature = "visualization")]
/// Plot 2D regression dataset with target values as colors
pub fn plot_2d_regression<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();

    if features.ncols() < 2 {
        return Err(VisualizationError::InvalidDimensions(
            "Need at least 2 features for 2D plot".to_string(),
        ));
    }

    let root = BitMapBackend::new(path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    // Find data ranges
    let x_min = features
        .column(0)
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let x_max = features
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = features
        .column(1)
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = features
        .column(1)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let t_min = targets.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    chart
        .configure_mesh()
        .x_desc(&config.xlabel)
        .y_desc(&config.ylabel)
        .draw()
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    // Plot points with color based on target value
    chart
        .draw_series((0..features.nrows()).map(|i| {
            let x = features[[i, 0]];
            let y = features[[i, 1]];
            let t = targets[i];

            // Normalize target value to 0-1 range for color mapping
            let normalized = if (t_max - t_min).abs() > 1e-10 {
                (t - t_min) / (t_max - t_min)
            } else {
                0.5
            };

            // Map to color (blue for low values, red for high values)
            let color = RGBColor(
                (normalized * 255.0) as u8,
                0,
                ((1.0 - normalized) * 255.0) as u8,
            );

            Circle::new((x, y), config.marker_size, color.filled())
        }))
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    root.present()
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    Ok(())
}

#[cfg(feature = "visualization")]
/// Plot feature distributions as histograms
pub fn plot_feature_distributions<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    feature_names: Option<&[String]>,
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();
    let n_features = features.ncols().min(4); // Plot up to 4 features

    let root = BitMapBackend::new(path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    let grid_rows = ((n_features as f64).sqrt().ceil()) as usize;
    let grid_cols = (n_features + grid_rows - 1) / grid_rows;

    let areas = root.split_evenly((grid_rows, grid_cols));

    for (idx, area) in areas.iter().enumerate().take(n_features) {
        let feature_data = features.column(idx);
        let default_name = format!("Feature {}", idx);
        let feature_name = feature_names
            .and_then(|names| names.get(idx))
            .map(|s| s.as_str())
            .unwrap_or(&default_name);

        // Calculate histogram
        let min_val = feature_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = feature_data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let n_bins = 20;
        let bin_width = (max_val - min_val) / n_bins as f64;

        let mut bins = vec![0usize; n_bins];
        for &val in feature_data.iter() {
            let bin_idx = ((val - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            bins[bin_idx] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&1);

        let mut chart = ChartBuilder::on(area)
            .caption(feature_name, ("sans-serif", 20).into_font())
            .margin(5)
            .x_label_area_size(20)
            .y_label_area_size(30)
            .build_cartesian_2d(min_val..max_val, 0usize..(max_count + 1))
            .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

        chart
            .configure_mesh()
            .draw()
            .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

        chart
            .draw_series(bins.iter().enumerate().map(|(i, &count)| {
                let x0 = min_val + i as f64 * bin_width;
                let x1 = x0 + bin_width;
                Rectangle::new([(x0, 0), (x1, count)], BLUE.mix(0.5).filled())
            }))
            .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;
    }

    root.present()
        .map_err(|e| VisualizationError::Plotting(format!("{}", e)))?;

    Ok(())
}

// Placeholder functions when visualization feature is not enabled
#[cfg(not(feature = "visualization"))]
pub fn plot_2d_classification<P: AsRef<Path>>(
    _path: P,
    _features: &Array2<f64>,
    _targets: &Array1<i32>,
    _config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    Err(VisualizationError::FeatureNotEnabled(
        "visualization feature is not enabled. Enable with --features visualization".to_string(),
    ))
}

#[cfg(not(feature = "visualization"))]
pub fn plot_2d_regression<P: AsRef<Path>>(
    _path: P,
    _features: &Array2<f64>,
    _targets: &Array1<f64>,
    _config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    Err(VisualizationError::FeatureNotEnabled(
        "visualization feature is not enabled. Enable with --features visualization".to_string(),
    ))
}

#[cfg(not(feature = "visualization"))]
pub fn plot_feature_distributions<P: AsRef<Path>>(
    _path: P,
    _features: &Array2<f64>,
    _feature_names: Option<&[String]>,
    _config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    Err(VisualizationError::FeatureNotEnabled(
        "visualization feature is not enabled. Enable with --features visualization".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[test]
    #[cfg(not(feature = "visualization"))]
    fn test_visualization_disabled() {
        use scirs2_core::ndarray::Array2;
        let features = Array2::zeros((10, 2));
        let targets = Array1::zeros(10);
        let int_targets: Array1<i32> = targets.mapv(|x: f64| x as i32);

        let result = plot_2d_classification("/tmp/test.png", &features, &int_targets, None);
        assert!(result.is_err());
    }
}
