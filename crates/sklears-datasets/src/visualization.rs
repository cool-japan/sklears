//! Dataset visualization utilities
//!
//! This module provides visualization capabilities for synthetic datasets and data analysis:
//! - 2D/3D scatter plots for classification and clustering datasets
//! - Distribution histograms and box plots
//! - Correlation matrix heatmaps
//! - Data quality metric visualizations
//! - Dataset comparison plots
//!
//! All visualization features require the `visualization` feature flag to be enabled.

#[cfg(feature = "visualization")]
use colorgrad::{Color as ColorGradColor, Gradient};
#[cfg(feature = "visualization")]
use image::{ImageBuffer, Rgb, RgbImage};
#[cfg(feature = "visualization")]
use scirs2_core::ndarray::s;
#[cfg(feature = "visualization")]
use plotters::prelude::*;
#[cfg(feature = "visualization")]
use plotters::style::Color as PlottersColor;

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Error types for visualization operations
#[derive(Error, Debug)]
pub enum VisualizationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Plotting error: {0}")]
    Plotting(String),
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    #[error("Color mapping error: {0}")]
    ColorMapping(String),
    #[cfg(feature = "visualization")]
    #[error("Plotters backend error: {0}")]
    PlottersBackend(String),
}

pub type VisualizationResult<T> = Result<T, VisualizationError>;

/// Configuration for visualization plots
#[cfg(feature = "visualization")]
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Plot width in pixels
    pub width: u32,
    /// Plot height in pixels
    pub height: u32,
    /// Plot title
    pub title: String,
    /// X-axis label
    pub xlabel: String,
    /// Y-axis label
    pub ylabel: String,
    /// Background color
    pub background_color: ColorGradColor,
    /// Point size for scatter plots
    pub point_size: i32,
    /// Use grid
    pub show_grid: bool,
    /// Color palette name
    pub color_palette: String,
}

#[cfg(feature = "visualization")]
impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "Dataset Visualization".to_string(),
            xlabel: "Feature 1".to_string(),
            ylabel: "Feature 2".to_string(),
            background_color: ColorGradColor::new(1.0, 1.0, 1.0, 1.0), // White
            point_size: 3,
            show_grid: true,
            color_palette: "viridis".to_string(),
        }
    }
}

/// Plot a 2D scatter plot for classification datasets
#[cfg(feature = "visualization")]
pub fn plot_2d_classification<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();

    if features.ncols() < 2 {
        return Err(VisualizationError::InvalidDimensions(
            "Dataset must have at least 2 features for 2D plotting".to_string(),
        ));
    }

    let root = BitMapBackend::new(&path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    // Extract first two features
    let x_values: Vec<f64> = features.column(0).to_vec();
    let y_values: Vec<f64> = features.column(1).to_vec();
    let targets_vec: Vec<i32> = targets.to_vec();

    // Find data ranges
    let x_min = x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Add padding
    let x_padding = (x_max - x_min) * 0.1;
    let y_padding = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("Arial", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            x_min - x_padding..x_max + x_padding,
            y_min - y_padding..y_max + y_padding,
        )
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    if config.show_grid {
        chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .draw()
            .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;
    }

    // Get unique classes for color mapping
    let unique_classes: std::collections::BTreeSet<i32> = targets_vec.iter().cloned().collect();
    let n_classes = unique_classes.len();

    // Create color gradient
    let gradient = get_color_gradient(&config.color_palette, n_classes)?;

    // Plot points for each class
    for (class_idx, &class_label) in unique_classes.iter().enumerate() {
        let class_color = gradient.at((class_idx as f64 / (n_classes - 1).max(1) as f64) as f32);
        let plot_color = RGBColor(
            (class_color.r * 255.0) as u8,
            (class_color.g * 255.0) as u8,
            (class_color.b * 255.0) as u8,
        );

        let class_points: Vec<(f64, f64)> = x_values
            .iter()
            .zip(y_values.iter())
            .zip(targets_vec.iter())
            .filter_map(|((&x, &y), &target)| {
                if target == class_label {
                    Some((x, y))
                } else {
                    None
                }
            })
            .collect();

        chart
            .draw_series(
                class_points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), config.point_size, plot_color.filled())),
            )
            .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?
            .label(format!("Class {}", class_label))
            .legend(move |(x, y)| Circle::new((x + 5, y), 3, plot_color.filled()));
    }

    chart
        .configure_series_labels()
        .draw()
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;
    root.present()
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    Ok(())
}

/// Plot a 2D scatter plot for regression datasets
#[cfg(feature = "visualization")]
pub fn plot_2d_regression<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();

    if features.ncols() < 2 {
        return Err(VisualizationError::InvalidDimensions(
            "Dataset must have at least 2 features for 2D plotting".to_string(),
        ));
    }

    let root = BitMapBackend::new(&path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    // Extract first two features
    let x_values: Vec<f64> = features.column(0).to_vec();
    let y_values: Vec<f64> = features.column(1).to_vec();
    let targets_vec: Vec<f64> = targets.to_vec();

    // Find data ranges
    let x_min = x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let target_min = targets_vec.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let target_max = targets_vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Add padding
    let x_padding = (x_max - x_min) * 0.1;
    let y_padding = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("Arial", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            x_min - x_padding..x_max + x_padding,
            y_min - y_padding..y_max + y_padding,
        )
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    if config.show_grid {
        chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .draw()
            .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;
    }

    // Create color gradient based on target values
    let gradient = get_color_gradient(&config.color_palette, 100)?;

    // Plot points colored by target value
    let points_with_colors: Vec<((f64, f64), RGBColor)> = x_values
        .iter()
        .zip(y_values.iter())
        .zip(targets_vec.iter())
        .map(|((&x, &y), &target)| {
            let normalized_target = if target_max > target_min {
                (target - target_min) / (target_max - target_min)
            } else {
                0.5
            };
            let color = gradient.at(normalized_target as f32);
            let plot_color = RGBColor(
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8,
            );
            ((x, y), plot_color)
        })
        .collect();

    chart
        .draw_series(
            points_with_colors
                .iter()
                .map(|&((x, y), color)| Circle::new((x, y), config.point_size, color.filled())),
        )
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    root.present()
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    Ok(())
}

/// Plot histograms for feature distributions
#[cfg(feature = "visualization")]
pub fn plot_feature_distributions<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    feature_names: Option<&[String]>,
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();
    let (n_samples, n_features) = features.dim();

    if n_samples == 0 {
        return Err(VisualizationError::InsufficientData(
            "Dataset contains no samples".to_string(),
        ));
    }

    let root = BitMapBackend::new(&path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    // Calculate subplot layout
    let cols = (n_features as f64).sqrt().ceil() as u32;
    let rows = (n_features as f64 / cols as f64).ceil() as u32;

    let sub_plots = root.split_evenly((rows as usize, cols as usize));

    for (feature_idx, subplot) in sub_plots.into_iter().enumerate().take(n_features) {
        if feature_idx >= n_features {
            break;
        }

        let feature_values: Vec<f64> = features.column(feature_idx).to_vec();
        let feature_name = feature_names
            .and_then(|names| names.get(feature_idx))
            .map(|s| s.clone())
            .unwrap_or_else(|| format!("Feature {}", feature_idx));

        // Calculate histogram
        let min_val = feature_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = feature_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let n_bins = 30;
        let bin_width = (max_val - min_val) / n_bins as f64;

        let mut histogram = vec![0u32; n_bins];
        for &value in &feature_values {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            histogram[bin_idx] += 1;
        }

        let max_count = *histogram.iter().max().unwrap_or(&1);

        let mut chart = ChartBuilder::on(&subplot)
            .caption(&feature_name, ("Arial", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_val..max_val, 0u32..max_count)
            .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .draw()
            .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

        chart
            .draw_series(histogram.iter().enumerate().map(|(i, &count)| {
                let x_start = min_val + i as f64 * bin_width;
                let x_end = x_start + bin_width;
                Rectangle::new([(x_start, 0), (x_end, count)], BLUE.filled())
            }))
            .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;
    }

    root.present()
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    Ok(())
}

/// Plot correlation matrix heatmap
#[cfg(feature = "visualization")]
pub fn plot_correlation_matrix<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    feature_names: Option<&[String]>,
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();
    let (n_samples, n_features) = features.dim();

    if n_samples < 2 {
        return Err(VisualizationError::InsufficientData(
            "Need at least 2 samples to compute correlation".to_string(),
        ));
    }

    // Compute correlation matrix
    let mut correlation_matrix = Array2::<f64>::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                correlation_matrix[[i, j]] = 1.0;
            } else {
                let feature_i = features.column(i);
                let feature_j = features.column(j);
                let corr = compute_correlation(&feature_i.to_vec(), &feature_j.to_vec());
                correlation_matrix[[i, j]] = corr;
            }
        }
    }

    let root = BitMapBackend::new(&path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("Arial", 30))
        .margin(50)
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(0f64..n_features as f64, 0f64..n_features as f64)
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    // Create color gradient for correlation values (-1 to 1)
    let gradient = colorgrad::preset::rd_bu();

    // Draw heatmap
    for i in 0..n_features {
        for j in 0..n_features {
            let corr_value = correlation_matrix[[i, j]];
            let color = gradient.at(corr_value as f32);
            let plot_color = RGBColor(
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8,
            );

            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(i as f64, j as f64), (i as f64 + 1.0, j as f64 + 1.0)],
                    plot_color.filled(),
                )))
                .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

            // Add correlation value text
            if config.width > 600 && n_features <= 10 {
                chart
                    .draw_series(std::iter::once(Text::new(
                        format!("{:.2}", corr_value),
                        (i as f64 + 0.5, j as f64 + 0.5),
                        ("Arial", 12),
                    )))
                    .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;
            }
        }
    }

    // Set axis labels if feature names are provided
    if let Some(names) = feature_names {
        chart
            .configure_mesh()
            .x_desc("Features")
            .y_desc("Features")
            .x_label_formatter(&|x| {
                let idx = *x as usize;
                if idx < names.len() {
                    names[idx].clone()
                } else {
                    format!("F{}", idx)
                }
            })
            .y_label_formatter(&|y| {
                let idx = *y as usize;
                if idx < names.len() {
                    names[idx].clone()
                } else {
                    format!("F{}", idx)
                }
            })
            .draw()
            .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;
    } else {
        chart
            .configure_mesh()
            .x_desc("Features")
            .y_desc("Features")
            .draw()
            .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;
    }

    root.present()
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    Ok(())
}

/// Plot data quality metrics as a bar chart
#[cfg(feature = "visualization")]
pub fn plot_quality_metrics<P: AsRef<Path>>(
    path: P,
    metrics: &HashMap<String, f64>,
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();

    if metrics.is_empty() {
        return Err(VisualizationError::InsufficientData(
            "No quality metrics provided".to_string(),
        ));
    }

    let root = BitMapBackend::new(&path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    let metric_names: Vec<String> = metrics.keys().cloned().collect();
    let metric_values: Vec<f64> = metric_names.iter().map(|k| metrics[k]).collect();

    let max_value = metric_values
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_value = metric_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("Arial", 30))
        .margin(10)
        .x_label_area_size(80)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f64..metric_names.len() as f64,
            (min_value - 0.1).max(0.0)..max_value + 0.1,
        )
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    chart
        .configure_mesh()
        .x_desc("Quality Metrics")
        .y_desc("Score")
        .x_label_formatter(&|x| {
            let idx = *x as usize;
            if idx < metric_names.len() {
                metric_names[idx].clone()
            } else {
                String::new()
            }
        })
        .draw()
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    // Draw bars
    chart
        .draw_series(metric_values.iter().enumerate().map(|(i, &value)| {
            let color = if value >= 0.8 {
                GREEN.filled()
            } else if value >= 0.6 {
                YELLOW.filled()
            } else {
                RED.filled()
            };

            Rectangle::new([(i as f64, 0.0), (i as f64 + 0.8, value)], color)
        }))
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    root.present()
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    Ok(())
}

/// Helper function to get color gradient by name
#[cfg(feature = "visualization")]
fn get_color_gradient(
    palette_name: &str,
    _n_colors: usize,
) -> VisualizationResult<Box<dyn Gradient>> {
    let gradient: Box<dyn Gradient> = match palette_name.to_lowercase().as_str() {
        "viridis" => Box::new(colorgrad::preset::viridis()),
        "plasma" => Box::new(colorgrad::preset::plasma()),
        "inferno" => Box::new(colorgrad::preset::inferno()),
        "magma" => Box::new(colorgrad::preset::magma()),
        "turbo" => Box::new(colorgrad::preset::turbo()),
        "rainbow" => Box::new(colorgrad::preset::rainbow()),
        "sinebow" => Box::new(colorgrad::preset::sinebow()),
        "cool" => Box::new(colorgrad::preset::cool()),
        "warm" => Box::new(colorgrad::preset::warm()),
        _ => Box::new(colorgrad::preset::viridis()), // Default fallback
    };

    Ok(gradient)
}

/// Helper function to compute Pearson correlation coefficient
#[cfg(feature = "visualization")]
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let numerator: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
        .sum();

    let x_var: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
    let y_var: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let denominator = (x_var * y_var).sqrt();

    if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compare two datasets visually
#[cfg(feature = "visualization")]
pub fn plot_dataset_comparison<P: AsRef<Path>>(
    path: P,
    dataset1: (&Array2<f64>, &Array1<i32>, &str),
    dataset2: (&Array2<f64>, &Array1<i32>, &str),
    config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    let config = config.unwrap_or_default();
    let (features1, targets1, name1) = dataset1;
    let (features2, targets2, name2) = dataset2;

    if features1.ncols() < 2 || features2.ncols() < 2 {
        return Err(VisualizationError::InvalidDimensions(
            "Both datasets must have at least 2 features for comparison".to_string(),
        ));
    }

    let root = BitMapBackend::new(&path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    let (left, right) = root.split_horizontally(0.5);

    // Plot dataset 1
    plot_2d_classification_on_area(
        &left,
        features1,
        targets1,
        &format!("{} - {}", config.title, name1),
        &config,
    )?;

    // Plot dataset 2
    plot_2d_classification_on_area(
        &right,
        features2,
        targets2,
        &format!("{} - {}", config.title, name2),
        &config,
    )?;

    root.present()
        .map_err(|e| VisualizationError::PlottersBackend(format!("{:?}", e)))?;

    Ok(())
}

/// Helper function to plot classification data on a specific drawing area
#[cfg(feature = "visualization")]
fn plot_2d_classification_on_area<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    title: &str,
    config: &PlotConfig,
) -> VisualizationResult<()>
where
    DB::ErrorType: 'static,
{
    // Extract first two features
    let x_values: Vec<f64> = features.column(0).to_vec();
    let y_values: Vec<f64> = features.column(1).to_vec();
    let targets_vec: Vec<i32> = targets.to_vec();

    // Find data ranges
    let x_min = x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Add padding
    let x_padding = (x_max - x_min) * 0.1;
    let y_padding = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("Arial", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            x_min - x_padding..x_max + x_padding,
            y_min - y_padding..y_max + y_padding,
        )
        .map_err(|e| VisualizationError::Plotting(format!("Chart build error: {}", e)))?;

    if config.show_grid {
        chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .draw()
            .map_err(|e| VisualizationError::Plotting(format!("Mesh error: {}", e)))?;
    }

    // Get unique classes for color mapping
    let unique_classes: std::collections::BTreeSet<i32> = targets_vec.iter().cloned().collect();
    let n_classes = unique_classes.len();

    // Create color gradient
    let gradient = get_color_gradient(&config.color_palette, n_classes)?;

    // Plot points for each class
    for (class_idx, &class_label) in unique_classes.iter().enumerate() {
        let class_color = gradient.at((class_idx as f64 / (n_classes - 1).max(1) as f64) as f32);
        let plot_color = RGBColor(
            (class_color.r * 255.0) as u8,
            (class_color.g * 255.0) as u8,
            (class_color.b * 255.0) as u8,
        );

        let class_points: Vec<(f64, f64)> = x_values
            .iter()
            .zip(y_values.iter())
            .zip(targets_vec.iter())
            .filter_map(|((&x, &y), &target)| {
                if target == class_label {
                    Some((x, y))
                } else {
                    None
                }
            })
            .collect();

        chart
            .draw_series(
                class_points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), config.point_size, plot_color.filled())),
            )
            .map_err(|e| VisualizationError::Plotting(format!("Draw series error: {}", e)))?;
    }

    Ok(())
}

#[cfg(not(feature = "visualization"))]
/// Placeholder function when visualization feature is not enabled
pub fn plot_2d_classification<P: AsRef<Path>>(
    _path: P,
    _features: &ndarray::Array2<f64>,
    _targets: &ndarray::Array1<i32>,
    _config: Option<PlotConfig>,
) -> VisualizationResult<()> {
    Err(VisualizationError::Plotting(
        "Visualization feature not enabled. Enable with --features visualization".to_string(),
    ))
}

// Add similar placeholder functions for when visualization is disabled
#[cfg(not(feature = "visualization"))]
pub use self::disabled::*;

#[cfg(not(feature = "visualization"))]
mod disabled {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};
    use std::collections::HashMap;

    pub fn plot_2d_regression<P: AsRef<std::path::Path>>(
        _path: P,
        _features: &Array2<f64>,
        _targets: &Array1<f64>,
        _config: Option<PlotConfig>,
    ) -> VisualizationResult<()> {
        Err(VisualizationError::Plotting(
            "Visualization feature not enabled. Enable with --features visualization".to_string(),
        ))
    }

    pub fn plot_feature_distributions<P: AsRef<std::path::Path>>(
        _path: P,
        _features: &Array2<f64>,
        _feature_names: Option<&[String]>,
        _config: Option<PlotConfig>,
    ) -> VisualizationResult<()> {
        Err(VisualizationError::Plotting(
            "Visualization feature not enabled. Enable with --features visualization".to_string(),
        ))
    }

    pub fn plot_correlation_matrix<P: AsRef<std::path::Path>>(
        _path: P,
        _features: &Array2<f64>,
        _feature_names: Option<&[String]>,
        _config: Option<PlotConfig>,
    ) -> VisualizationResult<()> {
        Err(VisualizationError::Plotting(
            "Visualization feature not enabled. Enable with --features visualization".to_string(),
        ))
    }

    pub fn plot_quality_metrics<P: AsRef<std::path::Path>>(
        _path: P,
        _metrics: &HashMap<String, f64>,
        _config: Option<PlotConfig>,
    ) -> VisualizationResult<()> {
        Err(VisualizationError::Plotting(
            "Visualization feature not enabled. Enable with --features visualization".to_string(),
        ))
    }

    pub fn plot_dataset_comparison<P: AsRef<std::path::Path>>(
        _path: P,
        _dataset1: (&Array2<f64>, &Array1<i32>, &str),
        _dataset2: (&Array2<f64>, &Array1<i32>, &str),
        _config: Option<PlotConfig>,
    ) -> VisualizationResult<()> {
        Err(VisualizationError::Plotting(
            "Visualization feature not enabled. Enable with --features visualization".to_string(),
        ))
    }

    #[derive(Debug, Clone)]
    pub struct PlotConfig {
        pub width: u32,
        pub height: u32,
        pub title: String,
        pub xlabel: String,
        pub ylabel: String,
        pub point_size: i32,
        pub show_grid: bool,
        pub color_palette: String,
    }

    impl Default for PlotConfig {
        fn default() -> Self {
            Self {
                width: 800,
                height: 600,
                title: "Dataset Visualization".to_string(),
                xlabel: "Feature 1".to_string(),
                ylabel: "Feature 2".to_string(),
                point_size: 3,
                show_grid: true,
                color_palette: "viridis".to_string(),
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::basic::{make_blobs, make_classification, make_regression};
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.color_palette, "viridis");
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_compute_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = compute_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10); // Perfect positive correlation

        let y_neg = vec![-2.0, -4.0, -6.0, -8.0, -10.0];
        let corr_neg = compute_correlation(&x, &y_neg);
        assert!((corr_neg + 1.0).abs() < 1e-10); // Perfect negative correlation
    }

    #[cfg(not(feature = "visualization"))]
    #[test]
    fn test_visualization_disabled() {
        let (features, targets) = make_classification(50, 2, 2, 0, 2, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.png");

        let result = plot_2d_classification(&file_path, &features, &targets, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_2d_classification_plot() {
        let (features, targets) = make_classification(100, 2, 2, 0, 2, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_classification.png");

        let config = PlotConfig {
            title: "Test Classification".to_string(),
            xlabel: "X1".to_string(),
            ylabel: "X2".to_string(),
            ..Default::default()
        };

        let result = plot_2d_classification(&file_path, &features, &targets, Some(config));
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_2d_regression_plot() {
        let (features, targets) = make_regression(100, 2, 1, 0.1, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_regression.png");

        let result = plot_2d_regression(&file_path, &features, &targets, None);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_feature_distributions_plot() {
        let (features, _) = make_blobs(200, 3, 2, 1.0, Some(42)).unwrap();
        let feature_names = vec!["F1".to_string(), "F2".to_string(), "F3".to_string()];

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_distributions.png");

        let result = plot_feature_distributions(&file_path, &features, Some(&feature_names), None);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_correlation_matrix_plot() {
        let (features, _) = make_classification(100, 4, 3, 1, 3, Some(42)).unwrap();
        let feature_names = vec![
            "Feature1".to_string(),
            "Feature2".to_string(),
            "Feature3".to_string(),
            "Feature4".to_string(),
        ];

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_correlation.png");

        let config = PlotConfig {
            title: "Correlation Matrix".to_string(),
            ..Default::default()
        };

        let result =
            plot_correlation_matrix(&file_path, &features, Some(&feature_names), Some(config));
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_quality_metrics_plot() {
        let mut metrics = HashMap::new();
        metrics.insert("Completeness".to_string(), 0.95);
        metrics.insert("Consistency".to_string(), 0.88);
        metrics.insert("Validity".to_string(), 0.92);
        metrics.insert("Accuracy".to_string(), 0.86);
        metrics.insert("Overall".to_string(), 0.90);

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_quality.png");

        let config = PlotConfig {
            title: "Dataset Quality Metrics".to_string(),
            ..Default::default()
        };

        let result = plot_quality_metrics(&file_path, &metrics, Some(config));
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_dataset_comparison_plot() {
        let (features1, targets1) = make_blobs(100, 2, 3, 1.0, Some(42)).unwrap();
        let (features2, targets2) = make_blobs(100, 2, 3, 1.2, Some(24)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_comparison.png");

        let config = PlotConfig {
            title: "Dataset Comparison".to_string(),
            width: 1200,
            ..Default::default()
        };

        let result = plot_dataset_comparison(
            &file_path,
            (&features1, &targets1, "Blobs"),
            (&features2, &targets2, "Classification"),
            Some(config),
        );
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_error_handling() {
        // Test with insufficient features
        let (features, targets) = make_blobs(50, 1, 2, 1.0, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_error.png");

        let result = plot_2d_classification(&file_path, &features, &targets, None);
        assert!(result.is_err());

        // When visualization feature is disabled, we get a different error
        #[cfg(not(feature = "visualization"))]
        {
            match result.unwrap_err() {
                VisualizationError::Plotting(ref msg) if msg.contains("not enabled") => {}
                _ => panic!("Expected plotting error about feature not being enabled"),
            }
        }

        // When visualization feature is enabled, we should get InvalidDimensions
        #[cfg(feature = "visualization")]
        {
            match result.unwrap_err() {
                VisualizationError::InvalidDimensions(_) => {}
                _ => panic!("Expected InvalidDimensions error"),
            }
        }
    }
}
