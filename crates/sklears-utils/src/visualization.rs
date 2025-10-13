//! Visualization utilities for machine learning data preparation
//!
//! This module provides utilities for preparing data for visualization,
//! chart data formatting, and plotting helpers for ML workflows.

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Chart data preparation utilities
pub struct ChartData;

impl ChartData {
    /// Prepare data for scatter plot visualization
    pub fn prepare_scatter_plot(
        x: &Array1<f64>,
        y: &Array1<f64>,
        labels: Option<&Array1<String>>,
    ) -> UtilsResult<ScatterPlotData> {
        if x.len() != y.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![x.len()],
                actual: vec![y.len()],
            });
        }

        let points: Vec<Point2D> = x
            .iter()
            .zip(y.iter())
            .map(|(&x_val, &y_val)| Point2D { x: x_val, y: y_val })
            .collect();

        let labels = labels
            .map(|l| l.to_vec())
            .unwrap_or_else(|| (0..x.len()).map(|i| format!("Point {i}")).collect());

        Ok(ScatterPlotData { points, labels })
    }

    /// Prepare data for line plot visualization
    pub fn prepare_line_plot(
        x: &Array1<f64>,
        y: &Array1<f64>,
        line_name: Option<String>,
    ) -> UtilsResult<LinePlotData> {
        if x.len() != y.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![x.len()],
                actual: vec![y.len()],
            });
        }

        let points: Vec<Point2D> = x
            .iter()
            .zip(y.iter())
            .map(|(&x_val, &y_val)| Point2D { x: x_val, y: y_val })
            .collect();

        Ok(LinePlotData {
            points,
            name: line_name.unwrap_or_else(|| "Line".to_string()),
        })
    }

    /// Prepare data for histogram visualization
    pub fn prepare_histogram(
        data: &Array1<f64>,
        bins: Option<usize>,
    ) -> UtilsResult<HistogramData> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let bins = bins.unwrap_or(10);
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if min_val == max_val {
            return Err(UtilsError::InvalidParameter(
                "All values are the same, cannot create histogram".to_string(),
            ));
        }

        let bin_width = (max_val - min_val) / bins as f64;
        let mut bin_counts = vec![0; bins];
        let mut bin_edges = Vec::with_capacity(bins + 1);

        // Create bin edges
        for i in 0..=bins {
            bin_edges.push(min_val + i as f64 * bin_width);
        }

        // Count values in each bin
        for &value in data.iter() {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1); // Handle edge case for max value
            bin_counts[bin_index] += 1;
        }

        Ok(HistogramData {
            counts: bin_counts,
            bin_edges,
            total_count: data.len(),
        })
    }

    /// Prepare data for heatmap visualization
    pub fn prepare_heatmap(
        data: &Array2<f64>,
        row_labels: Option<&[String]>,
        col_labels: Option<&[String]>,
    ) -> UtilsResult<HeatmapData> {
        let (rows, cols) = data.dim();

        if rows == 0 || cols == 0 {
            return Err(UtilsError::EmptyInput);
        }

        let values: Vec<Vec<f64>> = data.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();

        let row_labels = row_labels
            .map(|labels| labels.to_vec())
            .unwrap_or_else(|| (0..rows).map(|i| format!("Row {i}")).collect());

        let col_labels = col_labels
            .map(|labels| labels.to_vec())
            .unwrap_or_else(|| (0..cols).map(|i| format!("Col {i}")).collect());

        // Calculate min/max for color scaling
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Ok(HeatmapData {
            values,
            row_labels,
            col_labels,
            min_value: min_val,
            max_value: max_val,
        })
    }

    /// Prepare data for box plot visualization
    pub fn prepare_box_plot(data: &Array1<f64>, label: Option<String>) -> UtilsResult<BoxPlotData> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted_data.len();
        let q1 = Self::calculate_quantile(&sorted_data, 0.25);
        let median = Self::calculate_quantile(&sorted_data, 0.5);
        let q3 = Self::calculate_quantile(&sorted_data, 0.75);

        let iqr = q3 - q1;
        let lower_fence = q1 - 1.5 * iqr;
        let upper_fence = q3 + 1.5 * iqr;

        let outliers: Vec<f64> = sorted_data
            .iter()
            .copied()
            .filter(|&x| x < lower_fence || x > upper_fence)
            .collect();

        let whisker_low = sorted_data
            .iter()
            .find(|&&x| x >= lower_fence)
            .copied()
            .unwrap_or(sorted_data[0]);

        let whisker_high = sorted_data
            .iter()
            .rev()
            .find(|&&x| x <= upper_fence)
            .copied()
            .unwrap_or(sorted_data[len - 1]);

        Ok(BoxPlotData {
            q1,
            median,
            q3,
            whisker_low,
            whisker_high,
            outliers,
            label: label.unwrap_or_else(|| "Data".to_string()),
        })
    }

    fn calculate_quantile(sorted_data: &[f64], quantile: f64) -> f64 {
        let index = quantile * (sorted_data.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            sorted_data[lower_index]
        } else {
            let weight = index - index.floor();
            sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight
        }
    }
}

/// Plotting utilities for ML visualization
pub struct PlotUtils;

impl PlotUtils {
    /// Create color palette for categorical data
    pub fn create_color_palette(num_colors: usize) -> Vec<Color> {
        let base_colors = vec![
            Color::rgb(31, 119, 180),  // Blue
            Color::rgb(255, 127, 14),  // Orange
            Color::rgb(44, 160, 44),   // Green
            Color::rgb(214, 39, 40),   // Red
            Color::rgb(148, 103, 189), // Purple
            Color::rgb(140, 86, 75),   // Brown
            Color::rgb(227, 119, 194), // Pink
            Color::rgb(127, 127, 127), // Gray
            Color::rgb(188, 189, 34),  // Olive
            Color::rgb(23, 190, 207),  // Cyan
        ];

        if num_colors <= base_colors.len() {
            base_colors.into_iter().take(num_colors).collect()
        } else {
            // Generate additional colors using HSV color space
            let base_len = base_colors.len();
            let mut colors = base_colors;
            for i in base_len..num_colors {
                let hue = (i as f64 * 360.0 / num_colors as f64) % 360.0;
                let color = Color::from_hsv(hue, 0.8, 0.8);
                colors.push(color);
            }
            colors
        }
    }

    /// Create axis configuration for plots
    pub fn create_axis_config(
        label: &str,
        min_val: Option<f64>,
        max_val: Option<f64>,
        tick_count: Option<usize>,
    ) -> AxisConfig {
        AxisConfig {
            label: label.to_string(),
            min_value: min_val,
            max_value: max_val,
            tick_count: tick_count.unwrap_or(10),
            grid_lines: true,
            log_scale: false,
        }
    }

    /// Format data for JSON export
    pub fn to_json(plot_data: &PlotData) -> UtilsResult<String> {
        serde_json::to_string_pretty(plot_data)
            .map_err(|e| UtilsError::InvalidParameter(format!("JSON serialization error: {e}")))
    }

    /// Format data for CSV export
    pub fn to_csv(scatter_data: &ScatterPlotData) -> UtilsResult<String> {
        let mut csv = String::new();
        csv.push_str("x,y,label\n");

        for (point, label) in scatter_data.points.iter().zip(&scatter_data.labels) {
            csv.push_str(&format!("{},{},{}\n", point.x, point.y, label));
        }

        Ok(csv)
    }

    /// Create plot layout configuration
    pub fn create_layout(
        title: &str,
        x_axis: AxisConfig,
        y_axis: AxisConfig,
        width: Option<u32>,
        height: Option<u32>,
    ) -> PlotLayout {
        PlotLayout {
            title: title.to_string(),
            x_axis,
            y_axis,
            width: width.unwrap_or(800),
            height: height.unwrap_or(600),
            background_color: Color::rgb(255, 255, 255),
            margin: PlotMargin {
                top: 50,
                right: 50,
                bottom: 80,
                left: 80,
            },
        }
    }

    /// Generate plot summary statistics
    pub fn generate_plot_summary(plot_data: &PlotData) -> PlotSummary {
        match plot_data {
            PlotData::Scatter(data) => PlotSummary {
                plot_type: "scatter".to_string(),
                data_points: data.points.len(),
                summary_stats: Self::calculate_scatter_stats(&data.points),
            },
            PlotData::Line(data) => PlotSummary {
                plot_type: "line".to_string(),
                data_points: data.points.len(),
                summary_stats: Self::calculate_scatter_stats(&data.points),
            },
            PlotData::Histogram(data) => PlotSummary {
                plot_type: "histogram".to_string(),
                data_points: data.total_count,
                summary_stats: HashMap::from([
                    ("bins".to_string(), data.counts.len() as f64),
                    (
                        "max_count".to_string(),
                        *data.counts.iter().max().unwrap_or(&0) as f64,
                    ),
                ]),
            },
            PlotData::Heatmap(data) => PlotSummary {
                plot_type: "heatmap".to_string(),
                data_points: data.values.len() * data.values.first().map_or(0, |row| row.len()),
                summary_stats: HashMap::from([
                    ("rows".to_string(), data.values.len() as f64),
                    (
                        "cols".to_string(),
                        data.values.first().map_or(0.0, |row| row.len() as f64),
                    ),
                    ("min_value".to_string(), data.min_value),
                    ("max_value".to_string(), data.max_value),
                ]),
            },
            PlotData::BoxPlot(data) => PlotSummary {
                plot_type: "boxplot".to_string(),
                data_points: 1, // One box
                summary_stats: HashMap::from([
                    ("q1".to_string(), data.q1),
                    ("median".to_string(), data.median),
                    ("q3".to_string(), data.q3),
                    ("outliers".to_string(), data.outliers.len() as f64),
                ]),
            },
        }
    }

    fn calculate_scatter_stats(points: &[Point2D]) -> HashMap<String, f64> {
        if points.is_empty() {
            return HashMap::new();
        }

        let x_values: Vec<f64> = points.iter().map(|p| p.x).collect();
        let y_values: Vec<f64> = points.iter().map(|p| p.y).collect();

        let x_min = x_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        HashMap::from([
            ("x_min".to_string(), x_min),
            ("x_max".to_string(), x_max),
            ("y_min".to_string(), y_min),
            ("y_max".to_string(), y_max),
            ("x_range".to_string(), x_max - x_min),
            ("y_range".to_string(), y_max - y_min),
        ])
    }
}

/// Data structures for different plot types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterPlotData {
    pub points: Vec<Point2D>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinePlotData {
    pub points: Vec<Point2D>,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub counts: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    pub values: Vec<Vec<f64>>,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    pub min_value: f64,
    pub max_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxPlotData {
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub whisker_low: f64,
    pub whisker_high: f64,
    pub outliers: Vec<f64>,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotData {
    Scatter(ScatterPlotData),
    Line(LinePlotData),
    Histogram(HistogramData),
    Heatmap(HeatmapData),
    BoxPlot(BoxPlotData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: f64,
}

impl Color {
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub fn rgba(r: u8, g: u8, b: u8, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn from_hsv(h: f64, s: f64, v: f64) -> Self {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r_prime, g_prime, b_prime) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        Self {
            r: ((r_prime + m) * 255.0) as u8,
            g: ((g_prime + m) * 255.0) as u8,
            b: ((b_prime + m) * 255.0) as u8,
            a: 1.0,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.a < 1.0 {
            write!(f, "rgba({}, {}, {}, {:.2})", self.r, self.g, self.b, self.a)
        } else {
            write!(f, "rgb({}, {}, {})", self.r, self.g, self.b)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    pub label: String,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub tick_count: usize,
    pub grid_lines: bool,
    pub log_scale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotMargin {
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub left: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotLayout {
    pub title: String,
    pub x_axis: AxisConfig,
    pub y_axis: AxisConfig,
    pub width: u32,
    pub height: u32,
    pub background_color: Color,
    pub margin: PlotMargin,
}

#[derive(Debug, Clone)]
pub struct PlotSummary {
    pub plot_type: String,
    pub data_points: usize,
    pub summary_stats: HashMap<String, f64>,
}

impl fmt::Display for PlotSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Plot Summary:")?;
        writeln!(f, "  Type: {}", self.plot_type)?;
        writeln!(f, "  Data Points: {}", self.data_points)?;
        writeln!(f, "  Statistics:")?;
        for (key, value) in &self.summary_stats {
            writeln!(f, "    {key}: {value:.4}")?;
        }
        Ok(())
    }
}

/// ML visualization specific utilities
pub struct MLVisualizationUtils;

impl MLVisualizationUtils {
    /// Prepare confusion matrix for visualization
    pub fn prepare_confusion_matrix(
        y_true: &Array1<usize>,
        y_pred: &Array1<usize>,
        class_names: Option<&[String]>,
    ) -> UtilsResult<HeatmapData> {
        if y_true.len() != y_pred.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        let num_classes = y_true.iter().max().unwrap_or(&0) + 1;
        let mut matrix = Array2::zeros((num_classes, num_classes));

        for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
            matrix[(true_label, pred_label)] += 1.0;
        }

        let labels = class_names
            .map(|names| names.to_vec())
            .unwrap_or_else(|| (0..num_classes).map(|i| format!("Class {i}")).collect());

        ChartData::prepare_heatmap(&matrix, Some(&labels), Some(&labels))
    }

    /// Prepare learning curve data for visualization
    pub fn prepare_learning_curve(
        train_sizes: &Array1<usize>,
        train_scores: &Array1<f64>,
        val_scores: &Array1<f64>,
    ) -> UtilsResult<(LinePlotData, LinePlotData)> {
        if train_sizes.len() != train_scores.len() || train_sizes.len() != val_scores.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![train_sizes.len()],
                actual: vec![train_scores.len(), val_scores.len()],
            });
        }

        let x_values: Array1<f64> = train_sizes.mapv(|x| x as f64);

        let train_line = ChartData::prepare_line_plot(
            &x_values,
            train_scores,
            Some("Training Score".to_string()),
        )?;
        let val_line = ChartData::prepare_line_plot(
            &x_values,
            val_scores,
            Some("Validation Score".to_string()),
        )?;

        Ok((train_line, val_line))
    }

    /// Prepare feature importance visualization
    pub fn prepare_feature_importance(
        feature_names: &[String],
        importance_scores: &Array1<f64>,
    ) -> UtilsResult<ScatterPlotData> {
        if feature_names.len() != importance_scores.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![feature_names.len()],
                actual: vec![importance_scores.len()],
            });
        }

        let x_values: Array1<f64> = (0..feature_names.len()).map(|i| i as f64).collect();
        ChartData::prepare_scatter_plot(
            &x_values,
            importance_scores,
            Some(&feature_names.to_vec().into()),
        )
    }

    /// Prepare ROC curve data
    pub fn prepare_roc_curve(
        fpr: &Array1<f64>,
        tpr: &Array1<f64>,
        auc: f64,
    ) -> UtilsResult<LinePlotData> {
        if fpr.len() != tpr.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![fpr.len()],
                actual: vec![tpr.len()],
            });
        }

        ChartData::prepare_line_plot(fpr, tpr, Some(format!("ROC Curve (AUC = {auc:.3})")))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_scatter_plot_preparation() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let labels = array![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string()
        ];

        let scatter_data = ChartData::prepare_scatter_plot(&x, &y, Some(&labels)).unwrap();

        assert_eq!(scatter_data.points.len(), 4);
        assert_eq!(scatter_data.labels.len(), 4);
        assert_eq!(scatter_data.points[0].x, 1.0);
        assert_eq!(scatter_data.points[0].y, 2.0);
        assert_eq!(scatter_data.labels[0], "A");
    }

    #[test]
    fn test_scatter_plot_shape_mismatch() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![2.0, 4.0];

        let result = ChartData::prepare_scatter_plot(&x, &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_line_plot_preparation() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 4.0, 9.0];

        let line_data =
            ChartData::prepare_line_plot(&x, &y, Some("Quadratic".to_string())).unwrap();

        assert_eq!(line_data.points.len(), 3);
        assert_eq!(line_data.name, "Quadratic");
        assert_eq!(line_data.points[1].x, 2.0);
        assert_eq!(line_data.points[1].y, 4.0);
    }

    #[test]
    fn test_histogram_preparation() {
        let data = array![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0];

        let hist_data = ChartData::prepare_histogram(&data, Some(4)).unwrap();

        assert_eq!(hist_data.counts.len(), 4);
        assert_eq!(hist_data.bin_edges.len(), 5);
        assert_eq!(hist_data.total_count, 8);
        assert!(hist_data.bin_edges[0] <= 1.0);
        assert!(hist_data.bin_edges[4] >= 5.0);
    }

    #[test]
    fn test_histogram_empty_data() {
        let data = array![];
        let result = ChartData::prepare_histogram(&data, Some(10));
        assert!(result.is_err());
    }

    #[test]
    fn test_heatmap_preparation() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let row_labels = vec!["Row1".to_string(), "Row2".to_string()];
        let col_labels = vec!["Col1".to_string(), "Col2".to_string(), "Col3".to_string()];

        let heatmap_data =
            ChartData::prepare_heatmap(&data, Some(&row_labels), Some(&col_labels)).unwrap();

        assert_eq!(heatmap_data.values.len(), 2);
        assert_eq!(heatmap_data.values[0].len(), 3);
        assert_eq!(heatmap_data.row_labels.len(), 2);
        assert_eq!(heatmap_data.col_labels.len(), 3);
        assert_eq!(heatmap_data.min_value, 1.0);
        assert_eq!(heatmap_data.max_value, 6.0);
    }

    #[test]
    fn test_box_plot_preparation() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let box_data = ChartData::prepare_box_plot(&data, Some("Test Data".to_string())).unwrap();

        assert_eq!(box_data.label, "Test Data");
        assert_abs_diff_eq!(box_data.median, 5.5, epsilon = 1e-10);
        assert_abs_diff_eq!(box_data.q1, 3.25, epsilon = 1e-10);
        assert_abs_diff_eq!(box_data.q3, 7.75, epsilon = 1e-10);
        assert!(box_data.outliers.is_empty());
    }

    #[test]
    fn test_box_plot_with_outliers() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is an outlier

        let box_data = ChartData::prepare_box_plot(&data, None).unwrap();

        assert!(!box_data.outliers.is_empty());
        assert!(box_data.outliers.contains(&100.0));
    }

    #[test]
    fn test_color_palette_generation() {
        let colors = PlotUtils::create_color_palette(5);
        assert_eq!(colors.len(), 5);

        // Test that we get more colors than base palette
        let many_colors = PlotUtils::create_color_palette(15);
        assert_eq!(many_colors.len(), 15);
    }

    #[test]
    fn test_color_from_hsv() {
        let red = Color::from_hsv(0.0, 1.0, 1.0);
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);

        let green = Color::from_hsv(120.0, 1.0, 1.0);
        assert_eq!(green.r, 0);
        assert_eq!(green.g, 255);
        assert_eq!(green.b, 0);
    }

    #[test]
    fn test_color_display() {
        let color_rgb = Color::rgb(255, 128, 64);
        assert_eq!(color_rgb.to_string(), "rgb(255, 128, 64)");

        let color_rgba = Color::rgba(255, 128, 64, 0.5);
        assert_eq!(color_rgba.to_string(), "rgba(255, 128, 64, 0.50)");
    }

    #[test]
    fn test_axis_config_creation() {
        let axis = PlotUtils::create_axis_config("X Axis", Some(0.0), Some(10.0), Some(5));

        assert_eq!(axis.label, "X Axis");
        assert_eq!(axis.min_value, Some(0.0));
        assert_eq!(axis.max_value, Some(10.0));
        assert_eq!(axis.tick_count, 5);
        assert!(axis.grid_lines);
        assert!(!axis.log_scale);
    }

    #[test]
    fn test_plot_layout_creation() {
        let x_axis = PlotUtils::create_axis_config("X", None, None, None);
        let y_axis = PlotUtils::create_axis_config("Y", None, None, None);

        let layout = PlotUtils::create_layout("Test Plot", x_axis, y_axis, Some(1000), Some(800));

        assert_eq!(layout.title, "Test Plot");
        assert_eq!(layout.width, 1000);
        assert_eq!(layout.height, 800);
    }

    #[test]
    fn test_json_export() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let scatter_data = ChartData::prepare_scatter_plot(&x, &y, None).unwrap();
        let plot_data = PlotData::Scatter(scatter_data);

        let json_result = PlotUtils::to_json(&plot_data);
        assert!(json_result.is_ok());

        let json = json_result.unwrap();
        assert!(json.contains("Scatter"));
        assert!(json.contains("points"));
    }

    #[test]
    fn test_csv_export() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let scatter_data = ChartData::prepare_scatter_plot(&x, &y, None).unwrap();

        let csv = PlotUtils::to_csv(&scatter_data).unwrap();

        assert!(csv.contains("x,y,label"));
        assert!(csv.contains("1,3"));
        assert!(csv.contains("2,4"));
    }

    #[test]
    fn test_plot_summary_generation() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![2.0, 4.0, 6.0];
        let scatter_data = ChartData::prepare_scatter_plot(&x, &y, None).unwrap();
        let plot_data = PlotData::Scatter(scatter_data);

        let summary = PlotUtils::generate_plot_summary(&plot_data);

        assert_eq!(summary.plot_type, "scatter");
        assert_eq!(summary.data_points, 3);
        assert!(summary.summary_stats.contains_key("x_min"));
        assert!(summary.summary_stats.contains_key("x_max"));
        assert!(summary.summary_stats.contains_key("y_min"));
        assert!(summary.summary_stats.contains_key("y_max"));
    }

    #[test]
    fn test_confusion_matrix_preparation() {
        let y_true = array![0, 0, 1, 1, 2, 2];
        let y_pred = array![0, 1, 1, 1, 2, 0];
        let class_names = vec![
            "Class A".to_string(),
            "Class B".to_string(),
            "Class C".to_string(),
        ];

        let heatmap =
            MLVisualizationUtils::prepare_confusion_matrix(&y_true, &y_pred, Some(&class_names))
                .unwrap();

        assert_eq!(heatmap.values.len(), 3);
        assert_eq!(heatmap.values[0].len(), 3);
        assert_eq!(heatmap.row_labels[0], "Class A");
        assert_eq!(heatmap.col_labels[1], "Class B");

        // Check some values in confusion matrix
        assert_eq!(heatmap.values[0][0], 1.0); // True A, Pred A
        assert_eq!(heatmap.values[0][1], 1.0); // True A, Pred B
        assert_eq!(heatmap.values[1][1], 2.0); // True B, Pred B
    }

    #[test]
    fn test_learning_curve_preparation() {
        let train_sizes = array![100, 200, 300];
        let train_scores = array![0.8, 0.85, 0.87];
        let val_scores = array![0.75, 0.82, 0.83];

        let (train_line, val_line) =
            MLVisualizationUtils::prepare_learning_curve(&train_sizes, &train_scores, &val_scores)
                .unwrap();

        assert_eq!(train_line.name, "Training Score");
        assert_eq!(val_line.name, "Validation Score");
        assert_eq!(train_line.points.len(), 3);
        assert_eq!(val_line.points.len(), 3);

        assert_eq!(train_line.points[0].x, 100.0);
        assert_eq!(train_line.points[0].y, 0.8);
        assert_eq!(val_line.points[1].x, 200.0);
        assert_eq!(val_line.points[1].y, 0.82);
    }

    #[test]
    fn test_feature_importance_preparation() {
        let features = vec![
            "Feature1".to_string(),
            "Feature2".to_string(),
            "Feature3".to_string(),
        ];
        let importance = array![0.5, 0.3, 0.2];

        let scatter_data =
            MLVisualizationUtils::prepare_feature_importance(&features, &importance).unwrap();

        assert_eq!(scatter_data.points.len(), 3);
        assert_eq!(scatter_data.labels.len(), 3);
        assert_eq!(scatter_data.labels[0], "Feature1");
        assert_eq!(scatter_data.points[0].x, 0.0);
        assert_eq!(scatter_data.points[0].y, 0.5);
    }

    #[test]
    fn test_roc_curve_preparation() {
        let fpr = array![0.0, 0.2, 0.4, 1.0];
        let tpr = array![0.0, 0.6, 0.8, 1.0];
        let auc = 0.85;

        let roc_line = MLVisualizationUtils::prepare_roc_curve(&fpr, &tpr, auc).unwrap();

        assert_eq!(roc_line.points.len(), 4);
        assert!(roc_line.name.contains("ROC Curve"));
        assert!(roc_line.name.contains("0.850"));
        assert_eq!(roc_line.points[0].x, 0.0);
        assert_eq!(roc_line.points[0].y, 0.0);
        assert_eq!(roc_line.points[3].x, 1.0);
        assert_eq!(roc_line.points[3].y, 1.0);
    }

    #[test]
    fn test_plot_summary_display() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let scatter_data = ChartData::prepare_scatter_plot(&x, &y, None).unwrap();
        let plot_data = PlotData::Scatter(scatter_data);

        let summary = PlotUtils::generate_plot_summary(&plot_data);
        let display = format!("{summary}");

        assert!(display.contains("Plot Summary:"));
        assert!(display.contains("Type: scatter"));
        assert!(display.contains("Data Points: 2"));
        assert!(display.contains("Statistics:"));
    }
}
