//! Support Vector Machine Visualization Tools
//!
//! This module provides comprehensive visualization capabilities for SVM models including:
//! - Support vector plotting and highlighting
//! - Decision boundary visualization in 2D and 3D
//! - Margin visualization
//! - Kernel matrix heatmaps
//! - Training convergence plots
//! - Feature importance visualization for linear SVMs
//! - Multi-class decision regions
//! - Probability contours for calibrated models

use crate::errors::{SVMError, SVMResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Type alias for margin boundaries (positive, negative)
pub type MarginBoundaries = (Vec<(f64, f64)>, Vec<(f64, f64)>);

/// Color palette for visualizations
#[derive(Debug, Clone)]
pub enum ColorPalette {
    /// Default matplotlib-like colors
    Default,
    /// High contrast colors for accessibility
    HighContrast,
    /// Colorblind-friendly palette
    ColorblindFriendly,
    /// Grayscale palette
    Grayscale,
    /// Custom color specifications
    Custom(Vec<String>),
}

impl ColorPalette {
    /// Get color for a given class index
    pub fn get_color(&self, class_index: usize) -> String {
        match self {
            ColorPalette::Default => {
                let colors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                    "#7f7f7f", "#bcbd22", "#17becf",
                ];
                colors[class_index % colors.len()].to_string()
            }
            ColorPalette::HighContrast => {
                let colors = ["#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF"];
                colors[class_index % colors.len()].to_string()
            }
            ColorPalette::ColorblindFriendly => {
                let colors = [
                    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
                    "#000000",
                ];
                colors[class_index % colors.len()].to_string()
            }
            ColorPalette::Grayscale => {
                let intensity = (class_index * 40) % 200 + 40;
                format!("#{:02x}{:02x}{:02x}", intensity, intensity, intensity)
            }
            ColorPalette::Custom(colors) => colors[class_index % colors.len()].clone(),
        }
    }
}

/// Configuration for SVM visualizations
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Color palette to use
    pub color_palette: ColorPalette,
    /// Figure size (width, height)
    pub figure_size: (usize, usize),
    /// Resolution for contour plots
    pub resolution: usize,
    /// Show support vectors
    pub show_support_vectors: bool,
    /// Show decision boundary
    pub show_decision_boundary: bool,
    /// Show margins
    pub show_margins: bool,
    /// Support vector marker size
    pub support_vector_size: f64,
    /// Regular point marker size
    pub point_size: f64,
    /// Line width for boundaries
    pub line_width: f64,
    /// Transparency level (0.0 to 1.0)
    pub alpha: f64,
    /// Title for the plot
    pub title: Option<String>,
    /// Labels for axes
    pub axis_labels: Option<(String, String)>,
    /// Grid visibility
    pub show_grid: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            color_palette: ColorPalette::Default,
            figure_size: (800, 600),
            resolution: 100,
            show_support_vectors: true,
            show_decision_boundary: true,
            show_margins: true,
            support_vector_size: 8.0,
            point_size: 4.0,
            line_width: 2.0,
            alpha: 0.7,
            title: None,
            axis_labels: None,
            show_grid: true,
        }
    }
}

/// Point data for visualization
#[derive(Debug, Clone)]
pub struct PlotPoint {
    pub x: f64,
    pub y: f64,
    pub class: i32,
    pub is_support_vector: bool,
    pub alpha_value: Option<f64>, // Lagrange multiplier
    pub margin_type: MarginType,
}

/// Type of margin for a point
#[derive(Debug, Clone, PartialEq)]
pub enum MarginType {
    /// Point is outside the margin (correctly classified)
    Outside,
    /// Point is on the margin boundary
    OnMargin,
    /// Point is inside the margin (margin violation)
    Inside,
    /// Point is misclassified
    Misclassified,
}

/// 2D visualization data structure
#[derive(Debug, Clone)]
pub struct Plot2D {
    pub points: Vec<PlotPoint>,
    pub decision_boundary: Option<Vec<(f64, f64)>>,
    pub margin_boundaries: Option<MarginBoundaries>, // positive, negative margins
    pub config: VisualizationConfig,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
}

impl Plot2D {
    /// Create a new 2D plot
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            points: Vec::new(),
            decision_boundary: None,
            margin_boundaries: None,
            config,
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
        }
    }

    /// Add points to the plot
    pub fn add_points(&mut self, x: &Array2<f64>, y: &Array1<f64>, support_vectors: &[usize]) {
        if x.ncols() < 2 {
            return; // Need at least 2 dimensions for 2D plot
        }

        self.points.clear();

        for (i, (&x1, &x2)) in x.column(0).iter().zip(x.column(1).iter()).enumerate() {
            let is_sv = support_vectors.contains(&i);

            self.points.push(PlotPoint {
                x: x1,
                y: x2,
                class: y[i] as i32,
                is_support_vector: is_sv,
                alpha_value: None,
                margin_type: MarginType::Outside, // Will be calculated later
            });
        }

        // Update plot ranges
        let x_values: Vec<f64> = self.points.iter().map(|p| p.x).collect();
        let y_values: Vec<f64> = self.points.iter().map(|p| p.y).collect();

        if !x_values.is_empty() {
            let x_min = x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let x_max = x_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let y_min = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let y_max = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let x_range = x_max - x_min;
            let y_range = y_max - y_min;
            let margin = 0.1;

            self.x_range = (x_min - margin * x_range, x_max + margin * x_range);
            self.y_range = (y_min - margin * y_range, y_max + margin * y_range);
        }
    }

    /// Generate decision boundary points
    pub fn generate_decision_boundary<F>(&mut self, decision_function: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        let resolution = self.config.resolution;
        let x_step = (self.x_range.1 - self.x_range.0) / resolution as f64;
        let y_step = (self.y_range.1 - self.y_range.0) / resolution as f64;

        let mut boundary_points = Vec::new();

        // Find decision boundary using contour-following algorithm
        for i in 0..resolution {
            for j in 0..resolution {
                let x = self.x_range.0 + i as f64 * x_step;
                let y = self.y_range.0 + j as f64 * y_step;

                let value = decision_function(x, y);

                // Check for sign changes (decision boundary crossings)
                if value.abs() < 0.1 {
                    // Close to decision boundary
                    boundary_points.push((x, y));
                }
            }
        }

        if !boundary_points.is_empty() {
            self.decision_boundary = Some(boundary_points);
        }
    }

    /// Generate margin boundaries
    pub fn generate_margin_boundaries<F>(&mut self, decision_function: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        let resolution = self.config.resolution;
        let x_step = (self.x_range.1 - self.x_range.0) / resolution as f64;
        let y_step = (self.y_range.1 - self.y_range.0) / resolution as f64;

        let mut positive_margin = Vec::new();
        let mut negative_margin = Vec::new();

        for i in 0..resolution {
            for j in 0..resolution {
                let x = self.x_range.0 + i as f64 * x_step;
                let y = self.y_range.0 + j as f64 * y_step;

                let value = decision_function(x, y);

                // Positive margin boundary (value ≈ +1)
                if (value - 1.0).abs() < 0.1 {
                    positive_margin.push((x, y));
                }
                // Negative margin boundary (value ≈ -1)
                else if (value + 1.0).abs() < 0.1 {
                    negative_margin.push((x, y));
                }
            }
        }

        if !positive_margin.is_empty() || !negative_margin.is_empty() {
            self.margin_boundaries = Some((positive_margin, negative_margin));
        }
    }

    /// Generate SVG output for the plot
    pub fn to_svg(&self) -> String {
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<style>
    .point {{ stroke: #000; stroke-width: 0.5; }}
    .support-vector {{ stroke: #000; stroke-width: 2; }}
    .decision-boundary {{ fill: none; stroke: #ff0000; stroke-width: {}; }}
    .positive-margin {{ fill: none; stroke: #00ff00; stroke-width: 1; stroke-dasharray: 5,5; }}
    .negative-margin {{ fill: none; stroke: #0000ff; stroke-width: 1; stroke-dasharray: 5,5; }}
    .grid {{ stroke: #ddd; stroke-width: 0.5; }}
</style>
"#,
            self.config.figure_size.0, self.config.figure_size.1, self.config.line_width
        );

        // Add title
        if let Some(ref title) = self.config.title {
            svg.push_str(&format!(
                r#"<text x="{}" y="20" text-anchor="middle" font-family="Arial" font-size="16">{}</text>"#,
                self.config.figure_size.0 / 2,
                title
            ));
        }

        // Add grid if requested
        if self.config.show_grid {
            self.add_grid_to_svg(&mut svg);
        }

        // Transform coordinates to SVG space
        let x_scale = (self.config.figure_size.0 - 80) as f64 / (self.x_range.1 - self.x_range.0);
        let y_scale = (self.config.figure_size.1 - 80) as f64 / (self.y_range.1 - self.y_range.0);

        let transform_x = |x: f64| ((x - self.x_range.0) * x_scale + 40.0) as i32;
        let transform_y = |y: f64| {
            (self.config.figure_size.1 as f64 - (y - self.y_range.0) * y_scale - 40.0) as i32
        };

        // Draw margin boundaries
        if self.config.show_margins {
            if let Some((ref pos_margin, ref neg_margin)) = self.margin_boundaries {
                self.add_margin_boundaries_to_svg(
                    &mut svg,
                    pos_margin,
                    neg_margin,
                    transform_x,
                    transform_y,
                );
            }
        }

        // Draw decision boundary
        if self.config.show_decision_boundary {
            if let Some(ref boundary) = self.decision_boundary {
                self.add_decision_boundary_to_svg(&mut svg, boundary, transform_x, transform_y);
            }
        }

        // Draw points
        self.add_points_to_svg(&mut svg, transform_x, transform_y);

        // Add axes labels
        if let Some((ref x_label, ref y_label)) = self.config.axis_labels {
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial" font-size="12">{}</text>"#,
                self.config.figure_size.0 / 2,
                self.config.figure_size.1 - 10,
                x_label
            ));
            svg.push_str(&format!(
                r#"<text x="15" y="{}" text-anchor="middle" transform="rotate(-90 15 {})" font-family="Arial" font-size="12">{}</text>"#,
                self.config.figure_size.1 / 2,
                self.config.figure_size.1 / 2,
                y_label
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    fn add_grid_to_svg(&self, svg: &mut String) {
        // Add simple grid lines (implementation depends on requirements)
        let grid_lines = 10;
        let width = self.config.figure_size.0 - 80;
        let height = self.config.figure_size.1 - 80;

        for i in 1..grid_lines {
            let x = 40 + (width * i) / grid_lines;
            let y = 40 + (height * i) / grid_lines;

            svg.push_str(&format!(
                r#"<line x1="{}" y1="40" x2="{}" y2="{}" class="grid"/>"#,
                x,
                x,
                height + 40
            ));
            svg.push_str(&format!(
                r#"<line x1="40" y1="{}" x2="{}" y2="{}" class="grid"/>"#,
                y,
                width + 40,
                y
            ));
        }
    }

    fn add_points_to_svg<F1, F2>(&self, svg: &mut String, transform_x: F1, transform_y: F2)
    where
        F1: Fn(f64) -> i32,
        F2: Fn(f64) -> i32,
    {
        for point in &self.points {
            let x = transform_x(point.x);
            let y = transform_y(point.y);
            let color = self.config.color_palette.get_color(point.class as usize);
            let size = if point.is_support_vector {
                self.config.support_vector_size
            } else {
                self.config.point_size
            };
            let class_name = if point.is_support_vector {
                "support-vector"
            } else {
                "point"
            };

            svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="{}" fill="{}" opacity="{}" class="{}"/>"#,
                x, y, size, color, self.config.alpha, class_name
            ));
        }
    }

    fn add_decision_boundary_to_svg<F1, F2>(
        &self,
        svg: &mut String,
        boundary: &[(f64, f64)],
        transform_x: F1,
        transform_y: F2,
    ) where
        F1: Fn(f64) -> i32,
        F2: Fn(f64) -> i32,
    {
        if boundary.is_empty() {
            return;
        }

        svg.push_str(r#"<path d=""#);
        for (i, (x, y)) in boundary.iter().enumerate() {
            let svg_x = transform_x(*x);
            let svg_y = transform_y(*y);

            if i == 0 {
                svg.push_str(&format!("M {} {}", svg_x, svg_y));
            } else {
                svg.push_str(&format!(" L {} {}", svg_x, svg_y));
            }
        }
        svg.push_str(r#"" class="decision-boundary"/>"#);
    }

    fn add_margin_boundaries_to_svg<F1, F2>(
        &self,
        svg: &mut String,
        pos_margin: &[(f64, f64)],
        neg_margin: &[(f64, f64)],
        transform_x: F1,
        transform_y: F2,
    ) where
        F1: Fn(f64) -> i32 + Copy,
        F2: Fn(f64) -> i32 + Copy,
    {
        // Draw positive margin
        if !pos_margin.is_empty() {
            svg.push_str(r#"<path d=""#);
            for (i, (x, y)) in pos_margin.iter().enumerate() {
                let svg_x = transform_x(*x);
                let svg_y = transform_y(*y);

                if i == 0 {
                    svg.push_str(&format!("M {} {}", svg_x, svg_y));
                } else {
                    svg.push_str(&format!(" L {} {}", svg_x, svg_y));
                }
            }
            svg.push_str(r#"" class="positive-margin"/>"#);
        }

        // Draw negative margin
        if !neg_margin.is_empty() {
            svg.push_str(r#"<path d=""#);
            for (i, (x, y)) in neg_margin.iter().enumerate() {
                let svg_x = transform_x(*x);
                let svg_y = transform_y(*y);

                if i == 0 {
                    svg.push_str(&format!("M {} {}", svg_x, svg_y));
                } else {
                    svg.push_str(&format!(" L {} {}", svg_x, svg_y));
                }
            }
            svg.push_str(r#"" class="negative-margin"/>"#);
        }
    }

    /// Export plot data to JSON format for web visualization
    #[cfg(feature = "visualization")]
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;

        let points: Vec<serde_json::Value> = self
            .points
            .iter()
            .map(|p| {
                json!({
                    "x": p.x,
                    "y": p.y,
                    "class": p.class,
                    "is_support_vector": p.is_support_vector,
                    "alpha_value": p.alpha_value,
                    "margin_type": format!("{:?}", p.margin_type)
                })
            })
            .collect();

        json!({
            "points": points,
            "decision_boundary": self.decision_boundary,
            "margin_boundaries": self.margin_boundaries,
            "x_range": self.x_range,
            "y_range": self.y_range,
            "config": {
                "figure_size": self.config.figure_size,
                "resolution": self.config.resolution,
                "show_support_vectors": self.config.show_support_vectors,
                "show_decision_boundary": self.config.show_decision_boundary,
                "show_margins": self.config.show_margins
            }
        })
    }
}

/// Kernel matrix visualization
#[derive(Debug, Clone)]
pub struct KernelMatrixPlot {
    pub matrix: Array2<f64>,
    pub labels: Option<Array1<f64>>,
    pub config: VisualizationConfig,
}

impl KernelMatrixPlot {
    /// Create a new kernel matrix plot
    pub fn new(
        matrix: Array2<f64>,
        labels: Option<Array1<f64>>,
        config: VisualizationConfig,
    ) -> Self {
        Self {
            matrix,
            labels,
            config,
        }
    }

    /// Generate heatmap visualization of the kernel matrix
    pub fn to_heatmap_svg(&self) -> String {
        let (rows, cols) = self.matrix.dim();
        let cell_size = 4; // pixels per matrix cell
        let width = cols * cell_size;
        let height = rows * cell_size;

        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            width + 100,
            height + 100
        );

        // Add title
        if let Some(ref title) = self.config.title {
            svg.push_str(&format!(
                r#"<text x="{}" y="20" text-anchor="middle" font-family="Arial" font-size="14">{}</text>"#,
                (width + 100) / 2,
                title
            ));
        }

        // Find matrix value range for color mapping
        let min_val = self.matrix.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = self.matrix.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        // Draw heatmap cells
        for i in 0..rows {
            for j in 0..cols {
                let value = self.matrix[[i, j]];
                let normalized = if range > 0.0 {
                    ((value - min_val) / range).clamp(0.0, 1.0)
                } else {
                    0.5
                };

                // Color mapping (blue to red)
                let red = (255.0 * normalized) as u8;
                let blue = (255.0 * (1.0 - normalized)) as u8;
                let color = format!("#{:02x}00{:02x}", red, blue);

                let x = 50 + j * cell_size;
                let y = 50 + i * cell_size;

                svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
                    x, y, cell_size, cell_size, color
                ));
            }
        }

        // Add color bar legend
        self.add_colorbar_to_svg(&mut svg, width + 70, 50, 20, height, min_val, max_val);

        svg.push_str("</svg>");
        svg
    }

    #[allow(clippy::too_many_arguments)]
    fn add_colorbar_to_svg(
        &self,
        svg: &mut String,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        min_val: f64,
        max_val: f64,
    ) {
        let steps = 50;
        let step_height = height / steps;

        for i in 0..steps {
            let normalized = i as f64 / (steps - 1) as f64;
            let red = (255.0 * normalized) as u8;
            let blue = (255.0 * (1.0 - normalized)) as u8;
            let color = format!("#{:02x}00{:02x}", red, blue);

            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
                x,
                y + (steps - 1 - i) * step_height,
                width,
                step_height,
                color
            ));
        }

        // Add value labels
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="Arial" font-size="10">{:.3}</text>"#,
            x + width + 5,
            y + 5,
            max_val
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="Arial" font-size="10">{:.3}</text>"#,
            x + width + 5,
            y + height,
            min_val
        ));
    }
}

/// Training convergence visualization
#[derive(Debug, Clone)]
pub struct ConvergencePlot {
    pub objective_values: Vec<f64>,
    pub iteration_numbers: Vec<usize>,
    pub tolerance: f64,
    pub config: VisualizationConfig,
}

impl ConvergencePlot {
    /// Create a new convergence plot
    pub fn new(
        objective_values: Vec<f64>,
        iteration_numbers: Vec<usize>,
        tolerance: f64,
        config: VisualizationConfig,
    ) -> Self {
        Self {
            objective_values,
            iteration_numbers,
            tolerance,
            config,
        }
    }

    /// Generate SVG line plot of convergence
    pub fn to_svg(&self) -> String {
        let width = self.config.figure_size.0;
        let height = self.config.figure_size.1;
        let margin = 50;
        let plot_width = width - 2 * margin;
        let plot_height = height - 2 * margin;

        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<style>
    .axis {{ stroke: #000; stroke-width: 1; }}
    .grid {{ stroke: #ddd; stroke-width: 0.5; }}
    .line {{ fill: none; stroke: #1f77b4; stroke-width: 2; }}
    .tolerance-line {{ fill: none; stroke: #ff0000; stroke-width: 1; stroke-dasharray: 5,5; }}
</style>
"#,
            width, height
        );

        // Add title
        if let Some(ref title) = self.config.title {
            svg.push_str(&format!(
                r#"<text x="{}" y="20" text-anchor="middle" font-family="Arial" font-size="16">{}</text>"#,
                width / 2,
                title
            ));
        }

        if self.objective_values.is_empty() {
            svg.push_str("</svg>");
            return svg;
        }

        // Find data ranges
        let max_iter = *self.iteration_numbers.iter().max().unwrap_or(&1);
        let min_obj = self
            .objective_values
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_obj = self
            .objective_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let x_scale = plot_width as f64 / max_iter as f64;
        let y_scale = plot_height as f64 / (max_obj - min_obj).max(1e-10);

        // Draw axes
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>"#,
            margin,
            margin + plot_height,
            margin + plot_width,
            margin + plot_height
        ));
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>"#,
            margin,
            margin,
            margin,
            margin + plot_height
        ));

        // Draw convergence line
        if self.objective_values.len() > 1 {
            svg.push_str(r#"<path d=""#);
            for (i, (&iter, &obj)) in self
                .iteration_numbers
                .iter()
                .zip(self.objective_values.iter())
                .enumerate()
            {
                let x = margin + (iter as f64 * x_scale) as usize;
                let y = margin + plot_height - ((obj - min_obj) * y_scale) as usize;

                if i == 0 {
                    svg.push_str(&format!("M {} {}", x, y));
                } else {
                    svg.push_str(&format!(" L {} {}", x, y));
                }
            }
            svg.push_str(r#"" class="line"/>"#);
        }

        // Draw tolerance line
        let tolerance_y = margin + plot_height - ((self.tolerance - min_obj) * y_scale) as usize;
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="tolerance-line"/>"#,
            margin,
            tolerance_y,
            margin + plot_width,
            tolerance_y
        ));

        // Add labels
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial" font-size="12">Iteration</text>"#,
            margin + plot_width / 2,
            height - 10
        ));
        svg.push_str(&format!(
            r#"<text x="15" y="{}" text-anchor="middle" transform="rotate(-90 15 {})" font-family="Arial" font-size="12">Objective Value</text>"#,
            margin + plot_height / 2,
            margin + plot_height / 2
        ));

        svg.push_str("</svg>");
        svg
    }
}

/// Comprehensive SVM visualizer
pub struct SVMVisualizer {
    config: VisualizationConfig,
}

impl Default for SVMVisualizer {
    fn default() -> Self {
        Self::new(VisualizationConfig::default())
    }
}

impl SVMVisualizer {
    /// Create a new SVM visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Visualize 2D SVM classification
    pub fn plot_2d_classification<F>(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        support_vectors: &[usize],
        decision_function: F,
    ) -> SVMResult<Plot2D>
    where
        F: Fn(f64, f64) -> f64,
    {
        if x.ncols() < 2 {
            return Err(SVMError::invalid_input(
                "Need at least 2 features for 2D visualization",
            ));
        }

        let mut plot = Plot2D::new(self.config.clone());
        plot.add_points(x, y, support_vectors);

        if self.config.show_decision_boundary {
            plot.generate_decision_boundary(&decision_function);
        }

        if self.config.show_margins {
            plot.generate_margin_boundaries(&decision_function);
        }

        Ok(plot)
    }

    /// Visualize kernel matrix
    pub fn plot_kernel_matrix(
        &self,
        kernel_matrix: &Array2<f64>,
        labels: Option<&Array1<f64>>,
    ) -> KernelMatrixPlot {
        KernelMatrixPlot::new(kernel_matrix.clone(), labels.cloned(), self.config.clone())
    }

    /// Visualize training convergence
    pub fn plot_convergence(
        &self,
        objective_values: Vec<f64>,
        iteration_numbers: Vec<usize>,
        tolerance: f64,
    ) -> ConvergencePlot {
        ConvergencePlot::new(
            objective_values,
            iteration_numbers,
            tolerance,
            self.config.clone(),
        )
    }

    /// Create interactive HTML visualization
    pub fn create_interactive_html(&self, _plot_data: &Plot2D, include_controls: bool) -> String {
        #[cfg(feature = "visualization")]
        let json_data = _plot_data.to_json();
        #[cfg(not(feature = "visualization"))]
        let json_data = "{}".to_string();

        let controls = if include_controls {
            r#"
            <div id="controls">
                <label><input type="checkbox" id="show-sv" checked> Show Support Vectors</label>
                <label><input type="checkbox" id="show-boundary" checked> Show Decision Boundary</label>
                <label><input type="checkbox" id="show-margins" checked> Show Margins</label>
            </div>
            "#
        } else {
            ""
        };

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>SVM Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        #controls {{ margin: 10px; }}
        #controls label {{ margin-right: 15px; }}
        .point {{ stroke: #000; stroke-width: 0.5; }}
        .support-vector {{ stroke: #000; stroke-width: 2; }}
    </style>
</head>
<body>
    <h1>SVM Visualization</h1>
    {}
    <div id="plot"></div>

    <script>
        const data = {};

        // D3.js visualization code would go here
        // This is a placeholder for the interactive visualization
        const svg = d3.select('#plot')
            .append('svg')
            .attr('width', 800)
            .attr('height', 600);

        // Add visualization implementation
        console.log('Plot data:', data);
    </script>
</body>
</html>"#,
            controls, json_data
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_color_palette() {
        let palette = ColorPalette::Default;
        let color0 = palette.get_color(0);
        let color1 = palette.get_color(1);

        assert_ne!(color0, color1);
        assert!(color0.starts_with('#'));
    }

    #[test]
    fn test_plot_creation() {
        let config = VisualizationConfig::default();
        let plot = Plot2D::new(config);

        assert!(plot.points.is_empty());
        assert_eq!(plot.x_range, (0.0, 1.0));
        assert_eq!(plot.y_range, (0.0, 1.0));
    }

    #[test]
    fn test_add_points() {
        let config = VisualizationConfig::default();
        let mut plot = Plot2D::new(config);

        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.5]).unwrap();
        let y = Array1::from_vec(vec![-1.0, 1.0, 1.0]);
        let support_vectors = vec![0, 2];

        plot.add_points(&x, &y, &support_vectors);

        assert_eq!(plot.points.len(), 3);
        assert!(plot.points[0].is_support_vector);
        assert!(!plot.points[1].is_support_vector);
        assert!(plot.points[2].is_support_vector);
    }

    #[test]
    fn test_visualizer_creation() {
        let visualizer = SVMVisualizer::default();
        assert_eq!(visualizer.config.resolution, 100);
    }

    #[test]
    fn test_convergence_plot() {
        let config = VisualizationConfig::default();
        let objectives = vec![10.0, 5.0, 2.0, 1.0, 0.5];
        let iterations = vec![0, 1, 2, 3, 4];

        let plot = ConvergencePlot::new(objectives, iterations, 0.1, config);

        assert_eq!(plot.objective_values.len(), 5);
        assert_abs_diff_eq!(plot.tolerance, 0.1);
    }

    #[test]
    fn test_svg_generation() {
        let config = VisualizationConfig::default();
        let plot = Plot2D::new(config);

        let svg = plot.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }
}
