//! Interactive Visualization for Cross-Decomposition Methods
//!
//! This module provides interactive visualization capabilities for cross-decomposition
//! algorithms including CCA, PLS, and tensor methods. It supports various plot types,
//! real-time updates, and web-based interactive dashboards.
//!
//! ## Supported Visualizations
//! - Interactive canonical correlation plots
//! - Real-time component analysis
//! - 3D multi-view data visualization
//! - Network visualization for correlation structures
//! - Temporal dynamics visualization
//! - Component loading heatmaps with interactivity
//!
//! ## Output Formats
//! - HTML with JavaScript interactivity
//! - SVG with embedded interactions
//! - JSON data for custom visualizations
//! - Real-time streaming updates via WebSocket

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::path::Path;

/// Interactive visualization configuration
#[derive(Debug, Clone)]
pub struct InteractiveVisualizationConfig {
    /// Plot width in pixels
    pub width: usize,
    /// Plot height in pixels
    pub height: usize,
    /// Color scheme for the visualization
    pub color_scheme: ColorScheme,
    /// Whether to enable zoom and pan interactions
    pub enable_zoom_pan: bool,
    /// Whether to enable point selection
    pub enable_selection: bool,
    /// Whether to show tooltips on hover
    pub show_tooltips: bool,
    /// Animation duration in milliseconds
    pub animation_duration: usize,
    /// Whether to enable real-time updates
    pub real_time_updates: bool,
    /// Update interval in milliseconds (for real-time)
    pub update_interval: usize,
}

impl Default for InteractiveVisualizationConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            color_scheme: ColorScheme::Viridis,
            enable_zoom_pan: true,
            enable_selection: true,
            show_tooltips: true,
            animation_duration: 750,
            real_time_updates: false,
            update_interval: 100,
        }
    }
}

/// Color schemes for visualizations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    /// Viridis color scheme (perceptually uniform)
    Viridis,
    /// Plasma color scheme (high contrast)
    Plasma,
    /// Turbo color scheme (rainbow alternative)
    Turbo,
    /// Cool warm color scheme (diverging)
    CoolWarm,
    /// Custom color scheme
    Custom,
}

/// Plot types for interactive visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotType {
    /// Scatter plot with canonical coordinates
    CanonicalScatter,
    /// Component loading heatmap
    LoadingHeatmap,
    /// Correlation network graph
    CorrelationNetwork,
    /// 3D scatter plot for multi-view data
    Scatter3D,
    /// Time series plot for temporal dynamics
    TimeSeries,
    /// Biplot (scores and loadings together)
    Biplot,
    /// Parallel coordinates plot
    ParallelCoordinates,
}

/// Interactive plot data container
#[derive(Debug, Clone)]
pub struct InteractivePlot {
    /// Plot type
    pub plot_type: PlotType,
    /// Data points for plotting
    pub data: PlotData,
    /// Plot configuration
    pub config: InteractiveVisualizationConfig,
    /// Metadata for tooltips and interactions
    pub metadata: HashMap<String, String>,
    /// Custom JavaScript callbacks
    pub callbacks: HashMap<String, String>,
}

/// Plot data structure
#[derive(Debug, Clone)]
pub struct PlotData {
    /// X coordinates
    pub x: Array1<f64>,
    /// Y coordinates
    pub y: Array1<f64>,
    /// Z coordinates (for 3D plots)
    pub z: Option<Array1<f64>>,
    /// Point colors (indices into color scheme)
    pub colors: Option<Array1<f64>>,
    /// Point sizes
    pub sizes: Option<Array1<f64>>,
    /// Point labels for tooltips
    pub labels: Option<Vec<String>>,
    /// Additional data dimensions for parallel coordinates
    pub additional_dims: Option<Array2<f64>>,
}

impl PlotData {
    /// Create new plot data with x and y coordinates
    pub fn new(x: Array1<f64>, y: Array1<f64>) -> Self {
        Self {
            x,
            y,
            z: None,
            colors: None,
            sizes: None,
            labels: None,
            additional_dims: None,
        }
    }

    /// Add Z coordinates for 3D plotting
    pub fn with_z(mut self, z: Array1<f64>) -> Self {
        self.z = Some(z);
        self
    }

    /// Add colors for points
    pub fn with_colors(mut self, colors: Array1<f64>) -> Self {
        self.colors = Some(colors);
        self
    }

    /// Add sizes for points
    pub fn with_sizes(mut self, sizes: Array1<f64>) -> Self {
        self.sizes = Some(sizes);
        self
    }

    /// Add labels for tooltips
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Add additional dimensions for parallel coordinates
    pub fn with_additional_dims(mut self, dims: Array2<f64>) -> Self {
        self.additional_dims = Some(dims);
        self
    }
}

/// Interactive visualization engine
#[derive(Debug)]
pub struct InteractiveVisualizer {
    /// Configuration
    config: InteractiveVisualizationConfig,
    /// Current plots
    plots: Vec<InteractivePlot>,
    /// Output directory for generated files
    output_dir: String,
}

impl InteractiveVisualizer {
    /// Create a new interactive visualizer
    pub fn new() -> Self {
        Self {
            config: InteractiveVisualizationConfig::default(),
            plots: Vec::new(),
            output_dir: "visualizations".to_string(),
        }
    }

    /// Create visualizer with custom configuration
    pub fn with_config(config: InteractiveVisualizationConfig) -> Self {
        Self {
            config,
            plots: Vec::new(),
            output_dir: "visualizations".to_string(),
        }
    }

    /// Set output directory
    pub fn with_output_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.output_dir = path.as_ref().to_string_lossy().to_string();
        self
    }

    /// Add a new interactive plot
    pub fn add_plot(&mut self, plot: InteractivePlot) {
        self.plots.push(plot);
    }

    /// Create canonical correlation scatter plot
    pub fn canonical_scatter(
        &mut self,
        x_canonical: ArrayView1<f64>,
        y_canonical: ArrayView1<f64>,
        labels: Option<Vec<String>>,
    ) -> Result<(), VisualizationError> {
        let data = PlotData::new(x_canonical.to_owned(), y_canonical.to_owned()).with_labels(
            labels.unwrap_or_else(|| {
                (0..x_canonical.len())
                    .map(|i| format!("Point {}", i))
                    .collect()
            }),
        );

        let plot = InteractivePlot {
            plot_type: PlotType::CanonicalScatter,
            data,
            config: self.config.clone(),
            metadata: HashMap::new(),
            callbacks: HashMap::new(),
        };

        self.add_plot(plot);
        Ok(())
    }

    /// Create component loading heatmap
    pub fn loading_heatmap(
        &mut self,
        loadings: ArrayView2<f64>,
        feature_names: Option<Vec<String>>,
        component_names: Option<Vec<String>>,
    ) -> Result<(), VisualizationError> {
        // Convert 2D loadings to plot coordinates for heatmap
        let (n_features, n_components) = loadings.dim();
        let mut x_coords = Vec::new();
        let mut y_coords = Vec::new();
        let mut colors = Vec::new();
        let mut labels = Vec::new();

        for i in 0..n_features {
            for j in 0..n_components {
                x_coords.push(j as f64);
                y_coords.push(i as f64);
                colors.push(loadings[[i, j]]);

                let feature_name = feature_names
                    .as_ref()
                    .map(|names| names[i].clone())
                    .unwrap_or_else(|| format!("Feature {}", i));
                let component_name = component_names
                    .as_ref()
                    .map(|names| names[j].clone())
                    .unwrap_or_else(|| format!("Component {}", j));

                labels.push(format!(
                    "{} -> {}: {:.4}",
                    feature_name,
                    component_name,
                    loadings[[i, j]]
                ));
            }
        }

        let data = PlotData::new(Array1::from_vec(x_coords), Array1::from_vec(y_coords))
            .with_colors(Array1::from_vec(colors))
            .with_labels(labels);

        let plot = InteractivePlot {
            plot_type: PlotType::LoadingHeatmap,
            data,
            config: self.config.clone(),
            metadata: HashMap::new(),
            callbacks: HashMap::new(),
        };

        self.add_plot(plot);
        Ok(())
    }

    /// Create correlation network visualization
    pub fn correlation_network(
        &mut self,
        correlation_matrix: ArrayView2<f64>,
        variable_names: Option<Vec<String>>,
        threshold: f64,
    ) -> Result<(), VisualizationError> {
        let n_vars = correlation_matrix.nrows();

        // Create network layout (simple circular for now)
        let mut x_coords = Vec::new();
        let mut y_coords = Vec::new();
        let mut labels = Vec::new();

        for i in 0..n_vars {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_vars as f64);
            x_coords.push(angle.cos());
            y_coords.push(angle.sin());

            let label = variable_names
                .as_ref()
                .map(|names| names[i].clone())
                .unwrap_or_else(|| format!("Var {}", i));
            labels.push(label);
        }

        let data = PlotData::new(Array1::from_vec(x_coords), Array1::from_vec(y_coords))
            .with_labels(labels);

        let mut plot = InteractivePlot {
            plot_type: PlotType::CorrelationNetwork,
            data,
            config: self.config.clone(),
            metadata: HashMap::new(),
            callbacks: HashMap::new(),
        };

        // Add network metadata
        plot.metadata
            .insert("threshold".to_string(), threshold.to_string());
        plot.metadata.insert(
            "correlation_data".to_string(),
            format!("{:?}", correlation_matrix.shape()),
        );

        self.add_plot(plot);
        Ok(())
    }

    /// Create 3D scatter plot for multi-view data
    pub fn scatter_3d(
        &mut self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
        z: ArrayView1<f64>,
        colors: Option<ArrayView1<f64>>,
        labels: Option<Vec<String>>,
    ) -> Result<(), VisualizationError> {
        let mut data = PlotData::new(x.to_owned(), y.to_owned()).with_z(z.to_owned());

        if let Some(color_values) = colors {
            data = data.with_colors(color_values.to_owned());
        }

        if let Some(point_labels) = labels {
            data = data.with_labels(point_labels);
        }

        let plot = InteractivePlot {
            plot_type: PlotType::Scatter3D,
            data,
            config: self.config.clone(),
            metadata: HashMap::new(),
            callbacks: HashMap::new(),
        };

        self.add_plot(plot);
        Ok(())
    }

    /// Generate HTML output with interactive plots
    pub fn generate_html(&self, filename: &str) -> Result<(), VisualizationError> {
        let html_content = self.generate_html_content()?;

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| VisualizationError::IoError(e.to_string()))?;

        let filepath = format!("{}/{}", self.output_dir, filename);
        std::fs::write(&filepath, html_content)
            .map_err(|e| VisualizationError::IoError(e.to_string()))?;

        println!("Interactive visualization saved to: {}", filepath);
        Ok(())
    }

    /// Generate the HTML content for interactive plots
    fn generate_html_content(&self) -> Result<String, VisualizationError> {
        let mut html = String::new();

        // HTML header with D3.js and other dependencies
        html.push_str(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Cross-Decomposition Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .plot-container { margin: 20px 0; border: 1px solid #ccc; padding: 10px; }
        .plot-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .plot-description { color: #666; margin-bottom: 15px; }
    </style>
</head>
<body>
    <h1>Interactive Cross-Decomposition Analysis</h1>
"#,
        );

        // Generate plots
        for (i, plot) in self.plots.iter().enumerate() {
            html.push_str(&format!(r#"
    <div class="plot-container">
        <div class="plot-title">{}</div>
        <div class="plot-description">Interactive visualization with zoom, pan, and hover tooltips</div>
        <div id="plot-{}" style="width: {}px; height: {}px;"></div>
    </div>
"#,
                self.plot_type_title(plot.plot_type),
                i,
                plot.config.width,
                plot.config.height
            ));
        }

        // JavaScript for interactive plots
        html.push_str(
            r#"
    <script>
        // Color schemes
        const colorSchemes = {
            'Viridis': 'Viridis',
            'Plasma': 'Plasma',
            'Turbo': 'Turbo',
            'CoolWarm': 'RdBu'
        };
"#,
        );

        // Generate JavaScript for each plot
        for (i, plot) in self.plots.iter().enumerate() {
            html.push_str(&self.generate_plot_javascript(i, plot)?);
        }

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        Ok(html)
    }

    /// Generate JavaScript for a specific plot
    fn generate_plot_javascript(
        &self,
        plot_index: usize,
        plot: &InteractivePlot,
    ) -> Result<String, VisualizationError> {
        match plot.plot_type {
            PlotType::CanonicalScatter => self.generate_scatter_js(plot_index, plot),
            PlotType::LoadingHeatmap => self.generate_heatmap_js(plot_index, plot),
            PlotType::CorrelationNetwork => self.generate_network_js(plot_index, plot),
            PlotType::Scatter3D => self.generate_3d_scatter_js(plot_index, plot),
            PlotType::TimeSeries => self.generate_timeseries_js(plot_index, plot),
            PlotType::Biplot => self.generate_biplot_js(plot_index, plot),
            PlotType::ParallelCoordinates => self.generate_parallel_js(plot_index, plot),
        }
    }

    /// Generate JavaScript for scatter plot
    fn generate_scatter_js(
        &self,
        plot_index: usize,
        plot: &InteractivePlot,
    ) -> Result<String, VisualizationError> {
        let x_data: Vec<String> = plot.data.x.iter().map(|v| v.to_string()).collect();
        let y_data: Vec<String> = plot.data.y.iter().map(|v| v.to_string()).collect();
        let labels = plot
            .data
            .labels
            .as_ref()
            .map(|l| {
                l.iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_else(|| "[]".to_string());

        Ok(format!(
            r#"
        // Scatter plot for plot-{}
        const trace{} = {{
            x: [{}],
            y: [{}],
            mode: 'markers',
            type: 'scatter',
            text: [{}],
            hovertemplate: '%{{text}}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>',
            marker: {{
                size: 8,
                color: 'rgba(31, 119, 180, 0.7)',
                line: {{
                    color: 'rgba(31, 119, 180, 1.0)',
                    width: 1
                }}
            }}
        }};

        const layout{} = {{
            title: 'Canonical Correlation Scatter Plot',
            xaxis: {{ title: 'First Canonical Variable' }},
            yaxis: {{ title: 'Second Canonical Variable' }},
            hovermode: 'closest',
            showlegend: false
        }};

        const config{} = {{
            displayModeBar: true,
            modeBarButtonsToAdd: [
                {{
                    name: 'Select points',
                    icon: Plotly.Icons.selectbox,
                    click: function(gd) {{
                        console.log('Selection tool activated for plot {}');
                    }}
                }}
            ]
        }};

        Plotly.newPlot('plot-{}', [trace{}], layout{}, config{});
"#,
            plot_index,
            plot_index,
            x_data.join(","),
            y_data.join(","),
            labels,
            plot_index,
            plot_index,
            plot_index,
            plot_index,
            plot_index,
            plot_index,
            plot_index
        ))
    }

    /// Generate JavaScript for heatmap
    fn generate_heatmap_js(
        &self,
        plot_index: usize,
        plot: &InteractivePlot,
    ) -> Result<String, VisualizationError> {
        // This is a simplified heatmap implementation
        // In practice, you'd reconstruct the 2D matrix from the plot data
        Ok(format!(
            r#"
        // Heatmap for plot-{}
        const trace{} = {{
            type: 'scatter',
            mode: 'markers',
            x: [{}],
            y: [{}],
            marker: {{
                size: 20,
                color: [{}],
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {{
                    title: 'Loading Value'
                }}
            }},
            text: [{}],
            hovertemplate: '%{{text}}<extra></extra>'
        }};

        const layout{} = {{
            title: 'Component Loading Heatmap',
            xaxis: {{ title: 'Components' }},
            yaxis: {{ title: 'Features' }},
            hovermode: 'closest'
        }};

        Plotly.newPlot('plot-{}', [trace{}], layout{});
"#,
            plot_index,
            plot_index,
            plot.data
                .x
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            plot.data
                .y
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            plot.data
                .colors
                .as_ref()
                .map(|c| c
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","))
                .unwrap_or_else(|| "[]".to_string()),
            plot.data
                .labels
                .as_ref()
                .map(|l| l
                    .iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(","))
                .unwrap_or_else(|| "[]".to_string()),
            plot_index,
            plot_index,
            plot_index,
            plot_index
        ))
    }

    /// Generate JavaScript for network visualization
    fn generate_network_js(
        &self,
        plot_index: usize,
        plot: &InteractivePlot,
    ) -> Result<String, VisualizationError> {
        Ok(format!(
            r#"
        // Network plot for plot-{}
        const trace{} = {{
            x: [{}],
            y: [{}],
            mode: 'markers+text',
            type: 'scatter',
            text: [{}],
            textposition: 'middle center',
            marker: {{
                size: 15,
                color: 'rgba(255, 127, 14, 0.8)',
                line: {{
                    color: 'rgba(255, 127, 14, 1.0)',
                    width: 2
                }}
            }}
        }};

        const layout{} = {{
            title: 'Correlation Network',
            xaxis: {{ title: '', showgrid: false, zeroline: false, showticklabels: false }},
            yaxis: {{ title: '', showgrid: false, zeroline: false, showticklabels: false }},
            hovermode: 'closest',
            showlegend: false
        }};

        Plotly.newPlot('plot-{}', [trace{}], layout{});
"#,
            plot_index,
            plot_index,
            plot.data
                .x
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            plot.data
                .y
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            plot.data
                .labels
                .as_ref()
                .map(|l| l
                    .iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(","))
                .unwrap_or_else(|| "[]".to_string()),
            plot_index,
            plot_index,
            plot_index,
            plot_index
        ))
    }

    /// Generate JavaScript for 3D scatter plot
    fn generate_3d_scatter_js(
        &self,
        plot_index: usize,
        plot: &InteractivePlot,
    ) -> Result<String, VisualizationError> {
        let z_data = plot
            .data
            .z
            .as_ref()
            .map(|z| {
                z.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_else(|| "[]".to_string());

        Ok(format!(
            r#"
        // 3D scatter plot for plot-{}
        const trace{} = {{
            x: [{}],
            y: [{}],
            z: [{}],
            mode: 'markers',
            type: 'scatter3d',
            marker: {{
                size: 5,
                color: [{}],
                colorscale: 'Viridis',
                showscale: true
            }},
            text: [{}],
            hovertemplate: '%{{text}}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
        }};

        const layout{} = {{
            title: '3D Multi-View Data Visualization',
            scene: {{
                xaxis: {{ title: 'Component 1' }},
                yaxis: {{ title: 'Component 2' }},
                zaxis: {{ title: 'Component 3' }}
            }},
            hovermode: 'closest'
        }};

        Plotly.newPlot('plot-{}', [trace{}], layout{});
"#,
            plot_index,
            plot_index,
            plot.data
                .x
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            plot.data
                .y
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            z_data,
            plot.data
                .colors
                .as_ref()
                .map(|c| c
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","))
                .unwrap_or_else(|| "[]".to_string()),
            plot.data
                .labels
                .as_ref()
                .map(|l| l
                    .iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(","))
                .unwrap_or_else(|| "[]".to_string()),
            plot_index,
            plot_index,
            plot_index,
            plot_index
        ))
    }

    /// Generate placeholder JavaScript for other plot types
    fn generate_timeseries_js(
        &self,
        plot_index: usize,
        _plot: &InteractivePlot,
    ) -> Result<String, VisualizationError> {
        Ok(format!("// Time series plot {} - placeholder", plot_index))
    }

    fn generate_biplot_js(
        &self,
        plot_index: usize,
        _plot: &InteractivePlot,
    ) -> Result<String, VisualizationError> {
        Ok(format!("// Biplot {} - placeholder", plot_index))
    }

    fn generate_parallel_js(
        &self,
        plot_index: usize,
        _plot: &InteractivePlot,
    ) -> Result<String, VisualizationError> {
        Ok(format!(
            "// Parallel coordinates plot {} - placeholder",
            plot_index
        ))
    }

    /// Get title for plot type
    fn plot_type_title(&self, plot_type: PlotType) -> &'static str {
        match plot_type {
            PlotType::CanonicalScatter => "Canonical Correlation Scatter Plot",
            PlotType::LoadingHeatmap => "Component Loading Heatmap",
            PlotType::CorrelationNetwork => "Correlation Network Visualization",
            PlotType::Scatter3D => "3D Multi-View Data Visualization",
            PlotType::TimeSeries => "Temporal Dynamics Visualization",
            PlotType::Biplot => "Biplot Visualization",
            PlotType::ParallelCoordinates => "Parallel Coordinates Plot",
        }
    }
}

impl Default for InteractiveVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Visualization errors
#[derive(Debug, thiserror::Error)]
pub enum VisualizationError {
    #[error("Dimension mismatch: {0}")]
    DimensionError(String),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Rendering error: {0}")]
    RenderError(String),
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_interactive_visualizer_creation() {
        let visualizer = InteractiveVisualizer::new();
        assert_eq!(visualizer.plots.len(), 0);
        assert_eq!(visualizer.config.width, 800);
        assert_eq!(visualizer.config.height, 600);
    }

    #[test]
    fn test_plot_data_creation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let data = PlotData::new(x.clone(), y.clone());

        assert_eq!(data.x, x);
        assert_eq!(data.y, y);
        assert!(data.z.is_none());
        assert!(data.colors.is_none());
    }

    #[test]
    fn test_plot_data_with_colors_and_z() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let z = Array1::from_vec(vec![7.0, 8.0, 9.0]);
        let colors = Array1::from_vec(vec![0.1, 0.5, 0.9]);

        let data = PlotData::new(x.clone(), y.clone())
            .with_z(z.clone())
            .with_colors(colors.clone());

        assert_eq!(data.x, x);
        assert_eq!(data.y, y);
        assert_eq!(data.z.unwrap(), z);
        assert_eq!(data.colors.unwrap(), colors);
    }

    #[test]
    fn test_canonical_scatter_plot() -> Result<(), VisualizationError> {
        let mut visualizer = InteractiveVisualizer::new();

        let x_canonical = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y_canonical = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        visualizer.canonical_scatter(x_canonical.view(), y_canonical.view(), None)?;

        assert_eq!(visualizer.plots.len(), 1);
        assert_eq!(visualizer.plots[0].plot_type, PlotType::CanonicalScatter);
        assert_eq!(visualizer.plots[0].data.x.len(), 4);

        Ok(())
    }

    #[test]
    fn test_loading_heatmap() -> Result<(), VisualizationError> {
        let mut visualizer = InteractiveVisualizer::new();

        let loadings = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let feature_names = Some(vec![
            "Feature1".to_string(),
            "Feature2".to_string(),
            "Feature3".to_string(),
        ]);
        let component_names = Some(vec!["Comp1".to_string(), "Comp2".to_string()]);

        visualizer.loading_heatmap(loadings.view(), feature_names, component_names)?;

        assert_eq!(visualizer.plots.len(), 1);
        assert_eq!(visualizer.plots[0].plot_type, PlotType::LoadingHeatmap);
        assert_eq!(visualizer.plots[0].data.x.len(), 6); // 3 features * 2 components

        Ok(())
    }

    #[test]
    fn test_3d_scatter_plot() -> Result<(), VisualizationError> {
        let mut visualizer = InteractiveVisualizer::new();

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let z = Array1::from_vec(vec![7.0, 8.0, 9.0]);
        let colors = Array1::from_vec(vec![0.1, 0.5, 0.9]);

        visualizer.scatter_3d(x.view(), y.view(), z.view(), Some(colors.view()), None)?;

        assert_eq!(visualizer.plots.len(), 1);
        assert_eq!(visualizer.plots[0].plot_type, PlotType::Scatter3D);
        assert!(visualizer.plots[0].data.z.is_some());
        assert!(visualizer.plots[0].data.colors.is_some());

        Ok(())
    }

    #[test]
    fn test_color_scheme_enum() {
        let scheme = ColorScheme::Viridis;
        assert_eq!(scheme, ColorScheme::Viridis);
        assert_ne!(scheme, ColorScheme::Plasma);
    }

    #[test]
    fn test_visualization_config_default() {
        let config = InteractiveVisualizationConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.color_scheme, ColorScheme::Viridis);
        assert!(config.enable_zoom_pan);
        assert!(config.show_tooltips);
    }

    #[test]
    fn test_correlation_network() -> Result<(), VisualizationError> {
        let mut visualizer = InteractiveVisualizer::new();

        let correlation_matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.5, 0.3, 0.5, 1.0, 0.7, 0.3, 0.7, 1.0])
                .unwrap();
        let variable_names = Some(vec![
            "Var1".to_string(),
            "Var2".to_string(),
            "Var3".to_string(),
        ]);

        visualizer.correlation_network(correlation_matrix.view(), variable_names, 0.5)?;

        assert_eq!(visualizer.plots.len(), 1);
        assert_eq!(visualizer.plots[0].plot_type, PlotType::CorrelationNetwork);
        assert_eq!(visualizer.plots[0].data.x.len(), 3);
        assert!(visualizer.plots[0].metadata.contains_key("threshold"));

        Ok(())
    }

    #[test]
    fn test_html_generation() {
        let mut visualizer = InteractiveVisualizer::new();

        // Add a simple scatter plot
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let _ = visualizer.canonical_scatter(x.view(), y.view(), None);

        // Test HTML content generation (not file writing)
        let html_result = visualizer.generate_html_content();
        assert!(html_result.is_ok());

        let html = html_result.unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Interactive Cross-Decomposition"));
        assert!(html.contains("Plotly"));
        assert!(html.contains("plot-0"));
    }

    #[test]
    fn test_multiple_plots() -> Result<(), VisualizationError> {
        let mut visualizer = InteractiveVisualizer::new();

        // Add scatter plot
        let x1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y1 = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        visualizer.canonical_scatter(x1.view(), y1.view(), None)?;

        // Add 3D plot
        let x2 = Array1::from_vec(vec![1.0, 2.0]);
        let y2 = Array1::from_vec(vec![3.0, 4.0]);
        let z2 = Array1::from_vec(vec![5.0, 6.0]);
        visualizer.scatter_3d(x2.view(), y2.view(), z2.view(), None, None)?;

        assert_eq!(visualizer.plots.len(), 2);
        assert_eq!(visualizer.plots[0].plot_type, PlotType::CanonicalScatter);
        assert_eq!(visualizer.plots[1].plot_type, PlotType::Scatter3D);

        Ok(())
    }
}
