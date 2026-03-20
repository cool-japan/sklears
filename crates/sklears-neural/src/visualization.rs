//! Model Visualization Utilities
//!
//! This module provides comprehensive visualization capabilities for neural networks,
//! including architecture diagrams, training metrics, attention heatmaps, and
//! weight distributions.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

/// Configuration for visualization output
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Output directory for visualizations
    pub output_dir: String,
    /// Image format (SVG, PNG, etc.)
    pub format: ImageFormat,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// DPI for raster formats
    pub dpi: u32,
    /// Whether to show layer names
    pub show_layer_names: bool,
    /// Whether to show tensor shapes
    pub show_tensor_shapes: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            output_dir: "./visualizations".to_string(),
            format: ImageFormat::SVG,
            color_scheme: ColorScheme::Default,
            dpi: 300,
            show_layer_names: true,
            show_tensor_shapes: true,
        }
    }
}

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageFormat {
    SVG,
    PNG,
    HTML,
}

/// Color schemes for visualizations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorScheme {
    Default,
    Viridis,
    Plasma,
    Grayscale,
}

/// Model architecture visualizer
pub struct ModelVisualizer {
    config: VisualizationConfig,
}

impl ModelVisualizer {
    /// Create a new model visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate a model architecture diagram
    pub fn visualize_architecture(&self, layers: &[LayerInfo], filename: &str) -> NeuralResult<()> {
        let output_path = format!(
            "{}/{}.{}",
            self.config.output_dir,
            filename,
            self.format_extension()
        );

        match self.config.format {
            ImageFormat::SVG => self.generate_svg_architecture(layers, &output_path),
            ImageFormat::HTML => self.generate_html_architecture(layers, &output_path),
            ImageFormat::PNG => {
                // For PNG, we'll generate SVG first then mention it needs conversion
                self.generate_svg_architecture(layers, &output_path.replace(".png", ".svg"))?;
                println!("SVG generated. Use external tool to convert to PNG if needed.");
                Ok(())
            }
        }
    }

    /// Generate SVG architecture diagram
    fn generate_svg_architecture(
        &self,
        layers: &[LayerInfo],
        output_path: &str,
    ) -> NeuralResult<()> {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            "<svg width=\"800\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n            <defs>\n                <style>\n                    .layer-box {{ fill: #e1f5fe; stroke: #0277bd; stroke-width: 2; }}\n                    .layer-text {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}\n                    .layer-name {{ font-weight: bold; }}\n                    .layer-shape {{ font-size: 10px; fill: #666; }}\n                    .connection {{ stroke: #424242; stroke-width: 2; marker-end: url(#arrowhead); }}\n                </style>\n                <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" \n                    refX=\"10\" refY=\"3.5\" orient=\"auto\">\n                    <polygon points=\"0,0 10,3.5 0,7\" fill=\"#424242\" />\n                </marker>\n            </defs>\n            ", 
            layers.len() * 100 + 100
        ));

        // Draw layers
        for (i, layer) in layers.iter().enumerate() {
            let y = i * 100 + 50;
            let x = 400;

            // Layer box
            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="200" height="60" class="layer-box" />
                "#,
                x - 100,
                y - 30
            ));

            // Layer name
            if self.config.show_layer_names {
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" class="layer-text layer-name">{}</text>
                    "#,
                    x,
                    y - 10,
                    layer.name
                ));
            }

            // Layer type
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" class="layer-text">{}</text>
                "#,
                x,
                y + 5,
                layer.layer_type
            ));

            // Shape information
            if self.config.show_tensor_shapes {
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" class="layer-text layer-shape">{:?}</text>
                    "#,
                    x,
                    y + 20,
                    layer.output_shape
                ));
            }

            // Connection to next layer
            if i < layers.len() - 1 {
                svg.push_str(&format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connection" />
                    "#,
                    x,
                    y + 30,
                    x,
                    y + 70
                ));
            }
        }

        svg.push_str("</svg>");

        // Write to file
        let mut file = File::create(output_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;
        file.write_all(svg.as_bytes())
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Generate HTML architecture diagram with interactive features
    fn generate_html_architecture(
        &self,
        layers: &[LayerInfo],
        output_path: &str,
    ) -> NeuralResult<()> {
        let mut html = String::new();

        html.push_str(
            r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Network Architecture</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .architecture { display: flex; flex-direction: column; align-items: center; }
                .layer { 
                    background: #e1f5fe; 
                    border: 2px solid #0277bd; 
                    border-radius: 8px;
                    padding: 15px; 
                    margin: 10px;
                    min-width: 200px;
                    text-align: center;
                    transition: all 0.3s ease;
                }
                .layer:hover { 
                    background: #b3e5fc; 
                    transform: scale(1.05);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }
                .layer-name { font-weight: bold; font-size: 16px; color: #0277bd; }
                .layer-type { font-size: 14px; color: #424242; margin: 5px 0; }
                .layer-shape { font-size: 12px; color: #666; }
                .arrow { 
                    font-size: 24px; 
                    color: #424242; 
                    margin: 5px 0;
                }
                .layer-details {
                    display: none;
                    margin-top: 10px;
                    padding: 10px;
                    background: #f5f5f5;
                    border-radius: 4px;
                    font-size: 12px;
                }
            </style>
            <script>
                function toggleDetails(layerId) {
                    const details = document.getElementById(layerId);
                    details.style.display = details.style.display === 'none' ? 'block' : 'none';
                }
            </script>
        </head>
        <body>
            <h1>Neural Network Architecture</h1>
            <div class="architecture">
        "#,
        );

        for (i, layer) in layers.iter().enumerate() {
            html.push_str(&format!(
                r#"
                <div class="layer" onclick="toggleDetails('details_{}')">
                    <div class="layer-name">{}</div>
                    <div class="layer-type">{}</div>
                    <div class="layer-shape">Shape: {:?}</div>
                    <div id="details_{}" class="layer-details">
                        <strong>Parameters:</strong> {}<br>
                        <strong>Activation:</strong> {}<br>
                        <strong>Trainable:</strong> {}
                    </div>
                </div>
                "#,
                i,
                layer.name,
                layer.layer_type,
                layer.output_shape,
                i,
                layer.num_parameters,
                layer.activation.as_deref().unwrap_or("None"),
                layer.trainable
            ));

            if i < layers.len() - 1 {
                html.push_str(r#"<div class="arrow">↓</div>"#);
            }
        }

        html.push_str(
            r#"
            </div>
        </body>
        </html>
        "#,
        );

        // Write to file
        let mut file = File::create(output_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;
        file.write_all(html.as_bytes())
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Get file extension for current format
    fn format_extension(&self) -> &'static str {
        match self.config.format {
            ImageFormat::SVG => "svg",
            ImageFormat::PNG => "png",
            ImageFormat::HTML => "html",
        }
    }
}

/// Information about a layer for visualization
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub output_shape: Vec<usize>,
    pub num_parameters: usize,
    pub activation: Option<String>,
    pub trainable: bool,
}

/// Training metrics visualizer
pub struct TrainingVisualizer {
    config: VisualizationConfig,
}

impl TrainingVisualizer {
    /// Create a new training visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Plot training history (loss, accuracy, etc.)
    pub fn plot_training_history(
        &self,
        metrics: &TrainingMetrics,
        filename: &str,
    ) -> NeuralResult<()> {
        let output_path = format!("{}/{}.html", self.config.output_dir, filename);

        let mut html = String::new();
        html.push_str(
            r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training History</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .plot-container { width: 100%; height: 400px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Training History</h1>
        "#,
        );

        // Loss plot
        html.push_str(r#"<div id="loss-plot" class="plot-container"></div>"#);
        html.push_str(&format!(
            r#"
        <script>
            var lossData = [{{
                x: [{}],
                y: [{}],
                type: 'scatter',
                mode: 'lines',
                name: 'Training Loss',
                line: {{color: '#1f77b4'}}
            }}
        "#,
            (0..metrics.train_loss.len())
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
            metrics
                .train_loss
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ));

        if !metrics.val_loss.is_empty() {
            html.push_str(&format!(
                r#",{{
                x: [{}],
                y: [{}],
                type: 'scatter',
                mode: 'lines',
                name: 'Validation Loss',
                line: {{color: '#ff7f0e'}}
            }}"#,
                (0..metrics.val_loss.len())
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                metrics
                    .val_loss
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }

        html.push_str(
            r#"];
            var lossLayout = {
                title: 'Training Loss',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Loss' }
            };
            Plotly.newPlot('loss-plot', lossData, lossLayout);
        </script>
        "#,
        );

        // Accuracy plot (if available)
        if !metrics.train_accuracy.is_empty() {
            html.push_str(r#"<div id="accuracy-plot" class="plot-container"></div>"#);
            html.push_str(&format!(
                r#"
            <script>
                var accuracyData = [{{
                    x: [{}],
                    y: [{}],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Training Accuracy',
                    line: {{color: '#2ca02c'}}
                }}
            "#,
                (0..metrics.train_accuracy.len())
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                metrics
                    .train_accuracy
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));

            if !metrics.val_accuracy.is_empty() {
                html.push_str(&format!(
                    r#",{{
                    x: [{}],
                    y: [{}],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Validation Accuracy',
                    line: {{color: '#d62728'}}
                }}"#,
                    (0..metrics.val_accuracy.len())
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                    metrics
                        .val_accuracy
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }

            html.push_str(
                r#"];
                var accuracyLayout = {
                    title: 'Training Accuracy',
                    xaxis: { title: 'Epoch' },
                    yaxis: { title: 'Accuracy' }
                };
                Plotly.newPlot('accuracy-plot', accuracyData, accuracyLayout);
            </script>
            "#,
            );
        }

        html.push_str(
            r#"
        </body>
        </html>
        "#,
        );

        // Write to file
        let mut file = File::create(&output_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;
        file.write_all(html.as_bytes())
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))?;

        println!("Training history saved to: {}", output_path);
        Ok(())
    }
}

/// Training metrics for visualization
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub train_loss: Vec<f64>,
    pub val_loss: Vec<f64>,
    pub train_accuracy: Vec<f64>,
    pub val_accuracy: Vec<f64>,
    pub learning_rates: Vec<f64>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            train_accuracy: Vec::new(),
            val_accuracy: Vec::new(),
            learning_rates: Vec::new(),
        }
    }
}

/// Attention heatmap visualizer
pub struct AttentionVisualizer {
    config: VisualizationConfig,
}

impl AttentionVisualizer {
    /// Create a new attention visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate attention heatmap visualization
    pub fn visualize_attention_weights<T: FloatBounds>(
        &self,
        attention_weights: &Array3<T>,
        tokens: &[String],
        filename: &str,
    ) -> NeuralResult<()> {
        let output_path = format!("{}/{}.html", self.config.output_dir, filename);

        let mut html = String::new();
        html.push_str(
            r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Attention Heatmap</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .heatmap-container { width: 100%; height: 600px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Attention Weights Heatmap</h1>
            <div id="heatmap" class="heatmap-container"></div>
            <script>
        "#,
        );

        // Get the first head of the first layer for visualization
        let (batch_size, seq_len, _) = attention_weights.dim();
        if batch_size > 0 && seq_len > 0 {
            // Convert attention weights to JavaScript format
            let mut weights_js = String::new();
            weights_js.push('[');
            for i in 0..seq_len {
                weights_js.push('[');
                for j in 0..seq_len {
                    if j > 0 {
                        weights_js.push(',');
                    }
                    weights_js.push_str(
                        &attention_weights[[0, i, j]]
                            .to_f64()
                            .unwrap_or(0.0)
                            .to_string(),
                    );
                }
                weights_js.push(']');
                if i < seq_len - 1 {
                    weights_js.push(',');
                }
            }
            weights_js.push(']');

            // Convert tokens to JavaScript format
            let tokens_js = format!("[\"{}\"]", tokens.join("\",\""));

            html.push_str(&format!(
                r#"
                var data = [{{
                    z: {},
                    x: {},
                    y: {},
                    type: 'heatmap',
                    colorscale: 'Viridis'
                }}];
                
                var layout = {{
                    title: 'Attention Weights',
                    xaxis: {{ title: 'Key Tokens' }},
                    yaxis: {{ title: 'Query Tokens' }}
                }};
                
                Plotly.newPlot('heatmap', data, layout);
            "#,
                weights_js, tokens_js, tokens_js
            ));
        }

        html.push_str(
            r#"
            </script>
        </body>
        </html>
        "#,
        );

        // Write to file
        let mut file = File::create(&output_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;
        file.write_all(html.as_bytes())
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))?;

        println!("Attention heatmap saved to: {}", output_path);
        Ok(())
    }
}

/// Weight distribution visualizer
pub struct WeightVisualizer {
    config: VisualizationConfig,
}

impl WeightVisualizer {
    /// Create a new weight visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Visualize weight distributions across layers
    pub fn visualize_weight_distributions<T: FloatBounds>(
        &self,
        weights: &HashMap<String, Array2<T>>,
        filename: &str,
    ) -> NeuralResult<()> {
        let output_path = format!("{}/{}.html", self.config.output_dir, filename);

        let mut html = String::new();
        html.push_str(
            r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weight Distributions</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .histogram-container { width: 100%; height: 400px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Weight Distributions</h1>
        "#,
        );

        for (layer_name, weight_matrix) in weights.iter() {
            let div_id = format!("histogram-{}", layer_name.replace(".", "-"));
            html.push_str(&format!(r#"<h2>{}</h2>"#, layer_name));
            html.push_str(&format!(
                r#"<div id="{}" class="histogram-container"></div>"#,
                div_id
            ));

            // Flatten weights and convert to JavaScript array
            let flattened: Vec<f64> = weight_matrix
                .iter()
                .map(|&w| w.to_f64().unwrap_or(0.0))
                .collect();

            let weights_js = format!(
                "[{}]",
                flattened
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            html.push_str(&format!(
                r#"
            <script>
                var data_{} = [{{
                    x: {},
                    type: 'histogram',
                    nbinsx: 50,
                    name: '{}'
                }}];
                
                var layout_{} = {{
                    title: '{} Weight Distribution',
                    xaxis: {{ title: 'Weight Value' }},
                    yaxis: {{ title: 'Frequency' }}
                }};
                
                Plotly.newPlot('{}', data_{}, layout_{});
            </script>
            "#,
                div_id, weights_js, layer_name, div_id, layer_name, div_id, div_id, div_id
            ));
        }

        html.push_str(
            r#"
        </body>
        </html>
        "#,
        );

        // Write to file
        let mut file = File::create(&output_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;
        file.write_all(html.as_bytes())
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))?;

        println!("Weight distributions saved to: {}", output_path);
        Ok(())
    }
}

/// Create output directory if it doesn't exist
pub fn ensure_output_directory(path: &str) -> NeuralResult<()> {
    std::fs::create_dir_all(path)
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to create directory: {}", e)))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert_eq!(config.format, ImageFormat::SVG);
        assert_eq!(config.color_scheme, ColorScheme::Default);
        assert_eq!(config.dpi, 300);
    }

    #[test]
    fn test_layer_info_creation() {
        let layer_info = LayerInfo {
            name: "dense_1".to_string(),
            layer_type: "Dense".to_string(),
            output_shape: vec![128, 64],
            num_parameters: 8256,
            activation: Some("ReLU".to_string()),
            trainable: true,
        };

        assert_eq!(layer_info.name, "dense_1");
        assert_eq!(layer_info.num_parameters, 8256);
    }

    #[test]
    fn test_training_metrics_default() {
        let metrics = TrainingMetrics::default();
        assert!(metrics.train_loss.is_empty());
        assert!(metrics.val_loss.is_empty());
        assert!(metrics.train_accuracy.is_empty());
    }
}
