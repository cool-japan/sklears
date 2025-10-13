//! Extensible Visualization Backend System
//!
//! This module provides a trait-based system for pluggable visualization backends,
//! allowing for flexible rendering to different output formats and libraries.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

/// Trait for visualization backends
pub trait VisualizationBackend: Debug + Send + Sync {
    /// Render a feature importance plot
    fn render_feature_importance(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization>;

    /// Render a SHAP plot
    fn render_shap_plot(
        &self,
        data: &ShapData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization>;

    /// Render a partial dependence plot
    fn render_partial_dependence(
        &self,
        data: &PartialDependenceData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization>;

    /// Render a comparative plot
    fn render_comparative_plot(
        &self,
        data: &ComparativeData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization>;

    /// Render a custom plot
    fn render_custom_plot(
        &self,
        data: &CustomPlotData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization>;

    /// Get backend name
    fn name(&self) -> &str;

    /// Get supported output formats
    fn supported_formats(&self) -> Vec<OutputFormat>;

    /// Check if backend supports interactivity
    fn supports_interactivity(&self) -> bool;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Output format
    pub format: OutputFormat,
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// DPI for high-resolution output
    pub dpi: usize,
    /// Whether to enable interactivity
    pub interactive: bool,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Theme
    pub theme: Theme,
    /// Custom properties
    pub custom_properties: HashMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Html,
            width: 800,
            height: 600,
            dpi: 96,
            interactive: true,
            color_scheme: ColorScheme::Default,
            theme: Theme::Light,
            custom_properties: HashMap::new(),
        }
    }
}

/// Output formats supported by backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Html
    Html,
    /// Json
    Json,
    /// Svg
    Svg,
    /// Png
    Png,
    /// Jpeg
    Jpeg,
    /// Pdf
    Pdf,
    /// Ascii
    Ascii,
    /// Unicode
    Unicode,
}

/// Color schemes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Default
    Default,
    /// Viridis
    Viridis,
    /// Plasma
    Plasma,
    /// Magma
    Magma,
    /// Inferno
    Inferno,
    /// Blues
    Blues,
    /// Reds
    Reds,
    /// Greens
    Greens,
    /// Categorical
    Categorical,
    /// Diverging
    Diverging,
}

/// Visualization themes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Theme {
    /// Light
    Light,
    /// Dark
    Dark,
    /// HighContrast
    HighContrast,
    /// Minimal
    Minimal,
    /// Scientific
    Scientific,
}

/// Backend capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Supported output formats
    pub formats: Vec<OutputFormat>,
    /// Supports interactive features
    pub interactive: bool,
    /// Supports animations
    pub animations: bool,
    /// Supports 3D rendering
    pub three_d: bool,
    /// Supports custom themes
    pub custom_themes: bool,
    /// Supports real-time updates
    pub real_time_updates: bool,
    /// Maximum data points efficiently handled
    pub max_data_points: Option<usize>,
}

/// Rendered visualization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderedVisualization {
    /// Rendered content
    pub content: String,
    /// Output format
    pub format: OutputFormat,
    /// Metadata about the visualization
    pub metadata: VisualizationMetadata,
    /// Optional binary data (for images)
    pub binary_data: Option<Vec<u8>>,
}

/// Visualization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationMetadata {
    /// Backend used for rendering
    pub backend: String,
    /// Rendering time in milliseconds
    pub render_time_ms: u64,
    /// File size in bytes
    pub file_size_bytes: usize,
    /// Data points count
    pub data_points: usize,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Feature importance data for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceData {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Importance values
    pub importance_values: Vec<Float>,
    /// Standard deviations
    pub std_values: Option<Vec<Float>>,
    /// Plot type
    pub plot_type: FeatureImportanceType,
    /// Title
    pub title: String,
    /// Axis labels
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
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
    /// Waterfall
    Waterfall,
}

/// SHAP data for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapData {
    /// SHAP values matrix (instances x features)
    pub shap_values: Array2<Float>,
    /// Feature values matrix
    pub feature_values: Array2<Float>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Instance names
    pub instance_names: Vec<String>,
    /// Plot type
    pub plot_type: ShapPlotType,
    /// Title
    pub title: String,
}

/// Types of SHAP plots
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ShapPlotType {
    /// Waterfall
    Waterfall,
    /// ForceLayout
    ForceLayout,
    /// Summary
    Summary,
    /// Dependence
    Dependence,
    /// Beeswarm
    Beeswarm,
    /// DecisionPlot
    DecisionPlot,
    /// Violin
    Violin,
}

/// Partial dependence data for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialDependenceData {
    /// Feature values for x-axis
    pub feature_values: Array1<Float>,
    /// Partial dependence values
    pub pd_values: Array1<Float>,
    /// ICE curves (if available)
    pub ice_curves: Option<Array2<Float>>,
    /// Feature name
    pub feature_name: String,
    /// Title
    pub title: String,
    /// Show individual curves
    pub show_ice: bool,
}

/// Comparative data for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeData {
    /// Data for different models/methods
    pub model_data: HashMap<String, Array2<Float>>,
    /// Labels for comparison
    pub labels: Vec<String>,
    /// Comparison type
    pub comparison_type: ComparisonType,
    /// Title
    pub title: String,
}

/// Types of comparative plots
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComparisonType {
    /// SideBySide
    SideBySide,
    /// Overlay
    Overlay,
    /// Difference
    Difference,
    /// Ratio
    Ratio,
    /// Ranking
    Ranking,
}

/// Custom plot data for extensibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPlotData {
    /// Raw data as JSON value
    pub data: serde_json::Value,
    /// Plot type identifier
    pub plot_type: String,
    /// Title
    pub title: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Backend registry for managing multiple backends
#[derive(Debug, Default)]
pub struct BackendRegistry {
    backends: HashMap<String, Arc<dyn VisualizationBackend>>,
    default_backend: Option<String>,
}

impl BackendRegistry {
    /// Create a new backend registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new backend
    pub fn register_backend<B: VisualizationBackend + 'static>(&mut self, backend: B) {
        let name = backend.name().to_string();
        self.backends.insert(name.clone(), Arc::new(backend));

        // Set first backend as default if none set
        if self.default_backend.is_none() {
            self.default_backend = Some(name);
        }
    }

    /// Get backend by name
    pub fn get_backend(&self, name: &str) -> Option<Arc<dyn VisualizationBackend>> {
        self.backends.get(name).cloned()
    }

    /// Get default backend
    pub fn get_default_backend(&self) -> Option<Arc<dyn VisualizationBackend>> {
        self.default_backend
            .as_ref()
            .and_then(|name| self.backends.get(name).cloned())
    }

    /// Set default backend
    pub fn set_default_backend(&mut self, name: &str) -> SklResult<()> {
        if self.backends.contains_key(name) {
            self.default_backend = Some(name.to_string());
            Ok(())
        } else {
            Err(crate::SklearsError::InvalidInput(format!(
                "Backend '{}' not found",
                name
            )))
        }
    }

    /// List all registered backends
    pub fn list_backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }

    /// Get backend capabilities
    pub fn get_capabilities(&self, name: &str) -> Option<BackendCapabilities> {
        self.backends.get(name).map(|b| b.capabilities())
    }

    /// Find backends supporting a specific format
    pub fn find_backends_for_format(&self, format: OutputFormat) -> Vec<String> {
        self.backends
            .iter()
            .filter(|(_, backend)| backend.supported_formats().contains(&format))
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// Visualization renderer using pluggable backends
#[derive(Debug)]
pub struct VisualizationRenderer {
    registry: BackendRegistry,
}

impl VisualizationRenderer {
    /// Create a new visualization renderer
    pub fn new() -> Self {
        Self {
            registry: BackendRegistry::new(),
        }
    }

    /// Create with default backends
    pub fn with_default_backends() -> Self {
        let mut renderer = Self::new();
        renderer.register_default_backends();
        renderer
    }

    /// Register default backends
    pub fn register_default_backends(&mut self) {
        self.registry.register_backend(HtmlBackend::new());
        self.registry.register_backend(JsonBackend::new());
        self.registry.register_backend(AsciiBackend::new());
    }

    /// Register a custom backend
    pub fn register_backend<B: VisualizationBackend + 'static>(&mut self, backend: B) {
        self.registry.register_backend(backend);
    }

    /// Render with specific backend
    pub fn render_with_backend(
        &self,
        backend_name: &str,
        plot_type: PlotType,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let backend = self.registry.get_backend(backend_name).ok_or_else(|| {
            crate::SklearsError::InvalidInput(format!("Backend '{}' not found", backend_name))
        })?;

        match plot_type {
            PlotType::FeatureImportance(data) => backend.render_feature_importance(&data, config),
            PlotType::Shap(data) => backend.render_shap_plot(&data, config),
            PlotType::PartialDependence(data) => backend.render_partial_dependence(&data, config),
            PlotType::Comparative(data) => backend.render_comparative_plot(&data, config),
            PlotType::Custom(data) => backend.render_custom_plot(&data, config),
        }
    }

    /// Render with default backend
    pub fn render(
        &self,
        plot_type: PlotType,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let backend = self.registry.get_default_backend().ok_or_else(|| {
            crate::SklearsError::InvalidInput("No default backend available".to_string())
        })?;

        match plot_type {
            PlotType::FeatureImportance(data) => backend.render_feature_importance(&data, config),
            PlotType::Shap(data) => backend.render_shap_plot(&data, config),
            PlotType::PartialDependence(data) => backend.render_partial_dependence(&data, config),
            PlotType::Comparative(data) => backend.render_comparative_plot(&data, config),
            PlotType::Custom(data) => backend.render_custom_plot(&data, config),
        }
    }

    /// Get backend registry
    pub fn registry(&self) -> &BackendRegistry {
        &self.registry
    }

    /// Get mutable backend registry
    pub fn registry_mut(&mut self) -> &mut BackendRegistry {
        &mut self.registry
    }
}

impl Default for VisualizationRenderer {
    fn default() -> Self {
        Self::with_default_backends()
    }
}

/// Plot type enum for rendering
#[derive(Debug, Clone)]
pub enum PlotType {
    /// FeatureImportance
    FeatureImportance(FeatureImportanceData),
    /// Shap
    Shap(ShapData),
    /// PartialDependence
    PartialDependence(PartialDependenceData),
    /// Comparative
    Comparative(ComparativeData),
    /// Custom
    Custom(CustomPlotData),
}

/// HTML backend implementation
#[derive(Debug)]
pub struct HtmlBackend {
    name: String,
}

impl HtmlBackend {
    /// Create a new HTML backend
    pub fn new() -> Self {
        Self {
            name: "html".to_string(),
        }
    }

    /// Generate HTML template
    fn generate_html_template(&self, title: &str, content: &str) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ margin: 20px 0; }}
        .plot-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="plot-container">
        <div class="plot-title">{}</div>
        <div id="plot">{}</div>
    </div>
</body>
</html>"#,
            title, title, content
        )
    }
}

impl VisualizationBackend for HtmlBackend {
    fn render_feature_importance(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        // Generate Plotly.js bar chart
        let plot_data = match data.plot_type {
            FeatureImportanceType::Bar => {
                let x_data: Vec<String> = data
                    .feature_names
                    .iter()
                    .map(|name| format!("\"{}\"", name))
                    .collect();
                let y_data: Vec<String> = data
                    .importance_values
                    .iter()
                    .map(|val| val.to_string())
                    .collect();

                format!(
                    r#"
                    var data = [{{
                        x: [{}],
                        y: [{}],
                        type: 'bar',
                        marker: {{
                            color: '#1f77b4'
                        }}
                    }}];
                    
                    var layout = {{
                        title: '{}',
                        xaxis: {{ title: '{}' }},
                        yaxis: {{ title: '{}' }},
                        width: {},
                        height: {}
                    }};
                    
                    Plotly.newPlot('plot', data, layout);
                    "#,
                    x_data.join(", "),
                    y_data.join(", "),
                    data.title,
                    data.x_label,
                    data.y_label,
                    config.width,
                    config.height
                )
            }
            _ => {
                // Fallback to bar chart for other types
                let x_data: Vec<String> = data
                    .feature_names
                    .iter()
                    .map(|name| format!("\"{}\"", name))
                    .collect();
                let y_data: Vec<String> = data
                    .importance_values
                    .iter()
                    .map(|val| val.to_string())
                    .collect();

                format!(
                    r#"
                    var data = [{{
                        x: [{}],
                        y: [{}],
                        type: 'bar'
                    }}];
                    
                    var layout = {{
                        title: '{}',
                        width: {},
                        height: {}
                    }};
                    
                    Plotly.newPlot('plot', data, layout);
                    "#,
                    x_data.join(", "),
                    y_data.join(", "),
                    data.title,
                    config.width,
                    config.height
                )
            }
        };

        let html_content = self.generate_html_template(&data.title, &plot_data);
        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: html_content.clone(),
            format: OutputFormat::Html,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: html_content.len(),
                data_points: data.importance_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_shap_plot(
        &self,
        data: &ShapData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        // Generate basic SHAP visualization
        let plot_data = format!(
            r#"
            var data = [{{
                z: {},
                type: 'heatmap',
                colorscale: 'RdBu'
            }}];
            
            var layout = {{
                title: '{}',
                xaxis: {{ title: 'Features' }},
                yaxis: {{ title: 'Instances' }},
                width: {},
                height: {}
            }};
            
            Plotly.newPlot('plot', data, layout);
            "#,
            serde_json::to_string(&data.shap_values.to_owned().into_raw_vec()).unwrap(),
            data.title,
            config.width,
            config.height
        );

        let html_content = self.generate_html_template(&data.title, &plot_data);
        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: html_content.clone(),
            format: OutputFormat::Html,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: html_content.len(),
                data_points: data.shap_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_partial_dependence(
        &self,
        data: &PartialDependenceData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let x_data: Vec<String> = data
            .feature_values
            .iter()
            .map(|val| val.to_string())
            .collect();
        let y_data: Vec<String> = data.pd_values.iter().map(|val| val.to_string()).collect();

        let plot_data = format!(
            r#"
            var data = [{{
                x: [{}],
                y: [{}],
                type: 'scatter',
                mode: 'lines',
                name: 'Partial Dependence'
            }}];
            
            var layout = {{
                title: '{}',
                xaxis: {{ title: '{}' }},
                yaxis: {{ title: 'Partial Dependence' }},
                width: {},
                height: {}
            }};
            
            Plotly.newPlot('plot', data, layout);
            "#,
            x_data.join(", "),
            y_data.join(", "),
            data.title,
            data.feature_name,
            config.width,
            config.height
        );

        let html_content = self.generate_html_template(&data.title, &plot_data);
        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: html_content.clone(),
            format: OutputFormat::Html,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: html_content.len(),
                data_points: data.feature_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_comparative_plot(
        &self,
        data: &ComparativeData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        // Simple comparative plot implementation
        let plot_data = format!(
            r#"
            var data = [];
            var layout = {{
                title: '{}',
                width: {},
                height: {}
            }};
            
            Plotly.newPlot('plot', data, layout);
            "#,
            data.title, config.width, config.height
        );

        let html_content = self.generate_html_template(&data.title, &plot_data);
        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: html_content.clone(),
            format: OutputFormat::Html,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: html_content.len(),
                data_points: data.model_data.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_custom_plot(
        &self,
        data: &CustomPlotData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let plot_data = format!(
            r#"
            var data = {};
            var layout = {{
                title: '{}',
                width: {},
                height: {}
            }};
            
            Plotly.newPlot('plot', data, layout);
            "#,
            data.data.to_string(),
            data.title,
            config.width,
            config.height
        );

        let html_content = self.generate_html_template(&data.title, &plot_data);
        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: html_content.clone(),
            format: OutputFormat::Html,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: html_content.len(),
                data_points: 0,
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supported_formats(&self) -> Vec<OutputFormat> {
        vec![OutputFormat::Html]
    }

    fn supports_interactivity(&self) -> bool {
        true
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            formats: vec![OutputFormat::Html],
            interactive: true,
            animations: true,
            three_d: false,
            custom_themes: true,
            real_time_updates: true,
            max_data_points: Some(10000),
        }
    }
}

/// JSON backend implementation
#[derive(Debug)]
pub struct JsonBackend {
    name: String,
}

impl JsonBackend {
    /// Create a new JSON backend
    pub fn new() -> Self {
        Self {
            name: "json".to_string(),
        }
    }
}

impl VisualizationBackend for JsonBackend {
    fn render_feature_importance(
        &self,
        data: &FeatureImportanceData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let json_content = serde_json::to_string_pretty(data).map_err(|e| {
            crate::SklearsError::InvalidInput(format!("JSON serialization failed: {}", e))
        })?;

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: json_content.clone(),
            format: OutputFormat::Json,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: json_content.len(),
                data_points: data.importance_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_shap_plot(
        &self,
        data: &ShapData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let json_content = serde_json::to_string_pretty(data).map_err(|e| {
            crate::SklearsError::InvalidInput(format!("JSON serialization failed: {}", e))
        })?;

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: json_content.clone(),
            format: OutputFormat::Json,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: json_content.len(),
                data_points: data.shap_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_partial_dependence(
        &self,
        data: &PartialDependenceData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let json_content = serde_json::to_string_pretty(data).map_err(|e| {
            crate::SklearsError::InvalidInput(format!("JSON serialization failed: {}", e))
        })?;

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: json_content.clone(),
            format: OutputFormat::Json,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: json_content.len(),
                data_points: data.feature_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_comparative_plot(
        &self,
        data: &ComparativeData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let json_content = serde_json::to_string_pretty(data).map_err(|e| {
            crate::SklearsError::InvalidInput(format!("JSON serialization failed: {}", e))
        })?;

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: json_content.clone(),
            format: OutputFormat::Json,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: json_content.len(),
                data_points: data.model_data.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_custom_plot(
        &self,
        data: &CustomPlotData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let json_content = serde_json::to_string_pretty(data).map_err(|e| {
            crate::SklearsError::InvalidInput(format!("JSON serialization failed: {}", e))
        })?;

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: json_content.clone(),
            format: OutputFormat::Json,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: json_content.len(),
                data_points: 0,
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supported_formats(&self) -> Vec<OutputFormat> {
        vec![OutputFormat::Json]
    }

    fn supports_interactivity(&self) -> bool {
        false
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            formats: vec![OutputFormat::Json],
            interactive: false,
            animations: false,
            three_d: false,
            custom_themes: false,
            real_time_updates: false,
            max_data_points: None,
        }
    }
}

/// ASCII backend implementation for terminal output
#[derive(Debug)]
pub struct AsciiBackend {
    name: String,
}

impl AsciiBackend {
    /// Create a new ASCII backend
    pub fn new() -> Self {
        Self {
            name: "ascii".to_string(),
        }
    }

    /// Generate ASCII bar chart
    fn generate_ascii_bar_chart(
        &self,
        labels: &[String],
        values: &[Float],
        width: usize,
        height: usize,
    ) -> String {
        let max_value = values.iter().fold(0.0_f64, |acc, &x| acc.max(x as f64));
        let bar_width = (width - 20) / labels.len().max(1);
        let scale = (height - 5) as Float / max_value;

        let mut result = String::new();

        // Create horizontal bar chart
        for (i, (label, &value)) in labels.iter().zip(values.iter()).enumerate() {
            let bar_length = (value * scale / max_value * 50.0) as usize;
            let bar = "█".repeat(bar_length);
            result.push_str(&format!("{:15} │{:<50} {:.3}\n", label, bar, value));
        }

        result
    }
}

impl VisualizationBackend for AsciiBackend {
    fn render_feature_importance(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let ascii_content = format!(
            "{}\n{}\n{}\n{}",
            "=".repeat(60),
            data.title,
            "=".repeat(60),
            self.generate_ascii_bar_chart(
                &data.feature_names,
                &data.importance_values,
                config.width,
                config.height,
            )
        );

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: ascii_content.clone(),
            format: OutputFormat::Ascii,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: ascii_content.len(),
                data_points: data.importance_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_shap_plot(
        &self,
        data: &ShapData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let ascii_content = format!(
            "{}\n{}\n{}\nSHAP Values: {} instances x {} features\n",
            "=".repeat(60),
            data.title,
            "=".repeat(60),
            data.shap_values.nrows(),
            data.shap_values.ncols()
        );

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: ascii_content.clone(),
            format: OutputFormat::Ascii,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: ascii_content.len(),
                data_points: data.shap_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_partial_dependence(
        &self,
        data: &PartialDependenceData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let ascii_content = format!(
            "{}\n{}\n{}\nPartial Dependence for feature: {}\n",
            "=".repeat(60),
            data.title,
            "=".repeat(60),
            data.feature_name
        );

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: ascii_content.clone(),
            format: OutputFormat::Ascii,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: ascii_content.len(),
                data_points: data.feature_values.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_comparative_plot(
        &self,
        data: &ComparativeData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let ascii_content = format!(
            "{}\n{}\n{}\nComparative plot with {} models\n",
            "=".repeat(60),
            data.title,
            "=".repeat(60),
            data.model_data.len()
        );

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: ascii_content.clone(),
            format: OutputFormat::Ascii,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: ascii_content.len(),
                data_points: data.model_data.len(),
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_custom_plot(
        &self,
        data: &CustomPlotData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let start_time = std::time::Instant::now();

        let ascii_content = format!(
            "{}\n{}\n{}\nCustom plot type: {}\n",
            "=".repeat(60),
            data.title,
            "=".repeat(60),
            data.plot_type
        );

        let render_time = start_time.elapsed().as_millis() as u64;

        Ok(RenderedVisualization {
            content: ascii_content.clone(),
            format: OutputFormat::Ascii,
            metadata: VisualizationMetadata {
                backend: self.name.clone(),
                render_time_ms: render_time,
                file_size_bytes: ascii_content.len(),
                data_points: 0,
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supported_formats(&self) -> Vec<OutputFormat> {
        vec![OutputFormat::Ascii, OutputFormat::Unicode]
    }

    fn supports_interactivity(&self) -> bool {
        false
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            formats: vec![OutputFormat::Ascii, OutputFormat::Unicode],
            interactive: false,
            animations: false,
            three_d: false,
            custom_themes: false,
            real_time_updates: false,
            max_data_points: Some(1000),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_backend_registry() {
        let mut registry = BackendRegistry::new();

        // Register backends
        registry.register_backend(HtmlBackend::new());
        registry.register_backend(JsonBackend::new());
        registry.register_backend(AsciiBackend::new());

        // Test backend retrieval
        assert!(registry.get_backend("html").is_some());
        assert!(registry.get_backend("json").is_some());
        assert!(registry.get_backend("ascii").is_some());
        assert!(registry.get_backend("nonexistent").is_none());

        // Test default backend
        assert!(registry.get_default_backend().is_some());

        // Test backend listing
        let backends = registry.list_backends();
        assert_eq!(backends.len(), 3);
        assert!(backends.contains(&"html".to_string()));
        assert!(backends.contains(&"json".to_string()));
        assert!(backends.contains(&"ascii".to_string()));
    }

    #[test]
    fn test_visualization_renderer() {
        let mut renderer = VisualizationRenderer::with_default_backends();

        // Test feature importance rendering
        let data = FeatureImportanceData {
            feature_names: vec!["Feature1".to_string(), "Feature2".to_string()],
            importance_values: vec![0.6, 0.4],
            std_values: None,
            plot_type: FeatureImportanceType::Bar,
            title: "Test Plot".to_string(),
            x_label: "Features".to_string(),
            y_label: "Importance".to_string(),
        };

        let config = BackendConfig::default();
        let plot_type = PlotType::FeatureImportance(data);

        // Test HTML rendering
        let result = renderer.render_with_backend("html", plot_type.clone(), &config);
        assert!(result.is_ok());
        let rendered = result.unwrap();
        assert_eq!(rendered.format, OutputFormat::Html);
        assert!(rendered.content.contains("Test Plot"));

        // Test JSON rendering
        let result = renderer.render_with_backend("json", plot_type.clone(), &config);
        assert!(result.is_ok());
        let rendered = result.unwrap();
        assert_eq!(rendered.format, OutputFormat::Json);

        // Test ASCII rendering
        let result = renderer.render_with_backend("ascii", plot_type, &config);
        assert!(result.is_ok());
        let rendered = result.unwrap();
        assert_eq!(rendered.format, OutputFormat::Ascii);
    }

    #[test]
    fn test_html_backend_feature_importance() {
        let backend = HtmlBackend::new();
        let data = FeatureImportanceData {
            feature_names: vec!["Feature1".to_string(), "Feature2".to_string()],
            importance_values: vec![0.6, 0.4],
            std_values: None,
            plot_type: FeatureImportanceType::Bar,
            title: "Test Plot".to_string(),
            x_label: "Features".to_string(),
            y_label: "Importance".to_string(),
        };

        let config = BackendConfig::default();
        let result = backend.render_feature_importance(&data, &config);

        assert!(result.is_ok());
        let rendered = result.unwrap();
        assert_eq!(rendered.format, OutputFormat::Html);
        assert!(rendered.content.contains("Test Plot"));
        assert!(rendered.content.contains("Plotly"));
        assert!(rendered.metadata.data_points == 2);
    }

    #[test]
    fn test_json_backend_feature_importance() {
        let backend = JsonBackend::new();
        let data = FeatureImportanceData {
            feature_names: vec!["Feature1".to_string(), "Feature2".to_string()],
            importance_values: vec![0.6, 0.4],
            std_values: None,
            plot_type: FeatureImportanceType::Bar,
            title: "Test Plot".to_string(),
            x_label: "Features".to_string(),
            y_label: "Importance".to_string(),
        };

        let config = BackendConfig::default();
        let result = backend.render_feature_importance(&data, &config);

        assert!(result.is_ok());
        let rendered = result.unwrap();
        assert_eq!(rendered.format, OutputFormat::Json);
        assert!(rendered.content.contains("Feature1"));
        assert!(rendered.content.contains("Feature2"));
        assert!(rendered.metadata.data_points == 2);
    }

    #[test]
    fn test_ascii_backend_feature_importance() {
        let backend = AsciiBackend::new();
        let data = FeatureImportanceData {
            feature_names: vec!["Feature1".to_string(), "Feature2".to_string()],
            importance_values: vec![0.6, 0.4],
            std_values: None,
            plot_type: FeatureImportanceType::Bar,
            title: "Test Plot".to_string(),
            x_label: "Features".to_string(),
            y_label: "Importance".to_string(),
        };

        let config = BackendConfig::default();
        let result = backend.render_feature_importance(&data, &config);

        assert!(result.is_ok());
        let rendered = result.unwrap();
        assert_eq!(rendered.format, OutputFormat::Ascii);
        assert!(rendered.content.contains("Test Plot"));
        assert!(rendered.content.contains("Feature1"));
        assert!(rendered.content.contains("Feature2"));
        assert!(rendered.metadata.data_points == 2);
    }

    #[test]
    fn test_shap_data_creation() {
        let shap_values =
            Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let feature_values =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let data = ShapData {
            shap_values,
            feature_values,
            feature_names: vec!["F1".to_string(), "F2".to_string(), "F3".to_string()],
            instance_names: vec!["I1".to_string(), "I2".to_string()],
            plot_type: ShapPlotType::Summary,
            title: "SHAP Test".to_string(),
        };

        assert_eq!(data.shap_values.nrows(), 2);
        assert_eq!(data.shap_values.ncols(), 3);
        assert_eq!(data.feature_names.len(), 3);
        assert_eq!(data.instance_names.len(), 2);
    }

    #[test]
    fn test_backend_capabilities() {
        let html_backend = HtmlBackend::new();
        let json_backend = JsonBackend::new();
        let ascii_backend = AsciiBackend::new();

        let html_caps = html_backend.capabilities();
        assert!(html_caps.interactive);
        assert!(html_caps.animations);
        assert!(html_caps.real_time_updates);

        let json_caps = json_backend.capabilities();
        assert!(!json_caps.interactive);
        assert!(!json_caps.animations);
        assert!(!json_caps.real_time_updates);

        let ascii_caps = ascii_backend.capabilities();
        assert!(!ascii_caps.interactive);
        assert!(!ascii_caps.animations);
        assert!(!ascii_caps.real_time_updates);
    }

    #[test]
    fn test_find_backends_for_format() {
        let mut registry = BackendRegistry::new();
        registry.register_backend(HtmlBackend::new());
        registry.register_backend(JsonBackend::new());
        registry.register_backend(AsciiBackend::new());

        let html_backends = registry.find_backends_for_format(OutputFormat::Html);
        assert_eq!(html_backends.len(), 1);
        assert!(html_backends.contains(&"html".to_string()));

        let json_backends = registry.find_backends_for_format(OutputFormat::Json);
        assert_eq!(json_backends.len(), 1);
        assert!(json_backends.contains(&"json".to_string()));

        let ascii_backends = registry.find_backends_for_format(OutputFormat::Ascii);
        assert_eq!(ascii_backends.len(), 1);
        assert!(ascii_backends.contains(&"ascii".to_string()));
    }
}
