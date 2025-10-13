//! External Visualization Library Integrations
//!
//! This module provides integrations with popular external visualization libraries
//! such as Plotly, D3.js, Vega-Lite, and others for advanced interactive visualizations.

use crate::{
    visualization_backend::{
        BackendCapabilities, BackendConfig, ComparativeData, CustomPlotData, FeatureImportanceData,
        OutputFormat, PartialDependenceData, RenderedVisualization, ShapData, VisualizationBackend,
        VisualizationMetadata,
    },
    Float, SklResult, SklearsError,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Plotly backend for rich interactive visualizations
#[derive(Debug)]
pub struct PlotlyBackend {
    config: PlotlyConfig,
}

/// Configuration for Plotly backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyConfig {
    /// Plotly library version
    pub version: String,
    /// CDN URL for Plotly.js
    pub cdn_url: String,
    /// Default plot configuration
    pub default_config: PlotlyPlotConfig,
    /// Whether to include responsive behavior
    pub responsive: bool,
    /// Custom JavaScript code to include
    pub custom_js: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyPlotConfig {
    /// Display mode bar
    pub display_mode_bar: bool,
    /// Static plot (non-interactive)
    pub static_plot: bool,
    /// Show tips
    pub show_tips: bool,
    /// Editable
    pub editable: bool,
}

impl Default for PlotlyConfig {
    fn default() -> Self {
        Self {
            version: "2.27.0".to_string(),
            cdn_url: "https://cdn.plot.ly/plotly-2.27.0.min.js".to_string(),
            default_config: PlotlyPlotConfig {
                display_mode_bar: true,
                static_plot: false,
                show_tips: true,
                editable: false,
            },
            responsive: true,
            custom_js: None,
        }
    }
}

impl PlotlyBackend {
    /// Create a new Plotly backend
    pub fn new(config: PlotlyConfig) -> Self {
        Self { config }
    }

    /// Generate Plotly JavaScript for feature importance
    fn generate_feature_importance_plotly(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<String> {
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();
        let mut error_y = Vec::new();

        for (i, &importance) in data.importance_values.iter().enumerate() {
            x_values.push(
                data.feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature {}", i)),
            );
            y_values.push(importance);
            if let Some(ref std_values) = data.std_values {
                error_y.push(std_values.get(i).copied().unwrap_or(0.0));
            } else {
                error_y.push(0.0);
            }
        }

        let has_error = data.std_values.is_some() && !error_y.iter().all(|&x| x == 0.0);

        let plot_data = if has_error {
            format!(
                r#"{{
                    x: {:?},
                    y: {:?},
                    error_y: {{
                        type: 'data',
                        array: {:?},
                        visible: true
                    }},
                    type: 'bar',
                    name: 'Feature Importance',
                    marker: {{
                        color: 'rgba(158,202,225,0.8)',
                        line: {{
                            color: 'rgba(8,48,107,1.0)',
                            width: 1.5
                        }}
                    }}
                }}"#,
                x_values, y_values, error_y
            )
        } else {
            format!(
                r#"{{
                    x: {:?},
                    y: {:?},
                    type: 'bar',
                    name: 'Feature Importance',
                    marker: {{
                        color: 'rgba(158,202,225,0.8)',
                        line: {{
                            color: 'rgba(8,48,107,1.0)',
                            width: 1.5
                        }}
                    }}
                }}"#,
                x_values, y_values
            )
        };

        let layout = format!(
            r#"{{
                title: 'Feature Importance',
                xaxis: {{
                    title: 'Features',
                    tickangle: -45
                }},
                yaxis: {{
                    title: 'Importance Score'
                }},
                width: {},
                height: {},
                margin: {{
                    l: 60,
                    r: 30,
                    b: 120,
                    t: 60
                }}
            }}"#,
            config.width, config.height
        );

        let plot_config = format!(
            r#"{{
                displayModeBar: {},
                staticPlot: {},
                showTips: {},
                editable: {},
                responsive: {}
            }}"#,
            self.config.default_config.display_mode_bar,
            self.config.default_config.static_plot,
            self.config.default_config.show_tips,
            self.config.default_config.editable,
            self.config.responsive
        );

        Ok(format!(
            r#"
            var data = [{}];
            var layout = {};
            var config = {};
            Plotly.newPlot('plotly-div', data, layout, config);
            "#,
            plot_data, layout, plot_config
        ))
    }

    /// Generate Plotly JavaScript for SHAP values
    fn generate_shap_plotly(&self, data: &ShapData, config: &BackendConfig) -> SklResult<String> {
        let x_values: Vec<String> = data.feature_names.clone();
        // Use first instance's SHAP values for visualization
        let shap_values: Vec<Float> = if data.shap_values.nrows() > 0 {
            data.shap_values.row(0).to_vec()
        } else {
            vec![0.0; data.feature_names.len()]
        };

        let plot_data = format!(
            r#"{{
                x: {:?},
                y: {:?},
                type: 'bar',
                name: 'SHAP Values',
                marker: {{
                    color: {:?},
                    colorscale: 'RdBu',
                    line: {{
                        color: 'rgba(0,0,0,0.2)',
                        width: 1
                    }}
                }}
            }}"#,
            x_values,
            shap_values,
            shap_values
                .iter()
                .map(|&v| if v >= 0.0 {
                    "rgba(255,0,0,0.8)"
                } else {
                    "rgba(0,0,255,0.8)"
                })
                .collect::<Vec<_>>()
        );

        let layout = format!(
            r#"{{
                title: 'SHAP Values',
                xaxis: {{
                    title: 'Features',
                    tickangle: -45
                }},
                yaxis: {{
                    title: 'SHAP Value',
                    zeroline: true,
                    zerolinecolor: 'rgb(0,0,0)',
                    zerolinewidth: 2
                }},
                width: {},
                height: {},
                margin: {{
                    l: 60,
                    r: 30,
                    b: 120,
                    t: 60
                }}
            }}"#,
            config.width, config.height
        );

        let plot_config = format!(
            r#"{{
                displayModeBar: {},
                staticPlot: {},
                responsive: {}
            }}"#,
            self.config.default_config.display_mode_bar,
            self.config.default_config.static_plot,
            self.config.responsive
        );

        Ok(format!(
            r#"
            var data = [{}];
            var layout = {};
            var config = {};
            Plotly.newPlot('plotly-div', data, layout, config);
            "#,
            plot_data, layout, plot_config
        ))
    }

    /// Generate complete HTML with Plotly
    fn generate_complete_html(&self, js_code: &str, title: &str) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <script src="{}"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        #plotly-div {{
            width: 100%;
            height: 100%;
        }}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div id="plotly-div"></div>
    <script>
        {}
        {}
    </script>
</body>
</html>"#,
            title,
            self.config.cdn_url,
            title,
            js_code,
            self.config.custom_js.as_deref().unwrap_or("")
        )
    }
}

impl VisualizationBackend for PlotlyBackend {
    fn render_feature_importance(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let js_code = self.generate_feature_importance_plotly(data, config)?;

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(&js_code, "Feature Importance"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "plotly",
                "javascript": js_code,
                "title": "Feature Importance"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for Plotly backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "plotly".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
                data_points: 1,     // Custom plot data count
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
        let js_code = self.generate_shap_plotly(data, config)?;

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(&js_code, "SHAP Values"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "plotly",
                "javascript": js_code,
                "title": "SHAP Values"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for Plotly backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "plotly".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
                data_points: 1,     // Custom plot data count
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
        let plot_data = format!(
            r#"{{
                x: {:?},
                y: {:?},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Partial Dependence',
                line: {{
                    color: 'rgb(31, 119, 180)',
                    width: 3
                }},
                marker: {{
                    color: 'rgb(31, 119, 180)',
                    size: 6
                }}
            }}"#,
            data.feature_values.to_vec(),
            data.pd_values.to_vec()
        );

        let layout = format!(
            r#"{{
                title: 'Partial Dependence Plot',
                xaxis: {{
                    title: 'Feature Value'
                }},
                yaxis: {{
                    title: 'Partial Dependence'
                }},
                width: {},
                height: {}
            }}"#,
            config.width, config.height
        );

        let js_code = format!(
            r#"
            var data = [{}];
            var layout = {};
            Plotly.newPlot('plotly-div', data, layout);
            "#,
            plot_data, layout
        );

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(&js_code, "Partial Dependence Plot"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "plotly",
                "javascript": js_code,
                "title": "Partial Dependence Plot"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for Plotly backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "plotly".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
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
        let mut traces = Vec::new();

        for (i, (method_name, method_data)) in data.model_data.iter().enumerate() {
            let color = match i % 6 {
                0 => "rgb(31, 119, 180)",
                1 => "rgb(255, 127, 14)",
                2 => "rgb(44, 160, 44)",
                3 => "rgb(214, 39, 40)",
                4 => "rgb(148, 103, 189)",
                _ => "rgb(140, 86, 75)",
            };

            // Use first row of data for visualization
            let values: Vec<Float> = if method_data.nrows() > 0 {
                method_data.row(0).to_vec()
            } else {
                vec![0.0; data.labels.len()]
            };

            traces.push(format!(
                r#"{{
                    x: {:?},
                    y: {:?},
                    type: 'bar',
                    name: '{}',
                    marker: {{ color: '{}' }}
                }}"#,
                data.labels, values, method_name, color
            ));
        }

        let layout = format!(
            r#"{{
                title: 'Method Comparison',
                xaxis: {{
                    title: 'Features',
                    tickangle: -45
                }},
                yaxis: {{
                    title: 'Importance Score'
                }},
                barmode: 'group',
                width: {},
                height: {}
            }}"#,
            config.width, config.height
        );

        let js_code = format!(
            r#"
            var data = [{}];
            var layout = {};
            Plotly.newPlot('plotly-div', data, layout);
            "#,
            traces.join(","),
            layout
        );

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(&js_code, "Method Comparison"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "plotly",
                "javascript": js_code,
                "title": "Method Comparison"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for Plotly backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "plotly".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
                data_points: data.labels.len(),
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
        // For custom plots, we expect the data to contain Plotly-compatible JSON
        let js_code = format!(
            r#"
            var data = {};
            var layout = {};
            Plotly.newPlot('plotly-div', data, layout);
            "#,
            data.data
                .get("data")
                .unwrap_or(&serde_json::Value::Array(vec![])),
            data.data.get("layout").unwrap_or(&serde_json::json!({}))
        );

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(&js_code, "Custom Plot"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "plotly",
                "javascript": js_code,
                "title": "Custom Plot"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for Plotly backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "plotly".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
                data_points: 1,     // Custom plot
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn name(&self) -> &str {
        "plotly"
    }

    fn supported_formats(&self) -> Vec<OutputFormat> {
        vec![OutputFormat::Html, OutputFormat::Json]
    }

    fn supports_interactivity(&self) -> bool {
        !self.config.default_config.static_plot
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            formats: self.supported_formats(),
            interactive: self.supports_interactivity(),
            animations: true,
            three_d: false,
            custom_themes: true,
            real_time_updates: true,
            max_data_points: Some(100000),
        }
    }
}

/// D3.js backend for custom interactive visualizations
#[derive(Debug)]
pub struct D3Backend {
    config: D3Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Config {
    /// D3.js library version
    pub version: String,
    /// CDN URL for D3.js
    pub cdn_url: String,
    /// Custom CSS styles
    pub custom_css: Option<String>,
    /// Custom JavaScript code
    pub custom_js: Option<String>,
}

impl Default for D3Config {
    fn default() -> Self {
        Self {
            version: "7.8.5".to_string(),
            cdn_url: "https://cdn.jsdelivr.net/npm/d3@7".to_string(),
            custom_css: None,
            custom_js: None,
        }
    }
}

impl D3Backend {
    /// Create a new D3.js backend
    pub fn new(config: D3Config) -> Self {
        Self { config }
    }

    /// Generate D3.js code for feature importance
    fn generate_feature_importance_d3(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<String> {
        let data_json = serde_json::to_string(&serde_json::json!({
            "features": data.importance_values.iter().enumerate().map(|(i, &val)| {
                serde_json::json!({
                    "name": data.feature_names.get(i).cloned().unwrap_or_else(|| format!("Feature {}", i)),
                    "value": val,
                    "std": data.std_values.as_ref().map(|s| s.get(i).copied().unwrap_or(0.0))
                })
            }).collect::<Vec<_>>()
        })).map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?;

        Ok(format!(
            r##"
            const data = {};
            
            const margin = {{top: 20, right: 30, bottom: 70, left: 60}};
            const width = {} - margin.left - margin.right;
            const height = {} - margin.bottom - margin.top;

            const svg = d3.select("#d3-div")
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom);

            const g = svg.append("g")
                .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            const x = d3.scaleBand()
                .range([0, width])
                .domain(data.features.map(d => d.name))
                .padding(0.1);

            const y = d3.scaleLinear()
                .range([height, 0])
                .domain([0, d3.max(data.features, d => d.value)]);

            g.append("g")
                .attr("transform", `translate(0,${{height}})`)
                .call(d3.axisBottom(x))
                .selectAll("text")
                .style("text-anchor", "end")
                .attr("dx", "-.8em")
                .attr("dy", ".15em")
                .attr("transform", "rotate(-45)");

            g.append("g")
                .call(d3.axisLeft(y));

            g.selectAll(".bar")
                .data(data.features)
                .enter().append("rect")
                .attr("class", "bar")
                .attr("x", d => x(d.name))
                .attr("width", x.bandwidth())
                .attr("y", d => y(d.value))
                .attr("height", d => height - y(d.value))
                .attr("fill", "steelblue")
                .on("mouseover", function(event, d) {{
                    d3.select(this).attr("fill", "orange");
                    
                    const tooltip = d3.select("body").append("div")
                        .attr("class", "tooltip")
                        .style("opacity", 0)
                        .style("position", "absolute")
                        .style("background", "rgba(0,0,0,0.8)")
                        .style("color", "white")
                        .style("padding", "10px")
                        .style("border-radius", "5px")
                        .style("pointer-events", "none");

                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    
                    tooltip.html(`${{d.name}}: ${{d.value.toFixed(4)}}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                }})
                .on("mouseout", function(d) {{
                    d3.select(this).attr("fill", "steelblue");
                    d3.selectAll(".tooltip").remove();
                }});

            // Add title
            g.append("text")
                .attr("x", width / 2)
                .attr("y", 0 - (margin.top / 2))
                .attr("text-anchor", "middle")
                .style("font-size", "16px")
                .style("font-weight", "bold")
                .text("Feature Importance");

            // Add axis labels
            g.append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", 0 - margin.left)
                .attr("x", 0 - (height / 2))
                .attr("dy", "1em")
                .style("text-anchor", "middle")
                .text("Importance Score");
            "##,
            data_json, config.width, config.height
        ))
    }

    /// Generate complete HTML with D3.js
    fn generate_complete_html(&self, js_code: &str, title: &str) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <script src="{}"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        #d3-div {{
            width: 100%;
            height: 100%;
        }}
        .bar:hover {{
            fill: orange;
        }}
        {}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div id="d3-div"></div>
    <script>
        {}
        {}
    </script>
</body>
</html>"#,
            title,
            self.config.cdn_url,
            self.config.custom_css.as_deref().unwrap_or(""),
            title,
            js_code,
            self.config.custom_js.as_deref().unwrap_or("")
        )
    }
}

impl VisualizationBackend for D3Backend {
    fn render_feature_importance(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let js_code = self.generate_feature_importance_d3(data, config)?;

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(&js_code, "Feature Importance"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "d3",
                "javascript": js_code,
                "title": "Feature Importance"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for D3 backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "d3".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
                data_points: 1,     // Custom plot data count
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_shap_plot(
        &self,
        _data: &ShapData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        // Implementation would be similar to feature importance but with SHAP-specific styling
        Err(SklearsError::NotImplemented(
            "SHAP plot for D3 backend not yet implemented".to_string(),
        ))
    }

    fn render_partial_dependence(
        &self,
        _data: &PartialDependenceData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        // Implementation would create line plots for partial dependence
        Err(SklearsError::NotImplemented(
            "Partial dependence plot for D3 backend not yet implemented".to_string(),
        ))
    }

    fn render_comparative_plot(
        &self,
        _data: &ComparativeData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        // Implementation would create grouped bar charts
        Err(SklearsError::NotImplemented(
            "Comparative plot for D3 backend not yet implemented".to_string(),
        ))
    }

    fn render_custom_plot(
        &self,
        data: &CustomPlotData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        // For custom plots, we expect D3.js code in the data
        let js_code = data
            .data
            .get("d3_code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                SklearsError::InvalidInput("D3 code not found in custom plot data".to_string())
            })?;

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(js_code, "Custom D3 Plot"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "d3",
                "javascript": js_code,
                "title": "Custom D3 Plot"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for D3 backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "d3".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
                data_points: 1,     // Custom plot data count
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn name(&self) -> &str {
        "d3"
    }

    fn supported_formats(&self) -> Vec<OutputFormat> {
        vec![OutputFormat::Html, OutputFormat::Json]
    }

    fn supports_interactivity(&self) -> bool {
        true
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            formats: self.supported_formats(),
            interactive: true,
            animations: true,
            three_d: false,
            custom_themes: true,
            real_time_updates: true,
            max_data_points: Some(50000),
        }
    }
}

/// Vega-Lite backend for grammar of graphics visualizations
#[derive(Debug)]
pub struct VegaLiteBackend {
    config: VegaLiteConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegaLiteConfig {
    /// Vega-Lite version
    pub version: String,
    /// CDN URLs for Vega-Lite
    pub vega_url: String,
    /// vega_lite_url
    pub vega_lite_url: String,
    /// vega_embed_url
    pub vega_embed_url: String,
    /// Default theme
    pub theme: String,
    /// Default configuration
    pub default_config: VegaLiteDefaultConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegaLiteDefaultConfig {
    /// Actions to show in embed
    pub actions: Vec<String>,
    /// Whether to show tooltip
    pub tooltip: bool,
    /// Renderer type
    pub renderer: String,
}

impl Default for VegaLiteConfig {
    fn default() -> Self {
        Self {
            version: "5.8.0".to_string(),
            vega_url: "https://cdn.jsdelivr.net/npm/vega@5".to_string(),
            vega_lite_url: "https://cdn.jsdelivr.net/npm/vega-lite@5".to_string(),
            vega_embed_url: "https://cdn.jsdelivr.net/npm/vega-embed@6".to_string(),
            theme: "default".to_string(),
            default_config: VegaLiteDefaultConfig {
                actions: vec![
                    "export".to_string(),
                    "source".to_string(),
                    "compiled".to_string(),
                    "editor".to_string(),
                ],
                tooltip: true,
                renderer: "canvas".to_string(),
            },
        }
    }
}

impl VegaLiteBackend {
    /// Create a new Vega-Lite backend
    pub fn new(config: VegaLiteConfig) -> Self {
        Self { config }
    }

    /// Generate Vega-Lite specification for feature importance
    fn generate_feature_importance_vega(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<String> {
        let vega_data: Vec<serde_json::Value> = data.importance_values.iter().enumerate().map(|(i, &val)| {
            serde_json::json!({
                "feature": data.feature_names.get(i).cloned().unwrap_or_else(|| format!("Feature {}", i)),
                "importance": val,
                "std": data.std_values.as_ref().map(|s| s.get(i).copied().unwrap_or(0.0))
            })
        }).collect();

        let spec = serde_json::json!({
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Feature Importance Visualization",
            "width": config.width,
            "height": config.height,
            "data": {
                "values": vega_data
            },
            "mark": {
                "type": "bar",
                "tooltip": true
            },
            "encoding": {
                "x": {
                    "field": "feature",
                    "type": "nominal",
                    "axis": {
                        "title": "Features",
                        "labelAngle": -45
                    }
                },
                "y": {
                    "field": "importance",
                    "type": "quantitative",
                    "axis": {
                        "title": "Importance Score"
                    }
                },
                "color": {
                    "value": "steelblue"
                },
                "tooltip": [
                    {"field": "feature", "type": "nominal"},
                    {"field": "importance", "type": "quantitative", "format": ".4f"}
                ]
            },
            "title": {
                "text": "Feature Importance",
                "fontSize": 16,
                "anchor": "start"
            }
        });

        let embed_options = serde_json::json!({
            "actions": self.config.default_config.actions,
            "tooltip": self.config.default_config.tooltip,
            "renderer": self.config.default_config.renderer,
            "theme": self.config.theme
        });

        let js_code = format!(
            r#"
            const spec = {};
            const opt = {};
            vegaEmbed('#vega-div', spec, opt).catch(console.error);
            "#,
            serde_json::to_string_pretty(&spec)
                .map_err(|e| SklearsError::Other(format!("JSON error: {}", e)))?,
            serde_json::to_string_pretty(&embed_options)
                .map_err(|e| SklearsError::Other(format!("JSON error: {}", e)))?
        );

        Ok(js_code)
    }

    /// Generate complete HTML with Vega-Lite
    fn generate_complete_html(&self, js_code: &str, title: &str) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <script src="{}"></script>
    <script src="{}"></script>
    <script src="{}"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        #vega-div {{
            width: 100%;
            height: 100%;
        }}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div id="vega-div"></div>
    <script>
        {}
    </script>
</body>
</html>"#,
            title,
            self.config.vega_url,
            self.config.vega_lite_url,
            self.config.vega_embed_url,
            title,
            js_code
        )
    }
}

impl VisualizationBackend for VegaLiteBackend {
    fn render_feature_importance(
        &self,
        data: &FeatureImportanceData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        let js_code = self.generate_feature_importance_vega(data, config)?;

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(&js_code, "Feature Importance"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "vega-lite",
                "javascript": js_code,
                "title": "Feature Importance"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for Vega-Lite backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "vega-lite".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
                data_points: 1,     // Custom plot data count
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn render_shap_plot(
        &self,
        _data: &ShapData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        // Implementation would be similar to feature importance but with SHAP-specific styling
        Err(SklearsError::NotImplemented(
            "SHAP plot for Vega-Lite backend not yet implemented".to_string(),
        ))
    }

    fn render_partial_dependence(
        &self,
        _data: &PartialDependenceData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        // Implementation would create line plots for partial dependence
        Err(SklearsError::NotImplemented(
            "Partial dependence plot for Vega-Lite backend not yet implemented".to_string(),
        ))
    }

    fn render_comparative_plot(
        &self,
        _data: &ComparativeData,
        _config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        // Implementation would create grouped bar charts
        Err(SklearsError::NotImplemented(
            "Comparative plot for Vega-Lite backend not yet implemented".to_string(),
        ))
    }

    fn render_custom_plot(
        &self,
        data: &CustomPlotData,
        config: &BackendConfig,
    ) -> SklResult<RenderedVisualization> {
        // For custom plots, we expect Vega-Lite specification in the data
        let spec = data.data.get("vega_lite_spec").ok_or_else(|| {
            SklearsError::InvalidInput("Vega-Lite spec not found in custom plot data".to_string())
        })?;

        let js_code = format!(
            r#"
            const spec = {};
            vegaEmbed('#vega-div', spec, {{}}).catch(console.error);
            "#,
            serde_json::to_string_pretty(spec)
                .map_err(|e| SklearsError::Other(format!("JSON error: {}", e)))?
        );

        let content = match config.format {
            OutputFormat::Html => self.generate_complete_html(&js_code, "Custom Vega-Lite Plot"),
            OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "type": "vega-lite",
                "javascript": js_code,
                "title": "Custom Vega-Lite Plot"
            }))
            .map_err(|e| SklearsError::Other(format!("JSON serialization error: {}", e)))?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported format for Vega-Lite backend".to_string(),
                ))
            }
        };

        Ok(RenderedVisualization {
            content,
            format: config.format,
            metadata: VisualizationMetadata {
                backend: "vega-lite".to_string(),
                render_time_ms: 0,
                file_size_bytes: 0, // Will be calculated after content is moved
                data_points: 1,     // Custom plot data count
                created_at: chrono::Utc::now(),
            },
            binary_data: None,
        })
    }

    fn name(&self) -> &str {
        "vega-lite"
    }

    fn supported_formats(&self) -> Vec<OutputFormat> {
        vec![OutputFormat::Html, OutputFormat::Json]
    }

    fn supports_interactivity(&self) -> bool {
        true
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            formats: vec![OutputFormat::Html, OutputFormat::Json],
            interactive: true,
            animations: true,
            three_d: false,
            custom_themes: true,
            real_time_updates: false,
            max_data_points: Some(100000),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualization_backend::OutputFormat;

    fn create_test_feature_importance_data() -> FeatureImportanceData {
        FeatureImportanceData {
            feature_names: vec![
                "Feature 0".to_string(),
                "Feature 1".to_string(),
                "Feature 2".to_string(),
            ],
            importance_values: vec![0.5, 0.3, 0.2],
            std_values: Some(vec![0.1, 0.05, 0.03]),
            plot_type: crate::visualization_backend::FeatureImportanceType::Bar,
            title: "Feature Importance".to_string(),
            x_label: "Features".to_string(),
            y_label: "Importance Score".to_string(),
        }
    }

    fn create_test_config() -> BackendConfig {
        BackendConfig {
            format: OutputFormat::Html,
            width: 800,
            height: 600,
            ..Default::default()
        }
    }

    #[test]
    fn test_plotly_backend_creation() {
        let config = PlotlyConfig::default();
        let backend = PlotlyBackend::new(config);
        assert_eq!(backend.name(), "plotly");
        assert!(backend.supports_interactivity());
    }

    #[test]
    fn test_plotly_feature_importance_rendering() {
        let backend = PlotlyBackend::new(PlotlyConfig::default());
        let data = create_test_feature_importance_data();
        let config = create_test_config();

        let result = backend.render_feature_importance(&data, &config);
        assert!(result.is_ok());

        let visualization = result.unwrap();
        assert_eq!(visualization.format, OutputFormat::Html);
        assert!(visualization.content.contains("Plotly.newPlot"));
        assert_eq!(visualization.metadata.backend, "plotly");
    }

    #[test]
    fn test_d3_backend_creation() {
        let config = D3Config::default();
        let backend = D3Backend::new(config);
        assert_eq!(backend.name(), "d3");
        assert!(backend.supports_interactivity());
    }

    #[test]
    fn test_d3_feature_importance_rendering() {
        let backend = D3Backend::new(D3Config::default());
        let data = create_test_feature_importance_data();
        let config = create_test_config();

        let result = backend.render_feature_importance(&data, &config);
        assert!(result.is_ok());

        let visualization = result.unwrap();
        assert_eq!(visualization.format, OutputFormat::Html);
        assert!(visualization.content.contains("d3.select"));
        assert_eq!(visualization.metadata.backend, "d3");
    }

    #[test]
    fn test_vega_lite_backend_creation() {
        let config = VegaLiteConfig::default();
        let backend = VegaLiteBackend::new(config);
        assert_eq!(backend.name(), "vega-lite");
        assert!(backend.supports_interactivity());
    }

    #[test]
    fn test_vega_lite_feature_importance_rendering() {
        let backend = VegaLiteBackend::new(VegaLiteConfig::default());
        let data = create_test_feature_importance_data();
        let config = create_test_config();

        let result = backend.render_feature_importance(&data, &config);
        assert!(result.is_ok());

        let visualization = result.unwrap();
        assert_eq!(visualization.format, OutputFormat::Html);
        assert!(visualization.content.contains("vegaEmbed"));
        assert_eq!(visualization.metadata.backend, "vega-lite");
    }

    #[test]
    fn test_backend_capabilities() {
        let plotly = PlotlyBackend::new(PlotlyConfig::default());
        let d3 = D3Backend::new(D3Config::default());
        let vega = VegaLiteBackend::new(VegaLiteConfig::default());

        let plotly_caps = plotly.capabilities();
        let d3_caps = d3.capabilities();
        let vega_caps = vega.capabilities();

        assert!(plotly_caps.interactive);
        assert!(d3_caps.interactive);
        assert!(vega_caps.interactive);

        assert!(plotly_caps.animations);
        assert!(d3_caps.animations);
        assert!(vega_caps.animations);
    }

    #[test]
    fn test_supported_formats() {
        let plotly = PlotlyBackend::new(PlotlyConfig::default());
        let formats = plotly.supported_formats();

        assert!(formats.contains(&OutputFormat::Html));
        assert!(formats.contains(&OutputFormat::Json));
    }

    #[test]
    fn test_json_output_format() {
        let backend = PlotlyBackend::new(PlotlyConfig::default());
        let data = create_test_feature_importance_data();
        let mut config = create_test_config();
        config.format = OutputFormat::Json;

        let result = backend.render_feature_importance(&data, &config);
        assert!(result.is_ok());

        let visualization = result.unwrap();
        assert_eq!(visualization.format, OutputFormat::Json);

        // Verify it's valid JSON
        let json_value: serde_json::Value = serde_json::from_str(&visualization.content).unwrap();
        assert!(json_value.get("type").is_some());
        assert!(json_value.get("javascript").is_some());
    }
}
