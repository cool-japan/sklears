//! Graph export and visualization functionality
//!
//! This module provides comprehensive export capabilities for trait relationship
//! graphs, supporting multiple formats including interactive HTML, SVG, PNG, PDF,
//! DOT, JSON, and specialized formats for visualization tools.

use super::graph_config::{GraphConfig, GraphExportFormat, VisualizationTheme};
use super::graph_structures::{TraitGraph, TraitGraphNode, TraitGraphEdge, PerformanceMetrics};
use crate::error::{Result, SklearsError};

use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Main exporter for trait graphs with multiple format support
pub struct GraphExporter {
    config: GraphConfig,
    theme_templates: HashMap<VisualizationTheme, ThemeTemplate>,
    interactive_features: InteractiveFeatures,
}

/// Theme template for consistent styling
#[derive(Debug, Clone)]
pub struct ThemeTemplate {
    pub background_color: String,
    pub text_color: String,
    pub accent_color: String,
    pub node_colors: HashMap<String, String>,
    pub edge_colors: HashMap<String, String>,
    pub css_styles: String,
}

/// Interactive features configuration
#[derive(Debug, Clone)]
pub struct InteractiveFeatures {
    pub enable_zoom: bool,
    pub enable_pan: bool,
    pub enable_node_drag: bool,
    pub enable_tooltips: bool,
    pub enable_search: bool,
    pub enable_filtering: bool,
    pub enable_animation: bool,
    pub enable_physics: bool,
}

impl Default for InteractiveFeatures {
    fn default() -> Self {
        Self {
            enable_zoom: true,
            enable_pan: true,
            enable_node_drag: true,
            enable_tooltips: true,
            enable_search: true,
            enable_filtering: true,
            enable_animation: true,
            enable_physics: false,
        }
    }
}

impl GraphExporter {
    /// Create a new graph exporter
    pub fn new(config: GraphConfig) -> Self {
        let mut theme_templates = HashMap::new();

        // Initialize theme templates
        for theme in [
            VisualizationTheme::Light,
            VisualizationTheme::Dark,
            VisualizationTheme::HighContrast,
            VisualizationTheme::ColorblindFriendly,
            VisualizationTheme::Presentation,
            VisualizationTheme::Scientific,
            VisualizationTheme::Minimalist,
            VisualizationTheme::Vibrant,
        ] {
            theme_templates.insert(theme, Self::create_theme_template(theme));
        }

        Self {
            config,
            theme_templates,
            interactive_features: InteractiveFeatures::default(),
        }
    }

    /// Export graph to the specified format
    pub fn export_graph(&self, graph: &TraitGraph, format: GraphExportFormat) -> Result<String> {
        match format {
            GraphExportFormat::Svg => self.to_svg(graph),
            GraphExportFormat::Png => self.to_png(graph),
            GraphExportFormat::Pdf => self.to_pdf(graph),
            GraphExportFormat::Dot => self.to_dot(graph),
            GraphExportFormat::Json => self.to_json(graph),
            GraphExportFormat::InteractiveHtml => self.to_interactive_html(graph),
            GraphExportFormat::StaticHtml => self.to_static_html(graph),
            GraphExportFormat::Csv => self.to_csv(graph),
            GraphExportFormat::GraphMl => self.to_graphml(graph),
            GraphExportFormat::Gexf => self.to_gexf(graph),
        }
    }

    /// Export to interactive HTML with D3.js
    pub fn to_interactive_html(&self, graph: &TraitGraph) -> Result<String> {
        let theme = self.theme_templates.get(&self.config.theme)
            .ok_or_else(|| SklearsError::ValidationError("Theme not found".to_string()))?;

        let graph_data = self.to_json_object(graph)?;
        let css_styles = self.generate_css_styles(theme);
        let javascript_code = self.generate_interactive_javascript(graph);

        let html = format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        {}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <div id="header">
        <h1>{}</h1>
        <div id="controls">
            <button id="zoomIn">Zoom In</button>
            <button id="zoomOut">Zoom Out</button>
            <button id="resetView">Reset View</button>
            <input type="text" id="search" placeholder="Search nodes...">
            <select id="layoutSelect">
                <option value="force">Force-Directed</option>
                <option value="hierarchical">Hierarchical</option>
                <option value="circular">Circular</option>
                <option value="radial">Radial</option>
            </select>
        </div>
    </div>
    <div id="graph-container">
        <svg id="graph"></svg>
    </div>
    <div id="info-panel">
        <div id="node-info"></div>
        <div id="statistics">
            <h3>Graph Statistics</h3>
            <p>Nodes: {}</p>
            <p>Edges: {}</p>
            <p>Density: {:.3}</p>
            <p>Avg Degree: {:.2}</p>
        </div>
    </div>
    <script>
        const graphData = {};
        {}
    </script>
</body>
</html>"#,
            graph.metadata.title,
            css_styles,
            graph.metadata.title,
            graph.statistics.node_count,
            graph.statistics.edge_count,
            graph.statistics.density,
            graph.statistics.average_degree,
            graph_data,
            javascript_code
        );

        Ok(html)
    }

    /// Export to static HTML
    pub fn to_static_html(&self, graph: &TraitGraph) -> Result<String> {
        let theme = self.theme_templates.get(&self.config.theme)
            .ok_or_else(|| SklearsError::ValidationError("Theme not found".to_string()))?;

        let svg_content = self.generate_static_svg(graph)?;
        let css_styles = self.generate_css_styles(theme);

        let html = format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        {}
    </style>
</head>
<body>
    <div id="header">
        <h1>{}</h1>
        <p>{}</p>
    </div>
    <div id="graph-container">
        {}
    </div>
    <div id="statistics">
        <h3>Graph Statistics</h3>
        <ul>
            <li>Nodes: {}</li>
            <li>Edges: {}</li>
            <li>Density: {:.3}</li>
            <li>Average Degree: {:.2}</li>
            <li>Generated: {}</li>
        </ul>
    </div>
</body>
</html>"#,
            graph.metadata.title,
            css_styles,
            graph.metadata.title,
            graph.metadata.description.as_deref().unwrap_or(""),
            svg_content,
            graph.statistics.node_count,
            graph.statistics.edge_count,
            graph.statistics.density,
            graph.statistics.average_degree,
            graph.metadata.generated_at.format("%Y-%m-%d %H:%M:%S")
        );

        Ok(html)
    }

    /// Export to SVG format
    pub fn to_svg(&self, graph: &TraitGraph) -> Result<String> {
        self.generate_static_svg(graph)
    }

    /// Export to PNG format (placeholder - requires external rendering)
    pub fn to_png(&self, _graph: &TraitGraph) -> Result<String> {
        Err(SklearsError::ValidationError(
            "PNG export requires external rendering library. Use SVG export instead.".to_string()
        ))
    }

    /// Export to PDF format (placeholder - requires external rendering)
    pub fn to_pdf(&self, _graph: &TraitGraph) -> Result<String> {
        Err(SklearsError::ValidationError(
            "PDF export requires external rendering library. Use SVG export instead.".to_string()
        ))
    }

    /// Export to DOT format for Graphviz
    pub fn to_dot(&self, graph: &TraitGraph) -> Result<String> {
        let mut dot = String::new();

        dot.push_str("digraph TraitGraph {\n");
        dot.push_str("    rankdir=TB;\n");
        dot.push_str("    node [shape=ellipse, style=filled];\n");
        dot.push_str("    edge [fontsize=10];\n\n");

        // Add graph attributes
        dot.push_str(&format!("    label=\"{}\";\n", graph.metadata.title));
        dot.push_str("    labelloc=t;\n");
        dot.push_str("    fontsize=16;\n\n");

        // Add nodes
        for node in &graph.nodes {
            let color = node.color.as_deref().unwrap_or("#ffffff");
            let shape = match node.node_type {
                super::graph_config::TraitNodeType::Trait => "ellipse",
                super::graph_config::TraitNodeType::Implementation => "box",
                super::graph_config::TraitNodeType::AssociatedType => "diamond",
                super::graph_config::TraitNodeType::Method => "circle",
                _ => "ellipse",
            };

            dot.push_str(&format!(
                "    \"{}\" [label=\"{}\", fillcolor=\"{}\", shape={}];\n",
                Self::escape_dot_string(&node.id),
                Self::escape_dot_string(&node.label),
                color,
                shape
            ));
        }

        dot.push_str("\n");

        // Add edges
        for edge in &graph.edges {
            let style = if edge.directed { "->" } else { "--" };
            let label = edge.label.as_deref().unwrap_or("");
            let color = edge.color.as_deref().unwrap_or("#000000");
            let thickness = edge.thickness.unwrap_or(1.0);

            dot.push_str(&format!(
                "    \"{}\" {} \"{}\" [label=\"{}\", color=\"{}\", penwidth={}];\n",
                Self::escape_dot_string(&edge.from),
                style,
                Self::escape_dot_string(&edge.to),
                Self::escape_dot_string(label),
                color,
                thickness
            ));
        }

        dot.push_str("}\n");
        Ok(dot)
    }

    /// Export to JSON format
    pub fn to_json(&self, graph: &TraitGraph) -> Result<String> {
        self.to_json_object(graph)
    }

    /// Export to CSV format (edge list)
    pub fn to_csv(&self, graph: &TraitGraph) -> Result<String> {
        let mut csv = String::new();

        csv.push_str("Source,Target,Type,Weight,Label,Directed\n");

        for edge in &graph.edges {
            csv.push_str(&format!(
                "{},{},{},{},{},{}\n",
                Self::escape_csv_field(&edge.from),
                Self::escape_csv_field(&edge.to),
                edge.edge_type.display_name(),
                edge.weight,
                Self::escape_csv_field(edge.label.as_deref().unwrap_or("")),
                edge.directed
            ));
        }

        Ok(csv)
    }

    /// Export to GraphML format
    pub fn to_graphml(&self, graph: &TraitGraph) -> Result<String> {
        let mut graphml = String::new();

        graphml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="label" for="node" attr.name="label" attr.type="string"/>
  <key id="type" for="node" attr.name="type" attr.type="string"/>
  <key id="size" for="node" attr.name="size" attr.type="double"/>
  <key id="color" for="node" attr.name="color" attr.type="string"/>
  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>
  <key id="label" for="edge" attr.name="label" attr.type="string"/>
  <graph id="TraitGraph" edgedefault="directed">
"#);

        // Add nodes
        for node in &graph.nodes {
            graphml.push_str(&format!(
                r#"    <node id="{}">
      <data key="label">{}</data>
      <data key="type">{}</data>
      <data key="size">{}</data>
      <data key="color">{}</data>
    </node>
"#,
                Self::escape_xml(&node.id),
                Self::escape_xml(&node.label),
                node.node_type.display_name(),
                node.size,
                node.color.as_deref().unwrap_or("#ffffff")
            ));
        }

        // Add edges
        for (i, edge) in graph.edges.iter().enumerate() {
            graphml.push_str(&format!(
                r#"    <edge id="e{}" source="{}" target="{}">
      <data key="weight">{}</data>
      <data key="label">{}</data>
    </edge>
"#,
                i,
                Self::escape_xml(&edge.from),
                Self::escape_xml(&edge.to),
                edge.weight,
                Self::escape_xml(edge.label.as_deref().unwrap_or(""))
            ));
        }

        graphml.push_str("  </graph>\n</graphml>\n");
        Ok(graphml)
    }

    /// Export to GEXF format for Gephi
    pub fn to_gexf(&self, graph: &TraitGraph) -> Result<String> {
        let mut gexf = String::new();

        gexf.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
  <meta lastmodifieddate="2024-01-01">
    <creator>sklears-core graph visualization</creator>
    <description>Trait relationship graph</description>
  </meta>
  <graph mode="static" defaultedgetype="directed">
    <nodes>
"#);

        // Add nodes
        for node in &graph.nodes {
            gexf.push_str(&format!(
                r#"      <node id="{}" label="{}">
        <attvalues>
          <attvalue for="type" value="{}"/>
          <attvalue for="size" value="{}"/>
        </attvalues>
      </node>
"#,
                Self::escape_xml(&node.id),
                Self::escape_xml(&node.label),
                node.node_type.display_name(),
                node.size
            ));
        }

        gexf.push_str("    </nodes>\n    <edges>\n");

        // Add edges
        for (i, edge) in graph.edges.iter().enumerate() {
            gexf.push_str(&format!(
                r#"      <edge id="{}" source="{}" target="{}" weight="{}"/>
"#,
                i,
                Self::escape_xml(&edge.from),
                Self::escape_xml(&edge.to),
                edge.weight
            ));
        }

        gexf.push_str("    </edges>\n  </graph>\n</gexf>\n");
        Ok(gexf)
    }

    /// Save graph to file
    pub fn save_to_file<P: AsRef<Path>>(&self, graph: &TraitGraph, path: P, format: GraphExportFormat) -> Result<()> {
        let content = self.export_graph(graph, format)?;
        fs::write(path, content)
            .map_err(|e| SklearsError::ValidationError(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

    /// Generate JSON object representation
    fn to_json_object(&self, graph: &TraitGraph) -> Result<String> {
        let mut json_graph = serde_json::Map::new();

        // Add metadata
        json_graph.insert("metadata".to_string(), serde_json::to_value(&graph.metadata)?);
        json_graph.insert("statistics".to_string(), serde_json::to_value(&graph.statistics)?);
        json_graph.insert("performance".to_string(), serde_json::to_value(&graph.performance)?);

        // Add nodes
        let nodes: Vec<_> = graph.nodes.iter().map(|node| {
            let mut node_obj = serde_json::Map::new();
            node_obj.insert("id".to_string(), serde_json::Value::String(node.id.clone()));
            node_obj.insert("label".to_string(), serde_json::Value::String(node.label.clone()));
            node_obj.insert("type".to_string(), serde_json::Value::String(node.node_type.display_name().to_string()));
            node_obj.insert("size".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(node.size).expect("valid JSON operation")));

            if let Some(ref color) = node.color {
                node_obj.insert("color".to_string(), serde_json::Value::String(color.clone()));
            }

            if let Some((x, y)) = node.position_2d {
                node_obj.insert("x".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(x).expect("valid JSON operation")));
                node_obj.insert("y".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(y).expect("valid JSON operation")));
            }

            serde_json::Value::Object(node_obj)
        }).collect();

        // Add edges
        let edges: Vec<_> = graph.edges.iter().map(|edge| {
            let mut edge_obj = serde_json::Map::new();
            edge_obj.insert("source".to_string(), serde_json::Value::String(edge.from.clone()));
            edge_obj.insert("target".to_string(), serde_json::Value::String(edge.to.clone()));
            edge_obj.insert("type".to_string(), serde_json::Value::String(edge.edge_type.display_name().to_string()));
            edge_obj.insert("weight".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(edge.weight).expect("valid JSON operation")));

            if let Some(ref label) = edge.label {
                edge_obj.insert("label".to_string(), serde_json::Value::String(label.clone()));
            }

            if let Some(ref color) = edge.color {
                edge_obj.insert("color".to_string(), serde_json::Value::String(color.clone()));
            }

            serde_json::Value::Object(edge_obj)
        }).collect();

        json_graph.insert("nodes".to_string(), serde_json::Value::Array(nodes));
        json_graph.insert("edges".to_string(), serde_json::Value::Array(edges));

        Ok(serde_json::to_string_pretty(&json_graph)?)
    }

    /// Generate static SVG representation
    fn generate_static_svg(&self, graph: &TraitGraph) -> Result<String> {
        let width = 800;
        let height = 600;
        let theme = self.theme_templates.get(&self.config.theme)
            .ok_or_else(|| SklearsError::ValidationError("Theme not found".to_string()))?;

        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
  <style>
    .node {{ fill: {}; stroke: {}; stroke-width: 2; }}
    .edge {{ stroke: {}; stroke-width: 1; fill: none; }}
    .label {{ font-family: Arial, sans-serif; font-size: 12px; fill: {}; text-anchor: middle; }}
    .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: {}; text-anchor: middle; }}
  </style>
  <rect width="100%" height="100%" fill="{}"/>
  <text x="{}" y="30" class="title">{}</text>
"#,
            width, height,
            theme.accent_color, theme.text_color,
            theme.text_color, theme.text_color, theme.text_color,
            theme.background_color,
            width / 2, graph.metadata.title
        );

        // Simple layout for static SVG
        let node_positions = self.calculate_simple_layout(graph, width, height);

        // Draw edges first (so they appear behind nodes)
        for edge in &graph.edges {
            if let (Some(&(x1, y1)), Some(&(x2, y2))) = (
                node_positions.get(&edge.from),
                node_positions.get(&edge.to)
            ) {
                let color = edge.color.as_deref().unwrap_or(&theme.text_color);
                let thickness = edge.thickness.unwrap_or(1.0);

                if edge.directed {
                    svg.push_str(&format!(
                        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" marker-end="url(#arrowhead)"/>
"#,
                        x1, y1, x2, y2, color, thickness
                    ));
                } else {
                    svg.push_str(&format!(
                        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"/>
"#,
                        x1, y1, x2, y2, color, thickness
                    ));
                }
            }
        }

        // Add arrowhead marker for directed edges
        svg.push_str(r#"  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
"#);

        // Draw nodes
        for node in &graph.nodes {
            if let Some(&(x, y)) = node_positions.get(&node.id) {
                let color = node.color.as_deref().unwrap_or(&theme.accent_color);
                let radius = node.size * 15.0;

                svg.push_str(&format!(
                    r#"  <circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="2"/>
  <text x="{}" y="{}" class="label">{}</text>
"#,
                    x, y, radius, color, theme.text_color,
                    x, y + 4, Self::escape_xml(&node.label)
                ));
            }
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Calculate simple circular layout for static visualization
    fn calculate_simple_layout(&self, graph: &TraitGraph, width: usize, height: usize) -> HashMap<String, (f64, f64)> {
        let mut positions = HashMap::new();
        let n = graph.nodes.len();

        if n == 0 {
            return positions;
        }

        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;
        let radius = (width.min(height) as f64 * 0.3).min(200.0);

        // Use existing positions if available, otherwise create circular layout
        for (i, node) in graph.nodes.iter().enumerate() {
            let (x, y) = if let Some((px, py)) = node.position_2d {
                // Scale existing positions to fit in the SVG
                let scaled_x = (px + 200.0) / 400.0 * width as f64 * 0.8 + width as f64 * 0.1;
                let scaled_y = (py + 200.0) / 400.0 * height as f64 * 0.8 + height as f64 * 0.1;
                (scaled_x, scaled_y)
            } else {
                // Circular layout
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (
                    center_x + radius * angle.cos(),
                    center_y + radius * angle.sin()
                )
            };
            positions.insert(node.id.clone(), (x, y));
        }

        positions
    }

    /// Generate CSS styles for themes
    fn generate_css_styles(&self, theme: &ThemeTemplate) -> String {
        format!(r#"
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: {};
            color: {};
        }}

        #header {{
            background-color: {};
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid {};
        }}

        #header h1 {{
            margin: 0;
            color: {};
            font-size: 28px;
        }}

        #controls {{
            margin-top: 15px;
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }}

        #controls button, #controls input, #controls select {{
            padding: 8px 12px;
            border: 1px solid {};
            border-radius: 4px;
            background-color: {};
            color: {};
            font-size: 14px;
        }}

        #controls button:hover {{
            background-color: {};
            cursor: pointer;
        }}

        #graph-container {{
            position: relative;
            width: 100%;
            height: calc(100vh - 200px);
            overflow: hidden;
        }}

        #graph {{
            width: 100%;
            height: 100%;
        }}

        #info-panel {{
            position: fixed;
            top: 150px;
            right: 20px;
            width: 250px;
            background-color: {};
            border: 1px solid {};
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}

        #node-info {{
            margin-bottom: 20px;
            min-height: 100px;
        }}

        #statistics h3 {{
            margin-top: 0;
            color: {};
        }}

        .node {{
            cursor: pointer;
            stroke: {};
            stroke-width: 2px;
        }}

        .node:hover {{
            stroke-width: 3px;
            filter: brightness(1.2);
        }}

        .edge {{
            fill: none;
            stroke: {};
            stroke-width: 1px;
        }}

        .edge:hover {{
            stroke-width: 2px;
        }}

        .label {{
            font-size: 12px;
            text-anchor: middle;
            pointer-events: none;
            fill: {};
        }}

        .tooltip {{
            position: absolute;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
        }}
        "#,
            theme.background_color, theme.text_color,
            theme.background_color, theme.accent_color,
            theme.text_color,
            theme.accent_color, theme.background_color, theme.text_color,
            theme.accent_color,
            theme.background_color, theme.accent_color,
            theme.accent_color, theme.text_color,
            theme.text_color, theme.text_color
        )
    }

    /// Generate interactive JavaScript code
    fn generate_interactive_javascript(&self, graph: &TraitGraph) -> String {
        format!(r#"
        // Graph visualization with D3.js
        const width = document.getElementById('graph-container').clientWidth;
        const height = document.getElementById('graph-container').clientHeight;

        const svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height);

        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                container.attr("transform", event.transform);
            }});

        svg.call(zoom);

        const container = svg.append("g");

        // Create tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);

        // Initialize force simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size * 20 + 5));

        // Draw edges
        const link = container.append("g")
            .selectAll("line")
            .data(graphData.edges)
            .enter().append("line")
            .attr("class", "edge")
            .attr("stroke", d => d.color || "#999")
            .attr("stroke-width", d => Math.sqrt(d.weight) * 2);

        // Draw nodes
        const node = container.append("g")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => d.size * 15)
            .attr("fill", d => d.color || "#69b3a2")
            .call(d3.drag()
                .on("start", dragStarted)
                .on("drag", dragged)
                .on("end", dragEnded))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip)
            .on("click", showNodeInfo);

        // Add labels
        const label = container.append("g")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("class", "label")
            .text(d => d.label)
            .attr("dy", -3);

        // Update positions on simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});

        // Drag functions
        function dragStarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}

        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}

        function dragEnded(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}

        // Tooltip functions
        function showTooltip(event, d) {{
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html(`
                <strong>${{d.label}}</strong><br/>
                Type: ${{d.type}}<br/>
                Size: ${{d.size}}
            `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        }}

        function hideTooltip() {{
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        }}

        // Node info panel
        function showNodeInfo(event, d) {{
            const infoPanel = document.getElementById('node-info');
            infoPanel.innerHTML = `
                <h4>${{d.label}}</h4>
                <p><strong>Type:</strong> ${{d.type}}</p>
                <p><strong>Size:</strong> ${{d.size}}</p>
                <p><strong>ID:</strong> ${{d.id}}</p>
            `;
        }}

        // Control handlers
        document.getElementById('zoomIn').onclick = () => {{
            svg.transition().call(zoom.scaleBy, 1.5);
        }};

        document.getElementById('zoomOut').onclick = () => {{
            svg.transition().call(zoom.scaleBy, 0.75);
        }};

        document.getElementById('resetView').onclick = () => {{
            svg.transition().call(zoom.transform, d3.zoomIdentity);
        }};

        // Search functionality
        document.getElementById('search').oninput = function(event) {{
            const searchTerm = event.target.value.toLowerCase();
            node.style("opacity", d => {{
                if (searchTerm === '') return 1;
                return d.label.toLowerCase().includes(searchTerm) ? 1 : 0.3;
            }});
            label.style("opacity", d => {{
                if (searchTerm === '') return 1;
                return d.label.toLowerCase().includes(searchTerm) ? 1 : 0.3;
            }});
        }};

        // Layout selector
        document.getElementById('layoutSelect').onchange = function(event) {{
            const layout = event.target.value;
            switch(layout) {{
                case 'hierarchical':
                    applyHierarchicalLayout();
                    break;
                case 'circular':
                    applyCircularLayout();
                    break;
                case 'radial':
                    applyRadialLayout();
                    break;
                default:
                    restartForceSimulation();
            }}
        }};

        function applyHierarchicalLayout() {{
            simulation.stop();
            graphData.nodes.forEach((d, i) => {{
                d.fx = (i % 5) * 150 + 100;
                d.fy = Math.floor(i / 5) * 100 + 100;
            }});
            simulation.alpha(1).restart();
        }}

        function applyCircularLayout() {{
            simulation.stop();
            const radius = Math.min(width, height) / 3;
            graphData.nodes.forEach((d, i) => {{
                const angle = (i / graphData.nodes.length) * 2 * Math.PI;
                d.fx = width / 2 + radius * Math.cos(angle);
                d.fy = height / 2 + radius * Math.sin(angle);
            }});
            simulation.alpha(1).restart();
        }}

        function applyRadialLayout() {{
            simulation.stop();
            // Place most connected node at center
            const centerNode = graphData.nodes.reduce((max, node) =>
                (graphData.edges.filter(e => e.source.id === node.id || e.target.id === node.id).length >
                 graphData.edges.filter(e => e.source.id === max.id || e.target.id === max.id).length) ? node : max
            );
            centerNode.fx = width / 2;
            centerNode.fy = height / 2;

            // Place others in concentric circles
            let radius = 100;
            graphData.nodes.filter(n => n !== centerNode).forEach((d, i) => {{
                const angle = (i / (graphData.nodes.length - 1)) * 2 * Math.PI;
                d.fx = width / 2 + radius * Math.cos(angle);
                d.fy = height / 2 + radius * Math.sin(angle);
            }});
            simulation.alpha(1).restart();
        }}

        function restartForceSimulation() {{
            graphData.nodes.forEach(d => {{
                d.fx = null;
                d.fy = null;
            }});
            simulation.alpha(1).restart();
        }}
        "#)
    }

    /// Create theme template
    fn create_theme_template(theme: VisualizationTheme) -> ThemeTemplate {
        let mut node_colors = HashMap::new();
        let mut edge_colors = HashMap::new();

        // Set default colors based on theme
        match theme {
            VisualizationTheme::Dark => {
                node_colors.insert("trait".to_string(), "#66b3ff".to_string());
                node_colors.insert("implementation".to_string(), "#99ff99".to_string());
                edge_colors.insert("inherits".to_string(), "#ffcc99".to_string());
            },
            _ => {
                node_colors.insert("trait".to_string(), "#007bff".to_string());
                node_colors.insert("implementation".to_string(), "#28a745".to_string());
                edge_colors.insert("inherits".to_string(), "#fd7e14".to_string());
            }
        }

        ThemeTemplate {
            background_color: theme.background_color().to_string(),
            text_color: theme.text_color().to_string(),
            accent_color: theme.accent_color().to_string(),
            node_colors,
            edge_colors,
            css_styles: String::new(), // Generated dynamically
        }
    }

    /// Escape string for DOT format
    fn escape_dot_string(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
    }

    /// Escape string for CSV format
    fn escape_csv_field(s: &str) -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    }

    /// Escape string for XML format
    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }

    /// Set interactive features
    pub fn set_interactive_features(&mut self, features: InteractiveFeatures) {
        self.interactive_features = features;
    }

    /// Get supported export formats
    pub fn get_supported_formats() -> Vec<GraphExportFormat> {
        vec![
            GraphExportFormat::Svg,
            GraphExportFormat::Dot,
            GraphExportFormat::Json,
            GraphExportFormat::InteractiveHtml,
            GraphExportFormat::StaticHtml,
            GraphExportFormat::Csv,
            GraphExportFormat::GraphMl,
            GraphExportFormat::Gexf,
        ]
    }

    /// Validate export configuration
    pub fn validate_export_config(&self, format: GraphExportFormat) -> Result<()> {
        match format {
            GraphExportFormat::Png | GraphExportFormat::Pdf => {
                Err(SklearsError::ValidationError(
                    "PNG and PDF export require external rendering libraries".to_string()
                ))
            },
            _ => Ok(())
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph_config::GraphConfig;
    use super::super::graph_structures::{TraitGraph, TraitGraphNode, TraitGraphEdge};

    fn create_test_graph() -> TraitGraph {
        let mut graph = TraitGraph::new();

        let node1 = TraitGraphNode::new_trait("Trait1".to_string(), "Trait1".to_string())
            .with_position_2d(0.0, 0.0);
        let node2 = TraitGraphNode::new_implementation("Impl1".to_string(), "Impl1".to_string(), "Trait1".to_string())
            .with_position_2d(100.0, 0.0);

        graph.add_node(node1);
        graph.add_node(node2);

        let edge = TraitGraphEdge::new_implementation("Trait1".to_string(), "Impl1".to_string());
        graph.add_edge(edge);

        graph
    }

    #[test]
    fn test_graph_exporter_creation() {
        let config = GraphConfig::default();
        let exporter = GraphExporter::new(config);
        assert!(!exporter.theme_templates.is_empty());
    }

    #[test]
    fn test_json_export() {
        let config = GraphConfig::default();
        let exporter = GraphExporter::new(config);
        let graph = create_test_graph();

        let json_result = exporter.to_json(&graph);
        assert!(json_result.is_ok());

        let json_str = json_result.expect("expected valid value");
        assert!(json_str.contains("Trait1"));
        assert!(json_str.contains("Impl1"));
    }

    #[test]
    fn test_dot_export() {
        let config = GraphConfig::default();
        let exporter = GraphExporter::new(config);
        let graph = create_test_graph();

        let dot_result = exporter.to_dot(&graph);
        assert!(dot_result.is_ok());

        let dot_str = dot_result.expect("expected valid value");
        assert!(dot_str.contains("digraph TraitGraph"));
        assert!(dot_str.contains("Trait1"));
        assert!(dot_str.contains("Impl1"));
    }

    #[test]
    fn test_csv_export() {
        let config = GraphConfig::default();
        let exporter = GraphExporter::new(config);
        let graph = create_test_graph();

        let csv_result = exporter.to_csv(&graph);
        assert!(csv_result.is_ok());

        let csv_str = csv_result.expect("expected valid value");
        assert!(csv_str.contains("Source,Target,Type,Weight,Label,Directed"));
        assert!(csv_str.contains("Trait1,Impl1"));
    }

    #[test]
    fn test_svg_export() {
        let config = GraphConfig::default();
        let exporter = GraphExporter::new(config);
        let graph = create_test_graph();

        let svg_result = exporter.to_svg(&graph);
        assert!(svg_result.is_ok());

        let svg_str = svg_result.expect("expected valid value");
        assert!(svg_str.contains("<svg"));
        assert!(svg_str.contains("</svg>"));
    }

    #[test]
    fn test_html_export() {
        let config = GraphConfig::default();
        let exporter = GraphExporter::new(config);
        let graph = create_test_graph();

        let html_result = exporter.to_interactive_html(&graph);
        assert!(html_result.is_ok());

        let html_str = html_result.expect("expected valid value");
        assert!(html_str.contains("<!DOCTYPE html>"));
        assert!(html_str.contains("d3.js"));
    }

    #[test]
    fn test_graphml_export() {
        let config = GraphConfig::default();
        let exporter = GraphExporter::new(config);
        let graph = create_test_graph();

        let graphml_result = exporter.to_graphml(&graph);
        assert!(graphml_result.is_ok());

        let graphml_str = graphml_result.expect("expected valid value");
        assert!(graphml_str.contains("<?xml version"));
        assert!(graphml_str.contains("<graphml"));
    }

    #[test]
    fn test_escape_functions() {
        assert_eq!(GraphExporter::escape_dot_string("test\"quote"), "test\\\"quote");
        assert_eq!(GraphExporter::escape_csv_field("test,comma"), "\"test,comma\"");
        assert_eq!(GraphExporter::escape_xml("test<tag>"), "test&lt;tag&gt;");
    }

    #[test]
    fn test_supported_formats() {
        let formats = GraphExporter::get_supported_formats();
        assert!(formats.contains(&GraphExportFormat::Svg));
        assert!(formats.contains(&GraphExportFormat::Json));
        assert!(formats.contains(&GraphExportFormat::InteractiveHtml));
    }

    #[test]
    fn test_validation() {
        let config = GraphConfig::default();
        let exporter = GraphExporter::new(config);

        assert!(exporter.validate_export_config(GraphExportFormat::Svg).is_ok());
        assert!(exporter.validate_export_config(GraphExportFormat::Json).is_ok());
        assert!(exporter.validate_export_config(GraphExportFormat::Png).is_err());
        assert!(exporter.validate_export_config(GraphExportFormat::Pdf).is_err());
    }

    #[test]
    fn test_theme_templates() {
        let light_theme = GraphExporter::create_theme_template(VisualizationTheme::Light);
        let dark_theme = GraphExporter::create_theme_template(VisualizationTheme::Dark);

        assert_eq!(light_theme.background_color, "#ffffff");
        assert_eq!(dark_theme.background_color, "#1a1a1a");

        assert_ne!(light_theme.text_color, dark_theme.text_color);
    }
}