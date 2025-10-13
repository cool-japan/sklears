//! Core chart types and basic configuration structures
//!
//! This module provides fundamental chart type definitions and basic configuration
//! structures for the chart rendering system. It includes:
//! - Chart type enumeration with comprehensive chart varieties
//! - Rendering engine types and capabilities
//! - Basic data structures and formats
//! - Core configuration types used throughout the system

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

/// Comprehensive chart type enumeration
///
/// Defines all supported chart types with comprehensive visualization capabilities
/// for different data types and analytical requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart with multiple series support
    Line,
    /// Bar chart (vertical bars)
    Bar,
    /// Horizontal bar chart
    HorizontalBar,
    /// Pie chart with customizable segments
    Pie,
    /// Donut chart (pie with center hole)
    Donut,
    /// Scatter plot with correlation analysis
    Scatter,
    /// Area chart with stacking support
    Area,
    /// Histogram with bin customization
    Histogram,
    /// Box plot for statistical distribution
    BoxPlot,
    /// Heat map for matrix visualization
    Heatmap,
    /// Tree map for hierarchical data
    Treemap,
    /// Sankey diagram for flow visualization
    Sankey,
    /// Gantt chart for project management
    Gantt,
    /// Radar chart for multi-dimensional data
    Radar,
    /// Bubble chart with size dimension
    Bubble,
    /// Candlestick chart for financial data
    Candlestick,
    /// Violin plot for distribution visualization
    Violin,
    /// Waterfall chart for cumulative analysis
    Waterfall,
    /// Sunburst chart for hierarchical proportion
    Sunburst,
    /// Parallel coordinates for multivariate data
    ParallelCoordinates,
    /// Network graph for relationship visualization
    Network,
    /// Custom chart type with configuration
    Custom(String),
}

/// Rendering engine types with different capabilities
///
/// Defines the available rendering engines that can be used to generate
/// charts with different performance and feature characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RenderingEngine {
    /// SVG-based rendering for scalability
    SVG,
    /// HTML5 Canvas rendering for performance
    Canvas,
    /// WebGL rendering for high-performance graphics
    WebGL,
    /// D3.js integration for interactive visualizations
    D3,
    /// Chart.js integration for standard charts
    Chart,
    /// Plotly integration for scientific plotting
    Plotly,
    /// Three.js integration for 3D visualizations
    ThreeJS,
    /// Custom rendering engine with configuration
    Custom(String),
}

/// Data format types supported by the chart rendering system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    /// JavaScript Object Notation
    JSON,
    /// Comma-separated values
    CSV,
    /// Tab-separated values
    TSV,
    /// XML format
    XML,
    /// Binary format
    Binary,
    /// Custom data format
    Custom(String),
}

/// Chart data structure containing series and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    /// Data series collection
    pub series: Vec<DataSeries>,
    /// Metadata about the dataset
    pub metadata: DataMetadata,
    /// Data quality metrics
    pub quality_metrics: DataQualityMetrics,
    /// Data lineage information
    pub lineage: DataLineage,
}

/// Individual data series with styling and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSeries {
    /// Series identifier
    pub id: String,
    /// Series display name
    pub name: String,
    /// Data points in the series
    pub data: Vec<DataPoint>,
    /// Visual styling for the series
    pub style: SeriesStyle,
    /// Series-specific metadata
    pub metadata: SeriesMetadata,
}

/// Individual data point with coordinates and optional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// X-coordinate value
    pub x: f64,
    /// Y-coordinate value
    pub y: f64,
    /// Optional Z-coordinate for 3D charts
    pub z: Option<f64>,
    /// Optional label for the data point
    pub label: Option<String>,
    /// Optional metadata for the data point
    pub metadata: Option<HashMap<String, String>>,
}

/// Visual styling configuration for data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStyle {
    /// Primary color for the series
    pub color: String,
    /// Fill color (for area charts, bars, etc.)
    pub fill_color: Option<String>,
    /// Line width (for line charts)
    pub line_width: Option<f64>,
    /// Marker style for data points
    pub marker_style: MarkerStyle,
    /// Fill style for enclosed areas
    pub fill_style: FillStyle,
    /// Opacity level (0.0 to 1.0)
    pub opacity: f64,
}

/// Marker style enumeration for data points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkerStyle {
    /// Circular markers
    Circle,
    /// Square markers
    Square,
    /// Diamond markers
    Diamond,
    /// Triangle markers
    Triangle,
    /// Cross markers
    Cross,
    /// Plus markers
    Plus,
    /// No markers
    None,
    /// Custom marker with path definition
    Custom(String),
}

/// Fill style enumeration for areas and shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillStyle {
    /// Solid fill
    Solid,
    /// Striped pattern
    Striped,
    /// Dotted pattern
    Dotted,
    /// Gradient fill
    Gradient,
    /// No fill (transparent)
    None,
    /// Custom fill pattern
    Custom(String),
}

/// Series-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesMetadata {
    /// Data source information
    pub source: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Series description
    pub description: Option<String>,
    /// Units of measurement
    pub units: Option<String>,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMetadata {
    /// Dataset title
    pub title: String,
    /// Dataset description
    pub description: String,
    /// Data source information
    pub source: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Data license information
    pub license: Option<String>,
}

/// Data quality metrics for validation and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    /// Completeness ratio (0.0 to 1.0)
    pub completeness: f64,
    /// Accuracy score (0.0 to 1.0)
    pub accuracy: f64,
    /// Consistency score (0.0 to 1.0)
    pub consistency: f64,
    /// Validity score (0.0 to 1.0)
    pub validity: f64,
}

/// Data lineage tracking for audit and provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLineage {
    /// Transformation steps applied to the data
    pub transformations: Vec<TransformationStep>,
    /// Data dependencies
    pub dependencies: Vec<DataDependency>,
}

/// Individual transformation step in data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationStep {
    /// Step identifier
    pub id: String,
    /// Transformation name
    pub name: String,
    /// Step description
    pub description: String,
    /// Timestamp when transformation was applied
    pub timestamp: DateTime<Utc>,
    /// Parameters used in the transformation
    pub parameters: HashMap<String, String>,
}

/// Data dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDependency {
    /// Dependency identifier
    pub id: String,
    /// Source data identifier
    pub source_id: String,
    /// Dependency type
    pub dependency_type: String,
    /// Version or timestamp of the dependency
    pub version: String,
}

/// Interactive features available for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractiveFeature {
    /// Enable zooming functionality
    Zoom,
    /// Enable panning functionality
    Pan,
    /// Enable hover tooltips
    Hover,
    /// Enable click interactions
    Click,
    /// Enable brush selection
    Brush,
    /// Enable crossfilter interactions
    Crossfilter,
    /// Custom interactive feature
    Custom(String),
}

/// Export format options for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Portable Network Graphics
    PNG,
    /// JPEG image format
    JPEG,
    /// Scalable Vector Graphics
    SVG,
    /// Portable Document Format
    PDF,
    /// PowerPoint presentation
    PPTX,
    /// HTML with embedded chart
    HTML,
    /// Custom export format
    Custom(String),
}

/// Color format specification for chart elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorFormat {
    /// RGB color specification
    RGB(u8, u8, u8),
    /// RGBA color with alpha channel
    RGBA(u8, u8, u8, f64),
    /// Hexadecimal color code
    Hex(String),
    /// HSL color specification
    HSL(f64, f64, f64),
    /// Named color
    Named(String),
}

impl Default for ChartData {
    fn default() -> Self {
        Self {
            series: Vec::new(),
            metadata: DataMetadata::default(),
            quality_metrics: DataQualityMetrics::default(),
            lineage: DataLineage::default(),
        }
    }
}

impl Default for DataMetadata {
    fn default() -> Self {
        Self {
            title: "Untitled Dataset".to_string(),
            description: "No description provided".to_string(),
            source: "Unknown".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            license: None,
        }
    }
}

impl Default for DataQualityMetrics {
    fn default() -> Self {
        Self {
            completeness: 1.0,
            accuracy: 1.0,
            consistency: 1.0,
            validity: 1.0,
        }
    }
}

impl Default for DataLineage {
    fn default() -> Self {
        Self {
            transformations: Vec::new(),
            dependencies: Vec::new(),
        }
    }
}

impl Default for SeriesStyle {
    fn default() -> Self {
        Self {
            color: "#1f77b4".to_string(), // Default blue color
            fill_color: None,
            line_width: Some(2.0),
            marker_style: MarkerStyle::Circle,
            fill_style: FillStyle::None,
            opacity: 1.0,
        }
    }
}

impl Default for SeriesMetadata {
    fn default() -> Self {
        Self {
            source: "Unknown".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            description: None,
            units: None,
        }
    }
}