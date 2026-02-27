use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_types::*;


/// Visualization export engine for
/// generating output in various formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationExportEngine {
    /// Supported export formats
    export_formats: Vec<ExportFormat>,
    /// Quality presets for different use cases
    quality_presets: HashMap<String, ExportQualityPreset>,
    /// Batch processing configuration
    batch_processing: BatchProcessingConfig,
}

/// Export format specification for
/// output format capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFormat {
    format_name: String,
    file_extension: String,
    mime_type: String,
    supports_vector: bool,
    supports_animation: bool,
    compression_options: Vec<CompressionOption>,
}

/// Compression option configuration for
/// file size optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionOption {
    /// Compression algorithm name
    algorithm: String,
    /// Quality range (min, max)
    quality_range: (f64, f64),
    /// File size impact factor
    file_size_impact: f64,
}

/// Export quality preset for standardized
/// output quality configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportQualityPreset {
    /// Preset name identifier
    preset_name: String,
    /// Output resolution
    resolution: (u32, u32),
    /// Quality level (0.0 to 1.0)
    quality_level: f64,
    /// Color depth in bits
    color_depth: u8,
    /// Compression level
    compression_level: u8,
}

/// Batch processing configuration for
/// efficient bulk export operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingConfig {
    /// Maximum concurrent export operations
    max_concurrent_exports: usize,
    /// Memory limit per export operation
    memory_limit_per_export: usize,
    /// Timeout per export operation
    timeout_per_export: Duration,
    /// Whether to retry failed exports
    retry_failed_exports: bool,
}

/// Generated visualization structure containing
/// visualization data and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedVisualization {
    /// Unique visualization identifier
    pub visualization_id: String,
    /// Creation timestamp
    pub creation_timestamp: DateTime<Utc>,
    /// Chart type used
    pub chart_type: ChartType,
    /// Binary content of the visualization
    pub content: Vec<u8>,
    /// Interactive elements in the visualization
    pub interactive_elements: Vec<InteractiveElement>,
    /// Visualization metadata
    pub metadata: HashMap<String, String>,
}

/// Interactive element definition for
/// user interaction components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveElement {
    /// Element identifier
    pub element_id: String,
    /// Type of interactive element
    pub element_type: InteractiveElementType,
    /// Element position
    pub position: ElementPosition,
    /// Event handlers for the element
    pub event_handlers: Vec<ElementEventHandler>,
}

/// Interactive element type enumeration for
/// different interaction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractiveElementType {
    /// Button element
    Button,
    /// Tooltip element
    Tooltip,
    /// Legend element
    Legend,
    /// Filter element
    Filter,
    /// Zoom control element
    Zoom,
    /// Pan control element
    Pan,
    /// Custom element implementation
    Custom(String),
}

/// Element position for precise
/// interactive element placement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementPosition {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Element width
    pub width: f64,
    /// Element height
    pub height: f64,
}

/// Element event handler for
/// interactive element behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementEventHandler {
    /// Event type identifier
    pub event_type: String,
    /// Handler code or function
    pub handler_code: String,
    /// Handler parameters
    pub parameters: HashMap<String, String>,
}

/// Visualization data structure for
/// chart data input and processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    /// Data points for visualization
    pub data_points: Vec<DataPoint>,
    /// Data metadata
    pub metadata: HashMap<String, String>,
    /// Data schema definition
    pub schema: DataSchema,
}

/// Data point structure for individual
/// data entries in visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Optional timestamp for temporal data
    pub timestamp: Option<DateTime<Utc>>,
    /// Named values for the data point
    pub values: HashMap<String, DataValue>,
    /// Optional category classification
    pub category: Option<String>,
}

/// Data value enumeration for
/// different data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataValue {
    /// Numeric value
    Number(f64),
    /// String value
    String(String),
    /// Boolean value
    Boolean(bool),
    /// Date/time value
    Date(DateTime<Utc>),
    /// Null value
    Null,
}

/// Data schema definition for
/// data structure specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    /// Field definitions
    pub fields: Vec<FieldDefinition>,
    /// Primary key field
    pub primary_key: Option<String>,
    /// Data relationships
    pub relationships: Vec<Relationship>,
}

/// Field definition for data
/// schema field specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field name
    pub field_name: String,
    /// Field data type
    pub field_type: FieldType,
    /// Whether field is required
    pub required: bool,
    /// Field description
    pub description: String,
}

/// Field type enumeration for
/// data schema field types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    /// Numeric field type
    Numeric,
    /// Text field type
    Text,
    /// Date field type
    Date,
    /// Boolean field type
    Boolean,
    /// Category field type
    Category,
    /// Custom field type
    Custom(String),
}

/// Relationship definition for
/// data schema relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Source field name
    pub source_field: String,
    /// Target field name
    pub target_field: String,
    /// Relationship cardinality
    pub cardinality: Cardinality,
}

/// Relationship type enumeration for
/// different data relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// One-to-one relationship
    OneToOne,
    /// One-to-many relationship
    OneToMany,
    /// Many-to-many relationship
    ManyToMany,
    /// Hierarchical relationship
    Hierarchical,
    /// Custom relationship type
    Custom(String),
}

/// Cardinality enumeration for
/// relationship constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Cardinality {
    /// Required relationship
    Required,
    /// Optional relationship
    Optional,
    /// Multiple occurrence relationship
    Multiple,
}

