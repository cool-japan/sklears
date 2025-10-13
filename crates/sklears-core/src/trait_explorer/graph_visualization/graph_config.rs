//! Configuration and type definitions for graph visualization
//!
//! This module provides comprehensive configuration options and type definitions
//! for the graph visualization system, supporting various layout algorithms,
//! themes, export formats, and optimization strategies.

use serde::{Deserialize, Serialize};

/// Configuration for graph generation and visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Layout algorithm to use for graph positioning
    pub layout_algorithm: LayoutAlgorithm,
    /// Whether to enable 3D visualization
    pub enable_3d: bool,
    /// Whether to enable advanced graph analysis
    pub enable_analysis: bool,
    /// Maximum number of nodes to display
    pub max_nodes: usize,
    /// Maximum depth for trait exploration
    pub max_depth: usize,
    /// Whether to enable GPU acceleration
    pub enable_gpu: bool,
    /// Whether to enable SIMD acceleration
    pub enable_simd: bool,
    /// Visualization theme
    pub theme: VisualizationTheme,
    /// Export format preference
    pub export_format: GraphExportFormat,
    /// Filter configuration
    pub filter_config: FilterConfig,
    /// Performance optimization level
    pub optimization_level: OptimizationLevel,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            layout_algorithm: LayoutAlgorithm::ForceDirected,
            enable_3d: false,
            enable_analysis: true,
            max_nodes: 1000,
            max_depth: 10,
            enable_gpu: true,
            enable_simd: true,
            theme: VisualizationTheme::Light,
            export_format: GraphExportFormat::InteractiveHtml,
            filter_config: FilterConfig::default(),
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

impl GraphConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the layout algorithm
    pub fn with_layout_algorithm(mut self, algorithm: LayoutAlgorithm) -> Self {
        self.layout_algorithm = algorithm;
        self
    }

    /// Enable or disable 3D visualization
    pub fn with_3d_visualization(mut self, enable: bool) -> Self {
        self.enable_3d = enable;
        self
    }

    /// Enable or disable advanced analysis
    pub fn with_advanced_analysis(mut self, enable: bool) -> Self {
        self.enable_analysis = enable;
        self
    }

    /// Set maximum number of nodes
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Set visualization theme
    pub fn with_theme(mut self, theme: VisualizationTheme) -> Self {
        self.theme = theme;
        self
    }

    /// Enable GPU acceleration
    pub fn with_gpu_acceleration(mut self, enable: bool) -> Self {
        self.enable_gpu = enable;
        self
    }

    /// Enable SIMD acceleration
    pub fn with_simd_acceleration(mut self, enable: bool) -> Self {
        self.enable_simd = enable;
        self
    }

    /// Set export format
    pub fn with_export_format(mut self, format: GraphExportFormat) -> Self {
        self.export_format = format;
        self
    }

    /// Set filter configuration
    pub fn with_filter_config(mut self, filter_config: FilterConfig) -> Self {
        self.filter_config = filter_config;
        self
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_nodes == 0 {
            return Err("max_nodes must be greater than 0".to_string());
        }

        if self.max_depth == 0 {
            return Err("max_depth must be greater than 0".to_string());
        }

        if self.max_nodes > 10000 {
            return Err("max_nodes should not exceed 10000 for performance reasons".to_string());
        }

        if self.max_depth > 50 {
            return Err("max_depth should not exceed 50 to avoid infinite recursion".to_string());
        }

        Ok(())
    }

    /// Get recommended settings for large graphs
    pub fn for_large_graphs() -> Self {
        Self {
            layout_algorithm: LayoutAlgorithm::ForceDirected,
            enable_3d: false,
            enable_analysis: false,
            max_nodes: 5000,
            max_depth: 5,
            enable_gpu: true,
            enable_simd: true,
            theme: VisualizationTheme::Dark,
            export_format: GraphExportFormat::InteractiveHtml,
            filter_config: FilterConfig::performance_optimized(),
            optimization_level: OptimizationLevel::Performance,
        }
    }

    /// Get recommended settings for detailed analysis
    pub fn for_detailed_analysis() -> Self {
        Self {
            layout_algorithm: LayoutAlgorithm::Hierarchical,
            enable_3d: true,
            enable_analysis: true,
            max_nodes: 500,
            max_depth: 15,
            enable_gpu: true,
            enable_simd: true,
            theme: VisualizationTheme::Light,
            export_format: GraphExportFormat::InteractiveHtml,
            filter_config: FilterConfig::comprehensive(),
            optimization_level: OptimizationLevel::Quality,
        }
    }

    /// Get recommended settings for presentations
    pub fn for_presentation() -> Self {
        Self {
            layout_algorithm: LayoutAlgorithm::Hierarchical,
            enable_3d: false,
            enable_analysis: false,
            max_nodes: 100,
            max_depth: 8,
            enable_gpu: false,
            enable_simd: false,
            theme: VisualizationTheme::Presentation,
            export_format: GraphExportFormat::Svg,
            filter_config: FilterConfig::simple(),
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

/// Layout algorithms for graph visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    /// Force-directed layout using physics simulation
    ForceDirected,
    /// Hierarchical layout with levels
    Hierarchical,
    /// Circular layout around a center
    Circular,
    /// Grid-based layout
    Grid,
    /// Radial layout from a center node
    Radial,
    /// Spring embedder algorithm
    SpringEmbedder,
    /// Tree layout for hierarchical data
    Tree,
    /// Custom layout with user-defined parameters
    Custom(CustomLayoutParams),
}

/// Parameters for custom layout algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomLayoutParams {
    pub name: String,
    pub parameters: std::collections::HashMap<String, f64>,
}

impl CustomLayoutParams {
    /// Create new custom layout parameters
    pub fn new(name: String) -> Self {
        Self {
            name,
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Add a parameter to the custom layout
    pub fn with_parameter(mut self, key: String, value: f64) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Get a parameter value by key
    pub fn get_parameter(&self, key: &str) -> Option<f64> {
        self.parameters.get(key).copied()
    }
}

/// Visualization themes for graph appearance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationTheme {
    /// Light theme with bright background
    Light,
    /// Dark theme with dark background
    Dark,
    /// High contrast theme for accessibility
    HighContrast,
    /// Colorblind-friendly theme
    ColorblindFriendly,
    /// Professional presentation theme
    Presentation,
    /// Scientific publication theme
    Scientific,
    /// Minimalist theme with reduced visual elements
    Minimalist,
    /// Vibrant theme with bright colors
    Vibrant,
}

impl VisualizationTheme {
    /// Get the background color for this theme
    pub fn background_color(&self) -> &'static str {
        match self {
            VisualizationTheme::Light => "#ffffff",
            VisualizationTheme::Dark => "#1a1a1a",
            VisualizationTheme::HighContrast => "#000000",
            VisualizationTheme::ColorblindFriendly => "#f8f8f8",
            VisualizationTheme::Presentation => "#ffffff",
            VisualizationTheme::Scientific => "#ffffff",
            VisualizationTheme::Minimalist => "#fafafa",
            VisualizationTheme::Vibrant => "#f0f0f0",
        }
    }

    /// Get the primary text color for this theme
    pub fn text_color(&self) -> &'static str {
        match self {
            VisualizationTheme::Light => "#333333",
            VisualizationTheme::Dark => "#ffffff",
            VisualizationTheme::HighContrast => "#ffffff",
            VisualizationTheme::ColorblindFriendly => "#000000",
            VisualizationTheme::Presentation => "#000000",
            VisualizationTheme::Scientific => "#000000",
            VisualizationTheme::Minimalist => "#666666",
            VisualizationTheme::Vibrant => "#222222",
        }
    }

    /// Get the accent color for this theme
    pub fn accent_color(&self) -> &'static str {
        match self {
            VisualizationTheme::Light => "#007bff",
            VisualizationTheme::Dark => "#66b3ff",
            VisualizationTheme::HighContrast => "#ffff00",
            VisualizationTheme::ColorblindFriendly => "#0173b2",
            VisualizationTheme::Presentation => "#0056b3",
            VisualizationTheme::Scientific => "#000080",
            VisualizationTheme::Minimalist => "#888888",
            VisualizationTheme::Vibrant => "#ff6b35",
        }
    }
}

/// Graph export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphExportFormat {
    /// SVG vector format
    Svg,
    /// PNG raster format
    Png,
    /// PDF document format
    Pdf,
    /// DOT graph description language
    Dot,
    /// JSON data format
    Json,
    /// Interactive HTML with JavaScript
    InteractiveHtml,
    /// Static HTML
    StaticHtml,
    /// CSV edge list
    Csv,
    /// GraphML format
    GraphMl,
    /// GEXF format for Gephi
    Gexf,
}

impl GraphExportFormat {
    /// Get the file extension for this format
    pub fn file_extension(&self) -> &'static str {
        match self {
            GraphExportFormat::Svg => "svg",
            GraphExportFormat::Png => "png",
            GraphExportFormat::Pdf => "pdf",
            GraphExportFormat::Dot => "dot",
            GraphExportFormat::Json => "json",
            GraphExportFormat::InteractiveHtml => "html",
            GraphExportFormat::StaticHtml => "html",
            GraphExportFormat::Csv => "csv",
            GraphExportFormat::GraphMl => "graphml",
            GraphExportFormat::Gexf => "gexf",
        }
    }

    /// Check if this format supports interactivity
    pub fn is_interactive(&self) -> bool {
        matches!(self, GraphExportFormat::InteractiveHtml)
    }

    /// Check if this format is vector-based
    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            GraphExportFormat::Svg | GraphExportFormat::Pdf | GraphExportFormat::InteractiveHtml
        )
    }
}

/// Configuration for graph filtering
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Minimum node complexity to include
    pub min_complexity: f64,
    /// Maximum node complexity to include
    pub max_complexity: f64,
    /// Filter by edge types
    pub edge_types: Vec<EdgeType>,
    /// Filter by node types
    pub node_types: Vec<TraitNodeType>,
    /// Filter by stability levels
    pub stability_levels: Vec<StabilityLevel>,
    /// Include deprecated traits
    pub include_deprecated: bool,
    /// Include experimental traits
    pub include_experimental: bool,
    /// Filter by trait names (regex patterns)
    pub trait_name_patterns: Vec<String>,
    /// Filter by implementation names (regex patterns)
    pub implementation_patterns: Vec<String>,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_complexity: 0.0,
            max_complexity: 100.0,
            edge_types: vec![
                EdgeType::Inherits,
                EdgeType::Implements,
                EdgeType::Uses,
                EdgeType::Contains,
            ],
            node_types: vec![
                TraitNodeType::Trait,
                TraitNodeType::Implementation,
                TraitNodeType::AssociatedType,
            ],
            stability_levels: vec![
                StabilityLevel::Stable,
                StabilityLevel::Unstable,
                StabilityLevel::Experimental,
            ],
            include_deprecated: false,
            include_experimental: true,
            trait_name_patterns: Vec::new(),
            implementation_patterns: Vec::new(),
        }
    }
}

impl FilterConfig {
    /// Create a simple filter configuration
    pub fn simple() -> Self {
        Self {
            min_complexity: 0.0,
            max_complexity: 50.0,
            edge_types: vec![EdgeType::Inherits, EdgeType::Implements],
            node_types: vec![TraitNodeType::Trait, TraitNodeType::Implementation],
            stability_levels: vec![StabilityLevel::Stable],
            include_deprecated: false,
            include_experimental: false,
            trait_name_patterns: Vec::new(),
            implementation_patterns: Vec::new(),
        }
    }

    /// Create a comprehensive filter configuration
    pub fn comprehensive() -> Self {
        Self::default()
    }

    /// Create a performance-optimized filter configuration
    pub fn performance_optimized() -> Self {
        Self {
            min_complexity: 0.0,
            max_complexity: 30.0,
            edge_types: vec![EdgeType::Inherits, EdgeType::Implements],
            node_types: vec![TraitNodeType::Trait, TraitNodeType::Implementation],
            stability_levels: vec![StabilityLevel::Stable, StabilityLevel::Unstable],
            include_deprecated: false,
            include_experimental: false,
            trait_name_patterns: Vec::new(),
            implementation_patterns: Vec::new(),
        }
    }

    /// Add a trait name pattern filter
    pub fn with_trait_pattern(mut self, pattern: String) -> Self {
        self.trait_name_patterns.push(pattern);
        self
    }

    /// Add an implementation pattern filter
    pub fn with_implementation_pattern(mut self, pattern: String) -> Self {
        self.implementation_patterns.push(pattern);
        self
    }

    /// Set complexity range
    pub fn with_complexity_range(mut self, min: f64, max: f64) -> Self {
        self.min_complexity = min;
        self.max_complexity = max;
        self
    }

    /// Check if a trait name matches the filter patterns
    pub fn matches_trait_name(&self, name: &str) -> bool {
        if self.trait_name_patterns.is_empty() {
            return true;
        }

        for pattern in &self.trait_name_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if regex.is_match(name) {
                    return true;
                }
            } else {
                // Fallback to simple string matching
                if name.contains(pattern) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if an implementation name matches the filter patterns
    pub fn matches_implementation_name(&self, name: &str) -> bool {
        if self.implementation_patterns.is_empty() {
            return true;
        }

        for pattern in &self.implementation_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if regex.is_match(name) {
                    return true;
                }
            } else {
                // Fallback to simple string matching
                if name.contains(pattern) {
                    return true;
                }
            }
        }

        false
    }
}

/// Performance optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Prioritize quality over performance
    Quality,
    /// Balance quality and performance
    Balanced,
    /// Prioritize performance over quality
    Performance,
    /// Maximum performance with minimal quality
    Fast,
}

impl OptimizationLevel {
    /// Get the iteration count for layout algorithms
    pub fn layout_iterations(&self) -> usize {
        match self {
            OptimizationLevel::Quality => 1000,
            OptimizationLevel::Balanced => 500,
            OptimizationLevel::Performance => 200,
            OptimizationLevel::Fast => 50,
        }
    }

    /// Get the precision for calculations
    pub fn calculation_precision(&self) -> f64 {
        match self {
            OptimizationLevel::Quality => 1e-6,
            OptimizationLevel::Balanced => 1e-4,
            OptimizationLevel::Performance => 1e-3,
            OptimizationLevel::Fast => 1e-2,
        }
    }

    /// Whether to enable expensive analysis features
    pub fn enable_expensive_analysis(&self) -> bool {
        matches!(self, OptimizationLevel::Quality | OptimizationLevel::Balanced)
    }
}

/// Types of nodes in the trait graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraitNodeType {
    /// A trait definition
    Trait,
    /// A struct/enum implementing a trait
    Implementation,
    /// An associated type within a trait
    AssociatedType,
    /// A method within a trait
    Method,
    /// A constant within a trait
    Constant,
    /// A type alias
    TypeAlias,
    /// A generic parameter
    GenericParameter,
    /// A where clause constraint
    WhereClause,
    /// A macro
    Macro,
    /// A module
    Module,
}

impl TraitNodeType {
    pub fn default_color(&self) -> &'static str {
        match self {
            TraitNodeType::Trait => "#4CAF50",
            TraitNodeType::Implementation => "#2196F3",
            TraitNodeType::AssociatedType => "#FF9800",
            TraitNodeType::Method => "#9C27B0",
            TraitNodeType::Constant => "#795548",
            TraitNodeType::TypeAlias => "#607D8B",
            TraitNodeType::GenericParameter => "#E91E63",
            TraitNodeType::WhereClause => "#FFC107",
            TraitNodeType::Macro => "#8BC34A",
            TraitNodeType::Module => "#3F51B5",
        }
    }

    /// Get the default shape for this node type
    pub fn default_shape(&self) -> &'static str {
        match self {
            TraitNodeType::Trait => "ellipse",
            TraitNodeType::Implementation => "box",
            TraitNodeType::AssociatedType => "diamond",
            TraitNodeType::Method => "circle",
            TraitNodeType::Constant => "triangle",
            TraitNodeType::TypeAlias => "hexagon",
            TraitNodeType::GenericParameter => "star",
            TraitNodeType::WhereClause => "pentagon",
            TraitNodeType::Macro => "octagon",
            TraitNodeType::Module => "folder",
        }
    }

    /// Get the display name for this node type
    pub fn display_name(&self) -> &'static str {
        match self {
            TraitNodeType::Trait => "Trait",
            TraitNodeType::Implementation => "Implementation",
            TraitNodeType::AssociatedType => "Associated Type",
            TraitNodeType::Method => "Method",
            TraitNodeType::Constant => "Constant",
            TraitNodeType::TypeAlias => "Type Alias",
            TraitNodeType::GenericParameter => "Generic Parameter",
            TraitNodeType::WhereClause => "Where Clause",
            TraitNodeType::Macro => "Macro",
            TraitNodeType::Module => "Module",
        }
    }
}

/// Types of edges in the trait graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    Inherits,
    Implements,
    Uses,
    Contains,
    AssociatedWith,
    DefinesMethod,
    ConstrainedBy,
    DependsOn,
    ComposedOf,
    Aggregates,
}

impl EdgeType {
    /// Get the default color for this edge type
    pub fn default_color(&self) -> &'static str {
        match self {
            EdgeType::Inherits => "#4CAF50",
            EdgeType::Implements => "#2196F3",
            EdgeType::Uses => "#FF9800",
            EdgeType::Contains => "#9C27B0",
            EdgeType::AssociatedWith => "#795548",
            EdgeType::DefinesMethod => "#607D8B",
            EdgeType::ConstrainedBy => "#E91E63",
            EdgeType::DependsOn => "#FFC107",
            EdgeType::ComposedOf => "#8BC34A",
            EdgeType::Aggregates => "#3F51B5",
        }
    }

    /// Get the default line style for this edge type
    pub fn default_line_style(&self) -> &'static str {
        match self {
            EdgeType::Inherits => "solid",
            EdgeType::Implements => "solid",
            EdgeType::Uses => "dashed",
            EdgeType::Contains => "solid",
            EdgeType::AssociatedWith => "dotted",
            EdgeType::DefinesMethod => "dashed",
            EdgeType::ConstrainedBy => "dotted",
            EdgeType::DependsOn => "dashed",
            EdgeType::ComposedOf => "solid",
            EdgeType::Aggregates => "dashed",
        }
    }

    /// Get the display name for this edge type
    pub fn display_name(&self) -> &'static str {
        match self {
            EdgeType::Inherits => "inherits",
            EdgeType::Implements => "implements",
            EdgeType::Uses => "uses",
            EdgeType::Contains => "contains",
            EdgeType::AssociatedWith => "associated with",
            EdgeType::DefinesMethod => "defines method",
            EdgeType::ConstrainedBy => "constrained by",
            EdgeType::DependsOn => "depends on",
            EdgeType::ComposedOf => "composed of",
            EdgeType::Aggregates => "aggregates",
        }
    }

    /// Check if this edge type represents a hierarchical relationship
    pub fn is_hierarchical(&self) -> bool {
        matches!(
            self,
            EdgeType::Inherits | EdgeType::Contains | EdgeType::ComposedOf | EdgeType::Aggregates
        )
    }

    /// Check if this edge type should be directed
    pub fn is_directed(&self) -> bool {
        !matches!(self, EdgeType::Uses | EdgeType::AssociatedWith)
    }
}

/// Stability levels for traits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StabilityLevel {
    /// Stable API with backwards compatibility guarantees
    Stable,
    /// Unstable API that may change
    Unstable,
    /// Experimental API for testing new features
    Experimental,
    /// Deprecated API scheduled for removal
    Deprecated,
    /// Internal API not intended for public use
    Internal,
}

impl StabilityLevel {
    pub fn display_name(&self) -> &'static str {
        match self {
            StabilityLevel::Stable => "Stable",
            StabilityLevel::Unstable => "Unstable",
            StabilityLevel::Experimental => "Experimental",
            StabilityLevel::Deprecated => "Deprecated",
            StabilityLevel::Internal => "Internal",
        }
    }

    /// Get the badge color for this stability level
    pub fn badge_color(&self) -> &'static str {
        match self {
            StabilityLevel::Stable => "#4CAF50",
            StabilityLevel::Unstable => "#FF9800",
            StabilityLevel::Experimental => "#9C27B0",
            StabilityLevel::Deprecated => "#F44336",
            StabilityLevel::Internal => "#607D8B",
        }
    }

    /// Check if this stability level should be included by default
    pub fn include_by_default(&self) -> bool {
        !matches!(self, StabilityLevel::Deprecated | StabilityLevel::Internal)
    }
}

/// Community detection algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunityDetection {
    /// Louvain method for modularity optimization
    Louvain,
    /// Leiden algorithm for high-quality community detection
    Leiden,
    /// Label propagation algorithm
    LabelPropagation,
    /// Walktrap algorithm using random walks
    Walktrap,
    /// Girvan-Newman algorithm using edge betweenness
    GirvanNewman,
    /// Fast greedy algorithm
    FastGreedy,
}

impl CommunityDetection {
    /// Get the display name for this algorithm
    pub fn display_name(&self) -> &'static str {
        match self {
            CommunityDetection::Louvain => "Louvain",
            CommunityDetection::Leiden => "Leiden",
            CommunityDetection::LabelPropagation => "Label Propagation",
            CommunityDetection::Walktrap => "Walktrap",
            CommunityDetection::GirvanNewman => "Girvan-Newman",
            CommunityDetection::FastGreedy => "Fast Greedy",
        }
    }

    /// Get the time complexity description
    pub fn time_complexity(&self) -> &'static str {
        match self {
            CommunityDetection::Louvain => "O(n log n)",
            CommunityDetection::Leiden => "O(n log n)",
            CommunityDetection::LabelPropagation => "O(n + m)",
            CommunityDetection::Walktrap => "O(n^2 log n)",
            CommunityDetection::GirvanNewman => "O(m^2 n)",
            CommunityDetection::FastGreedy => "O(m log n)",
        }
    }

    /// Check if this algorithm is suitable for large graphs
    pub fn suitable_for_large_graphs(&self) -> bool {
        matches!(
            self,
            CommunityDetection::Louvain
                | CommunityDetection::Leiden
                | CommunityDetection::LabelPropagation
                | CommunityDetection::FastGreedy
        )
    }
}

/// Centrality measures for graph analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CentralityMeasure {
    /// Degree centrality (number of connections)
    Degree,
    /// Betweenness centrality (importance in paths)
    Betweenness,
    /// Closeness centrality (average distance to others)
    Closeness,
    /// Eigenvector centrality (importance based on neighbors)
    Eigenvector,
    /// PageRank algorithm
    PageRank,
    /// Katz centrality
    Katz,
    /// HITS authority score
    Authority,
    /// HITS hub score
    Hub,
}

impl CentralityMeasure {
    /// Get the display name for this centrality measure
    pub fn display_name(&self) -> &'static str {
        match self {
            CentralityMeasure::Degree => "Degree",
            CentralityMeasure::Betweenness => "Betweenness",
            CentralityMeasure::Closeness => "Closeness",
            CentralityMeasure::Eigenvector => "Eigenvector",
            CentralityMeasure::PageRank => "PageRank",
            CentralityMeasure::Katz => "Katz",
            CentralityMeasure::Authority => "Authority",
            CentralityMeasure::Hub => "Hub",
        }
    }

    /// Get a description of what this measure indicates
    pub fn description(&self) -> &'static str {
        match self {
            CentralityMeasure::Degree => "Number of direct connections",
            CentralityMeasure::Betweenness => "Importance as a bridge between other nodes",
            CentralityMeasure::Closeness => "How quickly information spreads from this node",
            CentralityMeasure::Eigenvector => "Importance based on well-connected neighbors",
            CentralityMeasure::PageRank => "Authority based on incoming links",
            CentralityMeasure::Katz => "Influence through all possible paths",
            CentralityMeasure::Authority => "Quality of information provided",
            CentralityMeasure::Hub => "Quality of links to authorities",
        }
    }

    /// Get the time complexity for this measure
    pub fn time_complexity(&self) -> &'static str {
        match self {
            CentralityMeasure::Degree => "O(n + m)",
            CentralityMeasure::Betweenness => "O(nm)",
            CentralityMeasure::Closeness => "O(nm)",
            CentralityMeasure::Eigenvector => "O(n^3)",
            CentralityMeasure::PageRank => "O(iterations × (n + m))",
            CentralityMeasure::Katz => "O(n^3)",
            CentralityMeasure::Authority => "O(iterations × (n + m))",
            CentralityMeasure::Hub => "O(iterations × (n + m))",
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_config_default() {
        let config = GraphConfig::default();
        assert_eq!(config.layout_algorithm, LayoutAlgorithm::ForceDirected);
        assert!(!config.enable_3d);
        assert!(config.enable_analysis);
        assert_eq!(config.max_nodes, 1000);
    }

    #[test]
    fn test_graph_config_builder() {
        let config = GraphConfig::new()
            .with_layout_algorithm(LayoutAlgorithm::Hierarchical)
            .with_3d_visualization(true)
            .with_max_nodes(500);

        assert_eq!(config.layout_algorithm, LayoutAlgorithm::Hierarchical);
        assert!(config.enable_3d);
        assert_eq!(config.max_nodes, 500);
    }

    #[test]
    fn test_graph_config_validation() {
        let valid_config = GraphConfig::default();
        assert!(valid_config.validate().is_ok());

        let invalid_config = GraphConfig {
            max_nodes: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_visualization_theme_colors() {
        assert_eq!(VisualizationTheme::Light.background_color(), "#ffffff");
        assert_eq!(VisualizationTheme::Dark.background_color(), "#1a1a1a");
        assert_eq!(VisualizationTheme::Light.text_color(), "#333333");
        assert_eq!(VisualizationTheme::Dark.text_color(), "#ffffff");
    }

    #[test]
    fn test_export_format_properties() {
        assert_eq!(GraphExportFormat::Svg.file_extension(), "svg");
        assert!(GraphExportFormat::InteractiveHtml.is_interactive());
        assert!(GraphExportFormat::Svg.is_vector());
        assert!(!GraphExportFormat::Png.is_vector());
    }

    #[test]
    fn test_filter_config_patterns() {
        let filter = FilterConfig::default()
            .with_trait_pattern("Estimator.*".to_string())
            .with_implementation_pattern(".*Impl".to_string());

        assert!(filter.matches_trait_name("EstimatorTrait"));
        assert!(filter.matches_implementation_name("MyImpl"));
        assert!(!filter.matches_trait_name("SomethingElse"));
    }

    #[test]
    fn test_optimization_level_settings() {
        assert_eq!(OptimizationLevel::Quality.layout_iterations(), 1000);
        assert_eq!(OptimizationLevel::Fast.layout_iterations(), 50);
        assert!(OptimizationLevel::Quality.enable_expensive_analysis());
        assert!(!OptimizationLevel::Fast.enable_expensive_analysis());
    }

    #[test]
    fn test_node_type_properties() {
        assert_eq!(TraitNodeType::Trait.display_name(), "Trait");
        assert_eq!(TraitNodeType::Trait.default_shape(), "ellipse");
        assert_eq!(TraitNodeType::Implementation.default_shape(), "box");
    }

    #[test]
    fn test_edge_type_properties() {
        assert_eq!(EdgeType::Inherits.display_name(), "inherits");
        assert!(EdgeType::Inherits.is_hierarchical());
        assert!(EdgeType::Inherits.is_directed());
        assert!(!EdgeType::Uses.is_directed());
    }

    #[test]
    fn test_stability_level_properties() {
        assert_eq!(StabilityLevel::Stable.display_name(), "Stable");
        assert!(StabilityLevel::Stable.include_by_default());
        assert!(!StabilityLevel::Deprecated.include_by_default());
    }

    #[test]
    fn test_community_detection_properties() {
        assert_eq!(CommunityDetection::Louvain.display_name(), "Louvain");
        assert!(CommunityDetection::Louvain.suitable_for_large_graphs());
        assert!(!CommunityDetection::GirvanNewman.suitable_for_large_graphs());
    }

    #[test]
    fn test_centrality_measure_properties() {
        assert_eq!(CentralityMeasure::Degree.display_name(), "Degree");
        assert_eq!(
            CentralityMeasure::Degree.description(),
            "Number of direct connections"
        );
        assert_eq!(CentralityMeasure::Degree.time_complexity(), "O(n + m)");
    }
}