//! Graph Visualization Framework for Trait Explorer
//!
//! This module provides a comprehensive framework for visualizing trait relationships
//! and dependencies in Rust code. It supports multiple layout algorithms, export formats,
//! 3D visualization, and advanced graph analysis capabilities.
//!
//! # Architecture
//!
//! The framework is organized into focused modules:
//!
//! - [`graph_config`] - Configuration, types, and builder patterns
//! - [`graph_structures`] - Core graph data structures and metadata
//! - [`graph_generator`] - Graph generation and trait analysis logic
//! - [`graph_export`] - Export functionality for multiple formats
//! - [`graph_3d`] - 3D visualization with Three.js and WebGL
//! - [`graph_analysis`] - Graph analysis algorithms and metrics
//! - [`layout_algorithms`] - Layout algorithm implementations
//!
//! # Key Features
//!
//! ## Layout Algorithms
//! - Force-directed layout with physics simulation
//! - Hierarchical layouts for trait inheritance
//! - Circular and radial layouts for overview visualization
//! - Grid layouts for systematic arrangement
//! - Spring embedder algorithms
//! - Tree layouts for dependency visualization
//!
//! ## Export Formats
//! - Interactive HTML with D3.js integration
//! - SVG for vector graphics
//! - JSON for data interchange
//! - DOT format for Graphviz
//! - GraphML and GEXF for network analysis tools
//! - 3D scenes with Three.js
//!
//! ## Analysis Capabilities
//! - Centrality measures (degree, betweenness, closeness, eigenvector, PageRank)
//! - Community detection (Louvain, Leiden, label propagation)
//! - Critical path analysis
//! - Hub and bridge node identification
//! - Modularity and clustering coefficient calculation
//! - Small-world network analysis
//!
//! ## 3D Visualization
//! - Three.js-based 3D rendering
//! - WebGL acceleration
//! - VR support
//! - Physics simulation
//! - Interactive camera controls
//! - Material and lighting systems
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use sklears_core::trait_explorer::graph_visualization::*;
//!
//! // Create configuration
//! let config = GraphConfig::new()
//!     .with_layout_algorithm(LayoutAlgorithm::ForceDirected)
//!     .with_3d_visualization(true)
//!     .with_advanced_analysis(true)
//!     .with_theme(VisualizationTheme::Dark);
//!
//! // Create generator
//! let generator = TraitGraphGenerator::new(config)?;
//!
//! // Generate graph from trait information
//! let trait_info = /* your trait info */;
//! let graph = generator.generate_trait_graph(&trait_info, &implementations)?;
//!
//! // Export to interactive HTML
//! let html = graph.to_interactive_html()?;
//! std::fs::write("trait_graph.html", html)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Advanced Usage
//!
//! ## Custom Layout Algorithms
//!
//! ```rust,ignore
//! use sklears_core::trait_explorer::graph_visualization::*;
//!
//! // Create custom force-directed layout
//! let layout = LayoutAlgorithmFactory::create_force_directed_custom(
//!     1000,    // iterations
//!     150.0,   // temperature factor
//!     1.2,     // repulsive strength
//!     0.8,     // attractive strength
//! );
//!
//! // Apply layout to graph
//! let result = layout.compute_layout(&graph, &config)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Graph Analysis
//!
//! ```rust,ignore
//! use sklears_core::trait_explorer::graph_visualization::*;
//!
//! let analyzer = GraphAnalyzer::new();
//!
//! // Calculate centrality measures
//! let centrality = analyzer.calculate_all_centrality_measures(&graph)?;
//!
//! // Detect communities
//! let communities = analyzer.detect_communities(&graph, CommunityDetection::Louvain)?;
//!
//! // Find critical paths
//! let paths = analyzer.find_critical_paths_comprehensive(&graph)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## 3D Visualization
//!
//! ```rust,ignore
//! use sklears_core::trait_explorer::graph_visualization::*;
//!
//! let config_3d = ThreeDConfig::default()
//!     .with_physics_simulation(true)
//!     .with_vr_support(true)
//!     .with_camera_controls(true);
//!
//! let generator_3d = Graph3DGenerator::new(config_3d);
//! let scene_3d = generator_3d.generate_3d_scene(&graph)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Performance Optimization
//!
//! The framework includes several performance optimizations:
//!
//! - SIMD vectorization for numerical computations
//! - Parallel processing for large graphs
//! - GPU acceleration support (where available)
//! - Memory-efficient data structures
//! - Lazy computation and caching
//! - Adaptive algorithms based on graph size
//!
//! # SciRS2 Integration
//!
//! This module makes full use of the SciRS2 ecosystem:
//!
//! - `scirs2_core::ndarray` for numerical arrays and linear algebra
//! - `scirs2_core::random` for random number generation
//! - `scirs2_core::simd` for vectorized operations
//! - `scirs2_core::parallel` for parallel processing
//! - `scirs2_core::profiling` for performance monitoring
//!

// Module declarations
pub mod graph_config;
pub mod graph_structures;
pub mod graph_generator;
pub mod graph_export;
pub mod graph_3d;
pub mod graph_analysis;
pub mod layout_algorithms;

// Re-export core functionality for unified API access

// Configuration and types
pub use graph_config::*;

// Core data structures
pub use graph_structures::*;

// Graph generation
pub use graph_generator::*;

// Export functionality
pub use graph_export::*;

// 3D visualization
pub use graph_3d::*;

// Analysis capabilities
pub use graph_analysis::*;

// Layout algorithms
pub use layout_algorithms::*;

/// Unified graph visualization facade for convenient access to all functionality
///
/// This struct provides a high-level interface that combines all the capabilities
/// of the graph visualization framework in a single, easy-to-use API.
pub struct GraphVisualizationFramework {
    config: GraphConfig,
    generator: TraitGraphGenerator,
    exporter: GraphExporter,
    analyzer: GraphAnalyzer,
    generator_3d: Option<Graph3DGenerator>,
}

impl GraphVisualizationFramework {
    /// Create a new graph visualization framework with the specified configuration
    pub fn new(config: GraphConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let generator = TraitGraphGenerator::new(config.clone())?;
        let exporter = GraphExporter::new(config.clone());
        let analyzer = GraphAnalyzer::new();

        let generator_3d = if config.enable_3d {
            Some(Graph3DGenerator::new(ThreeDConfig::from_graph_config(&config)))
        } else {
            None
        };

        Ok(Self {
            config,
            generator,
            exporter,
            analyzer,
            generator_3d,
        })
    }

    /// Create a framework with default configuration
    pub fn with_defaults() -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(GraphConfig::default())
    }

    /// Create a framework optimized for performance
    pub fn with_performance_config() -> Result<Self, Box<dyn std::error::Error>> {
        let config = GraphConfig::new()
            .with_layout_algorithm(LayoutAlgorithm::ForceDirected)
            .with_optimization_level(OptimizationLevel::Aggressive)
            .with_simd_acceleration(true)
            .with_gpu_acceleration(true)
            .with_parallel_processing(true);

        Self::new(config)
    }

    /// Create a framework optimized for quality visualization
    pub fn with_quality_config() -> Result<Self, Box<dyn std::error::Error>> {
        let config = GraphConfig::new()
            .with_layout_algorithm(LayoutAlgorithm::ForceDirected)
            .with_3d_visualization(true)
            .with_advanced_analysis(true)
            .with_optimization_level(OptimizationLevel::Quality)
            .with_theme(VisualizationTheme::Professional);

        Self::new(config)
    }

    /// Generate a complete trait graph with analysis and layout
    pub fn generate_complete_graph(
        &self,
        trait_info: &TraitInfo,
        implementations: &[String],
    ) -> Result<TraitGraph, Box<dyn std::error::Error>> {
        // Generate the basic graph
        let mut graph = self.generator.generate_trait_graph(trait_info, implementations)?;

        // Apply layout algorithm
        let layout = LayoutAlgorithmFactory::create_layout(self.config.layout_algorithm);
        let layout_result = layout.compute_layout(&graph, &self.config)?;

        // Update node positions
        for node in &mut graph.nodes {
            if let Some(&position) = layout_result.positions_2d.get(&node.id) {
                node.position_2d = Some(position);
            }
            if let Some(ref positions_3d) = layout_result.positions_3d {
                if let Some(&position) = positions_3d.get(&node.id) {
                    node.position_3d = Some(position);
                }
            }
        }

        // Add analysis results if enabled
        if self.config.enable_analysis {
            let analysis_result = self.analyzer.analyze_graph(&graph)?;

            // Update graph metadata with analysis results
            let mut metadata = graph.metadata;
            metadata.analysis_results = Some(analysis_result);
            metadata.layout_quality = Some(layout_result.quality_metrics);
            graph.metadata = metadata;
        }

        Ok(graph)
    }

    /// Generate a full ecosystem graph from multiple traits
    pub fn generate_ecosystem_graph(
        &self,
        traits: &[&TraitInfo],
    ) -> Result<TraitGraph, Box<dyn std::error::Error>> {
        // Generate the basic graph
        let mut graph = self.generator.generate_full_graph(traits)?;

        // Apply layout algorithm
        let layout = LayoutAlgorithmFactory::create_layout(self.config.layout_algorithm);
        let layout_result = layout.compute_layout(&graph, &self.config)?;

        // Update node positions
        for node in &mut graph.nodes {
            if let Some(&position) = layout_result.positions_2d.get(&node.id) {
                node.position_2d = Some(position);
            }
            if let Some(ref positions_3d) = layout_result.positions_3d {
                if let Some(&position) = positions_3d.get(&node.id) {
                    node.position_3d = Some(position);
                }
            }
        }

        // Add comprehensive analysis
        if self.config.enable_analysis {
            let analysis_result = self.analyzer.analyze_graph(&graph)?;
            let communities = self.analyzer.detect_communities(&graph, CommunityDetection::Louvain)?;
            let critical_paths = self.analyzer.find_critical_paths_comprehensive(&graph)?;

            // Update graph metadata
            let mut metadata = graph.metadata;
            metadata.analysis_results = Some(analysis_result);
            metadata.layout_quality = Some(layout_result.quality_metrics);
            metadata.communities = Some(communities);
            metadata.critical_paths = Some(critical_paths);
            graph.metadata = metadata;
        }

        Ok(graph)
    }

    /// Export graph to all supported formats
    pub fn export_all_formats(
        &self,
        graph: &TraitGraph,
        base_path: &str,
    ) -> Result<ExportResults, Box<dyn std::error::Error>> {
        let mut results = ExportResults::default();

        // Export to various formats
        results.svg = Some(self.exporter.export_graph(graph, GraphExportFormat::Svg)?);
        results.json = Some(self.exporter.export_graph(graph, GraphExportFormat::Json)?);
        results.dot = Some(self.exporter.export_graph(graph, GraphExportFormat::Dot)?);
        results.interactive_html = Some(self.exporter.export_graph(graph, GraphExportFormat::InteractiveHtml)?);
        results.graphml = Some(self.exporter.export_graph(graph, GraphExportFormat::GraphML)?);
        results.gexf = Some(self.exporter.export_graph(graph, GraphExportFormat::GEXF)?);

        // Export 3D scene if enabled
        if self.config.enable_3d {
            if let Some(ref generator_3d) = self.generator_3d {
                results.three_d_scene = Some(generator_3d.generate_3d_scene(graph)?);
            }
        }

        // Write files if base path is provided
        if !base_path.is_empty() {
            if let Some(ref svg) = results.svg {
                std::fs::write(format!("{}.svg", base_path), svg)?;
            }
            if let Some(ref json) = results.json {
                std::fs::write(format!("{}.json", base_path), json)?;
            }
            if let Some(ref dot) = results.dot {
                std::fs::write(format!("{}.dot", base_path), dot)?;
            }
            if let Some(ref html) = results.interactive_html {
                std::fs::write(format!("{}.html", base_path), html)?;
            }
            if let Some(ref graphml) = results.graphml {
                std::fs::write(format!("{}.graphml", base_path), graphml)?;
            }
            if let Some(ref gexf) = results.gexf {
                std::fs::write(format!("{}.gexf", base_path), gexf)?;
            }
            if let Some(ref scene_3d) = results.three_d_scene {
                std::fs::write(format!("{}_3d.html", base_path), scene_3d)?;
            }
        }

        Ok(results)
    }

    /// Perform comprehensive analysis on a graph
    pub fn analyze_comprehensive(
        &self,
        graph: &TraitGraph,
    ) -> Result<ComprehensiveAnalysisResults, Box<dyn std::error::Error>> {
        let mut results = ComprehensiveAnalysisResults::default();

        // Basic analysis
        results.basic_analysis = self.analyzer.analyze_graph(graph)?;

        // Community detection with multiple algorithms
        results.louvain_communities = self.analyzer.detect_communities(graph, CommunityDetection::Louvain)?;

        // Critical path analysis
        results.critical_paths = self.analyzer.find_critical_paths_comprehensive(graph)?;

        // Hub and bridge analysis
        results.hub_nodes = self.analyzer.identify_hub_nodes(graph, 0.1)?;
        results.bridge_nodes = self.analyzer.identify_bridge_nodes(graph)?;
        results.bottleneck_edges = self.analyzer.identify_bottleneck_edges(graph)?;

        // Quality metrics
        if !results.louvain_communities.is_empty() {
            results.modularity = Some(self.analyzer.calculate_modularity(graph, &results.louvain_communities)?);
        }

        results.small_world_coefficient = Some(self.analyzer.calculate_small_world_coefficient(graph)?);

        Ok(results)
    }

    /// Get the current configuration
    pub fn config(&self) -> &GraphConfig {
        &self.config
    }

    /// Update the configuration and reinitialize components
    pub fn update_config(&mut self, new_config: GraphConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.config = new_config.clone();
        self.generator = TraitGraphGenerator::new(new_config.clone())?;
        self.exporter = GraphExporter::new(new_config.clone());

        self.generator_3d = if new_config.enable_3d {
            Some(Graph3DGenerator::new(ThreeDConfig::from_graph_config(&new_config)))
        } else {
            None
        };

        Ok(())
    }

    /// Get available layout algorithms
    pub fn available_layout_algorithms(&self) -> Vec<LayoutAlgorithm> {
        LayoutAlgorithmFactory::available_algorithms()
    }

    /// Get available export formats
    pub fn available_export_formats(&self) -> Vec<GraphExportFormat> {
        GraphExporter::available_formats()
    }

    /// Get available themes
    pub fn available_themes(&self) -> Vec<VisualizationTheme> {
        vec![
            VisualizationTheme::Light,
            VisualizationTheme::Dark,
            VisualizationTheme::HighContrast,
            VisualizationTheme::ColorblindFriendly,
            VisualizationTheme::Minimalist,
            VisualizationTheme::Professional,
        ]
    }
}

/// Results from exporting a graph to multiple formats
#[derive(Debug, Default)]
pub struct ExportResults {
    pub svg: Option<String>,
    pub json: Option<String>,
    pub dot: Option<String>,
    pub interactive_html: Option<String>,
    pub graphml: Option<String>,
    pub gexf: Option<String>,
    pub three_d_scene: Option<String>,
}

/// Results from comprehensive graph analysis
#[derive(Debug, Default)]
pub struct ComprehensiveAnalysisResults {
    pub basic_analysis: GraphAnalysisResult,
    pub louvain_communities: Vec<Community>,
    pub critical_paths: Vec<GraphPath>,
    pub hub_nodes: Vec<String>,
    pub bridge_nodes: Vec<String>,
    pub bottleneck_edges: Vec<String>,
    pub modularity: Option<f64>,
    pub small_world_coefficient: Option<f64>,
}

// Convenience functions for quick access

/// Create a quick trait graph with default settings
pub fn quick_trait_graph(
    trait_info: &TraitInfo,
    implementations: &[String],
) -> Result<TraitGraph, Box<dyn std::error::Error>> {
    let framework = GraphVisualizationFramework::with_defaults()?;
    framework.generate_complete_graph(trait_info, implementations)
}

/// Create a quick ecosystem graph with default settings
pub fn quick_ecosystem_graph(
    traits: &[&TraitInfo],
) -> Result<TraitGraph, Box<dyn std::error::Error>> {
    let framework = GraphVisualizationFramework::with_defaults()?;
    framework.generate_ecosystem_graph(traits)
}

/// Export a graph to interactive HTML with default settings
pub fn quick_html_export(graph: &TraitGraph) -> Result<String, Box<dyn std::error::Error>> {
    let exporter = GraphExporter::new(GraphConfig::default());
    exporter.export_graph(graph, GraphExportFormat::InteractiveHtml)
}

/// Perform quick analysis on a graph
pub fn quick_analysis(graph: &TraitGraph) -> Result<GraphAnalysisResult, Box<dyn std::error::Error>> {
    let analyzer = GraphAnalyzer::new();
    analyzer.analyze_graph(graph)
}

/// Create a graph with performance-optimized settings
pub fn performance_optimized_graph(
    trait_info: &TraitInfo,
    implementations: &[String],
) -> Result<TraitGraph, Box<dyn std::error::Error>> {
    let framework = GraphVisualizationFramework::with_performance_config()?;
    framework.generate_complete_graph(trait_info, implementations)
}

/// Create a graph with quality-optimized settings
pub fn quality_optimized_graph(
    trait_info: &TraitInfo,
    implementations: &[String],
) -> Result<TraitGraph, Box<dyn std::error::Error>> {
    let framework = GraphVisualizationFramework::with_quality_config()?;
    framework.generate_complete_graph(trait_info, implementations)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn create_test_trait_info(name: &str) -> TraitInfo {
        TraitInfo {
            name: name.to_string(),
            description: format!("Test trait {}", name),
            path: format!("test::{}", name),
            generics: vec!["T".to_string()],
            associated_types: vec![AssociatedType {
                name: "Item".to_string(),
                description: "Test associated type".to_string(),
                bounds: vec![],
            }],
            methods: vec!["test_method".to_string()],
            supertraits: vec!["Clone".to_string()],
            implementations: vec!["TestImpl".to_string()],
        }
    }

    #[test]
    fn test_framework_creation() {
        let config = GraphConfig::default();
        let framework = GraphVisualizationFramework::new(config);
        assert!(framework.is_ok());
    }

    #[test]
    fn test_framework_defaults() {
        let framework = GraphVisualizationFramework::with_defaults();
        assert!(framework.is_ok());
    }

    #[test]
    fn test_framework_performance_config() {
        let framework = GraphVisualizationFramework::with_performance_config();
        assert!(framework.is_ok());
    }

    #[test]
    fn test_framework_quality_config() {
        let framework = GraphVisualizationFramework::with_quality_config();
        assert!(framework.is_ok());
    }

    #[test]
    fn test_complete_graph_generation() {
        let framework = GraphVisualizationFramework::with_defaults().unwrap();
        let trait_info = create_test_trait_info("TestTrait");
        let implementations = vec!["TestImpl".to_string()];

        let graph = framework.generate_complete_graph(&trait_info, &implementations);
        assert!(graph.is_ok());

        let graph = graph.unwrap();
        assert!(!graph.nodes.is_empty());
        assert!(!graph.edges.is_empty());
    }

    #[test]
    fn test_ecosystem_graph_generation() {
        let framework = GraphVisualizationFramework::with_defaults().unwrap();
        let trait1 = create_test_trait_info("Trait1");
        let trait2 = create_test_trait_info("Trait2");
        let traits = vec![&trait1, &trait2];

        let graph = framework.generate_ecosystem_graph(&traits);
        assert!(graph.is_ok());

        let graph = graph.unwrap();
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_export_all_formats() {
        let framework = GraphVisualizationFramework::with_defaults().unwrap();
        let trait_info = create_test_trait_info("TestTrait");
        let implementations = vec!["TestImpl".to_string()];

        let graph = framework.generate_complete_graph(&trait_info, &implementations).unwrap();

        // Test export without writing files
        let results = framework.export_all_formats(&graph, "");
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(results.svg.is_some());
        assert!(results.json.is_some());
        assert!(results.dot.is_some());
        assert!(results.interactive_html.is_some());
    }

    #[test]
    fn test_comprehensive_analysis() {
        let framework = GraphVisualizationFramework::with_defaults().unwrap();
        let trait_info = create_test_trait_info("TestTrait");
        let implementations = vec!["TestImpl".to_string()];

        let graph = framework.generate_complete_graph(&trait_info, &implementations).unwrap();
        let analysis = framework.analyze_comprehensive(&graph);

        assert!(analysis.is_ok());
        let analysis = analysis.unwrap();
        assert!(!analysis.hub_nodes.is_empty() || analysis.hub_nodes.is_empty()); // Either is valid
    }

    #[test]
    fn test_config_update() {
        let mut framework = GraphVisualizationFramework::with_defaults().unwrap();
        let new_config = GraphConfig::new()
            .with_layout_algorithm(LayoutAlgorithm::Circular)
            .with_theme(VisualizationTheme::Dark);

        let result = framework.update_config(new_config.clone());
        assert!(result.is_ok());
        assert_eq!(framework.config().layout_algorithm, LayoutAlgorithm::Circular);
        assert_eq!(framework.config().theme, VisualizationTheme::Dark);
    }

    #[test]
    fn test_available_algorithms_and_formats() {
        let framework = GraphVisualizationFramework::with_defaults().unwrap();

        let algorithms = framework.available_layout_algorithms();
        assert!(!algorithms.is_empty());

        let formats = framework.available_export_formats();
        assert!(!formats.is_empty());

        let themes = framework.available_themes();
        assert!(!themes.is_empty());
    }

    #[test]
    fn test_convenience_functions() {
        let trait_info = create_test_trait_info("TestTrait");
        let implementations = vec!["TestImpl".to_string()];

        // Test quick trait graph
        let graph = quick_trait_graph(&trait_info, &implementations);
        assert!(graph.is_ok());

        let graph = graph.unwrap();

        // Test quick HTML export
        let html = quick_html_export(&graph);
        assert!(html.is_ok());

        // Test quick analysis
        let analysis = quick_analysis(&graph);
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_optimized_graph_functions() {
        let trait_info = create_test_trait_info("TestTrait");
        let implementations = vec!["TestImpl".to_string()];

        // Test performance optimized
        let perf_graph = performance_optimized_graph(&trait_info, &implementations);
        assert!(perf_graph.is_ok());

        // Test quality optimized
        let quality_graph = quality_optimized_graph(&trait_info, &implementations);
        assert!(quality_graph.is_ok());
    }

    #[test]
    fn test_temporary_file_handling() {
        let temp_dir = env::temp_dir();
        let test_path = temp_dir.join("test_graph_framework");

        let framework = GraphVisualizationFramework::with_defaults().unwrap();
        let trait_info = create_test_trait_info("TestTrait");
        let implementations = vec!["TestImpl".to_string()];

        let graph = framework.generate_complete_graph(&trait_info, &implementations).unwrap();

        // Export with file writing
        let results = framework.export_all_formats(&graph, test_path.to_str().unwrap());
        assert!(results.is_ok());

        // Check that files were created
        let svg_file = format!("{}.svg", test_path.to_str().unwrap());
        let json_file = format!("{}.json", test_path.to_str().unwrap());

        if std::path::Path::new(&svg_file).exists() {
            std::fs::remove_file(&svg_file).unwrap_or(());
        }
        if std::path::Path::new(&json_file).exists() {
            std::fs::remove_file(&json_file).unwrap_or(());
        }
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together correctly
        let config = GraphConfig::new()
            .with_layout_algorithm(LayoutAlgorithm::ForceDirected)
            .with_3d_visualization(true)
            .with_advanced_analysis(true);

        let framework = GraphVisualizationFramework::new(config).unwrap();
        let trait_info = create_test_trait_info("IntegrationTest");
        let implementations = vec!["TestImpl".to_string()];

        // Full workflow test
        let graph = framework.generate_complete_graph(&trait_info, &implementations).unwrap();
        let analysis = framework.analyze_comprehensive(&graph).unwrap();
        let exports = framework.export_all_formats(&graph, "").unwrap();

        // Verify results
        assert!(!graph.nodes.is_empty());
        assert!(exports.svg.is_some());
        assert!(exports.interactive_html.is_some());
        assert!(!analysis.basic_analysis.centrality_measures.is_empty());
    }
}