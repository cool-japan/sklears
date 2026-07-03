//! Graph generation logic for trait relationship visualization
//!
//! This module provides the core functionality for generating trait relationship
//! graphs from source code analysis, with support for hierarchical relationships,
//! implementations, associated types, and complex trait dependencies.

use super::graph_config::{GraphConfig, TraitNodeType, EdgeType, StabilityLevel};
use super::graph_structures::{
    TraitGraph, TraitGraphNode, TraitGraphEdge, NodeMetadata, EdgeMetadata,
    TraitGraphMetadata, GraphStatistics, PerformanceMetrics, LayoutQualityMetrics,
};
use crate::api_reference_generator::{AssociatedType, MethodInfo, TraitInfo};
use crate::error::{Result, SklearsError};

// SciRS2 Core imports for full compliance
use scirs2_core::random::Random;
#[cfg(feature = "scirs2-gpu-reporting")]
use scirs2_core::gpu::{GpuBackend, GpuContext};

use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Main generator for trait relationship graphs with advanced capabilities
///
/// Implements [`std::fmt::Debug`] manually below (rather than deriving it)
/// since `layout_algorithms: HashMap<String, Box<dyn LayoutAlgorithmImpl + Send + Sync>>`
/// holds a trait object that does not itself implement `Debug`.
pub struct TraitGraphGenerator {
    /// Configuration for graph generation
    config: GraphConfig,
    /// GPU context for acceleration (optional; only tracked when the
    /// `scirs2-gpu-reporting` feature is enabled).
    #[cfg(feature = "scirs2-gpu-reporting")]
    gpu_context: Option<GpuContext>,
    /// Layout algorithms registry
    layout_algorithms: HashMap<String, Box<dyn LayoutAlgorithmImpl + Send + Sync>>,
    /// Layout computation cache
    layout_cache: Arc<RwLock<HashMap<String, LayoutResult>>>,
}

/// Trait for layout algorithm implementations
///
/// This is a private, minimal implementation used internally by
/// [`TraitGraphGenerator`] for its own layout application step. The public,
/// fully-featured layout algorithm surface (with configurable parameters and
/// quality metrics) lives in
/// [`crate::trait_explorer::graph_visualization::layout_algorithms`] and is
/// used by [`super::GraphVisualizationFramework`] instead; the two are kept
/// separate (rather than sharing one trait) so this module's placeholder
/// implementations don't collide with the public `layout_algorithms` API.
trait LayoutAlgorithmImpl: Send + Sync {
    fn compute_layout(&self, graph: &TraitGraph, config: &GraphConfig) -> Result<LayoutResult>;
    fn get_algorithm_name(&self) -> &str;
    fn supports_3d(&self) -> bool;
    fn get_quality_metrics(&self, result: &LayoutResult) -> LayoutQualityMetrics;
}

/// Result of layout computation
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// 2D positions for nodes
    pub positions_2d: HashMap<String, (f64, f64)>,
    /// 3D positions for nodes (if supported)
    pub positions_3d: Option<HashMap<String, (f64, f64, f64)>>,
    /// Quality metrics for the layout
    pub quality_metrics: LayoutQualityMetrics,
    /// Time taken for computation
    pub computation_time: Duration,
}

impl std::fmt::Debug for TraitGraphGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Summarize each registered layout algorithm as "name" or "name
        // (3d)" so the registry's actual contents are visible in Debug
        // output, rather than just a bare count.
        let mut algorithm_names: Vec<String> = self
            .layout_algorithms
            .values()
            .map(|algorithm| {
                if algorithm.supports_3d() {
                    format!("{} (3d)", algorithm.get_algorithm_name())
                } else {
                    algorithm.get_algorithm_name().to_string()
                }
            })
            .collect();
        algorithm_names.sort();

        f.debug_struct("TraitGraphGenerator")
            .field("config", &self.config)
            .field("has_gpu_context", &self.has_gpu_context())
            .field("layout_algorithms", &algorithm_names)
            .finish()
    }
}

impl TraitGraphGenerator {
    /// Create a new graph generator with the specified configuration
    pub fn new(config: GraphConfig) -> Result<Self> {
        // Initialize GPU context if enabled and the `scirs2-gpu-reporting`
        // feature is active.
        //
        // We use the SciRS2-Core GPU abstraction to select the best available
        // backend. `GpuBackend::preferred()` performs runtime device detection
        // and transparently yields `GpuBackend::Cpu` when no accelerator is
        // present, so the resulting context honestly reflects the hardware that
        // is actually in use rather than claiming a GPU that does not exist.
        #[cfg(feature = "scirs2-gpu-reporting")]
        let gpu_context = if config.enable_gpu {
            let preferred = GpuBackend::preferred();
            match GpuContext::new(preferred) {
                Ok(context) => Some(context),
                Err(_err) => {
                    // The preferred backend could not be initialized (for
                    // example, a detected device became unavailable). Fall
                    // back to a real CPU context so downstream code still has
                    // a valid execution context, rather than fabricating GPU
                    // availability.
                    GpuContext::new(GpuBackend::Cpu).ok()
                }
            }
        } else {
            None
        };

        // Initialize layout algorithms
        let mut layout_algorithms: HashMap<String, Box<dyn LayoutAlgorithmImpl + Send + Sync>> = HashMap::new();
        layout_algorithms.insert("force_directed".to_string(), Box::new(ForceDirectedLayout::new()));
        layout_algorithms.insert("hierarchical".to_string(), Box::new(HierarchicalLayout::new()));
        layout_algorithms.insert("circular".to_string(), Box::new(CircularLayout::new()));
        layout_algorithms.insert("grid".to_string(), Box::new(GridLayout::new()));
        layout_algorithms.insert("radial".to_string(), Box::new(RadialLayout::new()));

        let layout_cache = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            config,
            #[cfg(feature = "scirs2-gpu-reporting")]
            gpu_context,
            layout_algorithms,
            layout_cache,
        })
    }

    /// Generate a trait-specific graph focusing on a particular trait
    pub fn generate_trait_graph(
        &self,
        trait_info: &TraitInfo,
        implementations: &[String],
    ) -> Result<TraitGraph> {
        let start_time = Instant::now();

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create main trait node
        let main_node = self.create_trait_node(trait_info)?;
        nodes.push(main_node);

        // Add supertrait nodes and edges
        for supertrait in &trait_info.supertraits {
            let supertrait_node = self.create_supertrait_node(supertrait)?;
            nodes.push(supertrait_node);

            let edge = TraitGraphEdge {
                from: supertrait.clone(),
                to: trait_info.name.clone(),
                edge_type: EdgeType::Inherits,
                weight: 1.0,
                thickness: Some(2.0),
                color: None,
                label: Some("inherits".to_string()),
                directed: true,
                metadata: EdgeMetadata {
                    confidence: 1.0,
                    source: "source_code".to_string(),
                    source_line: None,
                    conditional: false,
                    conditions: Vec::new(),
                    feature_flags: Vec::new(),
                    attributes: HashMap::new(),
                },
            };
            edges.push(edge);
        }

        // Add implementation nodes and edges
        for implementation in implementations {
            let impl_node = self.create_implementation_node(implementation, &trait_info.name)?;
            nodes.push(impl_node);

            let edge = TraitGraphEdge {
                from: trait_info.name.clone(),
                to: implementation.clone(),
                edge_type: EdgeType::Implements,
                weight: 0.8,
                thickness: Some(1.5),
                color: None,
                label: Some("implements".to_string()),
                directed: true,
                metadata: EdgeMetadata {
                    confidence: 0.9,
                    source: "source_code".to_string(),
                    source_line: None,
                    conditional: false,
                    conditions: Vec::new(),
                    feature_flags: Vec::new(),
                    attributes: HashMap::new(),
                },
            };
            edges.push(edge);
        }

        // Add associated type nodes if any
        for associated_type in &trait_info.associated_types {
            let assoc_node = self.create_associated_type_node(&associated_type.name, &trait_info.name)?;
            // Read the node's actual (namespaced, e.g. "TraitName::AssocTypeName")
            // id before moving it into `nodes`, so the edge below points at
            // the node that really exists rather than the bare associated
            // type name.
            let assoc_node_id = assoc_node.id.clone();
            nodes.push(assoc_node);

            let edge = TraitGraphEdge {
                from: trait_info.name.clone(),
                to: assoc_node_id,
                edge_type: EdgeType::AssociatedWith,
                weight: 0.6,
                thickness: Some(1.0),
                color: None,
                label: Some("defines".to_string()),
                directed: true,
                metadata: EdgeMetadata {
                    confidence: 1.0,
                    source: "source_code".to_string(),
                    source_line: None,
                    conditional: false,
                    conditions: Vec::new(),
                    feature_flags: Vec::new(),
                    attributes: HashMap::new(),
                },
            };
            edges.push(edge);
        }

        // Add method nodes if configured
        if self.config.filter_config.node_types.contains(&TraitNodeType::Method) {
            for method in &trait_info.methods {
                let method_node = self.create_method_node(&method.name, &trait_info.name)?;
                // As with associated types above: the node's id is
                // namespaced ("TraitName::method_name"), so the edge must
                // target that id, not the bare method name.
                let method_node_id = method_node.id.clone();
                nodes.push(method_node);

                let edge = TraitGraphEdge {
                    from: trait_info.name.clone(),
                    to: method_node_id,
                    edge_type: EdgeType::DefinesMethod,
                    weight: 0.4,
                    thickness: Some(0.8),
                    color: None,
                    label: Some("defines".to_string()),
                    directed: true,
                    metadata: EdgeMetadata {
                        confidence: 1.0,
                        source: "source_code".to_string(),
                        source_line: None,
                        conditional: false,
                        conditions: Vec::new(),
                        feature_flags: Vec::new(),
                        attributes: HashMap::new(),
                    },
                };
                edges.push(edge);
            }
        }

        // Filter nodes and edges based on configuration
        self.apply_filters(&mut nodes, &mut edges)?;

        // Create graph metadata
        let metadata = TraitGraphMetadata {
            title: format!("Trait Graph: {}", trait_info.name),
            description: Some(format!("Relationship graph for trait {}", trait_info.name)),
            generated_at: Utc::now(),
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
            source_project: None,
            git_commit: None,
            tags: vec!["trait".to_string(), "relationships".to_string()],
            custom_metadata: HashMap::new(),
            ..Default::default()
        };

        // Calculate statistics
        let statistics = GraphStatistics::from_graph(&nodes, &edges);

        // Record performance metrics
        let generation_time = start_time.elapsed();
        let performance = PerformanceMetrics {
            generation_time,
            layout_time: Duration::from_secs(0), // Will be updated by layout computation
            analysis_time: Duration::from_secs(0),
            memory_usage: self.estimate_memory_usage(&nodes, &edges),
            layout_iterations: 0,
            cpu_utilization: 0.0,
            gpu_accelerated: self.is_gpu_accelerated(),
            simd_optimized: self.config.enable_simd,
        };

        let mut graph = TraitGraph {
            nodes,
            edges,
            metadata,
            statistics,
            performance,
        };

        // Apply layout if requested
        if self.config.enable_analysis {
            self.apply_layout(&mut graph)?;
        }

        Ok(graph)
    }

    /// Generate a comprehensive graph from multiple traits
    pub fn generate_full_graph(&self, traits: &[&TraitInfo]) -> Result<TraitGraph> {
        let start_time = Instant::now();

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut processed_traits = HashSet::new();

        // Process each trait
        for trait_info in traits {
            if processed_traits.contains(&trait_info.name) {
                continue;
            }

            // Add main trait node
            let trait_node = self.create_trait_node(trait_info)?;
            nodes.push(trait_node);
            processed_traits.insert(trait_info.name.clone());

            // Process supertraits
            for supertrait in &trait_info.supertraits {
                if !processed_traits.contains(supertrait) {
                    let supertrait_node = self.create_supertrait_node(supertrait)?;
                    nodes.push(supertrait_node);
                    processed_traits.insert(supertrait.clone());
                }

                // Add inheritance edge
                let edge = TraitGraphEdge {
                    from: supertrait.clone(),
                    to: trait_info.name.clone(),
                    edge_type: EdgeType::Inherits,
                    weight: 1.0,
                    thickness: Some(2.0),
                    color: None,
                    label: Some("inherits".to_string()),
                    directed: true,
                    metadata: EdgeMetadata::default(),
                };
                edges.push(edge);
            }

            // Process associated types
            for associated_type in &trait_info.associated_types {
                let assoc_node = self.create_associated_type_node(&associated_type.name, &trait_info.name)?;
                // See the identical comment in `generate_trait_graph`: the
                // edge must target the node's actual (namespaced) id.
                let assoc_node_id = assoc_node.id.clone();
                nodes.push(assoc_node);

                let edge = TraitGraphEdge {
                    from: trait_info.name.clone(),
                    to: assoc_node_id,
                    edge_type: EdgeType::AssociatedWith,
                    weight: 0.6,
                    thickness: Some(1.0),
                    color: None,
                    label: Some("defines".to_string()),
                    directed: true,
                    metadata: EdgeMetadata::default(),
                };
                edges.push(edge);
            }

            // Process methods if configured
            if self.config.filter_config.node_types.contains(&TraitNodeType::Method) {
                for method in &trait_info.methods {
                    let method_node = self.create_method_node(&method.name, &trait_info.name)?;
                    // See the identical comment in `generate_trait_graph`.
                    let method_node_id = method_node.id.clone();
                    nodes.push(method_node);

                    let edge = TraitGraphEdge {
                        from: trait_info.name.clone(),
                        to: method_node_id,
                        edge_type: EdgeType::DefinesMethod,
                        weight: 0.4,
                        thickness: Some(0.8),
                        color: None,
                        label: Some("defines".to_string()),
                        directed: true,
                        metadata: EdgeMetadata::default(),
                    };
                    edges.push(edge);
                }
            }
        }

        // Add cross-trait relationships
        self.add_cross_trait_relationships(&mut edges, traits)?;

        // Apply filters
        self.apply_filters(&mut nodes, &mut edges)?;

        // Create comprehensive metadata
        let metadata = TraitGraphMetadata {
            title: "Comprehensive Trait Relationship Graph".to_string(),
            description: Some(format!("Complete relationship graph for {} traits", traits.len())),
            generated_at: Utc::now(),
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
            source_project: None,
            git_commit: None,
            tags: vec!["traits".to_string(), "comprehensive".to_string(), "relationships".to_string()],
            custom_metadata: HashMap::new(),
            ..Default::default()
        };

        let statistics = GraphStatistics::from_graph(&nodes, &edges);
        let performance = PerformanceMetrics {
            generation_time: start_time.elapsed(),
            layout_time: Duration::from_secs(0),
            analysis_time: Duration::from_secs(0),
            memory_usage: self.estimate_memory_usage(&nodes, &edges),
            layout_iterations: 0,
            cpu_utilization: 0.0,
            gpu_accelerated: self.is_gpu_accelerated(),
            simd_optimized: self.config.enable_simd,
        };

        let mut graph = TraitGraph {
            nodes,
            edges,
            metadata,
            statistics,
            performance,
        };

        // Apply layout
        if self.config.enable_analysis {
            self.apply_layout(&mut graph)?;
        }

        Ok(graph)
    }

    /// Create a trait node from trait information
    fn create_trait_node(&self, trait_info: &TraitInfo) -> Result<TraitGraphNode> {
        let complexity = self.calculate_trait_complexity(trait_info);
        let stability = self.determine_stability_level(trait_info);

        let metadata = NodeMetadata {
            documentation: trait_info.docs.clone(),
            source_file: trait_info.source_file.clone(),
            source_line: trait_info.source_line,
            stability,
            complexity,
            created_at: None,
            modified_at: None,
            trait_name: Some(trait_info.name.clone()),
            generic_parameters: trait_info.generics.clone(),
            where_clauses: self.extract_where_clauses(trait_info),
            deprecation_note: None,
            feature_flags: trait_info.feature_flags.clone(),
            module_path: trait_info.module_path.clone(),
            visibility: Some(trait_info.visibility.to_string()),
            attributes: HashMap::new(),
        };

        let size = self.calculate_node_size(complexity, &trait_info.methods, &trait_info.associated_types);

        Ok(TraitGraphNode {
            id: trait_info.name.clone(),
            label: trait_info.name.clone(),
            node_type: TraitNodeType::Trait,
            position_2d: None,
            position_3d: None,
            size,
            color: Some(TraitNodeType::Trait.default_color().to_string()),
            shape: Some(TraitNodeType::Trait.default_shape().to_string()),
            visible: true,
            metadata,
        })
    }

    /// Extract the `where`-clause predicates declared by a trait.
    ///
    /// The [`TraitInfo`] representation stores generic parameters and associated
    /// types as bound-carrying strings rather than a structured `where` clause,
    /// so the predicates are reconstructed from the bounds that are actually
    /// recorded:
    ///
    /// * A generic parameter written with inline bounds (for example
    ///   `"T: Clone + Send"`) contributes the predicate `T: Clone + Send`.
    /// * An associated type with bounds (for example `Output: Debug`) contributes
    ///   the predicate `Self::Output: Debug`.
    ///
    /// Bounds are normalized (surrounding whitespace trimmed, internal spacing
    /// collapsed) and duplicate predicates are removed while preserving the order
    /// in which they were discovered. Parameters and associated types without
    /// bounds yield no predicate, so an unconstrained trait correctly produces an
    /// empty clause list.
    fn extract_where_clauses(&self, trait_info: &TraitInfo) -> Vec<String> {
        let mut clauses = Vec::new();
        let mut seen = HashSet::new();

        // Predicates carried inline on generic parameters.
        for generic in &trait_info.generics {
            if let Some((param, bounds)) = generic.split_once(':') {
                let param = param.trim();
                let bounds = Self::normalize_bound_list(bounds);
                if param.is_empty() || bounds.is_empty() {
                    continue;
                }
                let predicate = format!("{}: {}", param, bounds);
                if seen.insert(predicate.clone()) {
                    clauses.push(predicate);
                }
            }
        }

        // Predicates carried by associated types.
        for associated_type in &trait_info.associated_types {
            if associated_type.bounds.is_empty() {
                continue;
            }
            let bounds = Self::normalize_bound_list(&associated_type.bounds.join(" + "));
            if bounds.is_empty() {
                continue;
            }
            let predicate = format!("Self::{}: {}", associated_type.name.trim(), bounds);
            if seen.insert(predicate.clone()) {
                clauses.push(predicate);
            }
        }

        clauses
    }

    /// Normalize a `+`-separated list of trait bounds.
    ///
    /// Splits on `+`, trims each bound, discards empty fragments, and rejoins the
    /// remaining bounds with `" + "` so the resulting predicate uses canonical
    /// spacing regardless of how the source bounds were formatted.
    fn normalize_bound_list(bounds: &str) -> String {
        bounds
            .split('+')
            .map(|bound| bound.trim())
            .filter(|bound| !bound.is_empty())
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// Create a supertrait node
    fn create_supertrait_node(&self, supertrait_name: &str) -> Result<TraitGraphNode> {
        let metadata = NodeMetadata {
            trait_name: Some(supertrait_name.to_string()),
            stability: StabilityLevel::Stable, // Default assumption
            complexity: 5.0, // Default complexity
            ..Default::default()
        };

        Ok(TraitGraphNode {
            id: supertrait_name.to_string(),
            label: supertrait_name.to_string(),
            node_type: TraitNodeType::Trait,
            position_2d: None,
            position_3d: None,
            size: 1.0,
            color: Some(TraitNodeType::Trait.default_color().to_string()),
            shape: Some(TraitNodeType::Trait.default_shape().to_string()),
            visible: true,
            metadata,
        })
    }

    /// Create an implementation node
    fn create_implementation_node(&self, impl_name: &str, trait_name: &str) -> Result<TraitGraphNode> {
        let metadata = NodeMetadata {
            trait_name: Some(trait_name.to_string()),
            stability: StabilityLevel::Stable,
            complexity: 3.0,
            ..Default::default()
        };

        Ok(TraitGraphNode {
            id: impl_name.to_string(),
            label: impl_name.to_string(),
            node_type: TraitNodeType::Implementation,
            position_2d: None,
            position_3d: None,
            size: 0.8,
            color: Some(TraitNodeType::Implementation.default_color().to_string()),
            shape: Some(TraitNodeType::Implementation.default_shape().to_string()),
            visible: true,
            metadata,
        })
    }

    /// Create an associated type node
    fn create_associated_type_node(&self, type_name: &str, trait_name: &str) -> Result<TraitGraphNode> {
        let metadata = NodeMetadata {
            trait_name: Some(trait_name.to_string()),
            stability: StabilityLevel::Stable,
            complexity: 2.0,
            ..Default::default()
        };

        Ok(TraitGraphNode {
            id: format!("{}::{}", trait_name, type_name),
            label: type_name.to_string(),
            node_type: TraitNodeType::AssociatedType,
            position_2d: None,
            position_3d: None,
            size: 0.6,
            color: Some(TraitNodeType::AssociatedType.default_color().to_string()),
            shape: Some(TraitNodeType::AssociatedType.default_shape().to_string()),
            visible: true,
            metadata,
        })
    }

    /// Create a method node
    fn create_method_node(&self, method_name: &str, trait_name: &str) -> Result<TraitGraphNode> {
        let metadata = NodeMetadata {
            trait_name: Some(trait_name.to_string()),
            stability: StabilityLevel::Stable,
            complexity: 1.5,
            ..Default::default()
        };

        Ok(TraitGraphNode {
            id: format!("{}::{}", trait_name, method_name),
            label: method_name.to_string(),
            node_type: TraitNodeType::Method,
            position_2d: None,
            position_3d: None,
            size: 0.4,
            color: Some(TraitNodeType::Method.default_color().to_string()),
            shape: Some(TraitNodeType::Method.default_shape().to_string()),
            visible: true,
            metadata,
        })
    }

    /// Calculate trait complexity based on various factors
    fn calculate_trait_complexity(&self, trait_info: &TraitInfo) -> f64 {
        let method_complexity = trait_info.methods.len() as f64 * 2.0;
        let associated_type_complexity = trait_info.associated_types.len() as f64 * 3.0;
        let generic_complexity = trait_info.generics.len() as f64 * 1.5;
        let supertrait_complexity = trait_info.supertraits.len() as f64 * 2.5;

        let base_complexity = 1.0;
        let total = base_complexity + method_complexity + associated_type_complexity +
                   generic_complexity + supertrait_complexity;

        // Normalize to 0-100 scale
        total.min(100.0)
    }

    /// Determine stability level from trait information
    fn determine_stability_level(&self, trait_info: &TraitInfo) -> StabilityLevel {
        // Simple heuristics - in practice would analyze attributes and documentation
        if trait_info.feature_flags.contains(&"unstable".to_string()) {
            StabilityLevel::Unstable
        } else if trait_info.feature_flags.contains(&"experimental".to_string()) {
            StabilityLevel::Experimental
        } else if trait_info.docs.as_ref().is_some_and(|docs| docs.contains("deprecated")) {
            StabilityLevel::Deprecated
        } else {
            StabilityLevel::Stable
        }
    }

    /// Calculate appropriate node size based on complexity and content
    fn calculate_node_size(&self, complexity: f64, methods: &[MethodInfo], associated_types: &[AssociatedType]) -> f64 {
        let base_size = 1.0;
        let complexity_factor = complexity / 50.0; // Normalize complexity
        let method_factor = methods.len() as f64 * 0.1;
        let type_factor = associated_types.len() as f64 * 0.15;

        (base_size + complexity_factor + method_factor + type_factor).clamp(0.3, 3.0)
    }

    /// Apply filters to nodes and edges based on configuration
    fn apply_filters(&self, nodes: &mut Vec<TraitGraphNode>, edges: &mut Vec<TraitGraphEdge>) -> Result<()> {
        let filter_config = &self.config.filter_config;

        // Filter nodes by type
        nodes.retain(|node| filter_config.node_types.contains(&node.node_type));

        // Filter nodes by complexity
        nodes.retain(|node| {
            node.metadata.complexity >= filter_config.min_complexity &&
            node.metadata.complexity <= filter_config.max_complexity
        });

        // Filter nodes by stability
        nodes.retain(|node| filter_config.stability_levels.contains(&node.metadata.stability));

        // Filter nodes by deprecated status
        if !filter_config.include_deprecated {
            nodes.retain(|node| !node.metadata.is_deprecated());
        }

        // Filter nodes by experimental status
        if !filter_config.include_experimental {
            nodes.retain(|node| !node.metadata.is_experimental());
        }

        // Filter nodes by trait name patterns
        nodes.retain(|node| {
            if let Some(trait_name) = &node.metadata.trait_name {
                filter_config.matches_trait_name(trait_name)
            } else {
                filter_config.matches_trait_name(&node.label)
            }
        });

        // Create a set of valid node IDs
        let valid_node_ids: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();

        // Filter edges by type
        edges.retain(|edge| filter_config.edge_types.contains(&edge.edge_type));

        // Filter edges to only include those connecting valid nodes
        edges.retain(|edge| {
            valid_node_ids.contains(&edge.from) && valid_node_ids.contains(&edge.to)
        });

        // Apply node count limit
        if nodes.len() > self.config.max_nodes {
            // Sort by importance and keep the most important nodes
            nodes.sort_by(|a, b| {
                b.importance_score().partial_cmp(&a.importance_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            nodes.truncate(self.config.max_nodes);

            // Re-filter edges after node truncation
            let final_node_ids: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();
            edges.retain(|edge| {
                final_node_ids.contains(&edge.from) && final_node_ids.contains(&edge.to)
            });
        }

        Ok(())
    }

    /// Add cross-trait relationships (usage, dependencies, etc.)
    fn add_cross_trait_relationships(&self, edges: &mut Vec<TraitGraphEdge>, traits: &[&TraitInfo]) -> Result<()> {
        // Simple implementation - look for trait names mentioned in other traits
        for trait_info in traits {
            for other_trait in traits {
                if trait_info.name == other_trait.name {
                    continue;
                }

                // Check if trait is mentioned in documentation or other contexts
                let mut has_relationship = false;

                // Check supertraits
                if trait_info.supertraits.contains(&other_trait.name) {
                    continue; // Already handled in main generation
                }

                // Check if other trait is used in generic constraints
                for generic in &trait_info.generics {
                    if generic.contains(&other_trait.name) {
                        has_relationship = true;
                        break;
                    }
                }

                // Check associated types for references
                if !has_relationship {
                    for assoc_type in &trait_info.associated_types {
                        if assoc_type.bounds.iter().any(|bound| bound.contains(&other_trait.name)) {
                            has_relationship = true;
                            break;
                        }
                    }
                }

                if has_relationship {
                    let edge = TraitGraphEdge {
                        from: trait_info.name.clone(),
                        to: other_trait.name.clone(),
                        edge_type: EdgeType::Uses,
                        weight: 0.3,
                        thickness: Some(0.5),
                        color: None,
                        label: Some("uses".to_string()),
                        directed: true,
                        metadata: EdgeMetadata {
                            confidence: 0.7,
                            source: "analysis".to_string(),
                            source_line: None,
                            conditional: true,
                            conditions: vec!["generic_constraint".to_string()],
                            feature_flags: Vec::new(),
                            attributes: HashMap::new(),
                        },
                    };
                    edges.push(edge);
                }
            }
        }

        Ok(())
    }

    /// Apply layout algorithm to position nodes
    fn apply_layout(&self, graph: &mut TraitGraph) -> Result<()> {
        let layout_start = Instant::now();

        // Check cache first
        let cache_key = self.generate_cache_key(graph);
        if let Ok(cache) = self.layout_cache.read() {
            if let Some(cached_result) = cache.get(&cache_key) {
                self.apply_layout_result(graph, cached_result);
                return Ok(());
            }
        }

        // Determine which layout algorithm to use
        let algorithm_name = match self.config.layout_algorithm {
            super::graph_config::LayoutAlgorithm::ForceDirected => "force_directed",
            super::graph_config::LayoutAlgorithm::Hierarchical => "hierarchical",
            super::graph_config::LayoutAlgorithm::Circular => "circular",
            super::graph_config::LayoutAlgorithm::Grid => "grid",
            super::graph_config::LayoutAlgorithm::Radial => "radial",
            _ => "force_directed", // Default fallback
        };

        // Get the layout algorithm
        let layout_result = if let Some(algorithm) = self.layout_algorithms.get(algorithm_name) {
            algorithm.compute_layout(graph, &self.config)?
        } else {
            return Err(SklearsError::ValidationError(
                format!("Layout algorithm '{}' not found", algorithm_name)
            ));
        };

        // Cache the result
        if let Ok(mut cache) = self.layout_cache.write() {
            cache.insert(cache_key, layout_result.clone());
        }

        // Apply layout result to graph
        self.apply_layout_result(graph, &layout_result);

        // Update performance metrics
        graph.performance.layout_time = layout_start.elapsed();
        graph.performance.layout_iterations = self.config.optimization_level.layout_iterations() as u32;

        Ok(())
    }

    /// Generate a cache key for layout results
    fn generate_cache_key(&self, graph: &TraitGraph) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        graph.nodes.len().hash(&mut hasher);
        graph.edges.len().hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Apply layout result to graph nodes
    fn apply_layout_result(&self, graph: &mut TraitGraph, layout_result: &LayoutResult) {
        for node in &mut graph.nodes {
            if let Some(&(x, y)) = layout_result.positions_2d.get(&node.id) {
                node.position_2d = Some((x, y));
            }

            if let Some(ref positions_3d) = layout_result.positions_3d {
                if let Some(&(x, y, z)) = positions_3d.get(&node.id) {
                    node.position_3d = Some((x, y, z));
                }
            }
        }
    }

    /// Estimate memory usage of the graph
    fn estimate_memory_usage(&self, nodes: &[TraitGraphNode], edges: &[TraitGraphEdge]) -> u64 {
        (std::mem::size_of_val(nodes) + std::mem::size_of_val(edges)) as u64
    }

    /// Get configuration
    pub fn get_config(&self) -> &GraphConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: GraphConfig) {
        self.config = config;
    }

    /// Check if GPU acceleration is available
    ///
    /// Returns `true` only when a context backed by a real accelerator is
    /// active. A CPU fallback context (used when GPU acceleration was requested
    /// but no device is present) reports `false` so callers are never misled
    /// into believing the GPU is in use.
    ///
    /// Without the `scirs2-gpu-reporting` feature enabled, no GPU context is
    /// ever tracked and this always returns `false`.
    pub fn has_gpu_acceleration(&self) -> bool {
        self.is_gpu_accelerated()
    }

    /// Determine whether the active GPU context targets a real accelerator.
    ///
    /// This inspects the backend selected for the context and treats only
    /// non-CPU backends as genuinely GPU-accelerated.
    #[cfg(feature = "scirs2-gpu-reporting")]
    fn is_gpu_accelerated(&self) -> bool {
        self.gpu_context
            .as_ref()
            .map(|context| context.backend() != GpuBackend::Cpu)
            .unwrap_or(false)
    }

    /// Determine whether the active GPU context targets a real accelerator.
    ///
    /// The `scirs2-gpu-reporting` feature is disabled in this build, so no
    /// GPU context is ever tracked and acceleration is never reported.
    #[cfg(not(feature = "scirs2-gpu-reporting"))]
    fn is_gpu_accelerated(&self) -> bool {
        false
    }

    /// Whether a GPU context is currently tracked (for [`std::fmt::Debug`]
    /// reporting only; does not imply the context targets a real
    /// accelerator, see [`Self::is_gpu_accelerated`]).
    #[cfg(feature = "scirs2-gpu-reporting")]
    fn has_gpu_context(&self) -> bool {
        self.gpu_context.is_some()
    }

    /// Whether a GPU context is currently tracked (for [`std::fmt::Debug`]
    /// reporting only). The `scirs2-gpu-reporting` feature is disabled in
    /// this build, so no GPU context is ever tracked.
    #[cfg(not(feature = "scirs2-gpu-reporting"))]
    fn has_gpu_context(&self) -> bool {
        false
    }

    /// Get layout algorithm names
    pub fn get_available_layouts(&self) -> Vec<String> {
        self.layout_algorithms.keys().cloned().collect()
    }

    /// Clear layout cache
    pub fn clear_layout_cache(&self) -> Result<()> {
        if let Ok(mut cache) = self.layout_cache.write() {
            cache.clear();
        }
        Ok(())
    }
}

// Placeholder layout algorithm implementations.
//
// These are private to this module: they back `TraitGraphGenerator`'s own
// internal `layout_algorithms` registry only. The public, richer layout
// algorithm surface (same algorithm names, configurable parameters, full
// quality-metric computation) lives in `super::layout_algorithms` and is
// used by `super::GraphVisualizationFramework` instead. Keeping these
// private avoids an ambiguous-glob-export collision between the two
// same-named type sets when this module and `layout_algorithms` are both
// re-exported via `pub use *` from `graph_visualization/mod.rs`.

/// Force-directed layout algorithm using physics simulation
#[derive(Debug)]
struct ForceDirectedLayout;

impl ForceDirectedLayout {
    pub fn new() -> Self {
        Self
    }
}

impl LayoutAlgorithmImpl for ForceDirectedLayout {
    fn compute_layout(&self, graph: &TraitGraph, config: &GraphConfig) -> Result<LayoutResult> {
        let start_time = Instant::now();
        let n = graph.nodes.len();

        if n == 0 {
            return Ok(LayoutResult {
                positions_2d: HashMap::new(),
                positions_3d: None,
                quality_metrics: LayoutQualityMetrics {
                    edge_crossings: 0,
                    average_edge_length: 0.0,
                    distribution_uniformity: 1.0,
                    aesthetic_score: 1.0,
                },
                computation_time: start_time.elapsed(),
            });
        }

        // Initialize random positions. The maps are given explicit types so
        // that the `f64` arithmetic below (e.g. `.sqrt()`) has a concrete
        // numeric type to resolve against, rather than leaving `x`/`y`
        // dependent on inference alone (which `rng.random_range`'s generic
        // return type cannot pin down on its own).
        let mut rng = Random::seed(42);
        let mut positions_2d: HashMap<String, (f64, f64)> = HashMap::new();
        let mut positions_3d: Option<HashMap<String, (f64, f64, f64)>> = if config.enable_3d {
            Some(HashMap::new())
        } else {
            None
        };

        // Place nodes in random positions
        for node in &graph.nodes {
            let x = rng.random_range(-100.0..100.0);
            let y = rng.random_range(-100.0..100.0);
            positions_2d.insert(node.id.clone(), (x, y));

            if let Some(ref mut pos_3d) = positions_3d {
                let z = rng.random_range(-100.0..100.0);
                pos_3d.insert(node.id.clone(), (x, y, z));
            }
        }

        // Simple force-directed simulation (placeholder)
        let iterations = config.optimization_level.layout_iterations();
        let k = (400.0 / n as f64).sqrt(); // Optimal distance between nodes

        for _iteration in 0..iterations {
            // Apply forces (simplified)
            // In a real implementation, this would be much more sophisticated
            for node in &graph.nodes {
                if let Some((x, y)) = positions_2d.get(&node.id).copied() {
                    let mut fx = 0.0;
                    let mut fy = 0.0;

                    // Repulsive forces from other nodes
                    for other_node in &graph.nodes {
                        if other_node.id != node.id {
                            if let Some((ox, oy)) = positions_2d.get(&other_node.id).copied() {
                                let dx = x - ox;
                                let dy = y - oy;
                                let distance = (dx * dx + dy * dy).sqrt().max(0.1);
                                let force = k * k / distance;
                                fx += force * dx / distance;
                                fy += force * dy / distance;
                            }
                        }
                    }

                    // Attractive forces from connected nodes
                    for edge in &graph.edges {
                        if edge.from == node.id || edge.to == node.id {
                            let other_id = if edge.from == node.id { &edge.to } else { &edge.from };
                            if let Some((ox, oy)) = positions_2d.get(other_id).copied() {
                                let dx = ox - x;
                                let dy = oy - y;
                                let distance = (dx * dx + dy * dy).sqrt().max(0.1);
                                let force = distance * distance / k;
                                fx += force * dx / distance;
                                fy += force * dy / distance;
                            }
                        }
                    }

                    // Apply displacement with cooling
                    let temp = 1.0 - (_iteration as f64 / iterations as f64);
                    let displacement = temp * 10.0;
                    let force_magnitude = (fx * fx + fy * fy).sqrt();
                    if force_magnitude > 0.0 {
                        let new_x = x + fx / force_magnitude * displacement.min(force_magnitude);
                        let new_y = y + fy / force_magnitude * displacement.min(force_magnitude);
                        positions_2d.insert(node.id.clone(), (new_x, new_y));
                    }
                }
            }
        }

        let quality_metrics = self.get_quality_metrics(&LayoutResult {
            positions_2d: positions_2d.clone(),
            positions_3d: positions_3d.clone(),
            quality_metrics: LayoutQualityMetrics {
                edge_crossings: 0,
                average_edge_length: 0.0,
                distribution_uniformity: 0.0,
                aesthetic_score: 0.0,
            },
            computation_time: Duration::from_secs(0),
        });

        Ok(LayoutResult {
            positions_2d,
            positions_3d,
            quality_metrics,
            computation_time: start_time.elapsed(),
        })
    }

    fn get_algorithm_name(&self) -> &str {
        "force_directed"
    }

    fn supports_3d(&self) -> bool {
        true
    }

    fn get_quality_metrics(&self, _result: &LayoutResult) -> LayoutQualityMetrics {
        // Placeholder implementation
        LayoutQualityMetrics {
            edge_crossings: 0,
            average_edge_length: 50.0,
            distribution_uniformity: 0.7,
            aesthetic_score: 0.8,
        }
    }
}

// Additional placeholder layout implementations

/// Hierarchical layout algorithm
#[derive(Debug)]
struct HierarchicalLayout;

impl HierarchicalLayout {
    pub fn new() -> Self {
        Self
    }
}

impl LayoutAlgorithmImpl for HierarchicalLayout {
    fn compute_layout(&self, graph: &TraitGraph, _config: &GraphConfig) -> Result<LayoutResult> {
        let start_time = Instant::now();
        let mut positions_2d = HashMap::new();

        // Simple hierarchical layout: arrange nodes left-to-right, wrapping
        // into additional rows (levels) once a row fills up, so the
        // "hierarchical" structure is reflected in actual y-displacement
        // rather than placing every node on a single flat line.
        let x_spacing = 100.0;
        let y_spacing = 80.0;
        let nodes_per_row = (graph.nodes.len() as f64).sqrt().ceil().max(1.0) as usize;

        for (i, node) in graph.nodes.iter().enumerate() {
            let row = i / nodes_per_row;
            let col = i % nodes_per_row;
            let row_start = row * nodes_per_row;
            let row_len = nodes_per_row.min(graph.nodes.len() - row_start);
            let x = (col as f64) * x_spacing - ((row_len as f64 - 1.0) * x_spacing / 2.0);
            let y = (row as f64) * y_spacing;
            positions_2d.insert(node.id.clone(), (x, y));
        }

        Ok(LayoutResult {
            positions_2d,
            positions_3d: None,
            quality_metrics: LayoutQualityMetrics {
                edge_crossings: 0,
                average_edge_length: x_spacing,
                distribution_uniformity: 0.9,
                aesthetic_score: 0.7,
            },
            computation_time: start_time.elapsed(),
        })
    }

    fn get_algorithm_name(&self) -> &str {
        "hierarchical"
    }

    fn supports_3d(&self) -> bool {
        false
    }

    fn get_quality_metrics(&self, _result: &LayoutResult) -> LayoutQualityMetrics {
        LayoutQualityMetrics {
            edge_crossings: 0,
            average_edge_length: 100.0,
            distribution_uniformity: 0.9,
            aesthetic_score: 0.7,
        }
    }
}

/// Circular layout algorithm
#[derive(Debug)]
struct CircularLayout;

impl CircularLayout {
    pub fn new() -> Self {
        Self
    }
}

impl LayoutAlgorithmImpl for CircularLayout {
    fn compute_layout(&self, graph: &TraitGraph, _config: &GraphConfig) -> Result<LayoutResult> {
        let start_time = Instant::now();
        let mut positions_2d = HashMap::new();

        let n = graph.nodes.len();
        if n == 0 {
            return Ok(LayoutResult {
                positions_2d,
                positions_3d: None,
                quality_metrics: LayoutQualityMetrics {
                    edge_crossings: 0,
                    average_edge_length: 0.0,
                    distribution_uniformity: 1.0,
                    aesthetic_score: 1.0,
                },
                computation_time: start_time.elapsed(),
            });
        }

        let radius = 100.0;
        let angle_step = 2.0 * std::f64::consts::PI / n as f64;

        for (i, node) in graph.nodes.iter().enumerate() {
            let angle = i as f64 * angle_step;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            positions_2d.insert(node.id.clone(), (x, y));
        }

        Ok(LayoutResult {
            positions_2d,
            positions_3d: None,
            quality_metrics: LayoutQualityMetrics {
                edge_crossings: n / 4, // Rough estimate
                average_edge_length: radius,
                distribution_uniformity: 1.0,
                aesthetic_score: 0.6,
            },
            computation_time: start_time.elapsed(),
        })
    }

    fn get_algorithm_name(&self) -> &str {
        "circular"
    }

    fn supports_3d(&self) -> bool {
        false
    }

    fn get_quality_metrics(&self, _result: &LayoutResult) -> LayoutQualityMetrics {
        LayoutQualityMetrics {
            edge_crossings: 5,
            average_edge_length: 100.0,
            distribution_uniformity: 1.0,
            aesthetic_score: 0.6,
        }
    }
}

/// Grid layout algorithm
#[derive(Debug)]
struct GridLayout;

impl GridLayout {
    pub fn new() -> Self {
        Self
    }
}

impl LayoutAlgorithmImpl for GridLayout {
    fn compute_layout(&self, graph: &TraitGraph, _config: &GraphConfig) -> Result<LayoutResult> {
        let start_time = Instant::now();
        let mut positions_2d = HashMap::new();

        let n = graph.nodes.len();
        if n == 0 {
            return Ok(LayoutResult {
                positions_2d,
                positions_3d: None,
                quality_metrics: LayoutQualityMetrics {
                    edge_crossings: 0,
                    average_edge_length: 0.0,
                    distribution_uniformity: 1.0,
                    aesthetic_score: 1.0,
                },
                computation_time: start_time.elapsed(),
            });
        }

        let grid_size = (n as f64).sqrt().ceil() as usize;
        let spacing = 100.0;

        for (i, node) in graph.nodes.iter().enumerate() {
            let x = (i % grid_size) as f64 * spacing;
            let y = (i / grid_size) as f64 * spacing;
            positions_2d.insert(node.id.clone(), (x, y));
        }

        Ok(LayoutResult {
            positions_2d,
            positions_3d: None,
            quality_metrics: LayoutQualityMetrics {
                edge_crossings: n / 2, // Rough estimate
                average_edge_length: spacing,
                distribution_uniformity: 0.8,
                aesthetic_score: 0.5,
            },
            computation_time: start_time.elapsed(),
        })
    }

    fn get_algorithm_name(&self) -> &str {
        "grid"
    }

    fn supports_3d(&self) -> bool {
        false
    }

    fn get_quality_metrics(&self, _result: &LayoutResult) -> LayoutQualityMetrics {
        LayoutQualityMetrics {
            edge_crossings: 10,
            average_edge_length: 100.0,
            distribution_uniformity: 0.8,
            aesthetic_score: 0.5,
        }
    }
}

/// Radial layout algorithm
#[derive(Debug)]
struct RadialLayout;

impl RadialLayout {
    pub fn new() -> Self {
        Self
    }
}

impl LayoutAlgorithmImpl for RadialLayout {
    fn compute_layout(&self, graph: &TraitGraph, _config: &GraphConfig) -> Result<LayoutResult> {
        let start_time = Instant::now();
        let mut positions_2d = HashMap::new();

        // Place the most connected node at center
        let center_node = graph.nodes.iter()
            .max_by_key(|node| graph.get_degree(&node.id))
            .map(|node| node.id.clone())
            .unwrap_or_else(|| graph.nodes[0].id.clone());

        positions_2d.insert(center_node.clone(), (0.0, 0.0));

        // Place other nodes in concentric circles
        let mut radius = 80.0;
        let mut remaining_nodes: Vec<_> = graph.nodes.iter()
            .filter(|node| node.id != center_node)
            .collect();

        while !remaining_nodes.is_empty() {
            let nodes_in_ring = (2.0 * std::f64::consts::PI * radius / 60.0).floor() as usize;
            let nodes_to_place = nodes_in_ring.min(remaining_nodes.len());

            let angle_step = 2.0 * std::f64::consts::PI / nodes_to_place as f64;

            for (i, node) in remaining_nodes.iter().enumerate().take(nodes_to_place) {
                let angle = i as f64 * angle_step;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                positions_2d.insert(node.id.clone(), (x, y));
            }

            remaining_nodes.drain(0..nodes_to_place);
            radius += 80.0;
        }

        Ok(LayoutResult {
            positions_2d,
            positions_3d: None,
            quality_metrics: LayoutQualityMetrics {
                edge_crossings: graph.edges.len() / 8, // Rough estimate
                average_edge_length: radius / 2.0,
                distribution_uniformity: 0.7,
                aesthetic_score: 0.8,
            },
            computation_time: start_time.elapsed(),
        })
    }

    fn get_algorithm_name(&self) -> &str {
        "radial"
    }

    fn supports_3d(&self) -> bool {
        false
    }

    fn get_quality_metrics(&self, _result: &LayoutResult) -> LayoutQualityMetrics {
        LayoutQualityMetrics {
            edge_crossings: 3,
            average_edge_length: 80.0,
            distribution_uniformity: 0.7,
            aesthetic_score: 0.8,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_reference_generator::{MethodInfo, Visibility};

    fn create_test_trait_info() -> TraitInfo {
        TraitInfo {
            name: "TestTrait".to_string(),
            docs: Some("A test trait".to_string()),
            module_path: Some("test::module".to_string()),
            visibility: Visibility::Public,
            generics: vec!["T".to_string()],
            supertraits: vec!["SuperTrait".to_string()],
            associated_types: vec![AssociatedType {
                name: "Output".to_string(),
                bounds: vec!["Clone".to_string()],
                default: None,
            }],
            methods: vec![MethodInfo {
                name: "test_method".to_string(),
                signature: "fn test_method(&self) -> Self::Output".to_string(),
                docs: None,
                is_required: true,
                is_async: false,
                is_unsafe: false,
                generics: Vec::new(),
                return_type: Some("Self::Output".to_string()),
                arguments: Vec::new(),
            }],
            source_file: Some("test.rs".to_string()),
            source_line: Some(42),
            feature_flags: Vec::new(),
        }
    }

    #[test]
    fn test_graph_generator_creation() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_trait_graph_generation() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");
        let trait_info = create_test_trait_info();
        let implementations = vec!["TestImpl".to_string()];

        let graph = generator.generate_trait_graph(&trait_info, &implementations);
        assert!(graph.is_ok());

        let graph = graph.expect("expected valid value");
        assert!(!graph.nodes.is_empty());
        assert!(!graph.edges.is_empty());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_full_graph_generation() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");
        let trait_info = create_test_trait_info();
        let traits = vec![&trait_info];

        let graph = generator.generate_full_graph(&traits);
        assert!(graph.is_ok());

        let graph = graph.expect("expected valid value");
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_trait_complexity_calculation() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");
        let trait_info = create_test_trait_info();

        let complexity = generator.calculate_trait_complexity(&trait_info);
        assert!(complexity > 0.0);
        assert!(complexity <= 100.0);
    }

    #[test]
    fn test_layout_algorithms() {
        let force_directed = ForceDirectedLayout::new();
        assert_eq!(force_directed.get_algorithm_name(), "force_directed");
        assert!(force_directed.supports_3d());

        let hierarchical = HierarchicalLayout::new();
        assert_eq!(hierarchical.get_algorithm_name(), "hierarchical");
        assert!(!hierarchical.supports_3d());
    }

    #[test]
    fn test_node_creation() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");
        let trait_info = create_test_trait_info();

        let node = generator.create_trait_node(&trait_info).expect("create_trait_node should succeed");
        assert_eq!(node.id, "TestTrait");
        assert_eq!(node.node_type, TraitNodeType::Trait);
        assert!(node.visible);
    }

    #[test]
    fn test_extract_where_clauses_from_associated_type() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");
        let trait_info = create_test_trait_info();

        // The test trait has an unbounded generic `T` and an associated type
        // `Output: Clone`, so only the associated-type predicate is produced.
        let clauses = generator.extract_where_clauses(&trait_info);
        assert_eq!(clauses, vec!["Self::Output: Clone".to_string()]);
    }

    #[test]
    fn test_extract_where_clauses_from_bounded_generics() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");

        let mut trait_info = create_test_trait_info();
        trait_info.generics = vec![
            "T: Clone + Send".to_string(),
            "U".to_string(),
            "  V :  Debug  ".to_string(),
        ];
        trait_info.associated_types = vec![AssociatedType {
            name: "Item".to_string(),
            bounds: vec!["Eq".to_string(), "Hash".to_string()],
            default: None,
        }];

        let clauses = generator.extract_where_clauses(&trait_info);
        assert_eq!(
            clauses,
            vec![
                "T: Clone + Send".to_string(),
                "V: Debug".to_string(),
                "Self::Item: Eq + Hash".to_string(),
            ]
        );
    }

    #[test]
    fn test_extract_where_clauses_empty_when_unconstrained() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");

        let mut trait_info = create_test_trait_info();
        trait_info.generics = vec!["T".to_string(), "U".to_string()];
        trait_info.associated_types = vec![AssociatedType {
            name: "Output".to_string(),
            bounds: vec![],
            default: None,
        }];

        assert!(generator.extract_where_clauses(&trait_info).is_empty());
    }

    #[test]
    fn test_gpu_acceleration_reporting_without_gpu() {
        // With GPU disabled there is no context, so acceleration is false.
        let config = GraphConfig {
            enable_gpu: false,
            ..Default::default()
        };
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");
        assert!(!generator.has_gpu_acceleration());
    }

    #[test]
    fn test_filters() {
        let mut config = GraphConfig::default();
        config.filter_config.max_complexity = 5.0;

        let generator = TraitGraphGenerator::new(config).expect("expected valid value");
        let trait_info = create_test_trait_info();

        let mut nodes = vec![generator.create_trait_node(&trait_info).expect("create_trait_node should succeed")];
        let mut edges = Vec::new();

        // This should work since our test trait has low complexity
        let result = generator.apply_filters(&mut nodes, &mut edges);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_estimation() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");

        let nodes = vec![TraitGraphNode::new_trait("test".to_string(), "Test".to_string())];
        let edges = vec![TraitGraphEdge::new_inheritance("a".to_string(), "b".to_string())];

        let memory = generator.estimate_memory_usage(&nodes, &edges);
        assert!(memory > 0);
    }

    #[test]
    fn test_available_layouts() {
        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("expected valid value");

        let layouts = generator.get_available_layouts();
        assert!(layouts.contains(&"force_directed".to_string()));
        assert!(layouts.contains(&"hierarchical".to_string()));
        assert!(layouts.contains(&"circular".to_string()));
    }

    /// End-to-end smoke test: build a rich `TraitInfo` directly from
    /// `crate::api_reference_generator`'s public types (rather than through
    /// the shared `create_test_trait_info` helper) and drive it through
    /// `TraitGraphGenerator` to produce a `TraitGraph`, asserting the
    /// resulting node/edge structure matches the source trait information.
    #[test]
    fn test_smoke_rich_trait_info_end_to_end() {
        let trait_info = TraitInfo {
            name: "SmokeTestTrait".to_string(),
            docs: Some("Smoke-test trait for the rich API types.".to_string()),
            module_path: Some("smoke::module".to_string()),
            visibility: Visibility::Restricted("crate".to_string()),
            generics: vec!["T: Send".to_string()],
            supertraits: vec!["SmokeSuperTrait".to_string()],
            associated_types: vec![AssociatedType {
                name: "Item".to_string(),
                bounds: vec!["Clone".to_string(), "Debug".to_string()],
                default: Some("()".to_string()),
            }],
            methods: vec![
                MethodInfo {
                    name: "required_method".to_string(),
                    signature: "fn required_method(&self) -> Self::Item".to_string(),
                    docs: Some("A required method.".to_string()),
                    is_required: true,
                    is_async: false,
                    is_unsafe: false,
                    generics: Vec::new(),
                    return_type: Some("Self::Item".to_string()),
                    arguments: Vec::new(),
                },
                MethodInfo {
                    name: "provided_method".to_string(),
                    signature: "async fn provided_method(&self)".to_string(),
                    docs: None,
                    is_required: false,
                    is_async: true,
                    is_unsafe: false,
                    generics: Vec::new(),
                    return_type: None,
                    arguments: Vec::new(),
                },
            ],
            source_file: Some("smoke_test.rs".to_string()),
            source_line: Some(7),
            feature_flags: vec!["experimental".to_string()],
        };
        let implementations = vec!["SmokeImpl".to_string()];

        let config = GraphConfig::default();
        let generator = TraitGraphGenerator::new(config).expect("generator creation should succeed");

        let graph = generator
            .generate_trait_graph(&trait_info, &implementations)
            .expect("graph generation should succeed for a well-formed rich TraitInfo");

        // The graph must validate (no dangling edges, no duplicate node IDs).
        graph.validate().expect("generated graph should be internally consistent");

        // Expect at least: the trait node itself, its supertrait, its
        // implementation, and its associated type.
        assert!(graph.nodes.iter().any(|n| n.id == "SmokeTestTrait"));
        assert!(graph.nodes.iter().any(|n| n.id == "SmokeSuperTrait"));
        assert!(graph.nodes.iter().any(|n| n.id == "SmokeImpl"));
        assert!(graph
            .nodes
            .iter()
            .any(|n| n.node_type == TraitNodeType::AssociatedType));

        // The trait node should carry through the rich metadata via the
        // `Visibility::to_string()` conversion (Display impl), not a debug
        // dump of the enum.
        let trait_node = graph
            .find_node("SmokeTestTrait")
            .expect("trait node should be present");
        assert_eq!(
            trait_node.metadata.visibility.as_deref(),
            Some("restricted(crate)")
        );
        assert_eq!(trait_node.metadata.feature_flags, vec!["experimental".to_string()]);

        // Expect at least one edge for each relationship kind that was fed
        // in: inherits (supertrait), implements (implementation), and
        // associated-with (associated type).
        assert!(graph.edges.iter().any(|e| e.edge_type == EdgeType::Inherits));
        assert!(graph.edges.iter().any(|e| e.edge_type == EdgeType::Implements));
        assert!(graph
            .edges
            .iter()
            .any(|e| e.edge_type == EdgeType::AssociatedWith));
    }
}
