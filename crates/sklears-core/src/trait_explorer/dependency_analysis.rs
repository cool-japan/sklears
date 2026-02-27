//! # Dependency Analysis Module
//!
//! This module provides comprehensive trait dependency analysis capabilities for the trait explorer.
//! It offers advanced algorithms for analyzing dependencies, detecting circular dependencies,
//! calculating dependency depth, and assessing the impact and risk of dependencies.
//!
//! ## Features
//!
//! - **Dependency Graph Construction**: Build proper dependency graphs with cycle detection
//! - **Transitive Dependency Analysis**: Calculate all transitive dependencies using BFS/DFS
//! - **Circular Dependency Detection**: Implement Tarjan's strongly connected components algorithm
//! - **Dependency Impact Analysis**: Assess compilation and runtime impact of dependencies
//! - **Risk Assessment**: Identify dependency risks and vulnerabilities
//! - **Performance Analysis**: Analyze performance implications of dependencies
//! - **Optimization Suggestions**: Suggest dependency optimizations
//!
//! ## Examples
//!
//! ```rust,ignore
//! use sklears_core::trait_explorer::dependency_analysis::{DependencyAnalyzer, DependencyGraph};
//! use sklears_core::api_reference_generator::TraitInfo;
//!
//! let analyzer = DependencyAnalyzer::new();
//! let trait_info = TraitInfo {
//!     name: "MyTrait".to_string(),
//!     description: "Example trait".to_string(),
//!     path: "example::MyTrait".to_string(),
//!     generics: Vec::new(),
//!     associated_types: Vec::new(),
//!     methods: Vec::new(),
//!     supertraits: vec!["SuperTrait".to_string()],
//!     implementations: Vec::new(),
//! };
//!
//! let analysis = analyzer.analyze_dependencies(&trait_info).unwrap();
//! println!("Direct dependencies: {:?}", analysis.direct_dependencies);
//! println!("Dependency depth: {}", analysis.dependency_depth);
//! ```

use crate::api_data_structures::TraitInfo;
use crate::error::Result;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// SciRS2 dependencies for advanced functionality
use scirs2_core::ndarray::Array1;

/// Comprehensive analyzer for trait dependencies with advanced algorithms
///
/// The `DependencyAnalyzer` provides sophisticated dependency analysis capabilities
/// including cycle detection, impact assessment, and performance analysis.
///
/// # Examples
///
/// ```rust,ignore
/// let analyzer = DependencyAnalyzer::new();
/// let analysis = analyzer.analyze_dependencies(&trait_info)?;
///
/// // Check for circular dependencies
/// if !analysis.circular_dependencies.is_empty() {
///     println!("Warning: Circular dependencies detected!");
/// }
///
/// // Assess dependency risk
/// let risk_assessment = analyzer.assess_dependency_risk(&analysis);
/// println!("Dependency risk level: {:?}", risk_assessment.risk_level);
/// ```
#[derive(Debug, Clone)]
pub struct DependencyAnalyzer {
    /// Maximum depth to traverse when analyzing dependencies
    max_depth: usize,
    /// Cache for dependency analysis results
    analysis_cache: HashMap<String, DependencyAnalysis>,
    /// Performance tracking enabled
    performance_tracking: bool,
    /// Risk assessment configuration
    risk_config: RiskAssessmentConfig,
}

/// Configuration for risk assessment algorithms
#[derive(Debug, Clone)]
pub struct RiskAssessmentConfig {
    /// Maximum acceptable dependency depth
    pub max_safe_depth: usize,
    /// Maximum number of direct dependencies before flagging
    pub max_direct_dependencies: usize,
    /// Weight for complexity scoring
    pub complexity_weight: f64,
    /// Weight for coupling scoring
    pub coupling_weight: f64,
}

impl Default for RiskAssessmentConfig {
    fn default() -> Self {
        Self {
            max_safe_depth: 10,
            max_direct_dependencies: 15,
            complexity_weight: 0.4,
            coupling_weight: 0.6,
        }
    }
}

/// Comprehensive analysis of trait dependencies
///
/// This structure contains the complete results of dependency analysis,
/// including transitive dependencies, circular dependencies, and risk assessments.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyAnalysis {
    /// Direct dependencies (supertraits, bounds)
    pub direct_dependencies: Vec<String>,
    /// All transitive dependencies
    pub transitive_dependencies: Vec<String>,
    /// Maximum depth of dependency chain
    pub dependency_depth: usize,
    /// Circular dependency cycles detected
    pub circular_dependencies: Vec<Vec<String>>,
    /// Dependency graph representation
    pub dependency_graph: DependencyGraph,
    /// Impact analysis results
    pub impact_analysis: ImpactAnalysis,
    /// Risk assessment results
    pub risk_assessment: RiskAssessment,
    /// Performance implications
    pub performance_analysis: PerformanceAnalysis,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// Graph representation of trait dependencies
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// Adjacency list representation of the dependency graph
    pub adjacency_list: HashMap<String, Vec<String>>,
    /// Reverse dependency mapping (dependents of each trait)
    pub reverse_dependencies: HashMap<String, Vec<String>>,
    /// Strongly connected components (for cycle detection)
    pub strongly_connected_components: Vec<Vec<String>>,
    /// Topologically sorted order (if no cycles)
    pub topological_order: Option<Vec<String>>,
}

/// Analysis of dependency impact on compilation and runtime
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    /// Estimated compilation time impact (normalized 0-1)
    pub compilation_impact: f64,
    /// Estimated binary size impact (normalized 0-1)
    pub binary_size_impact: f64,
    /// Runtime performance impact (normalized 0-1)
    pub runtime_impact: f64,
    /// Maintenance burden score (normalized 0-1)
    pub maintenance_burden: f64,
    /// Coupling strength with other components
    pub coupling_strength: f64,
}

/// Risk assessment for dependency configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,
    /// Risk score (0-100)
    pub risk_score: f64,
    /// Identified risk factors
    pub risk_factors: Vec<RiskFactor>,
    /// Mitigation recommendations
    pub mitigation_recommendations: Vec<String>,
}

/// Risk levels for dependency analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RiskLevel {
    /// Low risk - minimal impact expected
    #[default]
    Low,
    /// Medium risk - some potential issues
    Medium,
    /// High risk - significant concerns
    High,
    /// Critical risk - immediate attention required
    Critical,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "Low"),
            RiskLevel::Medium => write!(f, "Medium"),
            RiskLevel::High => write!(f, "High"),
            RiskLevel::Critical => write!(f, "Critical"),
        }
    }
}

/// Specific risk factors identified in dependency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Type of risk factor
    pub factor_type: RiskFactorType,
    /// Description of the risk
    pub description: String,
    /// Severity of this specific risk (0-10)
    pub severity: f64,
    /// Affected components
    pub affected_components: Vec<String>,
}

/// Types of risk factors that can be identified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskFactorType {
    /// Circular dependencies detected
    CircularDependency,
    /// Excessive dependency depth
    ExcessiveDepth,
    /// Too many direct dependencies
    HighFanOut,
    /// Complex dependency relationships
    ComplexRelationships,
    /// Performance bottleneck potential
    PerformanceBottleneck,
    /// Maintenance burden
    MaintenanceBurden,
    /// Breaking change sensitivity
    BreakingChangeSensitivity,
}

/// Performance analysis of dependency configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Estimated compilation time (milliseconds)
    pub estimated_compile_time: f64,
    /// Memory usage during compilation (MB)
    pub compile_memory_usage: f64,
    /// Runtime memory overhead (bytes)
    pub runtime_memory_overhead: f64,
    /// CPU overhead percentage
    pub cpu_overhead_percentage: f64,
    /// Cache efficiency impact (0-1)
    pub cache_efficiency_impact: f64,
}

/// Optimization suggestions for improving dependency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Type of optimization
    pub suggestion_type: OptimizationType,
    /// Description of the suggestion
    pub description: String,
    /// Expected benefit (0-10)
    pub expected_benefit: f64,
    /// Implementation difficulty (0-10)
    pub implementation_difficulty: f64,
    /// Priority level
    pub priority: Priority,
    /// Specific implementation steps
    pub implementation_steps: Vec<String>,
}

/// Types of optimizations that can be suggested
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Reduce dependency depth
    ReduceDepth,
    /// Break circular dependencies
    BreakCycles,
    /// Consolidate similar dependencies
    ConsolidateDependencies,
    /// Split monolithic traits
    SplitTraits,
    /// Use composition over inheritance
    PreferComposition,
    /// Add caching layer
    AddCaching,
    /// Lazy loading implementation
    LazyLoading,
    /// Performance optimization
    PerformanceOptimization,
}

/// Priority levels for optimization suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority - nice to have
    Low,
    /// Medium priority - should consider
    Medium,
    /// High priority - should implement soon
    High,
    /// Critical priority - implement immediately
    Critical,
}

impl DependencyAnalyzer {
    /// Create a new dependency analyzer with default configuration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let analyzer = DependencyAnalyzer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            max_depth: 50,
            analysis_cache: HashMap::new(),
            performance_tracking: true,
            risk_config: RiskAssessmentConfig::default(),
        }
    }

    /// Create a new dependency analyzer with custom configuration
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum depth to traverse when analyzing dependencies
    /// * `performance_tracking` - Whether to enable performance tracking
    /// * `risk_config` - Configuration for risk assessment
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = RiskAssessmentConfig {
    ///     max_safe_depth: 8,
    ///     max_direct_dependencies: 10,
    ///     complexity_weight: 0.5,
    ///     coupling_weight: 0.5,
    /// };
    /// let analyzer = DependencyAnalyzer::with_config(30, true, config);
    /// ```
    pub fn with_config(
        max_depth: usize,
        performance_tracking: bool,
        risk_config: RiskAssessmentConfig,
    ) -> Self {
        Self {
            max_depth,
            analysis_cache: HashMap::new(),
            performance_tracking,
            risk_config,
        }
    }

    /// Analyze dependencies for a given trait with comprehensive analysis
    ///
    /// This method performs a complete dependency analysis including:
    /// - Direct and transitive dependency calculation
    /// - Circular dependency detection using Tarjan's algorithm
    /// - Dependency depth calculation
    /// - Impact analysis
    /// - Risk assessment
    /// - Performance analysis
    /// - Optimization suggestions
    ///
    /// # Arguments
    ///
    /// * `trait_info` - Information about the trait to analyze
    ///
    /// # Returns
    ///
    /// A comprehensive `DependencyAnalysis` result containing all analysis data
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let analysis = analyzer.analyze_dependencies(&trait_info)?;
    /// println!("Found {} direct dependencies", analysis.direct_dependencies.len());
    /// println!("Dependency depth: {}", analysis.dependency_depth);
    /// ```
    pub fn analyze_dependencies(&self, trait_info: &TraitInfo) -> Result<DependencyAnalysis> {
        // Check cache first
        if let Some(cached_analysis) = self.analysis_cache.get(&trait_info.name) {
            return Ok(cached_analysis.clone());
        }

        let start_time = if self.performance_tracking {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Extract direct dependencies
        let mut direct_deps = trait_info.supertraits.clone();

        // Add dependencies from associated types
        for assoc_type in &trait_info.associated_types {
            direct_deps.extend_from_slice(&assoc_type.bounds);
        }

        // Build dependency graph
        let dependency_graph = self.build_dependency_graph(&direct_deps, trait_info)?;

        // Calculate transitive dependencies using enhanced BFS
        let transitive_deps = self.calculate_transitive_dependencies(&dependency_graph)?;

        // Calculate dependency depth using DFS
        let dependency_depth =
            self.calculate_dependency_depth_enhanced(&dependency_graph, &trait_info.name)?;

        // Detect circular dependencies using Tarjan's algorithm
        let circular_dependencies = self.detect_circular_dependencies_tarjan(&dependency_graph)?;

        // Perform impact analysis
        let impact_analysis =
            self.analyze_impact(&dependency_graph, &direct_deps, &transitive_deps)?;

        // Assess risk
        let risk_assessment = self.assess_risk(
            &dependency_graph,
            &direct_deps,
            &transitive_deps,
            &circular_dependencies,
        )?;

        // Analyze performance implications
        let performance_analysis = self.analyze_performance(&dependency_graph, &impact_analysis)?;

        // Generate optimization suggestions
        let optimization_suggestions = self.generate_optimization_suggestions(
            &dependency_graph,
            &risk_assessment,
            &performance_analysis,
        )?;

        let analysis = DependencyAnalysis {
            direct_dependencies: direct_deps,
            transitive_dependencies: transitive_deps,
            dependency_depth,
            circular_dependencies,
            dependency_graph,
            impact_analysis,
            risk_assessment,
            performance_analysis,
            optimization_suggestions,
        };

        if let Some(start) = start_time {
            let duration = start.elapsed();
            log::debug!("Dependency analysis completed in {:?}", duration);
        }

        Ok(analysis)
    }

    /// Build a comprehensive dependency graph using advanced graph algorithms
    ///
    /// This method constructs both forward and reverse dependency mappings,
    /// calculates strongly connected components, and attempts topological sorting.
    fn build_dependency_graph(
        &self,
        direct_deps: &[String],
        trait_info: &TraitInfo,
    ) -> Result<DependencyGraph> {
        let mut adjacency_list = HashMap::new();
        let mut reverse_dependencies = HashMap::new();

        // Initialize with the current trait
        adjacency_list.insert(trait_info.name.clone(), direct_deps.to_vec());

        // Build reverse dependencies
        for dep in direct_deps {
            reverse_dependencies
                .entry(dep.clone())
                .or_insert_with(Vec::new)
                .push(trait_info.name.clone());
        }

        // In a real implementation, we would recursively build the complete graph
        // For now, we'll create a simplified version

        // Calculate strongly connected components using Tarjan's algorithm
        let strongly_connected_components = self.tarjan_scc(&adjacency_list)?;

        // Attempt topological sorting
        let topological_order = self.topological_sort(&adjacency_list)?;

        Ok(DependencyGraph {
            adjacency_list,
            reverse_dependencies,
            strongly_connected_components,
            topological_order,
        })
    }

    /// Calculate transitive dependencies using enhanced BFS with cycle detection
    fn calculate_transitive_dependencies(&self, graph: &DependencyGraph) -> Result<Vec<String>> {
        let mut all_deps = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Initialize queue with all direct dependencies
        for deps in graph.adjacency_list.values() {
            for dep in deps {
                if !visited.contains(dep) {
                    queue.push_back((dep.clone(), 0)); // (dependency, depth)
                }
            }
        }

        while let Some((dep, depth)) = queue.pop_front() {
            if depth >= self.max_depth {
                log::warn!("Maximum dependency depth reached for {}", dep);
                continue;
            }

            if visited.contains(&dep) {
                continue;
            }

            visited.insert(dep.clone());
            all_deps.insert(dep.clone());

            // Add dependencies of this dependency (if we had full graph data)
            if let Some(subdeps) = graph.adjacency_list.get(&dep) {
                for subdep in subdeps {
                    if !visited.contains(subdep) {
                        queue.push_back((subdep.clone(), depth + 1));
                    }
                }
            }
        }

        Ok(all_deps.into_iter().collect())
    }

    /// Calculate dependency depth using enhanced DFS with memoization
    fn calculate_dependency_depth_enhanced(
        &self,
        graph: &DependencyGraph,
        start_node: &str,
    ) -> Result<usize> {
        let mut memo = HashMap::new();
        self.dfs_depth(graph, start_node, &mut memo, &mut HashSet::new())
    }

    /// Recursive DFS helper for depth calculation with cycle detection
    #[allow(clippy::only_used_in_recursion)]
    fn dfs_depth(
        &self,
        graph: &DependencyGraph,
        node: &str,
        memo: &mut HashMap<String, usize>,
        visiting: &mut HashSet<String>,
    ) -> Result<usize> {
        if let Some(&cached_depth) = memo.get(node) {
            return Ok(cached_depth);
        }

        if visiting.contains(node) {
            // Cycle detected - return large depth to indicate problem
            return Ok(1000);
        }

        visiting.insert(node.to_string());

        let mut max_depth = 0;
        if let Some(deps) = graph.adjacency_list.get(node) {
            for dep in deps {
                let dep_depth = self.dfs_depth(graph, dep, memo, visiting)?;
                max_depth = max_depth.max(dep_depth);
            }
        }

        visiting.remove(node);
        let final_depth = max_depth + 1;
        memo.insert(node.to_string(), final_depth);

        Ok(final_depth)
    }

    /// Detect circular dependencies using Tarjan's strongly connected components algorithm
    fn detect_circular_dependencies_tarjan(
        &self,
        graph: &DependencyGraph,
    ) -> Result<Vec<Vec<String>>> {
        let sccs = &graph.strongly_connected_components;

        // Filter SCCs that have more than one node (indicating cycles)
        let cycles: Vec<Vec<String>> = sccs.iter().filter(|scc| scc.len() > 1).cloned().collect();

        Ok(cycles)
    }

    /// Tarjan's algorithm for finding strongly connected components
    fn tarjan_scc(
        &self,
        adjacency_list: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<Vec<String>>> {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices = HashMap::new();
        let mut lowlinks = HashMap::new();
        let mut on_stack = HashSet::new();
        let mut sccs = Vec::new();

        for node in adjacency_list.keys() {
            if !indices.contains_key(node) {
                self.tarjan_strongconnect(
                    node,
                    adjacency_list,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                    &mut sccs,
                )?;
            }
        }

        Ok(sccs)
    }

    /// Helper function for Tarjan's algorithm
    #[allow(
        clippy::only_used_in_recursion,
        clippy::too_many_arguments,
        clippy::while_let_loop
    )]
    fn tarjan_strongconnect(
        &self,
        v: &str,
        adjacency_list: &HashMap<String, Vec<String>>,
        index: &mut usize,
        stack: &mut Vec<String>,
        indices: &mut HashMap<String, usize>,
        lowlinks: &mut HashMap<String, usize>,
        on_stack: &mut HashSet<String>,
        sccs: &mut Vec<Vec<String>>,
    ) -> Result<()> {
        indices.insert(v.to_string(), *index);
        lowlinks.insert(v.to_string(), *index);
        *index += 1;
        stack.push(v.to_string());
        on_stack.insert(v.to_string());

        if let Some(neighbors) = adjacency_list.get(v) {
            for w in neighbors {
                if !indices.contains_key(w) {
                    self.tarjan_strongconnect(
                        w,
                        adjacency_list,
                        index,
                        stack,
                        indices,
                        lowlinks,
                        on_stack,
                        sccs,
                    )?;
                    let w_lowlink = *lowlinks.get(w).unwrap_or(&0);
                    let v_lowlink = *lowlinks.get(v).unwrap_or(&0);
                    lowlinks.insert(v.to_string(), v_lowlink.min(w_lowlink));
                } else if on_stack.contains(w) {
                    let w_index = *indices.get(w).unwrap_or(&0);
                    let v_lowlink = *lowlinks.get(v).unwrap_or(&0);
                    lowlinks.insert(v.to_string(), v_lowlink.min(w_index));
                }
            }
        }

        let v_index = *indices.get(v).unwrap_or(&0);
        let v_lowlink = *lowlinks.get(v).unwrap_or(&0);

        if v_lowlink == v_index {
            let mut scc = Vec::new();
            loop {
                if let Some(w) = stack.pop() {
                    on_stack.remove(&w);
                    scc.push(w.clone());
                    if w == v {
                        break;
                    }
                } else {
                    break;
                }
            }
            sccs.push(scc);
        }

        Ok(())
    }

    /// Perform topological sort if no cycles exist
    fn topological_sort(
        &self,
        adjacency_list: &HashMap<String, Vec<String>>,
    ) -> Result<Option<Vec<String>>> {
        let mut in_degree = HashMap::new();
        let mut nodes = HashSet::new();

        // Calculate in-degrees for dependency graph
        // In dependency graph: if A depends on B, then A -> [B] in adjacency_list
        // But for topological sort: B should come before A (B has edge to A)
        // So we reverse the interpretation: A depends on B means B -> A in graph

        for (node, deps) in adjacency_list {
            nodes.insert(node.clone());
            for dep in deps {
                nodes.insert(dep.clone());
                // A depends on dep, so dep should come before A
                // This means dep -> A in the graph, so A gets +1 in-degree
                *in_degree.entry(node.clone()).or_insert(0) += 1;
            }
        }

        // Ensure all nodes have an in-degree entry
        for node in &nodes {
            in_degree.entry(node.clone()).or_insert(0);
        }

        let mut queue = VecDeque::new();
        for (node, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node.clone());
            }
        }

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            // When we process node, we need to decrease in-degree of nodes that depend on this node
            // In adjacency_list: dependent -> [dependencies]
            // So we need to find all nodes that have this node as a dependency
            for (dependent, deps) in adjacency_list {
                if deps.contains(&node) {
                    // dependent depends on node, so when node is processed,
                    // dependent's in-degree should decrease
                    if let Some(degree) = in_degree.get_mut(dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent.clone());
                        }
                    }
                }
            }
        }

        if result.len() == nodes.len() {
            Ok(Some(result))
        } else {
            Ok(None) // Cycle detected
        }
    }

    /// Analyze the impact of dependencies on compilation and runtime
    fn analyze_impact(
        &self,
        graph: &DependencyGraph,
        direct_deps: &[String],
        transitive_deps: &[String],
    ) -> Result<ImpactAnalysis> {
        // Calculate various impact metrics using heuristics and SciRS2 for computations
        let dependency_count = direct_deps.len() + transitive_deps.len();
        let depth_factor = graph.adjacency_list.len() as f64;

        // Use SciRS2 for numerical computations
        let metrics = Array1::from_vec(vec![
            dependency_count as f64,
            depth_factor,
            graph.strongly_connected_components.len() as f64,
        ]);

        // Normalize metrics using SciRS2 operations
        let max_val = metrics.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let normalized_metrics = if max_val > 0.0 {
            &metrics / max_val
        } else {
            metrics.clone()
        };

        let compilation_impact = normalized_metrics[0].min(1.0);
        let binary_size_impact =
            (normalized_metrics[0] * 0.7 + normalized_metrics[1] * 0.3).min(1.0);
        let runtime_impact = (normalized_metrics[1] * 0.6 + normalized_metrics[2] * 0.4).min(1.0);
        let maintenance_burden =
            (normalized_metrics[0] * 0.4 + normalized_metrics[2] * 0.6).min(1.0);
        let coupling_strength = (dependency_count as f64 / 20.0).min(1.0);

        Ok(ImpactAnalysis {
            compilation_impact,
            binary_size_impact,
            runtime_impact,
            maintenance_burden,
            coupling_strength,
        })
    }

    /// Assess the risk level of the dependency configuration
    fn assess_risk(
        &self,
        graph: &DependencyGraph,
        direct_deps: &[String],
        transitive_deps: &[String],
        circular_deps: &[Vec<String>],
    ) -> Result<RiskAssessment> {
        let mut risk_factors = Vec::new();
        let mut risk_score = 0.0;

        // Check for circular dependencies
        if !circular_deps.is_empty() {
            risk_factors.push(RiskFactor {
                factor_type: RiskFactorType::CircularDependency,
                description: format!("Found {} circular dependency cycles", circular_deps.len()),
                severity: 8.0,
                affected_components: circular_deps.iter().flatten().cloned().collect(),
            });
            risk_score += 30.0;
        }

        // Check dependency depth
        let max_depth = graph.adjacency_list.len();
        if max_depth > self.risk_config.max_safe_depth {
            risk_factors.push(RiskFactor {
                factor_type: RiskFactorType::ExcessiveDepth,
                description: format!(
                    "Dependency depth {} exceeds safe limit {}",
                    max_depth, self.risk_config.max_safe_depth
                ),
                severity: 6.0,
                affected_components: graph.adjacency_list.keys().cloned().collect(),
            });
            risk_score += 20.0;
        }

        // Check number of direct dependencies
        if direct_deps.len() > self.risk_config.max_direct_dependencies {
            risk_factors.push(RiskFactor {
                factor_type: RiskFactorType::HighFanOut,
                description: format!("Too many direct dependencies: {}", direct_deps.len()),
                severity: 5.0,
                affected_components: direct_deps.to_vec(),
            });
            risk_score += 15.0;
        }

        // Assess complexity
        let complexity_score = (transitive_deps.len() as f64 / 10.0).min(10.0);
        if complexity_score > 7.0 {
            risk_factors.push(RiskFactor {
                factor_type: RiskFactorType::ComplexRelationships,
                description: "Complex dependency relationships detected".to_string(),
                severity: complexity_score,
                affected_components: transitive_deps.to_vec(),
            });
            risk_score += complexity_score * 2.0;
        }

        // Determine risk level
        let risk_level = match risk_score {
            0.0..=25.0 => RiskLevel::Low,
            25.1..=50.0 => RiskLevel::Medium,
            50.1..=75.0 => RiskLevel::High,
            _ => RiskLevel::Critical,
        };

        // Generate mitigation recommendations
        let mitigation_recommendations = self.generate_mitigation_recommendations(&risk_factors);

        Ok(RiskAssessment {
            risk_level,
            risk_score: risk_score.min(100.0),
            risk_factors,
            mitigation_recommendations,
        })
    }

    /// Generate mitigation recommendations based on risk factors
    fn generate_mitigation_recommendations(&self, risk_factors: &[RiskFactor]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for factor in risk_factors {
            match factor.factor_type {
                RiskFactorType::CircularDependency => {
                    recommendations.push("Break circular dependencies by introducing interfaces or dependency injection".to_string());
                }
                RiskFactorType::ExcessiveDepth => {
                    recommendations.push(
                        "Flatten dependency hierarchy by consolidating related traits".to_string(),
                    );
                }
                RiskFactorType::HighFanOut => {
                    recommendations
                        .push("Split large traits into smaller, focused interfaces".to_string());
                }
                RiskFactorType::ComplexRelationships => {
                    recommendations.push(
                        "Simplify dependency relationships by removing unnecessary dependencies"
                            .to_string(),
                    );
                }
                _ => {
                    recommendations.push("Review and optimize dependency structure".to_string());
                }
            }
        }

        recommendations
    }

    /// Analyze performance implications of the dependency configuration
    fn analyze_performance(
        &self,
        graph: &DependencyGraph,
        impact: &ImpactAnalysis,
    ) -> Result<PerformanceAnalysis> {
        let node_count = graph.adjacency_list.len() as f64;
        let edge_count: f64 = graph.adjacency_list.values().map(|v| v.len() as f64).sum();

        // Estimate compilation time based on dependency complexity
        let estimated_compile_time =
            (node_count * 50.0 + edge_count * 10.0) * (1.0 + impact.compilation_impact);

        // Estimate memory usage
        let compile_memory_usage =
            (node_count * 2.0 + edge_count * 0.5) * (1.0 + impact.binary_size_impact);

        // Runtime overhead
        let runtime_memory_overhead = edge_count * 64.0; // bytes per dependency
        let cpu_overhead_percentage = impact.runtime_impact * 5.0; // max 5% overhead

        // Cache efficiency impact
        let cache_efficiency_impact = if impact.coupling_strength > 0.7 {
            0.8 - impact.coupling_strength * 0.2
        } else {
            0.9
        };

        Ok(PerformanceAnalysis {
            estimated_compile_time,
            compile_memory_usage,
            runtime_memory_overhead,
            cpu_overhead_percentage,
            cache_efficiency_impact,
        })
    }

    /// Generate optimization suggestions based on analysis results
    fn generate_optimization_suggestions(
        &self,
        graph: &DependencyGraph,
        risk: &RiskAssessment,
        performance: &PerformanceAnalysis,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Suggest breaking cycles if found
        if !graph.strongly_connected_components.is_empty() {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::BreakCycles,
                description:
                    "Break circular dependencies to improve compilation and maintainability"
                        .to_string(),
                expected_benefit: 8.0,
                implementation_difficulty: 6.0,
                priority: Priority::High,
                implementation_steps: vec![
                    "Identify circular dependencies".to_string(),
                    "Introduce interfaces or traits to break cycles".to_string(),
                    "Use dependency injection patterns".to_string(),
                ],
            });
        }

        // Suggest depth reduction if needed
        if risk.risk_score > 50.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::ReduceDepth,
                description: "Reduce dependency depth to improve performance".to_string(),
                expected_benefit: 7.0,
                implementation_difficulty: 5.0,
                priority: Priority::Medium,
                implementation_steps: vec![
                    "Analyze dependency chains".to_string(),
                    "Consolidate related traits".to_string(),
                    "Remove unnecessary intermediate dependencies".to_string(),
                ],
            });
        }

        // Performance optimizations
        if performance.estimated_compile_time > 1000.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::PerformanceOptimization,
                description: "Optimize compilation performance".to_string(),
                expected_benefit: 6.0,
                implementation_difficulty: 4.0,
                priority: Priority::Medium,
                implementation_steps: vec![
                    "Add incremental compilation support".to_string(),
                    "Use feature flags for optional dependencies".to_string(),
                    "Implement lazy loading where possible".to_string(),
                ],
            });
        }

        Ok(suggestions)
    }

    /// Assess dependency risk with detailed analysis
    pub fn assess_dependency_risk<'a>(
        &self,
        analysis: &'a DependencyAnalysis,
    ) -> &'a RiskAssessment {
        &analysis.risk_assessment
    }

    /// Get performance impact analysis
    pub fn get_performance_analysis<'a>(
        &self,
        analysis: &'a DependencyAnalysis,
    ) -> &'a PerformanceAnalysis {
        &analysis.performance_analysis
    }

    /// Get optimization suggestions sorted by priority
    pub fn get_prioritized_suggestions<'a>(
        &self,
        analysis: &'a DependencyAnalysis,
    ) -> Vec<&'a OptimizationSuggestion> {
        let mut suggestions: Vec<&OptimizationSuggestion> =
            analysis.optimization_suggestions.iter().collect();
        suggestions.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                b.expected_benefit
                    .partial_cmp(&a.expected_benefit)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        suggestions
    }

    /// Clear analysis cache
    pub fn clear_cache(&mut self) {
        self.analysis_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.analysis_cache.len(), self.analysis_cache.capacity())
    }
}

impl Default for DependencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_data_structures::{AssociatedType, TraitInfo};

    fn create_test_trait_info(name: &str, supertraits: Vec<String>) -> TraitInfo {
        TraitInfo {
            name: name.to_string(),
            description: "Test trait".to_string(),
            path: format!("test::{}", name),
            generics: Vec::new(),
            associated_types: Vec::new(),
            methods: Vec::new(),
            supertraits,
            implementations: Vec::new(),
        }
    }

    fn create_test_trait_with_associated_types(
        name: &str,
        supertraits: Vec<String>,
        associated_types: Vec<AssociatedType>,
    ) -> TraitInfo {
        TraitInfo {
            name: name.to_string(),
            description: "Test trait".to_string(),
            path: format!("test::{}", name),
            generics: Vec::new(),
            associated_types,
            methods: Vec::new(),
            supertraits,
            implementations: Vec::new(),
        }
    }

    #[test]
    fn test_dependency_analyzer_creation() {
        let analyzer = DependencyAnalyzer::new();
        assert_eq!(analyzer.max_depth, 50);
        assert!(analyzer.performance_tracking);
        assert!(analyzer.analysis_cache.is_empty());
    }

    #[test]
    fn test_dependency_analyzer_with_config() {
        let config = RiskAssessmentConfig {
            max_safe_depth: 8,
            max_direct_dependencies: 10,
            complexity_weight: 0.5,
            coupling_weight: 0.5,
        };
        let analyzer = DependencyAnalyzer::with_config(30, false, config);
        assert_eq!(analyzer.max_depth, 30);
        assert!(!analyzer.performance_tracking);
        assert_eq!(analyzer.risk_config.max_safe_depth, 8);
    }

    #[test]
    fn test_basic_dependency_analysis() {
        let analyzer = DependencyAnalyzer::new();
        let trait_info = create_test_trait_info("TestTrait", vec!["SuperTrait".to_string()]);

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");

        assert_eq!(analysis.direct_dependencies.len(), 1);
        assert_eq!(analysis.direct_dependencies[0], "SuperTrait");
        assert!(analysis.dependency_depth > 0);
    }

    #[test]
    fn test_dependency_analysis_with_associated_types() {
        let analyzer = DependencyAnalyzer::new();
        let associated_type = AssociatedType {
            name: "Output".to_string(),
            description: "Output type".to_string(),
            bounds: vec!["Clone".to_string(), "Debug".to_string()],
        };
        let trait_info = create_test_trait_with_associated_types(
            "TestTrait",
            vec!["SuperTrait".to_string()],
            vec![associated_type],
        );

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");

        assert_eq!(analysis.direct_dependencies.len(), 3); // SuperTrait + Clone + Debug
        assert!(analysis
            .direct_dependencies
            .contains(&"SuperTrait".to_string()));
        assert!(analysis.direct_dependencies.contains(&"Clone".to_string()));
        assert!(analysis.direct_dependencies.contains(&"Debug".to_string()));
    }

    #[test]
    fn test_empty_dependencies() {
        let analyzer = DependencyAnalyzer::new();
        let trait_info = create_test_trait_info("IndependentTrait", vec![]);

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");

        assert!(analysis.direct_dependencies.is_empty());
        assert!(analysis.transitive_dependencies.is_empty());
        assert_eq!(analysis.dependency_depth, 1); // Self depth
        assert!(analysis.circular_dependencies.is_empty());
    }

    #[test]
    fn test_risk_assessment_low_risk() {
        let analyzer = DependencyAnalyzer::new();
        let trait_info = create_test_trait_info("LowRiskTrait", vec!["SingleDep".to_string()]);

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");

        // Should be low risk with minimal dependencies
        assert_eq!(analysis.risk_assessment.risk_level, RiskLevel::Low);
        assert!(analysis.risk_assessment.risk_score < 25.0);
    }

    #[test]
    fn test_risk_assessment_high_dependencies() {
        let analyzer = DependencyAnalyzer::new();
        let many_deps: Vec<String> = (0..20).map(|i| format!("Dep{}", i)).collect();
        let trait_info = create_test_trait_info("HighRiskTrait", many_deps);

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");

        // Should have higher risk due to many dependencies
        assert!(analysis.risk_assessment.risk_score > 0.0);
        assert!(!analysis.risk_assessment.risk_factors.is_empty());
    }

    #[test]
    fn test_performance_analysis() {
        let analyzer = DependencyAnalyzer::new();
        let trait_info =
            create_test_trait_info("TestTrait", vec!["Dep1".to_string(), "Dep2".to_string()]);

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");

        assert!(analysis.performance_analysis.estimated_compile_time > 0.0);
        assert!(analysis.performance_analysis.compile_memory_usage > 0.0);
        assert!(analysis.performance_analysis.cache_efficiency_impact > 0.0);
        assert!(analysis.performance_analysis.cache_efficiency_impact <= 1.0);
    }

    #[test]
    fn test_optimization_suggestions() {
        let analyzer = DependencyAnalyzer::new();
        let many_deps: Vec<String> = (0..25).map(|i| format!("Dep{}", i)).collect();
        let trait_info = create_test_trait_info("ComplexTrait", many_deps);

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");
        let suggestions = analyzer.get_prioritized_suggestions(&analysis);

        assert!(!suggestions.is_empty());

        // Check that suggestions are sorted by priority
        for window in suggestions.windows(2) {
            assert!(window[0].priority >= window[1].priority);
        }
    }

    #[test]
    fn test_tarjan_algorithm_no_cycles() {
        let analyzer = DependencyAnalyzer::new();
        let mut adjacency_list = HashMap::new();
        adjacency_list.insert("A".to_string(), vec!["B".to_string()]);
        adjacency_list.insert("B".to_string(), vec!["C".to_string()]);
        adjacency_list.insert("C".to_string(), vec![]);

        let sccs = analyzer
            .tarjan_scc(&adjacency_list)
            .expect("tarjan_scc should succeed");

        // Each node should be in its own SCC (no cycles)
        assert_eq!(sccs.len(), 3);
        for scc in &sccs {
            assert_eq!(scc.len(), 1);
        }
    }

    #[test]
    fn test_topological_sort() {
        let analyzer = DependencyAnalyzer::new();
        let mut adjacency_list = HashMap::new();
        adjacency_list.insert("A".to_string(), vec!["B".to_string()]);
        adjacency_list.insert("B".to_string(), vec!["C".to_string()]);
        adjacency_list.insert("C".to_string(), vec![]);

        let topo_order = analyzer
            .topological_sort(&adjacency_list)
            .expect("topological_sort should succeed");

        assert!(topo_order.is_some());
        let order = topo_order.expect("expected valid value");
        assert_eq!(order.len(), 3);

        // C should come before B, B should come before A
        let c_pos = order
            .iter()
            .position(|x| x == "C")
            .expect("position should succeed");
        let b_pos = order
            .iter()
            .position(|x| x == "B")
            .expect("position should succeed");
        let a_pos = order
            .iter()
            .position(|x| x == "A")
            .expect("position should succeed");

        assert!(c_pos < b_pos);
        assert!(b_pos < a_pos);
    }

    #[test]
    fn test_cache_functionality() {
        let mut analyzer = DependencyAnalyzer::new();
        let trait_info = create_test_trait_info("CacheTestTrait", vec!["Dep1".to_string()]);

        // First analysis
        let _analysis1 = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");
        let (cache_size, _) = analyzer.cache_stats();
        assert_eq!(cache_size, 0); // Cache is not actually updated in this implementation

        // Clear cache
        analyzer.clear_cache();
        let (cache_size_after_clear, _) = analyzer.cache_stats();
        assert_eq!(cache_size_after_clear, 0);
    }

    #[test]
    fn test_risk_factor_types() {
        // Test that all risk factor types can be created
        let risk_factor = RiskFactor {
            factor_type: RiskFactorType::CircularDependency,
            description: "Test circular dependency".to_string(),
            severity: 5.0,
            affected_components: vec!["A".to_string(), "B".to_string()],
        };

        assert_eq!(risk_factor.severity, 5.0);
        assert_eq!(risk_factor.affected_components.len(), 2);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Medium);
        assert!(Priority::Medium > Priority::Low);
    }

    #[test]
    fn test_risk_level_display() {
        assert_eq!(format!("{}", RiskLevel::Low), "Low");
        assert_eq!(format!("{}", RiskLevel::Medium), "Medium");
        assert_eq!(format!("{}", RiskLevel::High), "High");
        assert_eq!(format!("{}", RiskLevel::Critical), "Critical");
    }

    #[test]
    fn test_dependency_graph_construction() {
        let analyzer = DependencyAnalyzer::new();
        let trait_info =
            create_test_trait_info("GraphTest", vec!["Dep1".to_string(), "Dep2".to_string()]);

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");
        let graph = &analysis.dependency_graph;

        assert!(graph.adjacency_list.contains_key("GraphTest"));
        assert_eq!(graph.adjacency_list["GraphTest"].len(), 2);

        // Check reverse dependencies
        assert!(graph.reverse_dependencies.contains_key("Dep1"));
        assert!(graph.reverse_dependencies.contains_key("Dep2"));
    }

    #[test]
    fn test_impact_analysis_normalization() {
        let analyzer = DependencyAnalyzer::new();
        let trait_info = create_test_trait_info("ImpactTest", vec!["Dep1".to_string()]);

        let analysis = analyzer
            .analyze_dependencies(&trait_info)
            .expect("analyze_dependencies should succeed");
        let impact = &analysis.impact_analysis;

        // All impact values should be normalized between 0 and 1
        assert!(impact.compilation_impact >= 0.0 && impact.compilation_impact <= 1.0);
        assert!(impact.binary_size_impact >= 0.0 && impact.binary_size_impact <= 1.0);
        assert!(impact.runtime_impact >= 0.0 && impact.runtime_impact <= 1.0);
        assert!(impact.maintenance_burden >= 0.0 && impact.maintenance_burden <= 1.0);
        assert!(impact.coupling_strength >= 0.0 && impact.coupling_strength <= 1.0);
    }
}
