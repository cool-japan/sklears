//! Dependency Management System
//!
//! This module provides comprehensive dependency resolution, compatibility checking,
//! and dependency graph management for modular components including circular dependency
//! detection, version constraint solving, and dependency injection patterns.

use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use thiserror::Error;

use super::component_framework::{ComponentDependency, PluggableComponent};

/// Dependency resolver for managing component dependencies
///
/// Provides dependency resolution, topological sorting, circular dependency detection,
/// and version constraint satisfaction for complex component dependency graphs.
#[derive(Debug)]
pub struct DependencyResolver {
    /// Component dependency graph
    dependency_graph: Arc<RwLock<DependencyGraph>>,
    /// Version constraint solver
    version_solver: VersionConstraintSolver,
    /// Dependency injection registry
    injection_registry: Arc<RwLock<DependencyInjectionRegistry>>,
    /// Resolution configuration
    config: DependencyResolutionConfig,
    /// Resolution cache
    resolution_cache: Arc<RwLock<HashMap<String, ResolutionResult>>>,
    /// Resolver statistics
    stats: Arc<RwLock<DependencyStatistics>>,
}

impl DependencyResolver {
    /// Create a new dependency resolver
    #[must_use]
    pub fn new() -> Self {
        Self {
            dependency_graph: Arc::new(RwLock::new(DependencyGraph::new())),
            version_solver: VersionConstraintSolver::new(),
            injection_registry: Arc::new(RwLock::new(DependencyInjectionRegistry::new())),
            config: DependencyResolutionConfig::default(),
            resolution_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DependencyStatistics::new())),
        }
    }

    /// Add component dependencies to the graph
    pub fn add_component_dependencies(
        &self,
        component_id: &str,
        component_type: &str,
        version: &str,
        dependencies: Vec<ComponentDependency>,
    ) -> SklResult<()> {
        let mut graph = self.dependency_graph.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let node = DependencyNode {
            component_id: component_id.to_string(),
            component_type: component_type.to_string(),
            version: version.to_string(),
            dependencies: dependencies.clone(),
            resolved_dependencies: HashMap::new(),
            dependency_state: DependencyState::Unresolved,
        };

        graph.add_node(component_id.to_string(), node);

        // Add edges for dependencies
        for dependency in dependencies {
            graph.add_edge(component_id.to_string(), dependency.component_type.clone());
            stats.total_dependencies += 1;
        }

        stats.total_components += 1;
        Ok(())
    }

    /// Resolve dependencies for a component
    pub fn resolve_dependencies(&self, component_id: &str) -> SklResult<ResolutionResult> {
        let mut stats = self.stats.write().unwrap();
        stats.resolution_attempts += 1;

        // Check cache first
        {
            let cache = self.resolution_cache.read().unwrap();
            if let Some(cached_result) = cache.get(component_id) {
                if !self.is_cache_expired(&cached_result.resolution_time) {
                    stats.cache_hits += 1;
                    return Ok(cached_result.clone());
                }
            }
        }

        stats.cache_misses += 1;

        // Perform resolution
        let resolution_order = self.topological_sort(component_id)?;
        let version_constraints = self.collect_version_constraints(&resolution_order)?;
        let resolved_versions = self.version_solver.solve_constraints(version_constraints)?;

        let result = ResolutionResult {
            component_id: component_id.to_string(),
            resolution_order,
            resolved_versions,
            resolution_time: std::time::Instant::now(),
            dependency_conflicts: self.detect_conflicts(component_id)?,
            optional_dependencies: self.collect_optional_dependencies(component_id)?,
        };

        // Cache the result
        {
            let mut cache = self.resolution_cache.write().unwrap();
            cache.insert(component_id.to_string(), result.clone());
        }

        // Update dependency states
        self.update_dependency_states(component_id, &result)?;

        stats.successful_resolutions += 1;
        Ok(result)
    }

    /// Perform topological sort for dependency resolution order
    pub fn topological_sort(&self, component_id: &str) -> SklResult<Vec<String>> {
        let graph = self.dependency_graph.read().unwrap();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        let mut order = Vec::new();

        self.topological_sort_visit(
            &graph,
            component_id,
            &mut visited,
            &mut visiting,
            &mut order,
        )?;

        Ok(order)
    }

    /// Detect circular dependencies
    pub fn detect_circular_dependencies(&self) -> SklResult<Vec<CircularDependency>> {
        let graph = self.dependency_graph.read().unwrap();
        let mut circular_deps = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        for component_id in graph.nodes.keys() {
            if !visited.contains(component_id) {
                if let Err(DependencyError::CircularDependency { cycle }) = self.detect_cycles(
                    &graph,
                    component_id,
                    &mut visited,
                    &mut visiting,
                    &mut Vec::new(),
                ) {
                    circular_deps.push(CircularDependency {
                        cycle_components: cycle,
                        detection_time: std::time::Instant::now(),
                    });
                }
            }
        }

        Ok(circular_deps)
    }

    /// Check dependency compatibility
    pub fn check_compatibility(
        &self,
        component_a: &str,
        component_b: &str,
    ) -> SklResult<CompatibilityResult> {
        let graph = self.dependency_graph.read().unwrap();

        let node_a = graph.nodes.get(component_a).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Component {component_a} not found"))
        })?;
        let node_b = graph.nodes.get(component_b).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Component {component_b} not found"))
        })?;

        let mut compatibility_issues = Vec::new();

        // Check version compatibility
        for dep in &node_a.dependencies {
            if dep.component_type == node_b.component_type
                && !self
                    .version_solver
                    .is_version_compatible(&node_b.version, &dep.version_requirement)
            {
                compatibility_issues.push(CompatibilityIssue {
                    issue_type: CompatibilityIssueType::VersionMismatch,
                    component_a: component_a.to_string(),
                    component_b: component_b.to_string(),
                    description: format!(
                        "Version mismatch: {} requires {} but {} has version {}",
                        component_a, dep.version_requirement, component_b, node_b.version
                    ),
                });
            }
        }

        // Check capability compatibility
        self.check_capability_compatibility(node_a, node_b, &mut compatibility_issues)?;

        Ok(CompatibilityResult {
            compatible: compatibility_issues.is_empty(),
            issues: compatibility_issues,
            compatibility_score: self.calculate_compatibility_score(node_a, node_b),
        })
    }

    /// Register dependency injection provider
    pub fn register_injection_provider<T: 'static>(
        &self,
        provider: Box<dyn DependencyProvider<T>>,
    ) -> SklResult<()> {
        let mut registry = self.injection_registry.write().unwrap();
        registry.register_provider(std::any::TypeId::of::<T>(), provider)?;
        Ok(())
    }

    /// Inject dependencies into a component
    pub fn inject_dependencies(&self, component: &mut dyn PluggableComponent) -> SklResult<()> {
        let registry = self.injection_registry.read().unwrap();

        // Get component's dependency requirements
        let dependencies = component.dependencies();

        for dependency in dependencies {
            if let Some(_provider) = registry.get_provider(&dependency.component_type) {
                // TODO: Implement proper downcasting from Any to DependencyProvider
                // For now, skip injection since this is a placeholder implementation
            } else if !dependency.optional {
                return Err(SklearsError::InvalidInput(format!(
                    "Required dependency provider not found: {}",
                    dependency.component_type
                )));
            }
        }

        Ok(())
    }

    /// Get dependency statistics
    #[must_use]
    pub fn get_statistics(&self) -> DependencyStatistics {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Clear resolution cache
    pub fn clear_cache(&self) {
        let mut cache = self.resolution_cache.write().unwrap();
        cache.clear();
    }

    /// Private helper methods
    fn topological_sort_visit(
        &self,
        graph: &DependencyGraph,
        component_id: &str,
        visited: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) -> SklResult<()> {
        if visiting.contains(component_id) {
            return Err(DependencyError::CircularDependency {
                cycle: vec![component_id.to_string()],
            }
            .into());
        }

        if visited.contains(component_id) {
            return Ok(());
        }

        visiting.insert(component_id.to_string());

        if let Some(edges) = graph.edges.get(component_id) {
            for dependency in edges {
                self.topological_sort_visit(graph, dependency, visited, visiting, order)?;
            }
        }

        visiting.remove(component_id);
        visited.insert(component_id.to_string());
        order.push(component_id.to_string());

        Ok(())
    }

    fn detect_cycles(
        &self,
        graph: &DependencyGraph,
        component_id: &str,
        visited: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> Result<(), DependencyError> {
        if visiting.contains(component_id) {
            let cycle_start = path.iter().position(|id| id == component_id).unwrap_or(0);
            let cycle = path[cycle_start..].to_vec();
            return Err(DependencyError::CircularDependency { cycle });
        }

        if visited.contains(component_id) {
            return Ok(());
        }

        visiting.insert(component_id.to_string());
        path.push(component_id.to_string());

        if let Some(edges) = graph.edges.get(component_id) {
            for dependency in edges {
                self.detect_cycles(graph, dependency, visited, visiting, path)?;
            }
        }

        visiting.remove(component_id);
        visited.insert(component_id.to_string());
        path.pop();

        Ok(())
    }

    fn collect_version_constraints(
        &self,
        components: &[String],
    ) -> SklResult<Vec<VersionConstraint>> {
        let graph = self.dependency_graph.read().unwrap();
        let mut constraints = Vec::new();

        for component_id in components {
            if let Some(node) = graph.nodes.get(component_id) {
                for dependency in &node.dependencies {
                    constraints.push(VersionConstraint {
                        component_type: dependency.component_type.clone(),
                        constraint: dependency.version_requirement.clone(),
                        required_by: component_id.clone(),
                    });
                }
            }
        }

        Ok(constraints)
    }

    fn detect_conflicts(&self, component_id: &str) -> SklResult<Vec<DependencyConflict>> {
        // Implementation for conflict detection
        Ok(Vec::new())
    }

    fn collect_optional_dependencies(&self, component_id: &str) -> SklResult<Vec<String>> {
        let graph = self.dependency_graph.read().unwrap();
        let mut optional_deps = Vec::new();

        if let Some(node) = graph.nodes.get(component_id) {
            for dependency in &node.dependencies {
                if dependency.optional {
                    optional_deps.push(dependency.component_type.clone());
                }
            }
        }

        Ok(optional_deps)
    }

    fn update_dependency_states(
        &self,
        component_id: &str,
        result: &ResolutionResult,
    ) -> SklResult<()> {
        let mut graph = self.dependency_graph.write().unwrap();

        if let Some(node) = graph.nodes.get_mut(component_id) {
            node.dependency_state = if result.dependency_conflicts.is_empty() {
                DependencyState::Resolved
            } else {
                DependencyState::Conflicted
            };
        }

        Ok(())
    }

    fn is_cache_expired(&self, resolution_time: &std::time::Instant) -> bool {
        resolution_time.elapsed() > self.config.cache_expiry_duration
    }

    fn check_capability_compatibility(
        &self,
        node_a: &DependencyNode,
        node_b: &DependencyNode,
        issues: &mut Vec<CompatibilityIssue>,
    ) -> SklResult<()> {
        // Check if required capabilities are satisfied
        for dependency in &node_a.dependencies {
            if dependency.component_type == node_b.component_type {
                // In a real implementation, this would check if node_b provides
                // the capabilities required by dependency.required_capabilities
            }
        }
        Ok(())
    }

    fn calculate_compatibility_score(
        &self,
        node_a: &DependencyNode,
        node_b: &DependencyNode,
    ) -> f64 {
        // Simple compatibility scoring based on version proximity and capability overlap
        1.0 // Placeholder implementation
    }
}

/// Dependency graph representation
#[derive(Debug)]
pub struct DependencyGraph {
    /// Component nodes
    pub nodes: HashMap<String, DependencyNode>,
    /// Dependency edges (`component_id` -> dependencies)
    pub edges: HashMap<String, Vec<String>>,
}

impl DependencyGraph {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, component_id: String, node: DependencyNode) {
        self.nodes.insert(component_id, node);
    }

    pub fn add_edge(&mut self, from: String, to: String) {
        self.edges.entry(from).or_default().push(to);
    }
}

/// Dependency node in the graph
#[derive(Debug, Clone)]
pub struct DependencyNode {
    /// Component identifier
    pub component_id: String,
    /// Component type
    pub component_type: String,
    /// Component version
    pub version: String,
    /// Component dependencies
    pub dependencies: Vec<ComponentDependency>,
    /// Resolved dependency mappings
    pub resolved_dependencies: HashMap<String, String>,
    /// Current dependency state
    pub dependency_state: DependencyState,
}

/// Dependency resolution states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyState {
    /// Dependencies not yet resolved
    Unresolved,
    /// Dependencies resolved successfully
    Resolved,
    /// Dependency conflicts detected
    Conflicted,
    /// Dependencies partially resolved
    PartiallyResolved,
}

/// Version constraint solver
pub struct VersionConstraintSolver {
    /// Supported constraint operators
    constraint_operators: HashMap<String, Box<dyn Fn(&str, &str) -> bool + Send + Sync>>,
}

impl VersionConstraintSolver {
    #[must_use]
    pub fn new() -> Self {
        let mut operators = HashMap::new();

        // Add constraint operators
        operators.insert(
            "=".to_string(),
            Box::new(|version: &str, constraint: &str| version == constraint)
                as Box<dyn Fn(&str, &str) -> bool + Send + Sync>,
        );

        operators.insert(
            ">=".to_string(),
            Box::new(|version: &str, constraint: &str| {
                Self::compare_versions(version, constraint) >= 0
            }) as Box<dyn Fn(&str, &str) -> bool + Send + Sync>,
        );

        Self {
            constraint_operators: operators,
        }
    }

    pub fn solve_constraints(
        &self,
        constraints: Vec<VersionConstraint>,
    ) -> SklResult<HashMap<String, String>> {
        let mut resolved_versions = HashMap::new();
        let mut component_constraints: HashMap<String, Vec<String>> = HashMap::new();

        // Group constraints by component type
        for constraint in constraints {
            component_constraints
                .entry(constraint.component_type.clone())
                .or_default()
                .push(constraint.constraint);
        }

        // Resolve versions for each component type
        for (component_type, version_constraints) in component_constraints {
            let resolved_version = self.resolve_version_constraints(version_constraints)?;
            resolved_versions.insert(component_type, resolved_version);
        }

        Ok(resolved_versions)
    }

    #[must_use]
    pub fn is_version_compatible(&self, version: &str, constraint: &str) -> bool {
        // Simple version compatibility check
        // In a real implementation, this would parse constraint operators
        VersionConstraintSolver::compare_versions(version, constraint) >= 0
    }

    fn resolve_version_constraints(&self, constraints: Vec<String>) -> SklResult<String> {
        // For now, return the highest version constraint
        // In a real implementation, this would solve the constraint satisfaction problem
        Ok(constraints
            .into_iter()
            .max()
            .unwrap_or_else(|| "1.0.0".to_string()))
    }

    fn compare_versions(version_a: &str, version_b: &str) -> i32 {
        let parts_a: Vec<u32> = version_a
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let parts_b: Vec<u32> = version_b
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();

        for i in 0..std::cmp::max(parts_a.len(), parts_b.len()) {
            let a = parts_a.get(i).copied().unwrap_or(0);
            let b = parts_b.get(i).copied().unwrap_or(0);

            match a.cmp(&b) {
                std::cmp::Ordering::Less => return -1,
                std::cmp::Ordering::Greater => return 1,
                std::cmp::Ordering::Equal => continue,
            }
        }

        0
    }
}

impl std::fmt::Debug for VersionConstraintSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VersionConstraintSolver")
            .field(
                "constraint_operators",
                &format!("<{} operators>", self.constraint_operators.len()),
            )
            .finish()
    }
}

/// Version constraint
#[derive(Debug, Clone)]
pub struct VersionConstraint {
    /// Component type
    pub component_type: String,
    /// Version constraint string
    pub constraint: String,
    /// Component that requires this constraint
    pub required_by: String,
}

/// Dependency injection registry
#[derive(Debug)]
pub struct DependencyInjectionRegistry {
    /// Registered providers by type
    providers: HashMap<std::any::TypeId, Box<dyn std::any::Any + Send + Sync>>,
    /// Provider metadata
    provider_metadata: HashMap<String, ProviderMetadata>,
}

impl DependencyInjectionRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            provider_metadata: HashMap::new(),
        }
    }

    pub fn register_provider<T: 'static>(
        &mut self,
        type_id: std::any::TypeId,
        provider: Box<dyn DependencyProvider<T>>,
    ) -> SklResult<()> {
        self.providers.insert(type_id, Box::new(provider));
        Ok(())
    }

    #[must_use]
    pub fn get_provider(&self, component_type: &str) -> Option<&dyn std::any::Any> {
        // This would look up providers by component type
        // Placeholder implementation
        None
    }
}

/// Dependency provider trait
pub trait DependencyProvider<T>: Send + Sync {
    /// Inject dependency into component
    fn inject(&self, component: &mut dyn PluggableComponent) -> SklResult<()>;

    /// Get provider metadata
    fn metadata(&self) -> ProviderMetadata;
}

/// Provider metadata
#[derive(Debug, Clone)]
pub struct ProviderMetadata {
    /// Provider name
    pub name: String,
    /// Provided type
    pub provided_type: String,
    /// Provider version
    pub version: String,
}

/// Dependency resolution result
#[derive(Debug, Clone)]
pub struct ResolutionResult {
    /// Component identifier
    pub component_id: String,
    /// Resolution order
    pub resolution_order: Vec<String>,
    /// Resolved versions
    pub resolved_versions: HashMap<String, String>,
    /// Resolution timestamp
    pub resolution_time: std::time::Instant,
    /// Detected conflicts
    pub dependency_conflicts: Vec<DependencyConflict>,
    /// Optional dependencies
    pub optional_dependencies: Vec<String>,
}

/// Dependency conflict
#[derive(Debug, Clone)]
pub struct DependencyConflict {
    /// Conflicting components
    pub components: Vec<String>,
    /// Conflict type
    pub conflict_type: ConflictType,
    /// Conflict description
    pub description: String,
}

/// Conflict types
#[derive(Debug, Clone)]
pub enum ConflictType {
    /// Version constraint conflict
    VersionConflict,
    /// Capability conflict
    CapabilityConflict,
    /// Resource conflict
    ResourceConflict,
    /// Circular dependency
    CircularDependency,
}

/// Circular dependency detection result
#[derive(Debug, Clone)]
pub struct CircularDependency {
    /// Components in the cycle
    pub cycle_components: Vec<String>,
    /// Detection timestamp
    pub detection_time: std::time::Instant,
}

/// Compatibility check result
#[derive(Debug, Clone)]
pub struct CompatibilityResult {
    /// Whether components are compatible
    pub compatible: bool,
    /// Compatibility issues
    pub issues: Vec<CompatibilityIssue>,
    /// Compatibility score (0.0 to 1.0)
    pub compatibility_score: f64,
}

/// Compatibility issue
#[derive(Debug, Clone)]
pub struct CompatibilityIssue {
    /// Issue type
    pub issue_type: CompatibilityIssueType,
    /// First component
    pub component_a: String,
    /// Second component
    pub component_b: String,
    /// Issue description
    pub description: String,
}

/// Compatibility issue types
#[derive(Debug, Clone)]
pub enum CompatibilityIssueType {
    /// Version mismatch
    VersionMismatch,
    /// Missing capability
    MissingCapability,
    /// Resource conflict
    ResourceConflict,
    /// Configuration conflict
    ConfigurationConflict,
}

/// Dependency resolution configuration
#[derive(Debug, Clone)]
pub struct DependencyResolutionConfig {
    /// Enable circular dependency detection
    pub enable_circular_detection: bool,
    /// Maximum resolution depth
    pub max_resolution_depth: usize,
    /// Cache expiry duration
    pub cache_expiry_duration: std::time::Duration,
    /// Allow version downgrades
    pub allow_version_downgrades: bool,
    /// Strict version matching
    pub strict_version_matching: bool,
}

impl Default for DependencyResolutionConfig {
    fn default() -> Self {
        Self {
            enable_circular_detection: true,
            max_resolution_depth: 50,
            cache_expiry_duration: std::time::Duration::from_secs(300),
            allow_version_downgrades: false,
            strict_version_matching: false,
        }
    }
}

/// Dependency statistics
#[derive(Debug, Clone)]
pub struct DependencyStatistics {
    /// Total components in dependency graph
    pub total_components: u64,
    /// Total dependencies
    pub total_dependencies: u64,
    /// Resolution attempts
    pub resolution_attempts: u64,
    /// Successful resolutions
    pub successful_resolutions: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Circular dependencies detected
    pub circular_dependencies_detected: u64,
    /// Average resolution time
    pub average_resolution_time: std::time::Duration,
}

impl DependencyStatistics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_components: 0,
            total_dependencies: 0,
            resolution_attempts: 0,
            successful_resolutions: 0,
            cache_hits: 0,
            cache_misses: 0,
            circular_dependencies_detected: 0,
            average_resolution_time: std::time::Duration::from_secs(0),
        }
    }

    /// Get cache hit rate
    #[must_use]
    pub fn cache_hit_rate(&self) -> f64 {
        if self.resolution_attempts == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.resolution_attempts as f64
        }
    }

    /// Get resolution success rate
    #[must_use]
    pub fn resolution_success_rate(&self) -> f64 {
        if self.resolution_attempts == 0 {
            0.0
        } else {
            self.successful_resolutions as f64 / self.resolution_attempts as f64
        }
    }
}

/// Dependency management errors
#[derive(Debug, Error)]
pub enum DependencyError {
    #[error("Circular dependency detected: {cycle:?}")]
    CircularDependency { cycle: Vec<String> },

    #[error("Version constraint conflict: {0}")]
    VersionConstraintConflict(String),

    #[error("Dependency not found: {0}")]
    DependencyNotFound(String),

    #[error("Resolution failed: {0}")]
    ResolutionFailed(String),

    #[error("Injection failed: {0}")]
    InjectionFailed(String),
}

impl From<DependencyError> for SklearsError {
    fn from(err: DependencyError) -> Self {
        SklearsError::InvalidInput(err.to_string())
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for VersionConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DependencyInjectionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DependencyStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_resolver_creation() {
        let resolver = DependencyResolver::new();
        let stats = resolver.get_statistics();
        assert_eq!(stats.total_components, 0);
        assert_eq!(stats.total_dependencies, 0);
    }

    #[test]
    fn test_version_constraint_solver() {
        let solver = VersionConstraintSolver::new();

        assert!(solver.is_version_compatible("1.2.0", "1.0.0"));
        assert!(!solver.is_version_compatible("1.0.0", "1.2.0"));
        assert!(solver.is_version_compatible("1.0.0", "1.0.0"));
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();

        let node = DependencyNode {
            component_id: "comp1".to_string(),
            component_type: "type1".to_string(),
            version: "1.0.0".to_string(),
            dependencies: Vec::new(),
            resolved_dependencies: HashMap::new(),
            dependency_state: DependencyState::Unresolved,
        };

        graph.add_node("comp1".to_string(), node);
        graph.add_edge("comp1".to_string(), "comp2".to_string());

        assert!(graph.nodes.contains_key("comp1"));
        assert_eq!(graph.edges.get("comp1").unwrap().len(), 1);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let resolver = DependencyResolver::new();

        // Add components with circular dependencies
        let dep1 = ComponentDependency {
            component_type: "comp2".to_string(),
            version_requirement: "1.0.0".to_string(),
            optional: false,
            required_capabilities: Vec::new(),
        };

        let dep2 = ComponentDependency {
            component_type: "comp1".to_string(),
            version_requirement: "1.0.0".to_string(),
            optional: false,
            required_capabilities: Vec::new(),
        };

        resolver
            .add_component_dependencies("comp1", "type1", "1.0.0", vec![dep1])
            .unwrap();
        resolver
            .add_component_dependencies("comp2", "type2", "1.0.0", vec![dep2])
            .unwrap();

        let circular_deps = resolver.detect_circular_dependencies().unwrap();
        assert!(!circular_deps.is_empty());
    }

    #[test]
    fn test_topological_sort() {
        let resolver = DependencyResolver::new();

        // Add components in dependency order: comp3 -> comp2 -> comp1
        let dep1 = ComponentDependency {
            component_type: "comp2".to_string(),
            version_requirement: "1.0.0".to_string(),
            optional: false,
            required_capabilities: Vec::new(),
        };

        let dep2 = ComponentDependency {
            component_type: "comp3".to_string(),
            version_requirement: "1.0.0".to_string(),
            optional: false,
            required_capabilities: Vec::new(),
        };

        resolver
            .add_component_dependencies("comp1", "type1", "1.0.0", vec![dep1])
            .unwrap();
        resolver
            .add_component_dependencies("comp2", "type2", "1.0.0", vec![dep2])
            .unwrap();
        resolver
            .add_component_dependencies("comp3", "type3", "1.0.0", vec![])
            .unwrap();

        let order = resolver.topological_sort("comp1").unwrap();

        // Should be in order: comp3, comp2, comp1
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], "comp3");
        assert_eq!(order[1], "comp2");
        assert_eq!(order[2], "comp1");
    }
}
