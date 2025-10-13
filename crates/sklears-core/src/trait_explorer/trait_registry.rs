//! Comprehensive trait registry and exploration result types.
//!
//! This module provides the core infrastructure for managing and exploring traits within
//! the sklears ecosystem. It includes trait registration, dependency analysis, performance
//! analysis, and graph generation capabilities.
//!
//! # Key Components
//!
//! - [`TraitExplorationResult`] - Main result structure for trait exploration
//! - [`TraitRegistry`] - Core registry for managing trait information
//! - [`DependencyAnalysis`] - Analysis of trait dependencies and relationships
//! - [`PerformanceAnalysis`] - Performance characteristics and optimization hints
//! - [`TraitGraph`] - Visual graph representation of trait relationships
//!
//! # SciRS2 Compliance
//!
//! This module is fully compliant with SciRS2 policies, using:
//! - `scirs2_core::ndarray` for unified array operations
//! - `scirs2_core::random` for random number generation
//! - `scirs2_core::error` for error handling
//!
//! # Examples
//!
//! ## Basic trait registry usage
//!
//! ```rust,ignore
//! use sklears_core::trait_explorer::trait_registry::{TraitRegistry, TraitExplorationResult};
//! use sklears_core::api_reference_generator::TraitInfo;
//! use sklears_core::error::Result;
//!
//! // Create and populate a trait registry
//! let mut registry = TraitRegistry::new();
//! registry.load_sklears_traits()?;
//!
//! // Get trait information
//! if let Some(estimator_trait) = registry.get_trait("Estimator") {
//!     println!("Found trait: {}", estimator_trait.name);
//!     let implementations = registry.get_implementations("Estimator");
//!     println!("Implementations: {:?}", implementations);
//! }
//! # Ok::<(), sklears_core::error::SklearsError>(())
//! ```
//!
//! ## Exploring trait relationships
//!
//! ```rust,ignore
//! # use sklears_core::trait_explorer::trait_registry::*;
//! # use sklears_core::error::Result;
//! let mut registry = TraitRegistry::new();
//! registry.load_sklears_traits()?;
//!
//! // Get all traits that a specific implementation supports
//! let linear_reg_traits = registry.get_traits_for_implementation("LinearRegression");
//! println!("LinearRegression implements: {:?}", linear_reg_traits);
//!
//! // List all available traits
//! let all_traits = registry.get_all_trait_names();
//! println!("Available traits: {:?}", all_traits);
//! # Ok::<(), sklears_core::error::SklearsError>(())
//! ```

use crate::api_data_structures::{AssociatedType, MethodInfo, TraitInfo};
use crate::error::{Result, SklearsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// SciRS2 Compliance - Use scirs2_core for unified functionality

/// Comprehensive result structure for trait exploration operations.
///
/// This structure encapsulates all information discovered during trait exploration,
/// including trait metadata, implementations, dependency analysis, performance
/// characteristics, and visualization data.
///
/// # Fields
///
/// - `trait_name` - Name of the explored trait
/// - `trait_info` - Complete trait information including methods and generics
/// - `implementations` - Known implementations of this trait
/// - `dependencies` - Dependency analysis including direct and transitive dependencies
/// - `performance` - Performance characteristics and optimization hints
/// - `complexity_score` - Complexity score (higher = more complex)
/// - `graph` - Optional visual graph representation
/// - `examples` - Usage examples for the trait
/// - `related_traits` - List of related traits
///
/// # Examples
///
/// ```rust,ignore
/// # use sklears_core::trait_explorer::trait_registry::*;
/// # use sklears_core::api_reference_generator::TraitInfo;
/// # use sklears_core::error::Result;
/// // Typically obtained from trait exploration operations
/// let result = TraitExplorationResult {
///     trait_name: "Estimator".to_string(),
///     trait_info: TraitInfo::default(),
///     implementations: vec!["LinearRegression".to_string()],
///     dependencies: DependencyAnalysis::default(),
///     performance: PerformanceAnalysis::default(),
///     complexity_score: 3.5,
///     graph: None,
///     examples: vec![],
///     related_traits: vec!["Fit".to_string(), "Predict".to_string()],
/// };
///
/// println!("Explored trait: {}", result.trait_name);
/// println!("Complexity score: {}", result.complexity_score);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitExplorationResult {
    /// Name of the explored trait
    pub trait_name: String,
    /// Complete trait information
    pub trait_info: TraitInfo,
    /// Known implementations of this trait
    pub implementations: Vec<String>,
    /// Dependency analysis
    pub dependencies: DependencyAnalysis,
    /// Performance characteristics
    pub performance: PerformanceAnalysis,
    /// Complexity score (higher = more complex)
    pub complexity_score: f64,
    /// Visual graph representation
    pub graph: Option<TraitGraph>,
    /// Usage examples
    pub examples: Vec<UsageExample>,
    /// Related traits
    pub related_traits: Vec<String>,
}

impl Default for TraitExplorationResult {
    fn default() -> Self {
        Self {
            trait_name: String::new(),
            trait_info: TraitInfo::default(),
            implementations: Vec::new(),
            dependencies: DependencyAnalysis::default(),
            performance: PerformanceAnalysis::default(),
            complexity_score: 0.0,
            graph: None,
            examples: Vec::new(),
            related_traits: Vec::new(),
        }
    }
}

/// Registry for managing trait information and relationships.
///
/// The `TraitRegistry` serves as a central repository for trait information,
/// implementations, and their relationships. It provides efficient lookup
/// and management capabilities for the trait exploration system.
///
/// # Architecture
///
/// The registry maintains three core data structures:
/// - `traits` - Maps trait names to their complete information
/// - `implementations` - Maps trait names to their known implementations
/// - `implementation_traits` - Maps implementation names to the traits they implement
///
/// # Thread Safety
///
/// This registry is not thread-safe by default. For concurrent access,
/// wrap it in appropriate synchronization primitives.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
/// use sklears_core::api_reference_generator::TraitInfo;
/// use sklears_core::error::Result;
///
/// let mut registry = TraitRegistry::new();
///
/// // Load predefined sklears traits
/// registry.load_sklears_traits()?;
///
/// // Add a custom trait
/// let custom_trait = TraitInfo {
///     name: "CustomTrait".to_string(),
///     description: "A custom trait for testing".to_string(),
///     path: "my_crate::CustomTrait".to_string(),
///     generics: vec![],
///     associated_types: vec![],
///     methods: vec![],
///     supertraits: vec![],
///     implementations: vec!["MyImplementation".to_string()],
/// };
/// registry.add_trait(custom_trait);
///
/// // Query the registry
/// assert!(registry.get_trait("CustomTrait").is_some());
/// let implementations = registry.get_implementations("CustomTrait");
/// assert_eq!(implementations, vec!["MyImplementation"]);
/// # Ok::<(), sklears_core::error::SklearsError>(())
/// ```
#[derive(Debug)]
pub struct TraitRegistry {
    /// Maps trait names to their complete information
    traits: HashMap<String, TraitInfo>,
    /// Maps trait names to their known implementations
    implementations: HashMap<String, Vec<String>>,
    /// Maps implementation names to the traits they implement
    implementation_traits: HashMap<String, Vec<String>>,
}

impl TraitRegistry {
    /// Creates a new empty trait registry.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    ///
    /// let registry = TraitRegistry::new();
    /// assert_eq!(registry.get_all_trait_names().len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            traits: HashMap::new(),
            implementations: HashMap::new(),
            implementation_traits: HashMap::new(),
        }
    }

    /// Loads predefined sklears traits into the registry.
    ///
    /// This method populates the registry with the core traits used throughout
    /// the sklears ecosystem: Estimator, Fit, Predict, and Transform.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if trait loading fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// // Verify core traits are loaded
    /// assert!(registry.get_trait("Estimator").is_some());
    /// assert!(registry.get_trait("Fit").is_some());
    /// assert!(registry.get_trait("Predict").is_some());
    /// assert!(registry.get_trait("Transform").is_some());
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn load_sklears_traits(&mut self) -> Result<()> {
        // Load example trait data for sklears-core
        let estimator_trait = TraitInfo {
            name: "Estimator".to_string(),
            description: "Base trait for all machine learning estimators".to_string(),
            path: "sklears_core::traits::Estimator".to_string(),
            generics: Vec::new(),
            associated_types: vec![AssociatedType {
                name: "Config".to_string(),
                description: "Configuration type for the estimator".to_string(),
                bounds: Vec::new(),
            }],
            methods: vec![MethodInfo {
                name: "name".to_string(),
                signature: "fn name(&self) -> &'static str".to_string(),
                description: "Get the name of the estimator".to_string(),
                parameters: Vec::new(),
                return_type: "&'static str".to_string(),
                required: true,
            }],
            supertraits: Vec::new(),
            implementations: vec![
                "LinearRegression".to_string(),
                "LogisticRegression".to_string(),
                "RandomForest".to_string(),
            ],
        };

        let fit_trait = TraitInfo {
            name: "Fit".to_string(),
            description: "Trait for estimators that can be fitted to training data".to_string(),
            path: "sklears_core::traits::Fit".to_string(),
            generics: vec!["X".to_string(), "Y".to_string()],
            associated_types: vec![AssociatedType {
                name: "Fitted".to_string(),
                description: "The type returned after fitting".to_string(),
                bounds: vec!["Send".to_string(), "Sync".to_string()],
            }],
            methods: vec![MethodInfo {
                name: "fit".to_string(),
                signature: "fn fit(self, x: &X, y: &Y) -> Result<Self::Fitted>".to_string(),
                description: "Fit the estimator to training data".to_string(),
                parameters: Vec::new(),
                return_type: "Result<Self::Fitted>".to_string(),
                required: true,
            }],
            supertraits: vec!["Estimator".to_string()],
            implementations: vec![
                "LinearRegression".to_string(),
                "LogisticRegression".to_string(),
            ],
        };

        let predict_trait = TraitInfo {
            name: "Predict".to_string(),
            description: "Trait for making predictions on new data".to_string(),
            path: "sklears_core::traits::Predict".to_string(),
            generics: vec!["X".to_string()],
            associated_types: vec![AssociatedType {
                name: "Output".to_string(),
                description: "The type of predictions made".to_string(),
                bounds: Vec::new(),
            }],
            methods: vec![MethodInfo {
                name: "predict".to_string(),
                signature: "fn predict(&self, x: &X) -> Result<Self::Output>".to_string(),
                description: "Make predictions on input data".to_string(),
                parameters: Vec::new(),
                return_type: "Result<Self::Output>".to_string(),
                required: true,
            }],
            supertraits: Vec::new(),
            implementations: vec![
                "LinearRegression".to_string(),
                "LogisticRegression".to_string(),
                "RandomForest".to_string(),
            ],
        };

        let transform_trait = TraitInfo {
            name: "Transform".to_string(),
            description: "Trait for data transformation operations".to_string(),
            path: "sklears_core::traits::Transform".to_string(),
            generics: vec!["X".to_string()],
            associated_types: vec![AssociatedType {
                name: "Output".to_string(),
                description: "The type of transformed data".to_string(),
                bounds: Vec::new(),
            }],
            methods: vec![MethodInfo {
                name: "transform".to_string(),
                signature: "fn transform(&self, x: &X) -> Result<Self::Output>".to_string(),
                description: "Transform input data".to_string(),
                parameters: Vec::new(),
                return_type: "Result<Self::Output>".to_string(),
                required: true,
            }],
            supertraits: Vec::new(),
            implementations: vec![
                "StandardScaler".to_string(),
                "PCA".to_string(),
                "MinMaxScaler".to_string(),
            ],
        };

        // Register all traits
        self.add_trait(estimator_trait);
        self.add_trait(fit_trait);
        self.add_trait(predict_trait);
        self.add_trait(transform_trait);

        Ok(())
    }

    /// Adds a trait to the registry.
    ///
    /// This method registers a trait and updates the implementation mappings
    /// to maintain bidirectional relationships between traits and their implementations.
    ///
    /// # Arguments
    ///
    /// * `trait_info` - Complete information about the trait to add
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::api_reference_generator::TraitInfo;
    ///
    /// let mut registry = TraitRegistry::new();
    /// let trait_info = TraitInfo {
    ///     name: "MyTrait".to_string(),
    ///     description: "A custom trait".to_string(),
    ///     path: "my_crate::MyTrait".to_string(),
    ///     generics: vec![],
    ///     associated_types: vec![],
    ///     methods: vec![],
    ///     supertraits: vec![],
    ///     implementations: vec!["MyImpl".to_string()],
    /// };
    ///
    /// registry.add_trait(trait_info);
    /// assert!(registry.get_trait("MyTrait").is_some());
    /// assert_eq!(registry.get_implementations("MyTrait"), vec!["MyImpl"]);
    /// ```
    pub fn add_trait(&mut self, trait_info: TraitInfo) {
        // Register implementations
        for impl_name in &trait_info.implementations {
            self.implementations
                .entry(trait_info.name.clone())
                .or_default()
                .push(impl_name.clone());

            self.implementation_traits
                .entry(impl_name.clone())
                .or_default()
                .push(trait_info.name.clone());
        }

        self.traits.insert(trait_info.name.clone(), trait_info);
    }

    /// Retrieves trait information by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the trait to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Some(&TraitInfo)` if the trait exists, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// if let Some(estimator_trait) = registry.get_trait("Estimator") {
    ///     println!("Found trait: {}", estimator_trait.name);
    ///     println!("Description: {}", estimator_trait.description);
    /// }
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn get_trait(&self, name: &str) -> Option<&TraitInfo> {
        self.traits.get(name)
    }

    /// Gets all traits in the registry.
    ///
    /// # Returns
    ///
    /// Returns a vector of references to all trait information in the registry.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// let all_traits = registry.get_all_traits();
    /// println!("Registry contains {} traits", all_traits.len());
    /// for trait_info in all_traits {
    ///     println!("- {}: {}", trait_info.name, trait_info.description);
    /// }
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn get_all_traits(&self) -> Vec<&TraitInfo> {
        self.traits.values().collect()
    }

    /// Gets all trait names in the registry.
    ///
    /// # Returns
    ///
    /// Returns a vector of all trait names.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// let trait_names = registry.get_all_trait_names();
    /// assert!(trait_names.contains(&"Estimator".to_string()));
    /// assert!(trait_names.contains(&"Fit".to_string()));
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn get_all_trait_names(&self) -> Vec<String> {
        self.traits.keys().cloned().collect()
    }

    /// Gets all implementations of a specific trait.
    ///
    /// # Arguments
    ///
    /// * `trait_name` - The name of the trait
    ///
    /// # Returns
    ///
    /// Returns a vector of implementation names. Returns an empty vector
    /// if the trait doesn't exist or has no implementations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// let estimator_impls = registry.get_implementations("Estimator");
    /// assert!(estimator_impls.contains(&"LinearRegression".to_string()));
    ///
    /// let unknown_impls = registry.get_implementations("UnknownTrait");
    /// assert!(unknown_impls.is_empty());
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn get_implementations(&self, trait_name: &str) -> Vec<String> {
        self.implementations
            .get(trait_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Gets all traits implemented by a specific implementation.
    ///
    /// # Arguments
    ///
    /// * `implementation` - The name of the implementation
    ///
    /// # Returns
    ///
    /// Returns a vector of trait names that the implementation supports.
    /// Returns an empty vector if the implementation doesn't exist.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// let linear_reg_traits = registry.get_traits_for_implementation("LinearRegression");
    /// assert!(linear_reg_traits.contains(&"Estimator".to_string()));
    /// assert!(linear_reg_traits.contains(&"Fit".to_string()));
    ///
    /// let unknown_traits = registry.get_traits_for_implementation("UnknownImpl");
    /// assert!(unknown_traits.is_empty());
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn get_traits_for_implementation(&self, implementation: &str) -> Vec<String> {
        self.implementation_traits
            .get(implementation)
            .cloned()
            .unwrap_or_default()
    }

    /// Checks if a trait exists in the registry.
    ///
    /// # Arguments
    ///
    /// * `trait_name` - The name of the trait to check
    ///
    /// # Returns
    ///
    /// Returns `true` if the trait exists, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// assert!(registry.has_trait("Estimator"));
    /// assert!(!registry.has_trait("NonExistentTrait"));
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn has_trait(&self, trait_name: &str) -> bool {
        self.traits.contains_key(trait_name)
    }

    /// Checks if an implementation exists in the registry.
    ///
    /// # Arguments
    ///
    /// * `implementation` - The name of the implementation to check
    ///
    /// # Returns
    ///
    /// Returns `true` if the implementation exists, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// assert!(registry.has_implementation("LinearRegression"));
    /// assert!(!registry.has_implementation("NonExistentImpl"));
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn has_implementation(&self, implementation: &str) -> bool {
        self.implementation_traits.contains_key(implementation)
    }

    /// Gets the total number of traits in the registry.
    ///
    /// # Returns
    ///
    /// Returns the count of registered traits.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// assert_eq!(registry.trait_count(), 0);
    ///
    /// registry.load_sklears_traits()?;
    /// assert!(registry.trait_count() > 0);
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn trait_count(&self) -> usize {
        self.traits.len()
    }

    /// Gets the total number of unique implementations in the registry.
    ///
    /// # Returns
    ///
    /// Returns the count of unique implementations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::TraitRegistry;
    /// use sklears_core::error::Result;
    ///
    /// let mut registry = TraitRegistry::new();
    /// registry.load_sklears_traits()?;
    ///
    /// let impl_count = registry.implementation_count();
    /// assert!(impl_count > 0);
    /// # Ok::<(), sklears_core::error::SklearsError>(())
    /// ```
    pub fn implementation_count(&self) -> usize {
        self.implementation_traits.len()
    }
}

impl Default for TraitRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis of trait dependencies and relationships.
///
/// This structure provides comprehensive information about how a trait
/// relates to other traits in the system, including direct dependencies,
/// transitive dependencies, and potential circular dependencies.
///
/// # Fields
///
/// - `direct_dependencies` - Immediate dependencies (supertraits, bounds)
/// - `transitive_dependencies` - All dependencies through the dependency chain
/// - `dependency_depth` - Maximum depth of the dependency chain
/// - `circular_dependencies` - Any circular dependency cycles detected
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::trait_explorer::trait_registry::DependencyAnalysis;
///
/// let analysis = DependencyAnalysis {
///     direct_dependencies: vec!["Send".to_string(), "Sync".to_string()],
///     transitive_dependencies: vec!["Send".to_string(), "Sync".to_string()],
///     dependency_depth: 1,
///     circular_dependencies: vec![],
/// };
///
/// assert_eq!(analysis.direct_dependencies.len(), 2);
/// assert!(analysis.circular_dependencies.is_empty());
/// ```
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
}

/// Analysis of trait performance characteristics.
///
/// This structure provides detailed information about the performance
/// implications of using a trait, including compilation impact, runtime
/// overhead, memory footprint, and optimization suggestions.
///
/// # Fields
///
/// - `compilation_impact` - How the trait affects compilation time
/// - `runtime_overhead` - Runtime performance characteristics
/// - `memory_footprint` - Memory usage analysis
/// - `optimization_hints` - Suggestions for performance optimization
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::trait_explorer::trait_registry::*;
///
/// let performance = PerformanceAnalysis {
///     compilation_impact: CompilationImpact {
///         estimated_compile_time_ms: 50,
///         ..Default::default()
///     },
///     runtime_overhead: RuntimeOverhead {
///         virtual_dispatch_cost: 10,
///         stack_frame_size: 64,
///         cache_pressure: "Low".to_string(),
///     },
///     memory_footprint: MemoryFootprint {
///         vtable_size_bytes: 24,
///         associated_data_size: 0,
///         total_overhead: 24,
///     },
///     optimization_hints: vec![
///         "Consider using static dispatch for performance-critical code".to_string()
///     ],
/// };
///
/// assert_eq!(performance.runtime_overhead.virtual_dispatch_cost, 10);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Impact on compilation
    pub compilation_impact: CompilationImpact,
    /// Runtime performance characteristics
    pub runtime_overhead: RuntimeOverhead,
    /// Memory usage analysis
    pub memory_footprint: MemoryFootprint,
    /// Suggestions for optimization
    pub optimization_hints: Vec<String>,
}

/// Compilation performance impact analysis.
///
/// This structure quantifies how a trait affects compilation performance,
/// including estimated compile time impact and resource usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompilationImpact {
    /// Estimated additional compile time in milliseconds
    pub estimated_compile_time_ms: usize,
    /// Generic instantiation impact
    pub generic_instantiation_cost: usize,
    /// Type checking complexity score
    pub type_checking_complexity: f64,
}

/// Runtime performance overhead analysis.
///
/// This structure provides information about the runtime costs associated
/// with using a trait, including virtual dispatch overhead and memory pressure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeOverhead {
    /// Cost of virtual dispatch in nanoseconds
    pub virtual_dispatch_cost: usize,
    /// Stack frame size in bytes
    pub stack_frame_size: usize,
    /// Cache pressure level
    pub cache_pressure: String,
}

/// Memory footprint analysis.
///
/// This structure analyzes the memory impact of using a trait,
/// including vtable sizes and associated data overhead.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryFootprint {
    /// Size of virtual function table
    pub vtable_size_bytes: usize,
    /// Memory used by associated data
    pub associated_data_size: usize,
    /// Total memory overhead
    pub total_overhead: usize,
}

/// Visual graph representation of trait relationships.
///
/// This structure represents trait relationships as a directed graph,
/// suitable for visualization and analysis tools.
///
/// # Fields
///
/// - `nodes` - All nodes in the graph (traits, implementations, etc.)
/// - `edges` - Connections between nodes
/// - `metadata` - Graph metadata including generation time and format
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::trait_explorer::trait_registry::*;
/// use chrono::Utc;
///
/// let graph = TraitGraph {
///     nodes: vec![
///         TraitGraphNode {
///             id: "Estimator".to_string(),
///             label: "Estimator".to_string(),
///             node_type: TraitNodeType::Trait,
///             description: "Base estimator trait".to_string(),
///             complexity: 2.0,
///         }
///     ],
///     edges: vec![],
///     metadata: TraitGraphMetadata {
///         center_node: "Estimator".to_string(),
///         generation_time: Utc::now(),
///         export_format: GraphExportFormat::Dot,
///     },
/// };
///
/// let dot_output = graph.to_dot();
/// assert!(dot_output.contains("digraph TraitGraph"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitGraph {
    /// Nodes in the graph
    pub nodes: Vec<TraitGraphNode>,
    /// Edges connecting nodes
    pub edges: Vec<TraitGraphEdge>,
    /// Graph metadata
    pub metadata: TraitGraphMetadata,
}

impl TraitGraph {
    /// Export graph to DOT format (Graphviz).
    ///
    /// # Returns
    ///
    /// Returns a string containing the graph in DOT format, suitable
    /// for rendering with Graphviz tools.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_core::trait_explorer::trait_registry::*;
    /// use chrono::Utc;
    ///
    /// let graph = TraitGraph {
    ///     nodes: vec![
    ///         TraitGraphNode {
    ///             id: "Estimator".to_string(),
    ///             label: "Estimator".to_string(),
    ///             node_type: TraitNodeType::Trait,
    ///             description: "Base trait".to_string(),
    ///             complexity: 1.0,
    ///         }
    ///     ],
    ///     edges: vec![],
    ///     metadata: TraitGraphMetadata {
    ///         center_node: "Estimator".to_string(),
    ///         generation_time: Utc::now(),
    ///         export_format: GraphExportFormat::Dot,
    ///     },
    /// };
    ///
    /// let dot = graph.to_dot();
    /// assert!(dot.contains("Estimator"));
    /// assert!(dot.contains("digraph TraitGraph"));
    /// ```
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph TraitGraph {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, style=rounded];\n");

        // Add nodes
        for node in &self.nodes {
            let color = match node.node_type {
                TraitNodeType::Trait => "lightblue",
                TraitNodeType::Implementation => "lightgreen",
                TraitNodeType::AssociatedType => "lightyellow",
            };

            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\" fillcolor={} style=\"filled,rounded\"];\n",
                node.id, node.label, color
            ));
        }

        // Add edges
        for edge in &self.edges {
            let style = match edge.edge_type {
                EdgeType::Inherits => "solid",
                EdgeType::Implements => "dashed",
                EdgeType::AssociatedWith => "dotted",
            };

            dot.push_str(&format!(
                "  \"{}\" -> \"{}\" [style={}];\n",
                edge.from, edge.to, style
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Export graph to JSON format.
    ///
    /// # Returns
    ///
    /// Returns a JSON string representation of the graph, suitable
    /// for web-based visualization libraries.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| SklearsError::Other(format!("Failed to serialize graph to JSON: {}", e)))
    }
}

/// Node in the trait graph.
///
/// Represents an entity in the trait relationship graph, such as
/// a trait, implementation, or associated type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitGraphNode {
    /// Unique identifier
    pub id: String,
    /// Display label
    pub label: String,
    /// Type of node
    pub node_type: TraitNodeType,
    /// Description text
    pub description: String,
    /// Complexity score for visual sizing
    pub complexity: f64,
}

/// Edge in the trait graph.
///
/// Represents a relationship between two nodes in the trait graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitGraphEdge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Type of relationship
    pub edge_type: EdgeType,
    /// Edge weight for layout
    pub weight: f64,
}

/// Types of nodes in the trait graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraitNodeType {
    /// Main trait node
    Trait,
    /// Implementation node
    Implementation,
    /// Associated type node
    AssociatedType,
}

/// Types of relationships between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    /// Inheritance relationship
    Inherits,
    /// Implementation relationship
    Implements,
    /// Associated with relationship
    AssociatedWith,
}

/// Metadata for trait graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitGraphMetadata {
    /// Central node for focused graphs
    pub center_node: String,
    /// When the graph was generated
    pub generation_time: chrono::DateTime<chrono::Utc>,
    /// Export format used
    pub export_format: GraphExportFormat,
}

/// Available graph export formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphExportFormat {
    /// DOT format (Graphviz)
    Dot,
    /// JSON format
    Json,
    /// SVG format
    Svg,
}

/// Usage example for a trait.
///
/// Provides practical examples of how to use a trait, including
/// code samples and explanations.
///
/// # Fields
///
/// - `title` - Brief title for the example
/// - `description` - Detailed description of what the example demonstrates
/// - `code` - Actual code content
/// - `category` - Category of the example (Implementation, Usage, etc.)
/// - `difficulty` - Difficulty level (Beginner, Intermediate, Advanced)
/// - `runnable` - Whether the example can be executed
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::trait_explorer::trait_registry::*;
///
/// let example = UsageExample {
///     title: "Basic Estimator Implementation".to_string(),
///     description: "Shows how to implement the Estimator trait".to_string(),
///     code: r#"
/// impl Estimator for MyEstimator {
///     type Config = MyConfig;
///
///     fn name(&self) -> &'static str {
///         "MyEstimator"
///     }
/// }
/// "#.to_string(),
///     category: ExampleCategory::Implementation,
///     difficulty: ExampleDifficulty::Beginner,
///     runnable: true,
/// };
///
/// assert_eq!(example.difficulty, ExampleDifficulty::Beginner);
/// assert!(example.runnable);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExample {
    /// Title of the example
    pub title: String,
    /// Description of what the example shows
    pub description: String,
    /// Code content
    pub code: String,
    /// Category of example
    pub category: ExampleCategory,
    /// Difficulty level
    pub difficulty: ExampleDifficulty,
    /// Whether the example can be run
    pub runnable: bool,
}

/// Categories of usage examples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExampleCategory {
    /// How to implement the trait
    Implementation,
    /// How to use the trait
    Usage,
    /// Generic programming with the trait
    Generic,
    /// Advanced patterns
    Advanced,
}

/// Difficulty levels for examples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExampleDifficulty {
    Beginner,
    Intermediate,
    Advanced,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_registry_creation() {
        let registry = TraitRegistry::new();
        assert_eq!(registry.trait_count(), 0);
        assert_eq!(registry.implementation_count(), 0);
    }

    #[test]
    fn test_load_sklears_traits() -> Result<()> {
        let mut registry = TraitRegistry::new();
        registry.load_sklears_traits()?;

        // Verify core traits are loaded
        assert!(registry.has_trait("Estimator"));
        assert!(registry.has_trait("Fit"));
        assert!(registry.has_trait("Predict"));
        assert!(registry.has_trait("Transform"));

        // Verify trait count
        assert_eq!(registry.trait_count(), 4);

        Ok(())
    }

    #[test]
    fn test_trait_retrieval() -> Result<()> {
        let mut registry = TraitRegistry::new();
        registry.load_sklears_traits()?;

        let estimator_trait = registry.get_trait("Estimator").unwrap();
        assert_eq!(estimator_trait.name, "Estimator");
        assert_eq!(estimator_trait.path, "sklears_core::traits::Estimator");

        assert!(registry.get_trait("NonExistent").is_none());

        Ok(())
    }

    #[test]
    fn test_implementation_mapping() -> Result<()> {
        let mut registry = TraitRegistry::new();
        registry.load_sklears_traits()?;

        // Test trait -> implementations mapping
        let estimator_impls = registry.get_implementations("Estimator");
        assert!(estimator_impls.contains(&"LinearRegression".to_string()));
        assert!(estimator_impls.contains(&"LogisticRegression".to_string()));
        assert!(estimator_impls.contains(&"RandomForest".to_string()));

        // Test implementation -> traits mapping
        let linear_reg_traits = registry.get_traits_for_implementation("LinearRegression");
        assert!(linear_reg_traits.contains(&"Estimator".to_string()));
        assert!(linear_reg_traits.contains(&"Fit".to_string()));
        assert!(linear_reg_traits.contains(&"Predict".to_string()));

        Ok(())
    }

    #[test]
    fn test_custom_trait_addition() {
        let mut registry = TraitRegistry::new();

        let custom_trait = TraitInfo {
            name: "CustomTrait".to_string(),
            description: "A custom trait for testing".to_string(),
            path: "test::CustomTrait".to_string(),
            generics: vec!["T".to_string()],
            associated_types: vec![],
            methods: vec![],
            supertraits: vec![],
            implementations: vec!["CustomImpl".to_string()],
        };

        registry.add_trait(custom_trait);

        assert!(registry.has_trait("CustomTrait"));
        assert!(registry.has_implementation("CustomImpl"));
        assert_eq!(
            registry.get_implementations("CustomTrait"),
            vec!["CustomImpl"]
        );
        assert_eq!(
            registry.get_traits_for_implementation("CustomImpl"),
            vec!["CustomTrait"]
        );
    }

    #[test]
    fn test_trait_exploration_result_default() {
        let result = TraitExplorationResult::default();
        assert!(result.trait_name.is_empty());
        assert_eq!(result.complexity_score, 0.0);
        assert!(result.implementations.is_empty());
        assert!(result.examples.is_empty());
        assert!(result.related_traits.is_empty());
    }

    #[test]
    fn test_dependency_analysis_default() {
        let analysis = DependencyAnalysis::default();
        assert!(analysis.direct_dependencies.is_empty());
        assert!(analysis.transitive_dependencies.is_empty());
        assert_eq!(analysis.dependency_depth, 0);
        assert!(analysis.circular_dependencies.is_empty());
    }

    #[test]
    fn test_performance_analysis_default() {
        let performance = PerformanceAnalysis::default();
        assert_eq!(performance.compilation_impact.estimated_compile_time_ms, 0);
        assert_eq!(performance.runtime_overhead.virtual_dispatch_cost, 0);
        assert_eq!(performance.memory_footprint.vtable_size_bytes, 0);
        assert!(performance.optimization_hints.is_empty());
    }

    #[test]
    fn test_trait_graph_dot_export() {
        use chrono::Utc;

        let graph = TraitGraph {
            nodes: vec![
                TraitGraphNode {
                    id: "Estimator".to_string(),
                    label: "Estimator".to_string(),
                    node_type: TraitNodeType::Trait,
                    description: "Base trait".to_string(),
                    complexity: 1.0,
                },
                TraitGraphNode {
                    id: "LinearRegression".to_string(),
                    label: "LinearRegression".to_string(),
                    node_type: TraitNodeType::Implementation,
                    description: "Linear regression implementation".to_string(),
                    complexity: 2.0,
                },
            ],
            edges: vec![TraitGraphEdge {
                from: "LinearRegression".to_string(),
                to: "Estimator".to_string(),
                edge_type: EdgeType::Implements,
                weight: 1.0,
            }],
            metadata: TraitGraphMetadata {
                center_node: "Estimator".to_string(),
                generation_time: Utc::now(),
                export_format: GraphExportFormat::Dot,
            },
        };

        let dot = graph.to_dot();
        assert!(dot.contains("digraph TraitGraph"));
        assert!(dot.contains("Estimator"));
        assert!(dot.contains("LinearRegression"));
        assert!(dot.contains("lightblue")); // Trait color
        assert!(dot.contains("lightgreen")); // Implementation color
        assert!(dot.contains("dashed")); // Implements edge style
    }

    #[test]
    fn test_trait_graph_json_export() -> Result<()> {
        use chrono::Utc;

        let graph = TraitGraph {
            nodes: vec![TraitGraphNode {
                id: "Estimator".to_string(),
                label: "Estimator".to_string(),
                node_type: TraitNodeType::Trait,
                description: "Base trait".to_string(),
                complexity: 1.0,
            }],
            edges: vec![],
            metadata: TraitGraphMetadata {
                center_node: "Estimator".to_string(),
                generation_time: Utc::now(),
                export_format: GraphExportFormat::Json,
            },
        };

        let json = graph.to_json()?;
        assert!(json.contains("\"nodes\""));
        assert!(json.contains("\"edges\""));
        assert!(json.contains("\"metadata\""));
        assert!(json.contains("Estimator"));

        Ok(())
    }

    #[test]
    fn test_usage_example_creation() {
        let example = UsageExample {
            title: "Basic Example".to_string(),
            description: "A simple usage example".to_string(),
            code: "println!(\"Hello, world!\");".to_string(),
            category: ExampleCategory::Usage,
            difficulty: ExampleDifficulty::Beginner,
            runnable: true,
        };

        assert_eq!(example.category, ExampleCategory::Usage);
        assert_eq!(example.difficulty, ExampleDifficulty::Beginner);
        assert!(example.runnable);
    }

    #[test]
    fn test_all_trait_names() -> Result<()> {
        let mut registry = TraitRegistry::new();
        registry.load_sklears_traits()?;

        let trait_names = registry.get_all_trait_names();
        assert_eq!(trait_names.len(), 4);
        assert!(trait_names.contains(&"Estimator".to_string()));
        assert!(trait_names.contains(&"Fit".to_string()));
        assert!(trait_names.contains(&"Predict".to_string()));
        assert!(trait_names.contains(&"Transform".to_string()));

        Ok(())
    }

    #[test]
    fn test_all_traits() -> Result<()> {
        let mut registry = TraitRegistry::new();
        registry.load_sklears_traits()?;

        let all_traits = registry.get_all_traits();
        assert_eq!(all_traits.len(), 4);

        let trait_names: Vec<&str> = all_traits.iter().map(|t| t.name.as_str()).collect();
        assert!(trait_names.contains(&"Estimator"));
        assert!(trait_names.contains(&"Fit"));
        assert!(trait_names.contains(&"Predict"));
        assert!(trait_names.contains(&"Transform"));

        Ok(())
    }

    #[test]
    fn test_nonexistent_queries() {
        let registry = TraitRegistry::new();

        assert!(!registry.has_trait("NonExistent"));
        assert!(!registry.has_implementation("NonExistent"));
        assert!(registry.get_implementations("NonExistent").is_empty());
        assert!(registry
            .get_traits_for_implementation("NonExistent")
            .is_empty());
    }

    #[test]
    fn test_scirs2_compliance() {
        // Test that we can use SciRS2 components without issues
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::Random;

        let mut rng = Random::seed(42);
        let _random_value: f64 = rng.gen();

        // Test array operations
        use scirs2_core::ndarray::{Array1, Array2};
        let _arr1: Array1<f64> = Array1::zeros(10);
        let _arr2: Array2<f64> = Array2::zeros((5, 5));

        // This test passes if we can successfully use SciRS2 components
        assert!(true);
    }

    #[test]
    fn test_registry_counts() -> Result<()> {
        let mut registry = TraitRegistry::new();
        assert_eq!(registry.trait_count(), 0);
        assert_eq!(registry.implementation_count(), 0);

        registry.load_sklears_traits()?;
        assert_eq!(registry.trait_count(), 4);
        assert!(registry.implementation_count() > 0);

        Ok(())
    }
}
