//! Trait-based manifold learning framework
//!
//! This module provides a unified trait system for manifold learning algorithms,
//! enabling composable and extensible manifold learning pipelines.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
/// Core trait for manifold learning algorithms
use sklears_core::error::Result as SklResult;
use std::collections::HashMap;
pub trait ManifoldLearning {
    /// Get the intrinsic dimensionality of the learned manifold
    fn intrinsic_dimension(&self) -> Option<usize>;

    /// Get the embedding dimension
    fn embedding_dimension(&self) -> usize;

    /// Get algorithm-specific parameters
    fn parameters(&self) -> HashMap<String, f64>;

    /// Get algorithm name
    fn algorithm_name(&self) -> &'static str;

    /// Check if the algorithm supports out-of-sample extension
    fn supports_transform(&self) -> bool;

    /// Get the computational complexity
    fn complexity(&self) -> ManifoldComplexity;
}

/// Trait for algorithms that support distance metrics
pub trait DistanceMetric {
    /// Compute pairwise distances between points
    fn pairwise_distances(
        &self,
        x: ArrayView2<f64>,
        y: Option<ArrayView2<f64>>,
    ) -> SklResult<Array2<f64>>;

    /// Get the name of the distance metric
    fn metric_name(&self) -> &'static str;

    /// Check if the metric satisfies the triangle inequality
    fn is_metric(&self) -> bool;
}

/// Trait for neighborhood-based algorithms
pub trait NeighborhoodBased {
    /// Get the number of neighbors used
    fn n_neighbors(&self) -> usize;

    /// Set the number of neighbors
    fn set_n_neighbors(&mut self, n_neighbors: usize);

    /// Get the neighborhood connectivity matrix
    fn neighborhood_graph(&self) -> Option<Array2<f64>>;
}

/// Trait for algorithms with random initialization
pub trait RandomizedAlgorithm {
    /// Get the random state
    fn random_state(&self) -> Option<u64>;

    /// Set the random state for reproducibility
    fn set_random_state(&mut self, random_state: Option<u64>);
}

/// Trait for iterative optimization algorithms
pub trait IterativeOptimization {
    /// Get the number of iterations performed
    fn n_iter(&self) -> usize;

    /// Get the maximum number of iterations
    fn max_iter(&self) -> usize;

    /// Set the maximum number of iterations
    fn set_max_iter(&mut self, max_iter: usize);

    /// Get the convergence tolerance
    fn tolerance(&self) -> f64;

    /// Set the convergence tolerance
    fn set_tolerance(&mut self, tolerance: f64);

    /// Get the optimization history
    fn optimization_history(&self) -> Option<Vec<f64>>;
}

/// Trait for spectral embedding algorithms
pub trait SpectralEmbedding {
    /// Get the eigenvalues from the spectral decomposition
    fn eigenvalues(&self) -> Option<Array1<f64>>;

    /// Get the eigenvectors from the spectral decomposition
    fn eigenvectors(&self) -> Option<Array2<f64>>;

    /// Get the number of components to keep
    fn n_components(&self) -> usize;

    /// Set the number of components to keep
    fn set_n_components(&mut self, n_components: usize);
}

/// Trait for kernel-based methods
pub trait KernelBased {
    /// Get the kernel matrix
    fn kernel_matrix(&self) -> Option<Array2<f64>>;

    /// Get the kernel parameters
    fn kernel_params(&self) -> HashMap<String, f64>;

    /// Compute kernel values between new points and training data
    fn kernel_transform(&self, x: ArrayView2<f64>) -> SklResult<Array2<f64>>;
}

/// Trait for probabilistic manifold learning
pub trait ProbabilisticEmbedding {
    /// Get the joint probabilities in high-dimensional space
    fn high_dim_probabilities(&self) -> Option<Array2<f64>>;

    /// Get the joint probabilities in low-dimensional space
    fn low_dim_probabilities(&self) -> Option<Array2<f64>>;

    /// Get the perplexity or equivalent parameter
    fn perplexity(&self) -> f64;

    /// Set the perplexity
    fn set_perplexity(&mut self, perplexity: f64);
}

/// Trait for manifold learning quality assessment
pub trait EmbeddingQuality {
    /// Compute trustworthiness of the embedding
    fn trustworthiness(
        &self,
        x: ArrayView2<f64>,
        x_embedded: ArrayView2<f64>,
        k: usize,
    ) -> SklResult<f64>;

    /// Compute continuity of the embedding
    fn continuity(
        &self,
        x: ArrayView2<f64>,
        x_embedded: ArrayView2<f64>,
        k: usize,
    ) -> SklResult<f64>;

    /// Compute neighborhood hit rate
    fn neighborhood_hit_rate(
        &self,
        x: ArrayView2<f64>,
        x_embedded: ArrayView2<f64>,
        k: usize,
    ) -> SklResult<f64>;

    /// Compute reconstruction error
    fn reconstruction_error(
        &self,
        x: ArrayView2<f64>,
        x_embedded: ArrayView2<f64>,
    ) -> SklResult<f64>;
}

/// Computational complexity of manifold learning algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldComplexity {
    /// Linear complexity O(n)
    Linear,
    /// Quadratic complexity O(n²)
    Quadratic,
    /// Cubic complexity O(n³)
    Cubic,
    /// Log-linear complexity O(n log n)
    LogLinear,
    /// Custom complexity with description
    Custom(String),
}

impl ManifoldComplexity {
    /// Get a human-readable description of the complexity
    pub fn description(&self) -> &str {
        match self {
            ManifoldComplexity::Linear => "O(n) - Linear time complexity",
            ManifoldComplexity::Quadratic => "O(n²) - Quadratic time complexity",
            ManifoldComplexity::Cubic => "O(n³) - Cubic time complexity",
            ManifoldComplexity::LogLinear => "O(n log n) - Log-linear time complexity",
            ManifoldComplexity::Custom(desc) => desc,
        }
    }
}

/// Configuration builder for manifold learning algorithms
#[derive(Debug, Clone)]
pub struct ManifoldConfig {
    /// Number of components in the embedding
    pub n_components: usize,
    /// Number of neighbors for neighborhood-based algorithms
    pub n_neighbors: Option<usize>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Maximum number of iterations
    pub max_iter: Option<usize>,
    /// Convergence tolerance
    pub tolerance: Option<f64>,
    /// Distance metric name
    pub metric: Option<String>,
    /// Additional algorithm-specific parameters
    pub params: HashMap<String, f64>,
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: None,
            random_state: None,
            max_iter: None,
            tolerance: None,
            metric: None,
            params: HashMap::new(),
        }
    }
}

impl ManifoldConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = Some(n_neighbors);
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: impl Into<String>) -> Self {
        self.metric = Some(metric.into());
        self
    }

    /// Add a custom parameter
    pub fn param(mut self, key: impl Into<String>, value: f64) -> Self {
        self.params.insert(key.into(), value);
        self
    }
}

/// Preset configurations for common use cases
pub struct ManifoldPresets;

impl ManifoldPresets {
    /// Configuration for fast visualization (t-SNE with reduced iterations)
    pub fn fast_visualization() -> ManifoldConfig {
        ManifoldConfig::new()
            .n_components(2)
            .max_iter(250)
            .param("perplexity", 30.0)
            .param("learning_rate", 200.0)
    }

    /// Configuration for high-quality visualization (t-SNE with more iterations)
    pub fn high_quality_visualization() -> ManifoldConfig {
        ManifoldConfig::new()
            .n_components(2)
            .max_iter(1000)
            .param("perplexity", 30.0)
            .param("learning_rate", 200.0)
    }

    /// Configuration for clustering preprocessing (PCA-like)
    pub fn clustering_preprocessing() -> ManifoldConfig {
        ManifoldConfig::new()
            .n_components(50)
            .metric("euclidean".to_string())
    }

    /// Configuration for nonlinear dimensionality reduction (Isomap)
    pub fn nonlinear_reduction() -> ManifoldConfig {
        ManifoldConfig::new()
            .n_neighbors(12)
            .n_components(10)
            .metric("euclidean".to_string())
    }

    /// Configuration for local structure preservation (LLE)
    pub fn local_structure() -> ManifoldConfig {
        ManifoldConfig::new().n_neighbors(12).n_components(2)
    }

    /// Configuration for global structure preservation (MDS)
    pub fn global_structure() -> ManifoldConfig {
        ManifoldConfig::new()
            .n_components(2)
            .metric("euclidean".to_string())
            .max_iter(300)
    }
}

/// Factory trait for creating manifold learning algorithms
pub trait ManifoldFactory {
    /// The type of manifold algorithm this factory creates
    type Algorithm: ManifoldLearning;

    /// Create a new instance with default configuration
    fn default() -> Self::Algorithm;

    /// Create a new instance with custom configuration
    fn with_config(config: ManifoldConfig) -> Self::Algorithm;

    /// Create a new instance with a preset configuration
    fn with_preset(preset: fn() -> ManifoldConfig) -> Self::Algorithm {
        Self::with_config(preset())
    }
}

/// Pipeline for composing multiple manifold learning steps
/// This is a simplified version that stores step names and configurations
/// rather than trait objects for now
#[derive(Debug, Clone)]
pub struct ManifoldPipeline {
    step_configs: Vec<(String, ManifoldConfig)>,
}

impl Default for ManifoldPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl ManifoldPipeline {
    /// Create a new empty pipeline
    pub fn new() -> Self {
        Self {
            step_configs: Vec::new(),
        }
    }

    /// Add a step to the pipeline
    pub fn add_step(mut self, name: impl Into<String>, config: ManifoldConfig) -> Self {
        self.step_configs.push((name.into(), config));
        self
    }

    /// Get the number of steps in the pipeline
    pub fn len(&self) -> usize {
        self.step_configs.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.step_configs.is_empty()
    }

    /// Get the names of all steps
    pub fn step_names(&self) -> Vec<&str> {
        self.step_configs
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get all step configurations
    pub fn step_configs(&self) -> &[(String, ManifoldConfig)] {
        &self.step_configs
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifold_config_builder() {
        let config = ManifoldConfig::new()
            .n_components(3)
            .n_neighbors(10)
            .random_state(42)
            .max_iter(500)
            .tolerance(1e-6)
            .metric("euclidean")
            .param("perplexity", 50.0);

        assert_eq!(config.n_components, 3);
        assert_eq!(config.n_neighbors, Some(10));
        assert_eq!(config.random_state, Some(42));
        assert_eq!(config.max_iter, Some(500));
        assert_eq!(config.tolerance, Some(1e-6));
        assert_eq!(config.metric, Some("euclidean".to_string()));
        assert_eq!(config.params.get("perplexity"), Some(&50.0));
    }

    #[test]
    fn test_manifold_presets() {
        let fast_config = ManifoldPresets::fast_visualization();
        assert_eq!(fast_config.n_components, 2);
        assert_eq!(fast_config.max_iter, Some(250));

        let quality_config = ManifoldPresets::high_quality_visualization();
        assert_eq!(quality_config.n_components, 2);
        assert_eq!(quality_config.max_iter, Some(1000));

        let clustering_config = ManifoldPresets::clustering_preprocessing();
        assert_eq!(clustering_config.n_components, 50);

        let nonlinear_config = ManifoldPresets::nonlinear_reduction();
        assert_eq!(nonlinear_config.n_neighbors, Some(12));
        assert_eq!(nonlinear_config.n_components, 10);

        let local_config = ManifoldPresets::local_structure();
        assert_eq!(local_config.n_neighbors, Some(12));
        assert_eq!(local_config.n_components, 2);

        let global_config = ManifoldPresets::global_structure();
        assert_eq!(global_config.n_components, 2);
        assert_eq!(global_config.max_iter, Some(300));
    }

    #[test]
    fn test_manifold_complexity() {
        let linear = ManifoldComplexity::Linear;
        assert_eq!(linear.description(), "O(n) - Linear time complexity");

        let quadratic = ManifoldComplexity::Quadratic;
        assert_eq!(quadratic.description(), "O(n²) - Quadratic time complexity");

        let custom = ManifoldComplexity::Custom("O(n^1.5)".to_string());
        assert_eq!(custom.description(), "O(n^1.5)");
    }

    #[test]
    fn test_manifold_pipeline() {
        let pipeline = ManifoldPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);

        let config1 = ManifoldConfig::new().n_components(2);
        let config2 = ManifoldConfig::new().n_components(10);

        let pipeline = pipeline.add_step("tsne", config1).add_step("pca", config2);

        assert!(!pipeline.is_empty());
        assert_eq!(pipeline.len(), 2);
        assert_eq!(pipeline.step_names(), vec!["tsne", "pca"]);
        assert_eq!(pipeline.step_configs().len(), 2);
    }
}
