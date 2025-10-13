//! Fluent API for manifold learning configuration and execution
//!
//! This module provides a fluent, chainable API for configuring and executing
//! manifold learning algorithms, making it easier to work with complex parameter
//! combinations and preprocessing pipelines.

use crate::{ManifoldConfig, ManifoldPresets};

/// Fluent API builder for manifold learning
use sklears_core::traits::Estimator;
use std::collections::HashMap;
#[derive(Debug, Clone)]
pub struct ManifoldBuilder {
    algorithm: Option<String>,
    config: ManifoldConfig,
    preprocessing: Vec<PreprocessingStep>,
    validation: Option<ValidationConfig>,
    export_config: Option<ExportConfig>,
}

/// Preprocessing step configuration
#[derive(Debug, Clone)]
pub enum PreprocessingStep {
    /// Standardize features (zero mean, unit variance)
    Standardize,
    /// Normalize features to [0, 1] range
    MinMaxScale,
    /// Center features (subtract mean)
    Center,
    /// Apply PCA for initial dimensionality reduction
    PCA { n_components: usize },
    /// Remove features with low variance
    VarianceThreshold { threshold: f64 },
    /// Custom preprocessing function
    Custom { name: String, description: String },
}

/// Validation configuration for hyperparameter tuning
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Number of folds for cross-validation
    pub n_folds: usize,
    /// Metrics to compute during validation
    pub metrics: Vec<String>,
    /// Parameter grid for grid search
    pub param_grid: HashMap<String, Vec<f64>>,
    /// Whether to use random search instead of grid search
    pub random_search: bool,
    /// Number of random samples for random search
    pub n_random_samples: Option<usize>,
}

/// Export configuration for results
#[derive(Debug, Clone, Default)]
pub struct ExportConfig {
    pub csv: Option<String>,
    pub json: Option<String>,
    pub params: bool,
    pub metrics: bool,
}

impl Default for ManifoldBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ManifoldBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            algorithm: None,
            config: ManifoldConfig::new(),
            preprocessing: Vec::new(),
            validation: None,
            export_config: None,
        }
    }

    /// Start with a specific algorithm
    pub fn algorithm(mut self, algorithm: impl Into<String>) -> Self {
        self.algorithm = Some(algorithm.into());
        self
    }

    /// Use t-SNE algorithm
    pub fn tsne(self) -> TSNEBuilder {
        TSNEBuilder::from_builder(self)
    }

    /// Use UMAP algorithm
    pub fn umap(self) -> UMAPBuilder {
        UMAPBuilder::from_builder(self)
    }

    /// Use Isomap algorithm
    pub fn isomap(self) -> IsomapBuilder {
        IsomapBuilder::from_builder(self)
    }

    /// Use PCA algorithm
    pub fn pca(self) -> PCABuilder {
        PCABuilder::from_builder(self)
    }

    /// Use a preset configuration
    pub fn preset(mut self, preset: fn() -> ManifoldConfig) -> Self {
        self.config = preset();
        self
    }

    /// Use fast visualization preset
    pub fn fast_visualization(self) -> Self {
        self.preset(ManifoldPresets::fast_visualization)
    }

    /// Use high-quality visualization preset
    pub fn high_quality_visualization(self) -> Self {
        self.preset(ManifoldPresets::high_quality_visualization)
    }

    /// Use clustering preprocessing preset
    pub fn clustering_preprocessing(self) -> Self {
        self.preset(ManifoldPresets::clustering_preprocessing)
    }

    /// Use nonlinear reduction preset
    pub fn nonlinear_reduction(self) -> Self {
        self.preset(ManifoldPresets::nonlinear_reduction)
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config = self.config.n_components(n_components);
        self
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.config = self.config.n_neighbors(n_neighbors);
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config = self.config.random_state(random_state);
        self
    }

    /// Set the maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config = self.config.max_iter(max_iter);
        self
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.config = self.config.tolerance(tolerance);
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: impl Into<String>) -> Self {
        self.config = self.config.metric(metric);
        self
    }

    /// Add a custom parameter
    pub fn param(mut self, key: impl Into<String>, value: f64) -> Self {
        self.config = self.config.param(key, value);
        self
    }

    /// Add preprocessing step
    pub fn preprocess(mut self, step: PreprocessingStep) -> Self {
        self.preprocessing.push(step);
        self
    }

    /// Add standardization preprocessing
    pub fn standardize(self) -> Self {
        self.preprocess(PreprocessingStep::Standardize)
    }

    /// Add min-max scaling preprocessing
    pub fn min_max_scale(self) -> Self {
        self.preprocess(PreprocessingStep::MinMaxScale)
    }

    /// Add centering preprocessing
    pub fn center(self) -> Self {
        self.preprocess(PreprocessingStep::Center)
    }

    /// Add PCA preprocessing
    pub fn pca_preprocess(self, n_components: usize) -> Self {
        self.preprocess(PreprocessingStep::PCA { n_components })
    }

    /// Add variance threshold preprocessing
    pub fn variance_threshold(self, threshold: f64) -> Self {
        self.preprocess(PreprocessingStep::VarianceThreshold { threshold })
    }

    /// Configure validation
    pub fn validation(mut self, config: ValidationConfig) -> Self {
        self.validation = Some(config);
        self
    }

    /// Configure cross-validation
    pub fn cross_validate(mut self, n_folds: usize) -> Self {
        let config = ValidationConfig {
            n_folds,
            metrics: vec!["trustworthiness".to_string(), "continuity".to_string()],
            param_grid: HashMap::new(),
            random_search: false,
            n_random_samples: None,
        };
        self.validation = Some(config);
        self
    }

    /// Configure export options
    pub fn export(mut self, config: ExportConfig) -> Self {
        self.export_config = Some(config);
        self
    }

    /// Export to CSV
    pub fn to_csv(mut self, path: impl Into<String>) -> Self {
        let mut config = self.export_config.unwrap_or_default();
        config.csv = Some(path.into());
        self.export_config = Some(config);
        self
    }

    /// Export to JSON
    pub fn to_json(mut self, path: impl Into<String>) -> Self {
        let mut config = self.export_config.unwrap_or_default();
        config.json = Some(path.into());
        self.export_config = Some(config);
        self
    }

    /// Include parameters in export
    pub fn include_params(mut self) -> Self {
        let mut config = self.export_config.unwrap_or_default();
        config.params = true;
        self.export_config = Some(config);
        self
    }

    /// Include metrics in export
    pub fn include_metrics(mut self) -> Self {
        let mut config = self.export_config.unwrap_or_default();
        config.metrics = true;
        self.export_config = Some(config);
        self
    }

    /// Get the configuration
    pub fn config(&self) -> &ManifoldConfig {
        &self.config
    }

    /// Get the preprocessing steps
    pub fn preprocessing_steps(&self) -> &[PreprocessingStep] {
        &self.preprocessing
    }

    /// Get the validation configuration
    pub fn validation_config(&self) -> Option<&ValidationConfig> {
        self.validation.as_ref()
    }

    /// Get the export configuration
    pub fn export_config(&self) -> Option<&ExportConfig> {
        self.export_config.as_ref()
    }
}

/// Algorithm-specific builder for t-SNE
#[derive(Debug, Clone)]
pub struct TSNEBuilder {
    base: ManifoldBuilder,
}

impl TSNEBuilder {
    fn from_builder(builder: ManifoldBuilder) -> Self {
        let mut base = builder;
        base.algorithm = Some("tsne".to_string());
        Self { base }
    }

    /// Set perplexity parameter
    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.base.config = self.base.config.param("perplexity", perplexity);
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.base.config = self.base.config.param("learning_rate", learning_rate);
        self
    }

    /// Set early exaggeration
    pub fn early_exaggeration(mut self, early_exaggeration: f64) -> Self {
        self.base.config = self
            .base
            .config
            .param("early_exaggeration", early_exaggeration);
        self
    }

    /// Use Barnes-Hut approximation
    pub fn barnes_hut(mut self, angle: f64) -> Self {
        self.base.config = self.base.config.param("angle", angle);
        self
    }

    // Forward common methods
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.base = self.base.n_components(n_components);
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.base = self.base.random_state(random_state);
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.base = self.base.max_iter(max_iter);
        self
    }

    pub fn standardize(mut self) -> Self {
        self.base = self.base.standardize();
        self
    }

    pub fn to_csv(mut self, path: impl Into<String>) -> Self {
        self.base = self.base.to_csv(path);
        self
    }

    /// Get the underlying builder
    pub fn builder(&self) -> &ManifoldBuilder {
        &self.base
    }
}

/// Algorithm-specific builder for UMAP
#[derive(Debug, Clone)]
pub struct UMAPBuilder {
    base: ManifoldBuilder,
}

impl UMAPBuilder {
    fn from_builder(builder: ManifoldBuilder) -> Self {
        let mut base = builder;
        base.algorithm = Some("umap".to_string());
        Self { base }
    }

    /// Set minimum distance parameter
    pub fn min_dist(mut self, min_dist: f64) -> Self {
        self.base.config = self.base.config.param("min_dist", min_dist);
        self
    }

    /// Set spread parameter
    pub fn spread(mut self, spread: f64) -> Self {
        self.base.config = self.base.config.param("spread", spread);
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.base.config = self.base.config.param("learning_rate", learning_rate);
        self
    }

    // Forward common methods
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.base = self.base.n_components(n_components);
        self
    }

    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.base = self.base.n_neighbors(n_neighbors);
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.base = self.base.random_state(random_state);
        self
    }

    pub fn standardize(mut self) -> Self {
        self.base = self.base.standardize();
        self
    }

    pub fn to_csv(mut self, path: impl Into<String>) -> Self {
        self.base = self.base.to_csv(path);
        self
    }

    /// Get the underlying builder
    pub fn builder(&self) -> &ManifoldBuilder {
        &self.base
    }
}

/// Algorithm-specific builder for Isomap
#[derive(Debug, Clone)]
pub struct IsomapBuilder {
    base: ManifoldBuilder,
}

impl IsomapBuilder {
    fn from_builder(builder: ManifoldBuilder) -> Self {
        let mut base = builder;
        base.algorithm = Some("isomap".to_string());
        Self { base }
    }

    /// Set path method (geodesic distance computation)
    pub fn path_method(mut self, method: impl Into<String>) -> Self {
        self.base.config = self.base.config.param(
            "path_method",
            match method.into().as_str() {
                "auto" => 0.0,
                "FW" => 1.0,
                "D" => 2.0,
                _ => 0.0,
            },
        );
        self
    }

    // Forward common methods
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.base = self.base.n_components(n_components);
        self
    }

    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.base = self.base.n_neighbors(n_neighbors);
        self
    }

    pub fn metric(mut self, metric: impl Into<String>) -> Self {
        self.base = self.base.metric(metric);
        self
    }

    pub fn standardize(mut self) -> Self {
        self.base = self.base.standardize();
        self
    }

    pub fn to_csv(mut self, path: impl Into<String>) -> Self {
        self.base = self.base.to_csv(path);
        self
    }

    /// Get the underlying builder
    pub fn builder(&self) -> &ManifoldBuilder {
        &self.base
    }
}

/// Algorithm-specific builder for PCA
#[derive(Debug, Clone)]
pub struct PCABuilder {
    base: ManifoldBuilder,
}

impl PCABuilder {
    fn from_builder(builder: ManifoldBuilder) -> Self {
        let mut base = builder;
        base.algorithm = Some("pca".to_string());
        Self { base }
    }

    /// Set whether to whiten the components
    pub fn whiten(mut self, whiten: bool) -> Self {
        self.base.config = self
            .base
            .config
            .param("whiten", if whiten { 1.0 } else { 0.0 });
        self
    }

    /// Set the SVD solver
    pub fn svd_solver(mut self, solver: impl Into<String>) -> Self {
        self.base.config = self.base.config.param(
            "svd_solver",
            match solver.into().as_str() {
                "auto" => 0.0,
                "full" => 1.0,
                "arpack" => 2.0,
                "randomized" => 3.0,
                _ => 0.0,
            },
        );
        self
    }

    // Forward common methods
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.base = self.base.n_components(n_components);
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.base = self.base.random_state(random_state);
        self
    }

    pub fn standardize(mut self) -> Self {
        self.base = self.base.standardize();
        self
    }

    pub fn to_csv(mut self, path: impl Into<String>) -> Self {
        self.base = self.base.to_csv(path);
        self
    }

    /// Get the underlying builder
    pub fn builder(&self) -> &ManifoldBuilder {
        &self.base
    }
}

/// Convenience functions for quick setup
pub mod quick {
    use super::*;

    /// Quick t-SNE setup for visualization
    pub fn tsne_viz() -> TSNEBuilder {
        ManifoldBuilder::new()
            .fast_visualization()
            .tsne()
            .standardize()
    }

    /// Quick UMAP setup for visualization
    pub fn umap_viz() -> UMAPBuilder {
        ManifoldBuilder::new()
            .n_components(2)
            .umap()
            .n_neighbors(15)
            .min_dist(0.1)
            .standardize()
    }

    /// Quick Isomap setup for nonlinear reduction
    pub fn isomap_reduction() -> IsomapBuilder {
        ManifoldBuilder::new()
            .nonlinear_reduction()
            .isomap()
            .standardize()
    }

    /// Quick PCA setup for linear reduction
    pub fn pca_reduction() -> PCABuilder {
        ManifoldBuilder::new()
            .clustering_preprocessing()
            .pca()
            .standardize()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifold_builder_basic() {
        let builder = ManifoldBuilder::new()
            .n_components(3)
            .n_neighbors(10)
            .random_state(42)
            .max_iter(500)
            .tolerance(1e-6)
            .metric("euclidean")
            .param("perplexity", 50.0);

        assert_eq!(builder.config.n_components, 3);
        assert_eq!(builder.config.n_neighbors, Some(10));
        assert_eq!(builder.config.random_state, Some(42));
        assert_eq!(builder.config.max_iter, Some(500));
        assert_eq!(builder.config.tolerance, Some(1e-6));
        assert_eq!(builder.config.metric, Some("euclidean".to_string()));
        assert_eq!(builder.config.params.get("perplexity"), Some(&50.0));
    }

    #[test]
    fn test_tsne_builder() {
        let tsne = ManifoldBuilder::new()
            .tsne()
            .n_components(2)
            .perplexity(30.0)
            .learning_rate(200.0)
            .early_exaggeration(12.0)
            .barnes_hut(0.5)
            .standardize();

        assert_eq!(tsne.builder().algorithm, Some("tsne".to_string()));
        assert_eq!(tsne.builder().config.n_components, 2);
        assert_eq!(tsne.builder().config.params.get("perplexity"), Some(&30.0));
        assert_eq!(
            tsne.builder().config.params.get("learning_rate"),
            Some(&200.0)
        );
        assert_eq!(tsne.builder().preprocessing.len(), 1);
    }

    #[test]
    fn test_umap_builder() {
        let umap = ManifoldBuilder::new()
            .umap()
            .n_components(2)
            .n_neighbors(15)
            .min_dist(0.1)
            .spread(1.0)
            .learning_rate(1.0)
            .standardize();

        assert_eq!(umap.builder().algorithm, Some("umap".to_string()));
        assert_eq!(umap.builder().config.n_components, 2);
        assert_eq!(umap.builder().config.n_neighbors, Some(15));
        assert_eq!(umap.builder().config.params.get("min_dist"), Some(&0.1));
        assert_eq!(umap.builder().preprocessing.len(), 1);
    }

    #[test]
    fn test_presets() {
        let fast = ManifoldBuilder::new().fast_visualization();
        assert_eq!(fast.config.n_components, 2);
        assert_eq!(fast.config.max_iter, Some(250));

        let quality = ManifoldBuilder::new().high_quality_visualization();
        assert_eq!(quality.config.n_components, 2);
        assert_eq!(quality.config.max_iter, Some(1000));

        let clustering = ManifoldBuilder::new().clustering_preprocessing();
        assert_eq!(clustering.config.n_components, 50);

        let nonlinear = ManifoldBuilder::new().nonlinear_reduction();
        assert_eq!(nonlinear.config.n_neighbors, Some(12));
        assert_eq!(nonlinear.config.n_components, 10);
    }

    #[test]
    fn test_preprocessing() {
        let builder = ManifoldBuilder::new()
            .standardize()
            .min_max_scale()
            .center()
            .pca_preprocess(50)
            .variance_threshold(0.01);

        assert_eq!(builder.preprocessing.len(), 5);

        match &builder.preprocessing[0] {
            PreprocessingStep::Standardize => (),
            _ => panic!("Expected Standardize"),
        }

        match &builder.preprocessing[3] {
            PreprocessingStep::PCA { n_components } => assert_eq!(*n_components, 50),
            _ => panic!("Expected PCA"),
        }

        match &builder.preprocessing[4] {
            PreprocessingStep::VarianceThreshold { threshold } => assert_eq!(*threshold, 0.01),
            _ => panic!("Expected VarianceThreshold"),
        }
    }

    #[test]
    fn test_export_config() {
        let builder = ManifoldBuilder::new()
            .to_csv("embedding.csv")
            .to_json("embedding.json")
            .include_params()
            .include_metrics();

        let export = builder.export_config().unwrap();
        assert_eq!(export.csv, Some("embedding.csv".to_string()));
        assert_eq!(export.json, Some("embedding.json".to_string()));
        assert!(export.params);
        assert!(export.metrics);
    }

    #[test]
    fn test_quick_builders() {
        let tsne = quick::tsne_viz();
        assert_eq!(tsne.builder().algorithm, Some("tsne".to_string()));
        assert_eq!(tsne.builder().preprocessing.len(), 1);

        let umap = quick::umap_viz();
        assert_eq!(umap.builder().algorithm, Some("umap".to_string()));
        assert_eq!(umap.builder().config.n_components, 2);

        let isomap = quick::isomap_reduction();
        assert_eq!(isomap.builder().algorithm, Some("isomap".to_string()));

        let pca = quick::pca_reduction();
        assert_eq!(pca.builder().algorithm, Some("pca".to_string()));
    }
}
